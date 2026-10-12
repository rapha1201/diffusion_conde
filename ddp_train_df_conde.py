import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from conde import ConDeTrainer, AELoader
from conde.models import ConDe
from conde.data import ResidualDeepfault, SeismicAugmentation
from conde.losses import TripletLoss, CombinedLoss, HSIC_RBF_Multi, GradientLoss


SAVE_DIR = "conde_res/df_conde"
SAVE_CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")
EPOCHS = 251


def save_ckpt(save_dir, epoch, conde_model, decoder, ae):
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt")
    latest_path = os.path.join(save_dir, "ckpt_latest.pt")

    payload = {
        "epoch": epoch,
        "conde_state_dict": conde_model.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "ae_state_dict": ae.state_dict(),
    }

    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)

    print(f"[SAVE] {ckpt_path}")
    print(f"[SAVE] {latest_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_CKPT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check the GPU allocation, NVIDIA driver, container, and PyTorch CUDA installation.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    loader = AELoader("../configs/small_ae.yaml", model_only=True, load=False)
    ae = loader.model

    dataset = ResidualDeepfault("/pgeoprj/godeep/ej49/ConDe/dataset/WuSet.h5", split="train")
    test_dataset = ResidualDeepfault("/pgeoprj/godeep/ej49/ConDe/dataset/WuSet.h5", split="test")

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True, num_workers=2, persistent_workers=True)

    semantic_info = {
        "structure": {"head_index": 0},
    }

    conde_model = ConDe(latent_dim=1024, n_semantic_heads=1, encoder=ae.model.encoder, normalize=False).to(device)
    decoder = ae.model.decoder.to(device)

    contrastive_loss = TripletLoss()
    residual_loss = CombinedLoss(losses=[HSIC_RBF_Multi()])
    decoder_loss_fn = CombinedLoss(losses=[nn.L1Loss(), GradientLoss()], weights=[1, 0.1])
    coverage_loss_fn = nn.MSELoss()

    trainer = ConDeTrainer(conde_model, extract_embs=True, decoder_loss_fn=decoder_loss_fn, contrastive_loss_fn=contrastive_loss, residual_loss_fn=residual_loss, coverage_loss_fn=coverage_loss_fn, rgt_loss_fn=None, decoder=decoder, semantic_info=semantic_info, total_steps=len(dataloader) * EPOCHS, lambda_contrast=1, lambda_coverage=1, lambda_residual=1, lambda_decoder=1, device=device, decoder_warmup=0, coverage_mode="inverted_detach", decomp_mode="inverted_detach", lambda_decomp=1, require_labels=False)

    augment = SeismicAugmentation(
        aug_p=0.5,
        init_blur_sigma_range=(1.2, 2.5),
        p_init_blur=1,
        noise_phase_sigma_range=(0.2, 0.4),
        p_noise_phase=0.3,
        noise_amp_sigma_range=(0.5, 0.8),
        p_noise_amp=0.75,
        blur_amp_sigma_range=(0.8, 1.5),
        p_blur_amp=0.55,
        freq_low_energy_remove_range=(0.0, 0.35),
        freq_high_energy_remove_range=(0.0, 0.35),
        max_freq_energy_remove=0.5,
        p_freq_filter=0.7,
        p_hflip=0,
        zoom_range=(1.0, 1.6),
        p_zoom=0,
        p_aniso=0,
        blur_kernel_size=15,
    ).to(device)

    for epoch in range(EPOCHS):
        print(f"[EPOCH {epoch}/{EPOCHS}]")

        conde_model.train()
        decoder.train()

        for batch in tqdm(dataloader):
            x1, x2 = batch

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            pair = x1.clone()
            x1, x2, x1p = augment(x1, pair, x2, residual=True)

            pos_idx = torch.zeros((x1.shape[0], 1), dtype=torch.long, device=device)
            structure_label = torch.ones(1, device=device)
            residual_label = torch.ones(1, device=device)
            coverage_label = torch.ones(1, device=device)

            train_batch = (x1, x2, x1p, structure_label, residual_label, coverage_label, pos_idx)
            trainer.train_step(train_batch)

        trainer.on_epoch_end(do_print=True)

        if epoch % 10 == 0:
            conde_model.eval()
            decoder.eval()

            n_show = 10

            with torch.no_grad():
                for val_batch in test_dataloader:
                    x1, x2 = val_batch

                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)

                    pair = x1.clone()
                    x1, x2, x1p = augment(x1, pair, x2, residual=True)

                    pos_idx = torch.zeros((x1.shape[0], 1), dtype=torch.long, device=device)
                    structure_label = torch.ones(1, device=device)
                    residual_label = torch.ones(1, device=device)
                    coverage_label = torch.ones(1, device=device)

                    validation_batch = (x1, x2, x1p, structure_label, residual_label, coverage_label, pos_idx)

                    trainer.validation_step(validation_batch, n_show=n_show, save_name=os.path.join(SAVE_DIR, f"{epoch}.png"))
                    n_show = 0

            trainer.on_validation_epoch_end()
            save_ckpt(SAVE_CKPT_DIR, epoch, conde_model, decoder, ae)


if __name__ == "__main__":
    main()
