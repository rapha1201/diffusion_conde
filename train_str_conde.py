import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from conde import ConDeTrainer, AELoader
from conde.models import ConDe, Autoencoder
from conde.data import WuResidualCondeDataset, SeismicAugmentation
from conde.losses import *

SAVE_DIR = 'conde_res/wu_str'
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")

def save_ckpt(
    save_dir: str,
    epoch: int,
    conde_model,
    ae,
    trainer=None,
):
    """
    Saves:
      - ConDe model weights
      - LoRA adapters for encoder/decoder via PEFT
      - (optional) trainer optimizer/scheduler/scaler states if available
    """
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt")
    latest_path = os.path.join(save_dir, "ckpt_latest.pt")


    # ---- Generic torch checkpoint (model + training state) ----
    payload = {
        "epoch": epoch,
        "conde_state_dict": conde_model.state_dict(),
        "ae": ae.state_dict(),
    }


    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    print(f"[SAVE] {ckpt_path}")
    print(f"[SAVE] {latest_path}")
    

if __name__ == '__main__':
    
    loader = AELoader('../configs/wu_ae.yaml', model_only=True, load=True)
    ae = loader.model.cuda()


    dataset = WuResidualCondeDataset('/pgeoprj/godeep/ej49/ConDe/dataset/WuSet.h5', split='train', emb_model=ae)
    dataloader = DataLoader(dataset, batch_size = 256, shuffle=True, drop_last=True)
    
    seap_dataset = WuResidualCondeDataset('/pgeoprj/godeep/ej49/ConDe/dataset/WuSet.h5', split='test', emb_model=ae)
    seap_dataloader = DataLoader(seap_dataset, batch_size = 128, shuffle=True, drop_last=True)


    semantic_info = {
        "structure":  {"head_index": 0},
    }
    
    EPOCHS=251
    
    conde_model = ConDe(latent_dim = 1024, n_semantic_heads = 1, encoder=ae.model.encoder)
    
    contrastive_loss = NtXentLoss()
    residual_loss = CombinedLoss(losses=[CrossCorrelationLoss(), HSIC_RBF()])
    decoder_loss_fn = CombinedLoss(losses=[nn.L1Loss(), GradientLoss()], weights=[1, 0.1])
    
    trainer = ConDeTrainer(conde_model, extract_embs = True, decoder_loss_fn=decoder_loss_fn, contrastive_loss_fn=contrastive_loss, residual_loss_fn=residual_loss, decoder=ae.model.decoder, semantic_info=semantic_info, total_steps = len(dataloader)*EPOCHS, lambda_contrast=1, lambda_coverage=2, lambda_residual=2, lambda_decoder=0.1)

    augment = SeismicAugmentation(aug_p=0.5,
                        init_blur_sigma_range=(1.2, 2.5), p_init_blur=1,                      
                        noise_phase_sigma_range=(0.2, 0.4), p_noise_phase=0.3,
                        noise_amp_sigma_range=(0.5, 0.8), p_noise_amp=0.75,
                        blur_amp_sigma_range=(0.8, 1.5),   p_blur_amp=0.55,
                        freq_low_energy_remove_range=(0.0, 0.35),
                        freq_high_energy_remove_range=(0.0, 0.35),
                        max_freq_energy_remove=0.5,
                        p_freq_filter=0.7,
                        p_hflip=0,
                        zoom_range=(1.0, 1.6), p_zoom=0,
                        p_aniso=0,
                        blur_kernel_size=15)
    
    for epoch in range(EPOCHS):
        print(f"[EPOCH {epoch}/{EPOCHS}]", end = '  ')
        for batch in tqdm(dataloader):
            
            x1, x2, x1p, z1, z2, z1p = batch
            pair = x1.clone()
            x1, x2, x1p = augment(x1, pair, x2, residual=True)
            pos_idx = torch.zeros((z1.shape[0], 1))
            batch = (x1, x2, x1p, z1.squeeze(), z2.squeeze(), z1p.squeeze(), pos_idx)
            stats = trainer.train_step(batch)
    
        trainer.on_epoch_end()
    
    
        if epoch%5 == 0:
            n_show = 6
            for val_batch in dataloader:
                x1, x2, x1p, z1, z2, z1p = val_batch
                pair = x1.clone()
                x1, x2, x1p = augment(x1, pair, x2, residual=True)
                pos_idx = torch.zeros((z1.shape[0], 1))
                val_batch = (x1, x2, x1p, z1.squeeze(), z2.squeeze(), z1p.squeeze(), pos_idx)
                trainer.validation_step(val_batch, n_show=n_show, save_name=f'{SAVE_DIR}/{epoch}.png')
                n_show=0
            trainer.on_validation_epoch_end()

            save_ckpt(
                save_dir=SAVE_CKPT_DIR,
                epoch=epoch,
                conde_model=conde_model,
                ae=ae
            )
