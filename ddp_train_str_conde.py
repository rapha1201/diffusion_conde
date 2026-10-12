import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from conde import ConDeTrainer, AELoader
from conde.models import ConDe
from conde.data import (
    H5FourierTargetPool,
    ResidualCondeDataset,
    SeismicAugmentation,
)
from conde.losses import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import copy

def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

SAVE_DIR = 'conde_res/deep_fault_rgt_conde'
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")
DATASET_PATH = '/pgeoprj/godeep/ej49/ConDe/dataset/deep_fault.h5'
FOURIER_TARGET_FRACTION = 0.05
FOURIER_TARGET_SEED = 0
FOURIER_BETA_A = 15.0
FOURIER_BETA_B = 10.0


def build_augmentation(fourier_target_pool=None, p_fourier_mix=0.0):
    """Build the current augmentation stack with optional Buzios FFT mixing."""
    return SeismicAugmentation(
        aug_p=0.5,
        init_blur_sigma_range=(1.2, 2.5),
        p_init_blur=1,
        noise_phase_sigma_range=(0.2, 0.4),
        p_noise_phase=0.3,
        noise_amp_sigma_range=(0.5, 0.8),
        p_noise_amp=0.75,
        blur_amp_sigma_range=(0.8, 1.5),
        p_blur_amp=0.55,
        # The current augmenter bounds removed spectral energy rather than
        # accepting legacy pixel-frequency cutoffs.
        freq_low_energy_remove_range=(0.0, 0.35),
        freq_high_energy_remove_range=(0.0, 0.35),
        max_freq_energy_remove=0.5,
        p_freq_filter=0.7,
        p_hflip=0,
        # Preserve the old script's disabled scale/crop augmentation using the
        # corresponding controls in the current affine implementation.
        zoom_range=(1.0, 1.6),
        p_zoom=0,
        p_aniso=0,
        blur_kernel_size=15,
        fourier_target_pool=fourier_target_pool,
        p_fourier_mix=p_fourier_mix,
        fourier_beta_a=FOURIER_BETA_A,
        fourier_beta_b=FOURIER_BETA_B,
    )

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


    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    payload = {
        "epoch": epoch,
        "conde_state_dict": unwrap(conde_model).state_dict(),
        "decoder_state_dict": unwrap(decoder).state_dict(),   # if decoder is trained
        "ae_state_dict": unwrap(ae).state_dict(),             # optional
    }


    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    print(f"[SAVE] {ckpt_path}")
    print(f"[SAVE] {latest_path}")


if __name__ == '__main__':

    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    
    loader = AELoader('../configs/small_ae.yaml', model_only=True, load=False)
    ae = loader.model

    rgt_decoder = copy.deepcopy(ae.model.decoder)

    fourier_target_pool = H5FourierTargetPool(
        DATASET_PATH,
        target_model='buzios',
        fraction=FOURIER_TARGET_FRACTION,
        seed=FOURIER_TARGET_SEED,
        min_held_out_units=2,
    )

    # Source training is synthetic-only. Real Buzios data can enter training
    # solely through the selected Fourier appearance pool.
    dataset = ResidualCondeDataset(
        DATASET_PATH,
        model_names=['Modelo2', 'Modelo3', 'Modelo4', 'Modelo5', 'Modelo8', 'Modelo9'],
        emb_model=ae,
        embedding_device=device,
    )

    # Whole Buzios HDF5 units selected for Fourier mixing are excluded here.
    test_dataset = ResidualCondeDataset(
        DATASET_PATH,
        dataset_names=fourier_target_pool.held_out_dataset_names,
        emb_model=ae,
        embedding_device=device,
    )

    if is_main_process():
        fourier_target_pool.write_manifest(
            os.path.join(SAVE_DIR, 'fourier_target_split.json'),
            fourier_beta_a=FOURIER_BETA_A,
            fourier_beta_b=FOURIER_BETA_B,
            p_fourier_mix=1.0,
            global_aug_p=0.5,
        )
    

    train_sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset,batch_size=256,sampler=train_sampler,drop_last=True,
                            pin_memory=True,num_workers=4,persistent_workers=True,)
    

    val_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_dataloader = DataLoader(test_dataset,batch_size=128,sampler=val_sampler,drop_last=False,pin_memory=True,num_workers=2,persistent_workers=True)


    semantic_info = {
        "structure":  {"head_index": 0},
    }
    
    EPOCHS=251
    
    conde_model = ConDe(latent_dim = 1024, n_semantic_heads = 1, encoder=ae.model.encoder, normalize=False)
    conde_model = DDP(conde_model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=False)
    decoder = DDP(ae.model.decoder.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=False)
    
    contrastive_loss = TripletLoss()
    residual_loss = CombinedLoss(losses=[CrossCorrelationLoss(), HSIC_RBF_Multi()])
    decoder_loss_fn = CombinedLoss(losses=[nn.L1Loss(), GradientLoss()], weights=[1, 0.1])
    coverage_loss_fn = nn.L1Loss()
    rgt_loss_fn = RGTLoss()
    
    trainer = ConDeTrainer(conde_model, extract_embs = True, decoder_loss_fn=decoder_loss_fn, contrastive_loss_fn=contrastive_loss, residual_loss_fn=residual_loss, coverage_loss_fn=coverage_loss_fn, rgt_loss_fn=rgt_loss_fn, decoder=decoder, semantic_info=semantic_info, total_steps = len(dataloader)*EPOCHS, lambda_contrast=5, lambda_coverage=0.2, lambda_residual=5, lambda_decoder=0.2, device=device, decoder_warmup=0, coverage_mode='inverted_detach', decomp_mode='inverted_detach', lambda_decomp=1, require_labels=True, rgt_decoder = rgt_decoder)

    # All globally gated training samples receive Fourier amplitude mixing.
    augment = build_augmentation(
        fourier_target_pool=fourier_target_pool,
        p_fourier_mix=1.0,
    ).to(device)

    # Validation retains the existing augmentation/preprocessing contract but
    # never consumes the real target style pool.
    val_augment = build_augmentation(p_fourier_mix=0.0).to(device)
    
    for epoch in range(EPOCHS):
        if is_main_process():
            print(f"[EPOCH {epoch}/{EPOCHS}]", end="  ")
        train_sampler.set_epoch(epoch)
        for batch in tqdm(dataloader, disable=not is_main_process()):
            
            x1, x2, x1p, rgt, z1, z2, z1p = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            pair = x1.clone()
            x1, x2, x1p, rgt = augment(x1, pair, x2, residual=True, rgt=rgt)
            pos_idx = torch.zeros((z1.shape[0], 1))
            batch = (x1, x2, x1p, rgt, z1.squeeze(), z2.squeeze(), z1p.squeeze(), pos_idx)
            stats = trainer.train_step(batch)

        
        trainer.on_epoch_end(do_print=is_main_process())
        dist.barrier()
    
    
        if epoch%10 == 0:
            n_show = 10
            val_sampler.set_epoch(epoch)
            for val_batch in test_dataloader:
                                   
                x1, x2, x1p, rgt, z1, z2, z1p = val_batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                pair = x1.clone()
                x1, x2, x1p, rgt = val_augment(x1, pair, x2, residual=True, rgt=rgt)
                pos_idx = torch.zeros((z1.shape[0], 1))
                val_batch = (x1, x2, x1p, rgt, z1.squeeze(), z2.squeeze(), z1p.squeeze(), pos_idx)
                trainer.validation_step(val_batch, n_show=n_show if is_main_process() else 0, save_name=f'{SAVE_DIR}/{epoch}.png')
                n_show=0
            trainer.on_validation_epoch_end()
            dist.barrier()
            if is_main_process():
                save_ckpt(
                    save_dir=SAVE_CKPT_DIR,
                    epoch=epoch,
                    conde_model=conde_model,
                    ae=ae
                )
