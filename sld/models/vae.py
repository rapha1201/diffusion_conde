"""SD-style KL autoencoder (diffusers ``AutoencoderKL``) for single-channel 2D sections.

One instance per modality (seismic, RGT). Both must share the spatial
downsampling factor ``2 ** (len(block_out_channels) - 1)`` so their latents can
be concatenated along channels; the number of latent channels may differ.

``scale_factor`` is a buffer (saved in the state dict) set after training by
compute_latent_scale.py so that scaled latents have unit variance, which the
diffusion noise schedule assumes. ``encode``/``decode`` work in scaled units.
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class LatentVAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4, block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, norm_num_groups=32):
        super().__init__()
        n = len(block_out_channels)
        self.model_cfg = dict(in_channels=in_channels, latent_channels=latent_channels,
                              block_out_channels=list(block_out_channels), layers_per_block=layers_per_block,
                              norm_num_groups=norm_num_groups)
        self.vae = AutoencoderKL(
            in_channels=in_channels,
            out_channels=in_channels,
            down_block_types=("DownEncoderBlock2D",) * n,
            up_block_types=("UpDecoderBlock2D",) * n,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
        )
        self.register_buffer("scale_factor", torch.tensor(1.0))

    @property
    def downsample_factor(self):
        return 2 ** (len(self.model_cfg["block_out_channels"]) - 1)

    @property
    def latent_channels(self):
        return self.model_cfg["latent_channels"]

    def posterior(self, x):
        return self.vae.encode(x).latent_dist

    def encode(self, x, sample=False, generator=None):
        """Scaled latent: posterior sample or mode, multiplied by ``scale_factor``."""
        post = self.posterior(x)
        z = post.sample(generator=generator) if sample else post.mode()
        return z * self.scale_factor

    def decode(self, z_scaled):
        return self.vae.decode(z_scaled / self.scale_factor).sample

    def forward(self, x, sample_posterior=True):
        """Unscaled reconstruction path used for AE training. Returns (recon, posterior)."""
        post = self.posterior(x)
        z = post.sample() if sample_posterior else post.mode()
        return self.vae.decode(z).sample, post
