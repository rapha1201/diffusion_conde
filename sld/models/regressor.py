"""Deterministic UNet regressor (baselines).

B1: pixel space, seismic [1,H,W] -> RGT [1,H,W].
B2 (Phase 2): the same class in latent space, z_x -> z_y.

Built on diffusers ``UNet2DModel`` with a constant timestep, i.e. a plain UNet,
so the baseline and the diffusion denoiser share the same architecture family.
"""
import torch
import torch.nn as nn
from diffusers import UNet2DModel


class UNetRegressor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, block_out_channels=(64, 128, 256, 512), layers_per_block=2,
                 attention_levels=(3,), norm_num_groups=32):
        super().__init__()
        n = len(block_out_channels)
        attention_levels = set(attention_levels or ())
        self.model_cfg = dict(in_channels=in_channels, out_channels=out_channels,
                              block_out_channels=list(block_out_channels), layers_per_block=layers_per_block,
                              attention_levels=sorted(attention_levels), norm_num_groups=norm_num_groups)
        down = tuple("AttnDownBlock2D" if i in attention_levels else "DownBlock2D" for i in range(n))
        up = tuple("AttnUpBlock2D" if (n - 1 - j) in attention_levels else "UpBlock2D" for j in range(n))
        self.unet = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down,
            up_block_types=up,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

    @property
    def downsample_factor(self):
        return 2 ** (len(self.model_cfg["block_out_channels"]) - 1)

    def forward(self, x):
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        return self.unet(x, t).sample
