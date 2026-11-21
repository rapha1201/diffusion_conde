"""Small resolution-preserving CNN on a latent map, shared by the CLIP encoders and the LDM condition encoder.

The LDM loads the pretrained CLIP seismic encoder into its condition encoder, so both must be
built by the same function with the same arguments (the state-dict keys then match).
"""
import math

import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, ch, groups):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                   nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


def make_latent_encoder(in_channels, dim, n_blocks, norm_num_groups):
    """[B, in_channels, H, W] -> [B, dim, H, W]. With 4 blocks at 16x16 the receptive field covers the latent."""
    groups = math.gcd(norm_num_groups, dim)
    return nn.Sequential(nn.Conv2d(in_channels, dim, 3, padding=1), *[_ResBlock(dim, groups) for _ in range(n_blocks)],
                         nn.GroupNorm(groups, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, padding=1))
