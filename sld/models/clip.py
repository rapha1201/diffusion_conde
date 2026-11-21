"""Phase A: CLIP-style pretraining of the seismic condition encoder (the Stable Diffusion analogue of
the pretrained CLIP text encoder).

Two encoders on the frozen, scaled VAE latents of a pair:
    seismic_encoder  z_x [8, 16, 16] -> features [embed_dim, 16, 16]   (later loaded, frozen, into the LDM)
    rgt_encoder      z_y [4, 16, 16] -> features [embed_dim, 16, 16]   (only used here, discarded afterwards)
Projection heads (1x1 conv) map the features to ``proj_dim``-d embeddings, L2-normalized, and a
symmetric InfoNCE loss with a learned temperature makes each seismic pick its own RGT among the
B sections of the batch, and vice versa:
    mode="dense"   one B-way classification per latent position (i, j), averaged over the H*W
                   positions: aligns *where* the structures are, which is what the LDM needs
    mode="global"  classic CLIP: mean-pooled embeddings, one B-way classification per section
As in SD, the LDM uses the encoder features before the projection head.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_encoder import make_latent_encoder


class LatentCLIP(nn.Module):
    def __init__(self, seismic_channels=8, rgt_channels=4, embed_dim=128, encoder_blocks=4, norm_num_groups=32,
                 proj_dim=128, mode="dense", temperature=0.07):
        super().__init__()
        if mode not in ("dense", "global"):
            raise ValueError(f"mode must be dense or global, got {mode!r}")
        self.model_cfg = dict(seismic_channels=seismic_channels, rgt_channels=rgt_channels, embed_dim=embed_dim,
                              encoder_blocks=encoder_blocks, norm_num_groups=norm_num_groups, proj_dim=proj_dim,
                              mode=mode, temperature=temperature)
        self.mode = mode
        self.seismic_encoder = make_latent_encoder(seismic_channels, embed_dim, encoder_blocks, norm_num_groups)
        self.rgt_encoder = make_latent_encoder(rgt_channels, embed_dim, encoder_blocks, norm_num_groups)
        self.seismic_head = nn.Conv2d(embed_dim, proj_dim, 1)
        self.rgt_head = nn.Conv2d(embed_dim, proj_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def _embed(self, encoder, head, z):
        """[B, P, D] unit vectors; P = H*W (dense) or 1 (global)."""
        e = head(encoder(z)).float()
        e = e.mean(dim=(2, 3), keepdim=True) if self.mode == "global" else e
        return F.normalize(e.flatten(2).transpose(1, 2), dim=-1)

    def embed_seismic(self, z_x):
        return self._embed(self.seismic_encoder, self.seismic_head, z_x)

    def embed_rgt(self, z_y):
        return self._embed(self.rgt_encoder, self.rgt_head, z_y)

    @property
    def temperature(self):
        return float(1.0 / self.logit_scale.exp())

    def logits(self, ex, ey):
        """[P, B, B]: seismic i (rows) vs RGT j (columns), at each position. Always fp32 (bf16 is too coarse
        for cosine similarities scaled by up to 100)."""
        with torch.autocast(device_type=ex.device.type, enabled=False):
            return self.logit_scale.float().exp().clamp(max=100.0) * torch.einsum("ipd,jpd->pij", ex.float(), ey.float())

    def loss_from_embeddings(self, ex, ey):
        """Symmetric InfoNCE (diagonal = own pair). Returns (loss, top-1 accuracy averaged over both directions)."""
        b = ex.shape[0]
        logits = self.logits(ex, ey)
        labels = torch.arange(b, device=ex.device).expand(logits.shape[0], b).reshape(-1)
        loss = 0.5 * (F.cross_entropy(logits.reshape(-1, b), labels)
                      + F.cross_entropy(logits.transpose(1, 2).reshape(-1, b), labels))
        acc = 0.5 * ((logits.argmax(-1).reshape(-1) == labels).float().mean()
                     + (logits.argmax(-2).reshape(-1) == labels).float().mean())
        return loss, acc

    def forward(self, z_x, z_y):
        if z_x.shape[0] < 2:
            raise ValueError("The contrastive loss needs at least 2 pairs per batch")
        return self.loss_from_embeddings(self.embed_seismic(z_x), self.embed_rgt(z_y))

    @torch.no_grad()
    def retrieval(self, ex, ey, keys):
        """Evaluation on one group of sections. ``keys`` [B, k] identify the geology of each section
        (e.g. model, slice index): realizations of the same slice share the RGT, so picking a twin counts as
        correct. Returns (seismic->RGT accuracy, RGT->seismic accuracy, InfoNCE loss with diagonal labels)."""
        logits = self.logits(ex, ey)
        b = ex.shape[0]
        same = (keys[:, None, :] == keys[None, :, :]).all(-1)            # [B, B]
        s2r = same[torch.arange(b, device=ex.device), logits.argmax(-1)]  # [P, B]
        r2s = same[torch.arange(b, device=ex.device), logits.argmax(-2)]
        loss, _ = self.loss_from_embeddings(ex, ey)
        return s2r.float().mean(), r2s.float().mean(), loss
