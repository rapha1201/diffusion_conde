"""GPU-side pair transform: shared geometry for seismic and RGT, then per-crop renormalization.

Reuses ``conde.data.augmentation.SeismicAugmentation``: its pair path applies the
appearance augmentations (blur, spectral, noise, gain, polarity...) to the
seismic only and the geometric ones (statics, synthetic fault, affine, random
crop) identically to both. The RGT and its validity mask travel together as a
2-channel label, so the mask is warped exactly like the RGT.

Depth is axis -2 (rows), which is the axis the augmentation treats as vertical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from conde.data.augmentation import SeismicAugmentation


def center_crop(x, size):
    h, w = x.shape[-2:]
    if h < size or w < size:
        raise ValueError(f"Input {h}x{w} is smaller than crop size {size}.")
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def pad_to_multiple(x, multiple, mode="reflect"):
    """Pad H and W at the bottom/right to a multiple. Returns (padded, (pad_h, pad_w))."""
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (pad_h, pad_w)


def unpad(x, pads):
    pad_h, pad_w = pads
    h, w = x.shape[-2:]
    return x[..., : h - pad_h, : w - pad_w]


def standardize(x, eps=1e-6):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    return (x - mean) / (std + eps)


def renormalize_rgt(rgt, valid, eps=1e-6):
    """Per-sample min-max over valid pixels; invalid pixels set to 0. Returns (rgt01, value_range)."""
    big = torch.finfo(rgt.dtype).max
    vmin = torch.where(valid, rgt, torch.full_like(rgt, big)).amin(dim=(-3, -2, -1), keepdim=True)
    vmax = torch.where(valid, rgt, torch.full_like(rgt, -big)).amax(dim=(-3, -2, -1), keepdim=True)
    has_valid = valid.flatten(1).any(dim=1).view(-1, 1, 1, 1)
    value_range = torch.where(has_valid, vmax - vmin, torch.zeros_like(vmax))
    out = (rgt - torch.where(has_valid, vmin, torch.zeros_like(vmin))) / value_range.clamp_min(eps)
    out = torch.where(valid, out.clamp(0.0, 1.0), torch.zeros_like(out))
    return out, value_range.view(-1)


class PairTransform(nn.Module):
    """Batch transform applied on the GPU after the DataLoader.

    train=True : SeismicAugmentation pair path (random crop to ``output_size``).
    train=False: deterministic center crop to ``output_size`` (or no crop if None).

    Returns (seismic, rgt, valid, keep):
        seismic standardized per crop; rgt in [0,1] over valid pixels, 0 elsewhere;
        valid as float {0,1}; keep = bool [B], False for crops with too few valid
        pixels or a degenerate RGT range (exclude them from losses).
    """

    def __init__(self, aug_kwargs=None, output_size=128, train=True, min_valid_frac=0.5, valid_threshold=0.999):
        super().__init__()
        self.train_mode = train
        self.output_size = output_size
        self.min_valid_frac = min_valid_frac
        self.valid_threshold = valid_threshold
        kwargs = dict(aug_kwargs or {})
        # A vertical flip would invert RGT monotonicity; real-data style mixing is out of scope here.
        kwargs.update(p_vflip=0.0, invert_rgt_on_vflip=False, p_fourier_mix=0.0, fourier_target_pool=None)
        if output_size is not None:
            kwargs["output_size"] = output_size
        self.augment = SeismicAugmentation(**kwargs) if train else None

    @torch.no_grad()
    def forward(self, seismic, rgt, valid):
        label = torch.cat([rgt, valid], dim=1)
        if self.train_mode:
            seismic, label = self.augment(seismic, label, label=True)
        elif self.output_size is not None:
            seismic, label = center_crop(seismic, self.output_size), center_crop(label, self.output_size)

        valid = label[:, 1:2] > self.valid_threshold
        rgt, value_range = renormalize_rgt(label[:, 0:1], valid)
        valid_frac = valid.flatten(1).float().mean(dim=1)
        keep = (valid_frac >= self.min_valid_frac) & (value_range > 1e-3)
        return standardize(seismic), rgt, valid.float(), keep
