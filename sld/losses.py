"""Reconstruction / regression losses with optional validity masks.

Masks are [B,1,H,W] in {0,1}; masked pixels contribute nothing. Gradient-type
terms only use neighbour pairs where both pixels are valid.

Notes from the real data (tools/preview_pairs.py on fault_aware.h5):
- the RGT has large jumps (unconformities, fault offsets) -> L1-type terms;
- it is partly stair-stepped -> no second-derivative term;
- it is NOT monotonic everywhere (reverse/strike-slip faults repeat sections),
  so the monotonicity penalty only acts where the target itself increases.

``conde.losses.FourierLoss`` (MSE of real + imaginary parts of the orthonormal
FFT) equals the pixel-space MSE exactly by Parseval, so it is not used; ``spectral_l1`` compares log-amplitude spectra
instead, which does penalize lost high frequencies in the seismic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    mask = mask.expand_as(x).to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _pair_masks(mask):
    if mask is None:
        return None, None
    return mask[..., 1:, :] * mask[..., :-1, :], mask[..., :, 1:] * mask[..., :, :-1]


def masked_l1(pred, target, mask=None):
    return masked_mean((pred - target).abs(), mask)


def masked_grad_l1(pred, target, mask=None):
    mz, mx = _pair_masks(mask)
    dz = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    return masked_mean(dz.abs(), mz) + masked_mean(dx.abs(), mx)


def _gaussian_window(size, sigma, device, dtype):
    ax = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def masked_ssim_loss(pred, target, mask=None, window_size=11, sigma=1.5, data_range=1.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype).expand(pred.shape[1], 1, -1, -1)
    pad = window_size // 2
    conv = lambda t: F.conv2d(F.pad(t, (pad,) * 4, mode="reflect"), w, groups=pred.shape[1])
    mu_x, mu_y = conv(pred), conv(target)
    sxx = conv(pred * pred) - mu_x ** 2
    syy = conv(target * target) - mu_y ** 2
    sxy = conv(pred * target) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return 1.0 - masked_mean(ssim, mask)


def monotonic_penalty(pred, target, mask=None):
    """Penalize decreasing pred along depth (rows) only where the target increases."""
    mz, _ = _pair_masks(mask)
    dp = pred[..., 1:, :] - pred[..., :-1, :]
    dt = target[..., 1:, :] - target[..., :-1, :]
    where = (dt >= 0).to(pred.dtype)
    where = where if mz is None else where * mz
    return masked_mean(F.relu(-dp), where)


def spectral_l1(pred, target, eps=1e-6):
    fp = torch.fft.rfft2(pred, norm="ortho").abs()
    ft = torch.fft.rfft2(target, norm="ortho").abs()
    return (torch.log(fp + eps) - torch.log(ft + eps)).abs().mean()


def kl_loss(posterior):
    """Mean over the batch of the KL summed over latent dimensions (LDM convention)."""
    return posterior.kl().mean()


class ReconLoss(nn.Module):
    """Weighted sum of terms. Unknown or zero-weight terms are skipped.

    rgt terms:     l1, grad, ssim, monotonic   (masked)
    seismic terms: l1, grad, spectral, mse
    both:          kl (when a posterior is given)
    """

    RGT_TERMS = ("l1", "grad", "ssim", "monotonic")
    SEISMIC_TERMS = ("l1", "grad", "spectral", "mse")

    def __init__(self, modality, weights):
        super().__init__()
        if modality not in ("rgt", "seismic"):
            raise ValueError(f"modality must be 'rgt' or 'seismic', got {modality!r}")
        allowed = set(self.RGT_TERMS if modality == "rgt" else self.SEISMIC_TERMS) | {"kl"}
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(f"Unknown loss terms for {modality}: {sorted(unknown)}")
        self.modality = modality
        self.weights = {k: float(v) for k, v in weights.items() if float(v) != 0.0}

    def forward(self, pred, target, mask=None, posterior=None):
        pred, target = pred.float(), target.float()
        terms = {}
        w = self.weights
        if "l1" in w:
            terms["l1"] = masked_l1(pred, target, mask)
        if "grad" in w:
            terms["grad"] = masked_grad_l1(pred, target, mask)
        if "ssim" in w:
            terms["ssim"] = masked_ssim_loss(pred, target, mask)
        if "monotonic" in w:
            terms["monotonic"] = monotonic_penalty(pred, target, mask)
        if "spectral" in w:
            terms["spectral"] = spectral_l1(pred, target)
        if "mse" in w:
            terms["mse"] = masked_mean((pred - target) ** 2, mask)
        if "kl" in w and posterior is not None:
            terms["kl"] = kl_loss(posterior)
        total = sum(w[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
