"""Seismic-RGT dip consistency: a label-free score and its use as sampling guidance.

RGT isolines must run parallel to the reflectors, so the RGT gradient must be parallel to the
reflector normal. The normal comes from the seismic alone (structure tensor), so the score needs
no RGT and can be used on a new survey:

    T     = G_st * (grad s  grad s^T)              grad s after a G_grad blur
    n     = main eigenvector of T                  reflector normal, (n_z, n_x)
    c     = (l1 - l2) / (l1 + l2)                  coherence in [0, 1]: ~0 in chaotic zones and at faults
    dip   = sum c * sin^2(angle(grad tau, n)) / sum c          in [0, 1], 0.5 for a random orientation
    mono  = mean relu(-d tau / dz) / mean |d tau / dz|          tau must increase with depth (rows)

The score only sees the local orientation: it cannot tell which horizon connects to which across
a fault, nor the spacing of the isolines. Depth is axis -2, as everywhere in sld.
"""
import copy
import math

import torch
import torch.nn.functional as F


def _gauss_kernel(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of [B,C,H,W], reflect padding; identity for sigma <= 0. Differentiable."""
    if sigma <= 0:
        return x
    b, c, h, w = x.shape
    k = _gauss_kernel(sigma, x.device, x.dtype)
    r = len(k) // 2
    y = x.reshape(b * c, 1, h, w)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return y.view(b, c, h, w)


def gradients(x):
    """Central differences (d/dz, d/dx) of [B,1,H,W], replicate padding at the border. Differentiable."""
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dz = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    dx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    return dz, dx


def erode(mask):
    """3x3 erosion of a {0,1} [B,1,H,W] mask: the pixels whose central differences only use valid pixels."""
    return 1.0 - F.max_pool2d(1.0 - mask, 3, stride=1, padding=1)


@torch.no_grad()
def structure_tensor(seismic, sigma_grad=1.0, sigma_tensor=3.0, eps=1e-8):
    """seismic [B,1,H,W] -> (n_z, n_x, coherence), each [B,1,H,W]. n is the unit reflector normal."""
    s = gaussian_blur(seismic.float(), sigma_grad)
    gz, gx = gradients(s)
    jzz, jzx, jxx = (gaussian_blur(t, sigma_tensor) for t in (gz * gz, gz * gx, gx * gx))
    # Quadratic form along (cos p, sin p): (jzz+jxx)/2 + (jzz-jxx)/2 cos 2p + jzx sin 2p, max at 2p = atan2(2 jzx, jzz-jxx).
    phi = 0.5 * torch.atan2(2 * jzx, jzz - jxx)
    coherence = torch.sqrt((jzz - jxx) ** 2 + 4 * jzx ** 2) / (jzz + jxx + eps)
    return torch.cos(phi), torch.sin(phi), coherence.clamp(0, 1)


class DipScore:
    """Dip-consistency score of RGT maps against the seismic they should follow.

    Built once per batch of seismic (the reflector field is fixed); ``__call__(tau)`` returns the
    per-section score [B] = dip + mono_weight * mono, differentiable in tau (for guidance).
    ``mask`` restricts the score (e.g. to the valid pixels when scoring against a GT);
    guidance passes none, so it uses nothing but the seismic.
    """

    def __init__(self, seismic, mask=None, sigma_grad=1.0, sigma_tensor=3.0, sigma_tau=1.0, coherence_power=1.0,
                 mono_weight=1.0, rel_eps=1e-2):
        nz, nx, c = structure_tensor(seismic, sigma_grad, sigma_tensor)
        self.tz, self.tx = -nx, nz  # reflector tangent
        m = torch.ones_like(c) if mask is None else erode(mask.float())
        self.weight = (c ** coherence_power) * m
        self.mask = m
        self.sigma_tau, self.mono_weight, self.rel_eps = sigma_tau, mono_weight, rel_eps

    def _sin2(self, tau):
        t = gaussian_blur(tau.float(), self.sigma_tau)
        dz, dx = gradients(t)
        g2 = dz * dz + dx * dx
        eps = self.rel_eps * ((g2 * self.mask).flatten(1).sum(1) / self.mask.flatten(1).sum(1).clamp_min(1)).detach()
        return (dz * self.tz + dx * self.tx) ** 2 / (g2 + eps.view(-1, 1, 1, 1) + 1e-12), dz

    def misfit_map(self, tau):
        """[B,1,H,W] per-pixel sin^2 of the angle between grad tau and the reflector normal, times the weight."""
        return self._sin2(tau)[0] * self.weight

    def terms(self, tau):
        """(dip [B], mono [B]) for tau [B,1,H,W]."""
        sin2, dz = self._sin2(tau)
        w = self.weight
        dip = (w * sin2).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-8)
        m = self.mask
        mono = (F.relu(-dz) * m).flatten(1).sum(1) / ((dz.abs() * m).flatten(1).sum(1) + 1e-8)
        return dip, mono

    def __call__(self, tau):
        dip, mono = self.terms(tau)
        return dip + self.mono_weight * mono if self.mono_weight else dip

    def _map(self, fn):
        out = copy.copy(self)
        for k in ("tz", "tx", "weight", "mask"):
            setattr(out, k, fn(getattr(self, k)))
        return out

    def repeat_interleave(self, n):
        """The same score for n samples per section (rows repeated like z_x.repeat_interleave(n))."""
        return self._map(lambda t: t.repeat_interleave(n, dim=0))

    def index(self, idx):
        return self._map(lambda t: t[idx])
