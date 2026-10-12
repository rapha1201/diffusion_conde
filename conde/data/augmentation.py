import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_mix import mix_fourier_amplitudes


class _ShapeCache:
    def __init__(self):
        self._dist = {}      # key: (H,W,device,dtype) -> [1,1,H,W]
        self._gauss1d = {}   # key: (ks,sigma,device,dtype) -> [ks]
        self._fftfreq = {}   # key: (H,device,dtype) -> [H]
        self._freqdist = {}   # key: (H,W,device,dtype) -> [1,1,H,W//2+1]

    def dist(self, H, W, device, dtype=torch.float32):
        k = (H, W, device, dtype)
        out = self._dist.get(k)
        if out is None:
            y = torch.arange(H, device=device, dtype=dtype)
            x = torch.arange(W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            d = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).unsqueeze(0).unsqueeze(0)
            self._dist[k] = d
            out = d
        return out

    def gauss1d(self, ks, sigma, device, dtype):
        k = (ks, float(sigma), device, dtype)
        out = self._gauss1d.get(k)
        if out is None:
            ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
            g = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
            g = g / g.sum()
            self._gauss1d[k] = g
            out = g
        return out

    def fftfreq(self, H, device, dtype):
        k = (H, device, dtype)
        out = self._fftfreq.get(k)
        if out is None:
            out = torch.fft.fftfreq(H, device=device, dtype=dtype)
            self._fftfreq[k] = out
        return out

    def freqdist(self, H, W, device, dtype=torch.float32):
        k = (H, W, device, dtype)
        out = self._freqdist.get(k)
        if out is None:
            fy = torch.fft.fftfreq(H, device=device, dtype=dtype) * H
            fx = torch.fft.rfftfreq(W, device=device, dtype=dtype) * W
            yy, xx = torch.meshgrid(fy, fx, indexing="ij")
            out = torch.sqrt(xx.square() + yy.square()).unsqueeze(0).unsqueeze(0)
            self._freqdist[k] = out
        return out


_CACHE = _ShapeCache()


# -----------------------------------------------------------------------------
# FFT helpers
# -----------------------------------------------------------------------------
def rfft2(img):
    return torch.fft.rfft2(img)


def irfft2(fft, H, W):
    return torch.fft.irfft2(fft, s=(H, W))


def apply_phase_noise_(fft, sigma_map):
    delta = torch.randn_like(fft.real) * sigma_map
    rot = torch.complex(torch.cos(delta), torch.sin(delta))
    fft.mul_(rot)


def apply_amp_noise_(fft, sigma_map):
    scale = 1.0 + torch.randn_like(fft.real) * sigma_map
    fft.mul_(scale)


def apply_energy_frequency_filter_(fft, low_frac, high_frac, H, W, eps=1e-12):
    """Remove low/high radial-frequency tails while bounding removed spectral energy."""
    n = fft.shape[0]
    d = _CACHE.freqdist(H, W, fft.device, fft.real.dtype).reshape(-1)
    order = torch.argsort(d)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)

    # rfft2 stores only one side of the horizontal spectrum. Double interior
    # columns so the power sum matches the full real-valued FFT energy.
    wx = torch.ones(W // 2 + 1, device=fft.device, dtype=fft.real.dtype)
    if W % 2 == 0:
        if wx.numel() > 2:
            wx[1:-1] = 2.0
    elif wx.numel() > 1:
        wx[1:] = 2.0
    weights = wx.view(1, 1, 1, -1)

    power = fft.abs().square() * weights
    power = power.sum(dim=1).reshape(n, -1)
    power_sorted = power[:, order]
    total = power_sorted.sum(dim=1, keepdim=True).clamp_min(eps)

    low_budget = low_frac.view(-1, 1).clamp(0.0, 1.0) * total
    high_budget = high_frac.view(-1, 1).clamp(0.0, 1.0) * total

    low_cum = torch.cumsum(power_sorted, dim=1)
    high_cum = torch.cumsum(power_sorted.flip(1), dim=1).flip(1)

    # Strict '<=' means the actually discarded energy never exceeds the sampled budget.
    remove_low = low_cum <= low_budget
    remove_high = high_cum <= high_budget
    keep_sorted = ~(remove_low | remove_high)

    keep = keep_sorted[:, inv_order].reshape(n, 1, H, W // 2 + 1).to(fft.real.dtype)
    fft.mul_(keep)


# -----------------------------------------------------------------------------
# Blur helpers
# -----------------------------------------------------------------------------
def gaussian_blur_separable(x, sigma, kernel_size=15):
    if kernel_size % 2 == 0:
        kernel_size += 1

    if not torch.is_tensor(sigma):
        sigma_t = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    else:
        assert sigma.numel() == 1, "Provide scalar sigma here (bucket per group if needed)."
        sigma_t = sigma.to(x.device, x.dtype)

    sigma_t = torch.clamp(sigma_t, min=torch.finfo(x.dtype).eps)
    _, C, _, _ = x.shape

    g = _CACHE.gauss1d(kernel_size, float(sigma_t), x.device, x.dtype)
    g = g.view(1, 1, kernel_size, 1)
    pad_v = (0, 0, kernel_size // 2, kernel_size // 2)
    x = F.pad(x, pad_v, mode="replicate")
    x = F.conv2d(x, g.expand(C, 1, -1, 1), groups=C)

    g2 = g.transpose(2, 3)
    pad_h = (kernel_size // 2, kernel_size // 2, 0, 0)
    x = F.pad(x, pad_h, mode="replicate")
    x = F.conv2d(x, g2.expand(C, 1, 1, -1), groups=C)
    return x


@torch.no_grad()
def blur_batched_by_sigma(x, sigmas, kernel_size, bucket_step=0.05):
    B = x.shape[0]
    if sigmas.dim() != 1:
        sigmas = sigmas.view(B)

    s = torch.clamp(sigmas, min=0.0)
    s_q = torch.round(s / bucket_step) * bucket_step

    out = x.clone()
    for v in torch.unique(s_q):
        idx = (s_q == v).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0 or float(v) <= 0.0:
            continue
        out[idx] = gaussian_blur_separable(out[idx], v, kernel_size=kernel_size)
    return out


# -----------------------------------------------------------------------------
# New signal-domain augmentations
# -----------------------------------------------------------------------------
def apply_random_wavelet_to_fft_(fft, peak_freq, bandwidth_frac, phase_deg, strength, H):
    """
    Apply a randomized band-limited wavelet response along the vertical/time axis.

    peak_freq is in cycles/sample, bandwidth_frac is relative to peak_freq, and
    phase_deg is a constant seismic phase rotation. Horizontal wavenumber is free.
    """
    f = _CACHE.fftfreq(H, fft.device, fft.real.dtype).view(1, 1, H, 1)
    af = f.abs()
    f0 = peak_freq.view(-1, 1, 1, 1).clamp_min(1e-4)
    bw = (f0 * bandwidth_frac.view(-1, 1, 1, 1)).clamp_min(1e-4)
    amp = torch.exp(-0.5 * ((af - f0) / bw).square())

    phase = torch.deg2rad(phase_deg).view(-1, 1, 1, 1) * torch.sign(f)
    phase = torch.where(af > 0, phase, torch.zeros_like(phase))
    wavelet = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))

    mix = strength.view(-1, 1, 1, 1).clamp(0.0, 1.0)
    response = (1.0 - mix) + mix * wavelet
    fft.mul_(response)


def apply_gain_contrast_clip_agc(
    x,
    do_gain, gain_db,
    do_contrast, contrast_gamma,
    do_clip, clip_rms,
    do_agc, agc_window,
    eps=1e-6,
):
    """All masks/parameters are [B]."""
    y = x.clone()
    B = y.shape[0]

    if do_gain.any():
        idx = do_gain.nonzero(as_tuple=False).squeeze(1)
        g = torch.pow(
            torch.tensor(10.0, device=y.device, dtype=y.dtype),
            gain_db[idx] / 20.0,
        ).view(-1, 1, 1, 1)
        y[idx] = y[idx] * g

    if do_contrast.any():
        idx = do_contrast.nonzero(as_tuple=False).squeeze(1)
        sub = y[idx]
        rms = sub.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
        z = sub / rms
        gamma = contrast_gamma[idx].view(-1, 1, 1, 1)
        z = torch.sign(z) * torch.pow(torch.abs(z).clamp_min(eps), gamma)
        y[idx] = z * rms

    if do_clip.any():
        idx = do_clip.nonzero(as_tuple=False).squeeze(1)
        sub = y[idx]
        rms = sub.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
        lim = clip_rms[idx].view(-1, 1, 1, 1) * rms
        y[idx] = torch.maximum(torch.minimum(sub, lim), -lim)

    if do_agc.any():
        idx_all = do_agc.nonzero(as_tuple=False).squeeze(1)
        windows = agc_window[idx_all]
        # Bucket by integer window size. AGC is vertical/time-axis only.
        for w in torch.unique(windows):
            idx = idx_all[windows == w]
            if idx.numel() == 0:
                continue
            wi = int(w.item())
            wi = max(3, wi)
            if wi % 2 == 0:
                wi += 1
            wi = min(wi, y.shape[-2] if y.shape[-2] % 2 == 1 else y.shape[-2] - 1)
            sub = y[idx]
            local_power = F.avg_pool2d(
                sub.square(), kernel_size=(wi, 1), stride=1,
                padding=(wi // 2, 0),
            )
            env = torch.sqrt(local_power + eps)
            # Keep roughly the original global RMS so AGC does not always imply
            # an unrelated global normalization.
            global_rms = sub.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
            y[idx] = sub / env * global_rms

    return y


def add_patterned_noise_batched(
    x, mask, strength, kind, corr_sigma, coherent_freq, coherent_slope, coherent_phase,
    blur_kernel_size=15, eps=1e-6,
):
    """
    kind:
      0 white
      1 horizontally coherent (same noise across traces)
      2 vertical/acquisition stripes
      3 smooth correlated 2-D noise
      4 dipping coherent sinusoidal event field

    strength is noise RMS as a fraction of input RMS.
    """
    if not mask.any():
        return x

    y = x.clone()
    B, C, H, W = y.shape
    yy = torch.arange(H, device=y.device, dtype=y.dtype).view(1, 1, H, 1)
    xx = torch.arange(W, device=y.device, dtype=y.dtype).view(1, 1, 1, W)

    for k in range(5):
        idx = ((kind == k) & mask).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        n = idx.numel()

        if k == 0:
            noise = torch.randn((n, C, H, W), device=y.device, dtype=y.dtype)
        elif k == 1:
            noise = torch.randn((n, C, H, 1), device=y.device, dtype=y.dtype).expand(-1, -1, -1, W)
        elif k == 2:
            noise = torch.randn((n, C, 1, W), device=y.device, dtype=y.dtype).expand(-1, -1, H, -1)
        elif k == 3:
            noise = torch.randn((n, C, H, W), device=y.device, dtype=y.dtype)
            noise = blur_batched_by_sigma(
                noise, corr_sigma[idx], kernel_size=blur_kernel_size, bucket_step=0.25
            )
        else:
            freq = coherent_freq[idx].view(-1, 1, 1, 1)
            slope = coherent_slope[idx].view(-1, 1, 1, 1)
            phase = coherent_phase[idx].view(-1, 1, 1, 1)
            arg = 2.0 * math.pi * freq * (yy + slope * xx) + phase
            base = torch.sin(arg)
            noise = base.expand(-1, C, -1, -1)

        nrms = noise.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
        xrms = y[idx].square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
        amp = strength[idx].view(-1, 1, 1, 1)
        y[idx] = y[idx] + noise / nrms * xrms * amp

    return y


def sample_statics_profiles(B, W, max_knots, device, dtype):
    """Random smooth statics profiles with per-sample knot count and positions."""
    profiles = torch.empty(B, W, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    max_knots = max(3, int(max_knots))
    for b in range(B):
        k = int(torch.randint(3, max_knots + 1, (1,), device=device).item())
        inner = torch.rand(k - 2, device=device, dtype=dtype) * (W - 1)
        pos = torch.cat((torch.zeros(1, device=device, dtype=dtype), inner, torch.full((1,), W - 1, device=device, dtype=dtype))).sort().values
        vals = torch.randn(k, device=device, dtype=dtype)
        j = torch.searchsorted(pos, x, right=True).clamp(1, k - 1)
        x0, x1 = pos[j - 1], pos[j]
        v0, v1 = vals[j - 1], vals[j]
        profiles[b] = v0 + (v1 - v0) * (x - x0) / (x1 - x0).clamp_min(1e-6)
    return profiles


def apply_statics_batched(x, mask, max_shift, profile):
    """
    Smooth tracewise vertical statics. This changes trace geometry, so apply the
    exact same warp to seismic and continuous RGT labels.

    profile: [B,W], sampled with the rest of the augmentation parameters so paired
    views can share exactly the same static field when desired.
    """
    if not mask.any():
        return x

    y = x.clone()
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    sub = y[idx]
    n, C, H, W = sub.shape

    profile = profile[idx]
    profile = profile / profile.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    shift = profile * max_shift[idx].view(-1, 1)  # [n,W], pixels

    ypix = torch.arange(H, device=sub.device, dtype=sub.dtype).view(1, H, 1).expand(n, -1, W)
    xpix = torch.arange(W, device=sub.device, dtype=sub.dtype).view(1, 1, W).expand(n, H, -1)
    src_y = ypix - shift.view(n, 1, W)

    gx = 2.0 * (xpix + 0.5) / W - 1.0
    gy = 2.0 * (src_y + 0.5) / H - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    sub = F.grid_sample(sub, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y[idx] = sub
    return y


# -----------------------------------------------------------------------------
# Structural augmentations
# -----------------------------------------------------------------------------
def apply_fault_batched(x, mask, fault_x_frac, fault_slope, fault_throw, fault_side):
    """
    Simple block-fault transform on one side of a line
        x_fault(z) = x0 + slope * (z - z_center)
    with a vertical throw in pixels.

    Apply this identically to seismic and RGT.
    """
    if not mask.any():
        return x

    y = x.clone()
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    sub = y[idx]
    n, C, H, W = sub.shape

    ypix = torch.arange(H, device=sub.device, dtype=sub.dtype).view(1, H, 1).expand(n, -1, W)
    xpix = torch.arange(W, device=sub.device, dtype=sub.dtype).view(1, 1, W).expand(n, H, -1)

    x0 = fault_x_frac[idx].view(-1, 1, 1) * (W - 1)
    slope = fault_slope[idx].view(-1, 1, 1)
    line = x0 + slope * (ypix - (H - 1) / 2.0)

    right = xpix >= line
    side_right = fault_side[idx].view(-1, 1, 1)
    moved = torch.where(side_right, right, ~right)

    throw = fault_throw[idx].view(-1, 1, 1)
    src_y = ypix - moved.to(sub.dtype) * throw

    gx = 2.0 * (xpix + 0.5) / W - 1.0
    gy = 2.0 * (src_y + 0.5) / H - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    sub = F.grid_sample(sub, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y[idx] = sub
    return y


def affine_and_crop_batched(
    x,
    do_aug,
    do_aniso, scale_x, scale_z,
    do_shear, shear_deg,
    do_rotation, rotation_deg,
    do_hflip, do_vflip,
    do_zoom, zoom,
    crop_left, crop_top,
    out_size,
    guard_y_px=None,
):
    """
    Compose anisotropic scale + shear + rotation + flips + isotropic zoom, then
    directly sample the final out_size crop from the larger source image.

    This deliberately fuses affine resampling and the 196->128 crop into ONE
    grid_sample. The crop center is pulled toward the image center only when a
    particular transform combination would otherwise sample outside the source.
    If the transformed crop footprint itself is too large, it is minimally fit
    inside the source (equivalent to the smallest extra zoom-in required).
    """
    B, C, H, W = x.shape
    OH = OW = int(out_size)
    if H < OH or W < OW:
        raise ValueError(f"Input {H}x{W} is smaller than output crop {OH}x{OW}.")

    dev, dt = x.device, x.dtype
    one = torch.ones(B, device=dev, dtype=dt)

    active_aniso = do_aniso & do_aug
    active_shear = do_shear & do_aug
    active_rot = do_rotation & do_aug
    active_hf = do_hflip & do_aug
    active_vf = do_vflip & do_aug
    active_zoom = do_zoom & do_aug

    sx = torch.where(active_aniso, scale_x, one)
    sz = torch.where(active_aniso, scale_z, one)
    zz = torch.where(active_zoom, zoom, one)
    fx = torch.where(active_hf, -one, one)
    fz = torch.where(active_vf, -one, one)

    # Forward transform: F = R @ Sh_x @ S
    ax = sx * zz * fx
    az = sz * zz * fz
    sh = torch.tan(torch.deg2rad(torch.where(active_shear, shear_deg, torch.zeros_like(shear_deg))))
    ang = torch.deg2rad(torch.where(active_rot, rotation_deg, torch.zeros_like(rotation_deg)))
    ca, sa = torch.cos(ang), torch.sin(ang)

    # Sh @ S = [[ax, sh*az], [0, az]]
    f00 = ca * ax
    f01 = ca * sh * az - sa * az
    f10 = sa * ax
    f11 = sa * sh * az + ca * az

    det = (f00 * f11 - f01 * f10).clamp(min=-1e12, max=1e12)
    det = torch.where(det.abs() < 1e-6, torch.sign(det) * 1e-6 + (det == 0).to(dt) * 1e-6, det)

    # F^{-1}
    i00 = f11 / det
    i01 = -f01 / det
    i10 = -f10 / det
    i11 = f00 / det

    # An exact out_size crop from an untransformed HxW image corresponds to this
    # normalized linear scale with align_corners=False.
    dx = torch.tensor(OW / W, device=dev, dtype=dt)
    dy = torch.tensor(OH / H, device=dev, dtype=dt)

    a00 = i00 * dx
    a01 = i01 * dy
    a10 = i10 * dx
    a11 = i11 * dy

    # Crop center in the *transformed* full-image coordinate system.
    cx = 2.0 * (crop_left.to(dt) + OW / 2.0) / W - 1.0
    cy = 2.0 * (crop_top.to(dt) + OH / 2.0) / H - 1.0
    b0 = i00 * cx + i01 * cy
    b1 = i10 * cx + i11 * cy

    # Keep away from top/bottom borders that may have been touched by fault/statics.
    if guard_y_px is None:
        guard_y = torch.zeros(B, device=dev, dtype=dt)
    else:
        guard_y = (2.0 * guard_y_px.to(dt) / H).clamp(0.0, 0.45)
    limit_x = torch.ones(B, device=dev, dtype=dt)
    limit_y = 1.0 - guard_y

    # Corner extents of affine footprint. If an extreme combination does not fit,
    # minimally shrink the footprint rather than create black/reflected borders.
    ex = a00.abs() + a01.abs()
    ey = a10.abs() + a11.abs()
    fit = torch.maximum(ex / limit_x.clamp_min(1e-6), ey / limit_y.clamp_min(1e-6)).clamp_min(1.0)
    a00, a01, a10, a11 = a00 / fit, a01 / fit, a10 / fit, a11 / fit

    ex = a00.abs() + a01.abs()
    ey = a10.abs() + a11.abs()
    allow_x = (limit_x - ex).clamp_min(0.0)
    allow_y = (limit_y - ey).clamp_min(0.0)

    # Preserve the random crop direction, but pull toward center when necessary.
    alphax = torch.where(b0.abs() > 1e-6, allow_x / b0.abs(), torch.ones_like(b0))
    alphay = torch.where(b1.abs() > 1e-6, allow_y / b1.abs(), torch.ones_like(b1))
    alpha = torch.minimum(torch.minimum(alphax, alphay), torch.ones_like(b0)).clamp(0.0, 1.0)
    b0, b1 = b0 * alpha, b1 * alpha

    theta = torch.stack(
        (
            torch.stack((a00, a01, b0), dim=1),
            torch.stack((a10, a11, b1), dim=1),
        ),
        dim=1,
    )

    grid = F.affine_grid(theta, size=(B, C, OH, OW), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


# -----------------------------------------------------------------------------
# Main augmentation module
# -----------------------------------------------------------------------------
class SeismicAugmentation(nn.Module):
    def __init__(
        self,
        aug_p=0.7,
        output_size=128,

        # Existing augmentations
        init_blur_sigma_range=(1.2, 2.0), p_init_blur=0.7,                      
        noise_phase_sigma_range=(0.2, 0.35), p_noise_phase=0.3,
        noise_amp_sigma_range=(0.5, 0.7), p_noise_amp=0.65,
        blur_amp_sigma_range=(0.8, 1.3),   p_blur_amp=0.45,
        freq_low_energy_remove_range=(0.0, 0.25), freq_high_energy_remove_range=(0.0, 0.25), max_freq_energy_remove=0.5, p_freq_filter=0.3,

        # Random wavelet / polarity
        wavelet_peak_freq_range=(0.04, 0.22),
        wavelet_bandwidth_frac_range=(0.25, 0.80),
        wavelet_phase_deg_range=(-45.0, 45.0),
        wavelet_strength_range=(0.55, 1.0), p_wavelet=0.0,
        p_polarity=0.20,

        # Gain / contrast / clipping / AGC
        gain_db_range=(-4.0, 4.0), p_gain=0.35,
        contrast_gamma_range=(0.95, 1.05), p_contrast=0.30,
        clip_rms_range=(1.5, 5.0), p_clip=0.0,
        agc_window_range=(9, 61), p_agc=0.0,

        # Patterned additive noise
        noise_strength_range=(0.01, 0.1), p_noise_pattern=0.45,
        noise_corr_sigma_range=(1.0, 3.0),
        coherent_freq_range=(0.015, 0.10),
        coherent_slope_range=(-1.0, 1.0),

        # Statics / tracewise vertical jitter (image only)
        statics_shift_range=(1.0, 10.0), p_statics=0.30,
        statics_knots=16,
        
        # Structural/global geometry
        aniso_x_range=(0.85, 1.15),
        aniso_z_range=(0.85, 1.15), p_aniso=0.35,
        shear_deg_range=(-12.0, 12.0), p_shear=0.30,
        rotation_deg_range=(-25.0, 25.0), p_rotation=0.30,
        p_hflip=0.50,
        p_vflip=0.0,
        zoom_range=(0.75, 1.35), p_zoom=0.35,

        # Synthetic fault / block displacement
        fault_x_frac_range=(0.20, 0.80),
        fault_slope_range=(-0.6, 0.6),
        fault_throw_range=(4.0, 28.0), p_fault=0.5,

        blur_kernel_size=15,
        keep_exact_phase_mult=False,
        invert_rgt_on_vflip=False,
        compile_ready=False,

        # Optional arguments are appended to preserve positional compatibility.
        fourier_target_pool=None,
        p_fourier_mix=0.0,
        fourier_beta_a=15.0,
        fourier_beta_b=10.0,
    ):
        super().__init__()
        self.aug_p = aug_p
        self.output_size = int(output_size)

        self.init_blur_sigma_range = init_blur_sigma_range
        self.p_init_blur = p_init_blur
        self.noise_phase_sigma_range = noise_phase_sigma_range
        self.p_noise_phase = p_noise_phase
        self.noise_amp_sigma_range = noise_amp_sigma_range
        self.p_noise_amp = p_noise_amp
        self.blur_amp_sigma_range = blur_amp_sigma_range
        self.p_blur_amp = p_blur_amp
        self.freq_low_energy_remove_range = freq_low_energy_remove_range
        self.freq_high_energy_remove_range = freq_high_energy_remove_range
        self.max_freq_energy_remove = float(max_freq_energy_remove)
        self.p_freq_filter = p_freq_filter

        self.fourier_target_pool = fourier_target_pool
        self.p_fourier_mix = float(p_fourier_mix)
        self.fourier_beta_a = float(fourier_beta_a)
        self.fourier_beta_b = float(fourier_beta_b)
        if not 0.0 <= self.p_fourier_mix <= 1.0:
            raise ValueError("p_fourier_mix must lie in [0, 1].")
        if self.fourier_beta_a <= 0.0 or self.fourier_beta_b <= 0.0:
            raise ValueError("Fourier Beta distribution parameters must be positive.")
        if self.p_fourier_mix > 0.0 and (
            self.fourier_target_pool is None or len(self.fourier_target_pool) == 0
        ):
            raise ValueError(
                "An enabled Fourier mix requires a nonempty fourier_target_pool."
            )

        self.wavelet_peak_freq_range = wavelet_peak_freq_range
        self.wavelet_bandwidth_frac_range = wavelet_bandwidth_frac_range
        self.wavelet_phase_deg_range = wavelet_phase_deg_range
        self.wavelet_strength_range = wavelet_strength_range
        self.p_wavelet = p_wavelet
        self.p_polarity = p_polarity

        self.gain_db_range = gain_db_range
        self.p_gain = p_gain
        self.contrast_gamma_range = contrast_gamma_range
        self.p_contrast = p_contrast
        self.clip_rms_range = clip_rms_range
        self.p_clip = p_clip
        self.agc_window_range = agc_window_range
        self.p_agc = p_agc

        self.noise_strength_range = noise_strength_range
        self.p_noise_pattern = p_noise_pattern
        self.noise_corr_sigma_range = noise_corr_sigma_range
        self.coherent_freq_range = coherent_freq_range
        self.coherent_slope_range = coherent_slope_range

        self.statics_shift_range = statics_shift_range
        self.p_statics = p_statics
        self.statics_knots = int(statics_knots)

        self.aniso_x_range = aniso_x_range
        self.aniso_z_range = aniso_z_range
        self.p_aniso = p_aniso
        self.shear_deg_range = shear_deg_range
        self.p_shear = p_shear
        self.rotation_deg_range = rotation_deg_range
        self.p_rotation = p_rotation
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.zoom_range = zoom_range
        self.p_zoom = p_zoom

        self.fault_x_frac_range = fault_x_frac_range
        self.fault_slope_range = fault_slope_range
        self.fault_throw_range = fault_throw_range
        self.p_fault = p_fault

        self.blur_kernel_size = blur_kernel_size
        self.keep_exact_phase_mult = keep_exact_phase_mult
        self.invert_rgt_on_vflip = invert_rgt_on_vflip

        torch.backends.cudnn.benchmark = True
        self.compile_ready = compile_ready

    def _sample_params(self, B, H, W, device, dtype):
        u = lambda p: (torch.rand(B, device=device) < p)
        ur = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=device, dtype=dtype)

        toggles = {
            "do_aug": u(self.aug_p),
            "do_init_blur": u(self.p_init_blur),
            "do_np": u(self.p_noise_phase),
            "do_na": u(self.p_noise_amp),
            "do_ba": u(self.p_blur_amp),
            "do_freq": u(self.p_freq_filter),
            "do_wavelet": u(self.p_wavelet),
            "do_polarity": u(self.p_polarity),
            "do_gain": u(self.p_gain),
            "do_contrast": u(self.p_contrast),
            "do_clip": u(self.p_clip),
            "do_agc": u(self.p_agc),
            "do_noise_pattern": u(self.p_noise_pattern),
            "do_statics": u(self.p_statics),
            "do_aniso": u(self.p_aniso),
            "do_shear": u(self.p_shear),
            "do_rotation": u(self.p_rotation),
            "do_hflip": u(self.p_hflip),
            "do_vflip": u(self.p_vflip),
            "do_zoom": u(self.p_zoom),
            "do_fault": u(self.p_fault),
        }

        # Signed fault throw with nonzero sampled magnitude.
        throw_mag = ur(*self.fault_throw_range)
        throw_sign = torch.where(u(0.5), -torch.ones(B, device=device, dtype=dtype), torch.ones(B, device=device, dtype=dtype))

        lo_agc, hi_agc = self.agc_window_range
        agc_window = torch.randint(int(lo_agc), int(hi_agc) + 1, (B,), device=device)
        agc_window = agc_window + (agc_window % 2 == 0).to(agc_window.dtype)

        max_left = max(0, W - self.output_size)
        max_top = max(0, H - self.output_size)
        crop_left = torch.randint(0, max_left + 1, (B,), device=device) if max_left > 0 else torch.zeros(B, device=device, dtype=torch.long)
        crop_top = torch.randint(0, max_top + 1, (B,), device=device) if max_top > 0 else torch.zeros(B, device=device, dtype=torch.long)

        params = {
            "sigma_init": ur(*self.init_blur_sigma_range),
            "sigma_np": ur(*self.noise_phase_sigma_range),
            "sigma_na": ur(*self.noise_amp_sigma_range),
            "sigma_ba": ur(*self.blur_amp_sigma_range),
            "freq_low_energy_remove": ur(*self.freq_low_energy_remove_range),
            "freq_high_energy_remove": ur(*self.freq_high_energy_remove_range),
            "wavelet_peak": ur(*self.wavelet_peak_freq_range),
            "wavelet_bandwidth_frac": ur(*self.wavelet_bandwidth_frac_range),
            "wavelet_phase_deg": ur(*self.wavelet_phase_deg_range),
            "wavelet_strength": ur(*self.wavelet_strength_range),
            "gain_db": ur(*self.gain_db_range),
            "contrast_gamma": ur(*self.contrast_gamma_range),
            "clip_rms": ur(*self.clip_rms_range),
            "agc_window": agc_window,
            "noise_strength": ur(*self.noise_strength_range),
            "noise_kind": torch.randint(0, 5, (B,), device=device),
            "noise_corr_sigma": ur(*self.noise_corr_sigma_range),
            "coherent_freq": ur(*self.coherent_freq_range),
            "coherent_slope": ur(*self.coherent_slope_range),
            "coherent_phase": ur(0.0, 2.0 * math.pi),
            "static_max_shift": ur(*self.statics_shift_range),
            "static_profile": sample_statics_profiles(B, W, self.statics_knots, device, dtype),
            "scale_x": ur(*self.aniso_x_range),
            "scale_z": ur(*self.aniso_z_range),
            "shear_deg": ur(*self.shear_deg_range),
            "rotation_deg": ur(*self.rotation_deg_range),
            "zoom": ur(*self.zoom_range),
            "fault_x_frac": ur(*self.fault_x_frac_range),
            "fault_slope": ur(*self.fault_slope_range),
            "fault_throw": throw_mag * throw_sign,
            "fault_side": u(0.5),
            "crop_left": crop_left,
            "crop_top": crop_top,
        }

        total_remove = params["freq_low_energy_remove"] + params["freq_high_energy_remove"]
        scale = torch.clamp(self.max_freq_energy_remove / total_remove.clamp_min(1e-12), max=1.0)
        params["freq_low_energy_remove"] = params["freq_low_energy_remove"] * scale
        params["freq_high_energy_remove"] = params["freq_high_energy_remove"] * scale

        fourier_toggle, fourier_index, fourier_alpha = self._sample_fourier_params(
            B, device, dtype
        )
        toggles["do_fourier_mix"] = fourier_toggle
        params["fourier_target_index"] = fourier_index
        params["fourier_alpha"] = fourier_alpha
        return toggles, params

    def _sample_fourier_params(self, B, device, dtype):
        """Sample per-image target identities and Beta-distributed mix weights."""
        # Preserve the exact legacy RNG stream when this optional augmentation
        # is disabled by avoiding any additional random draws.
        if self.p_fourier_mix <= 0.0:
            return (
                torch.zeros(B, device=device, dtype=torch.bool),
                torch.zeros(B, device=device, dtype=torch.long),
                torch.zeros(B, device=device, dtype=dtype),
            )

        do_fourier_mix = torch.rand(B, device=device) < self.p_fourier_mix
        target_index = torch.randint(
            0,
            len(self.fourier_target_pool),
            (B,),
            device=device,
        )
        beta = torch.distributions.Beta(
            torch.tensor(self.fourier_beta_a, device=device, dtype=torch.float32),
            torch.tensor(self.fourier_beta_b, device=device, dtype=torch.float32),
        )
        alpha = beta.sample((B,)).to(dtype=dtype)
        return do_fourier_mix, target_index, alpha

    # Which operations define "structure" vs "resolution/appearance" for your
    # existing triplet coupling logic.
    def _sr_keys(self):
        struct_toggles = [
            "do_statics",
            "do_aniso", "do_shear", "do_rotation", "do_hflip", "do_vflip",
            "do_zoom", "do_fault",
        ]
        struct_params = [
            "static_max_shift", "static_profile",
            "scale_x", "scale_z", "shear_deg", "rotation_deg", "zoom",
            "fault_x_frac", "fault_slope", "fault_throw", "fault_side",
            "crop_left", "crop_top",
        ]

        resol_toggles = [
            "do_init_blur", "do_na", "do_ba", "do_freq", "do_np",
            "do_wavelet", "do_polarity", "do_gain", "do_contrast", "do_clip",
            "do_agc", "do_noise_pattern", "do_fourier_mix",
        ]
        resol_params = [
            "sigma_init", "sigma_na", "sigma_ba", "freq_low_energy_remove", "freq_high_energy_remove", "sigma_np",
            "wavelet_peak", "wavelet_bandwidth_frac", "wavelet_phase_deg", "wavelet_strength", "gain_db", "contrast_gamma", "clip_rms",
            "agc_window", "noise_strength", "noise_kind", "noise_corr_sigma",
            "coherent_freq", "coherent_slope", "coherent_phase",
            "fourier_target_index", "fourier_alpha",
        ]
        return struct_toggles, struct_params, resol_toggles, resol_params

    def _clone_tp(self, toggles, params):
        t2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in toggles.items()}
        p2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in params.items()}
        return t2, p2

    def _recompute_do_aug(self, toggles):
        # Keep aug_p as the global gate. Operations already have their own toggles,
        # so changing tied toggles in the triplet path does not require replacing it.
        return toggles

    @torch.no_grad()
    def forward(self, img, img2=None, img3=None, label=False, per_image=False, residual=False, rgt=None):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img2 is not None and img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        if img3 is not None and img3.dim() == 3:
            img3 = img3.unsqueeze(0)
        if rgt is not None and rgt.dim() == 3:
            rgt = rgt.unsqueeze(1)

        if residual:
            return self._forward_residual(img, img2, img3, per_image=per_image, rgt=rgt)

        if img2 is None:
            return self._forward_single(img, per_image=per_image)
        elif img3 is None:
            return self._forward_pair(img, img2, per_image=per_image, label=label)
        else:
            return self.forward_triplet(img, img2, img3, per_image=per_image)

    @torch.no_grad()
    def _forward_pair(self, x1, x2, per_image=False, label=False):
        B, _, H, W = x1.shape
        device, dtype = x1.device, x1.dtype
        toggles, params = self._sample_params(B, H, W, device, dtype)
        y1 = self._apply_with_params(x1, toggles, params, per_image, label=False)
        y2 = self._apply_with_params(x2, toggles, params, per_image, label=label)
        return y1, y2

    @torch.no_grad()
    def _forward_single(self, x, per_image=False):
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        toggles, params = self._sample_params(B, H, W, device, dtype)
        return self._apply_with_params(x, toggles, params, per_image, label=False)

    @torch.no_grad()
    def forward_triplet(self, x_s1r1, x_s2r1, x_s1r2, per_image=False):
        B, _, H, W = x_s1r1.shape
        dev, dt = x_s1r1.device, x_s1r1.dtype

        t1, p1 = self._sample_params(B, H, W, dev, dt)
        t2, p2 = self._sample_params(B, H, W, dev, dt)
        t3, p3 = self._sample_params(B, H, W, dev, dt)

        t2, p2 = self._clone_tp(t2, p2)
        t3, p3 = self._clone_tp(t3, p3)
        t2["do_aug"] = t1["do_aug"].clone()
        t3["do_aug"] = t1["do_aug"].clone()
        struct_tog, struct_par, resol_tog, resol_par = self._sr_keys()

        # Tie resolution/appearance between S1R1 and S2R1.
        for k in resol_tog:
            t2[k] = t1[k].clone()
        for k in resol_par:
            p2[k] = p1[k].clone()

        # Tie structure between S1R1 and S1R2.
        for k in struct_tog:
            t3[k] = t1[k].clone()
        for k in struct_par:
            p3[k] = p1[k].clone()

        y11 = self._apply_with_params(x_s1r1, t1, p1, per_image=per_image, label=False)
        y21 = self._apply_with_params(x_s2r1, t2, p2, per_image=per_image, label=False)
        y12 = self._apply_with_params(x_s1r2, t3, p3, per_image=per_image, label=False)
        return y11, y21, y12

    @torch.no_grad()
    def _forward_residual(self, x1, x2, x3, per_image=False, rgt=None):
        B, _, H, W = x1.shape
        device, dtype = x1.device, x1.dtype
    
        t1, p1 = self._sample_params(B, H, W, device, dtype)
        y1 = self._apply_with_params(x1, t1, p1, per_image, label=False)
    
        t2, p2 = self._sample_params(B, H, W, device, dtype)
        struct_tog, struct_par, resol_tog, _ = self._sr_keys()
    
        # x2 always enters the augmentation pipeline.
        t2["do_aug"] = torch.ones_like(t1["do_aug"])
    
        # If x1 was augmented, x2 gets exactly the same structural augmentation.
        # If x1 was not augmented, x2 gets no structural augmentation.
        for k in struct_tog:
            t2[k] = t1[k] & t1["do_aug"]
    
        for k in struct_par:
            p2[k] = p1[k].clone()
    
        # Resolution is complementary:
        # x1 augmented -> x2 gets no resolution augmentation
        # x1 untouched  -> x2 gets independently sampled resolution augmentation
        for k in resol_tog:
            t2[k] = t2[k] & ~t1["do_aug"]
    
        y2 = self._apply_with_params(x2, t2, p2, per_image, label=False)
    
        t3, p3 = self._sample_params(B, H, W, device, dtype)
        y3 = self._apply_with_params(x3, t3, p3, per_image, label=False)

        if rgt is None:
            return y1, y2, y3

        # The anchor RGT gets x1's geometry only (statics, fault, affine, crop),
        # which is also the structural geometry shared by x2.
        rgt = self._apply_with_params(rgt.to(device=device, dtype=dtype), t1, p1, per_image, label=True)
        return y1, y2, y3, rgt

    @torch.no_grad()
    def _apply_with_params(self, x, toggles, params, per_image, label=False):
        B, C, H, W = x.shape
        dev, dt = x.device, x.dtype
        gate = toggles["do_aug"]

        # ------------------------------------------------------------------
        # 1) SIGNAL / RESOLUTION / ACQUISITION AUGMENTATIONS (image only; no spatial remapping)
        # ------------------------------------------------------------------
        if not label:
            m0 = toggles["do_init_blur"] & gate
            if m0.any():
                idx = m0.nonzero(as_tuple=False).squeeze(1)
                x = x.clone()
                x[idx] = blur_batched_by_sigma(x[idx], params["sigma_init"][idx], self.blur_kernel_size)

            # One 2-D FFT for the existing phase/amplitude operations, band-pass,
            # and the new vertical random-wavelet shaping.
            fft = rfft2(x)

            # Cross-domain appearance transfer: linearly blend synthetic and
            # real target amplitudes while retaining the synthetic phase. This
            # runs before other spectral perturbations so they continue to
            # compose instead of being overwritten by the blend.
            m_fourier = toggles["do_fourier_mix"] & gate
            if m_fourier.any():
                idx = m_fourier.nonzero(as_tuple=False).squeeze(1)
                real = self.fourier_target_pool.fetch(
                    params["fourier_target_index"][idx],
                    image_shape=(C, H, W),
                    device=dev,
                    dtype=dt,
                )
                fft[idx] = mix_fourier_amplitudes(
                    fft[idx],
                    real,
                    params["fourier_alpha"][idx],
                )

            m_w = toggles["do_wavelet"] & gate
            if m_w.any():
                idx = m_w.nonzero(as_tuple=False).squeeze(1)
                sub = fft[idx]
                apply_random_wavelet_to_fft_(
                    sub,
                    params["wavelet_peak"][idx],
                    params["wavelet_bandwidth_frac"][idx],
                    params["wavelet_phase_deg"][idx],
                    params["wavelet_strength"][idx],
                    H,
                )
                fft[idx] = sub

            m_np = toggles["do_np"] & gate
            if m_np.any():
                idx = m_np.nonzero(as_tuple=False).squeeze(1)
                sub = fft[idx]
                s = params["sigma_np"][idx].view(-1, 1, 1, 1)
                if self.keep_exact_phase_mult:
                    amp = torch.abs(sub)
                    phase = torch.angle(sub)
                    noise = 1.0 + torch.randn_like(phase) * s
                    phase = phase * noise
                    sub = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
                else:
                    apply_phase_noise_(sub, s.expand_as(sub.real))
                fft[idx] = sub

            m_na = toggles["do_na"] & gate
            if m_na.any():
                idx = m_na.nonzero(as_tuple=False).squeeze(1)
                sub = fft[idx]
                s = params["sigma_na"][idx].view(-1, 1, 1, 1)
                apply_amp_noise_(sub, s.expand_as(sub.real))
                fft[idx] = sub

            m_ba = toggles["do_ba"] & gate
            if m_ba.any():
                idx = m_ba.nonzero(as_tuple=False).squeeze(1)
                sub = fft[idx]
                amp = torch.abs(sub)
                phase = torch.angle(sub)
                amp = blur_batched_by_sigma(amp, params["sigma_ba"][idx], self.blur_kernel_size)
                fft[idx] = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))

            # Remove bounded fractions of total spectral energy from the low/high
            # radial-frequency tails. Cutoffs adapt independently to each sample.
            m_f = toggles["do_freq"] & gate
            if m_f.any():
                idx = m_f.nonzero(as_tuple=False).squeeze(1)
                sub = fft[idx]
                apply_energy_frequency_filter_(sub, params["freq_low_energy_remove"][idx], params["freq_high_energy_remove"][idx], H, W)
                fft[idx] = sub

            y = irfft2(fft, H, W)

            # Polarity.
            m_pol = toggles["do_polarity"] & gate
            if m_pol.any():
                y = y.clone()
                y[m_pol] = -y[m_pol]

            # Gain / contrast / clipping / AGC.
            y = apply_gain_contrast_clip_agc(
                y,
                toggles["do_gain"] & gate, params["gain_db"],
                toggles["do_contrast"] & gate, params["contrast_gamma"],
                toggles["do_clip"] & gate, params["clip_rms"],
                toggles["do_agc"] & gate, params["agc_window"],
            )

            # Varying noise families.
            y = add_patterned_noise_batched(
                y,
                toggles["do_noise_pattern"] & gate,
                params["noise_strength"], params["noise_kind"], params["noise_corr_sigma"],
                params["coherent_freq"], params["coherent_slope"], params["coherent_phase"],
                blur_kernel_size=self.blur_kernel_size,
            )

        else:
            y = x

        # ------------------------------------------------------------------
        # 2) STRUCTURAL / GEOMETRIC AUGMENTATIONS (image + label)
        # ------------------------------------------------------------------
        m_statics = toggles["do_statics"] & gate
        y = apply_statics_batched(
            y,
            m_statics,
            params["static_max_shift"],
            params["static_profile"],
        )

        m_fault = toggles["do_fault"] & gate
        y = apply_fault_batched(
            y, m_fault,
            params["fault_x_frac"], params["fault_slope"],
            params["fault_throw"], params["fault_side"],
        )

        # ------------------------------------------------------------------
        # 3) GLOBAL GEOMETRY + FINAL RANDOM 128x128 CROP
        # ------------------------------------------------------------------
        # Keep the final crop away from source rows potentially touched by fault
        # padding. Statics use border sampling and do not constrain crop position,
        # avoiding a statics-dependent bias toward vertically central crops.
        fault_guard = torch.where(m_fault, params["fault_throw"].abs(), torch.zeros_like(params["fault_throw"]))
        guard_y = fault_guard + 1.0

        y = affine_and_crop_batched(
            y,
            gate,
            toggles["do_aniso"], params["scale_x"], params["scale_z"],
            toggles["do_shear"], params["shear_deg"],
            toggles["do_rotation"], params["rotation_deg"],
            toggles["do_hflip"], toggles["do_vflip"],
            toggles["do_zoom"], params["zoom"],
            params["crop_left"], params["crop_top"],
            self.output_size,
            guard_y_px=guard_y,
        )

        # Optional convention-specific RGT inversion. Leave False unless your RGT
        # target is normalized so that vertical orientation itself defines 0->1.
        if label and self.invert_rgt_on_vflip:
            m_v = toggles["do_vflip"] & gate
            if m_v.any():
                y = y.clone()
                y[m_v] = 1.0 - y[m_v]

        return y
