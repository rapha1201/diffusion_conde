"""Per-section descriptors, to find which properties of a section make the sampling fail.

Seismic-only descriptors (available on a real survey, so they could flag hard sections at inference):
    coherence           mean structure-tensor coherence (low: chaotic, faulted, noisy)
    chaotic_frac        fraction of pixels with coherence < 0.5
    dip_deg             coherence-weighted mean |reflector dip| from horizontal, degrees
    dip_std_deg         coherence-weighted std of the signed dip (folding / dip variability)
    wavenumber          spectral centroid of the traces along depth, cycles/px (frequency content)
    lateral_ratio       mean |d s/dx| / mean |d s/dz| (lateral heterogeneity)
GT descriptors (need the RGT: only to explain failures, never available at inference):
    gt_overturn         fraction of the |d tau/dz| mass where tau decreases with depth (reverse faults, overturning)
    gt_dip              dip misfit of the GT against its own seismic (seismic and RGT disagree, e.g. at faults)
    gt_spacing_cv       coefficient of variation of d tau/dz (isochore thickness variability, pinch-outs)
    gt_disc_frac        fraction of pixels with |grad tau| > 3x the section median (RGT discontinuities: faults)
    valid_frac          fraction of valid RGT pixels in the crop
All return [B] tensors for [B,1,H,W] inputs; depth = rows.
"""
import re

import torch

from sld.guidance import DipScore, erode, gaussian_blur, gradients, structure_tensor

SEISMIC_KEYS = ("coherence", "chaotic_frac", "dip_deg", "dip_std_deg", "wavenumber", "lateral_ratio")
GT_KEYS = ("gt_overturn", "gt_dip", "gt_spacing_cv", "gt_disc_frac", "valid_frac")


def _wmean(x, w):
    return (x * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-8)


@torch.no_grad()
def seismic_descriptors(seismic, sigma_tensor=3.0):
    s = seismic.float()
    nz, nx, c = structure_tensor(s, sigma_tensor=sigma_tensor)
    sign = torch.where(nz < 0, -1.0, 1.0)  # normal pointing down: the dip sign is then well defined
    dip = torch.rad2deg(torch.atan2(nx * sign, nz * sign))  # 0 = flat reflectors, +-90 = vertical
    w = c
    mean_dip = _wmean(dip, w)
    dip_std = (_wmean((dip - mean_dip.view(-1, 1, 1, 1)) ** 2, w)).sqrt()
    spec = torch.fft.rfft(s - s.mean(dim=2, keepdim=True), dim=2).abs() ** 2  # per trace, along depth
    power = spec.mean(dim=(1, 3))  # [B, F]
    freqs = torch.fft.rfftfreq(s.shape[2], device=s.device)
    centroid = (power * freqs).sum(1) / power.sum(1).clamp_min(1e-12)
    gz, gx = gradients(gaussian_blur(s, 1.0))
    return {"coherence": c.flatten(1).mean(1), "chaotic_frac": (c < 0.5).float().flatten(1).mean(1),
            "dip_deg": _wmean(dip.abs(), w), "dip_std_deg": dip_std, "wavenumber": centroid,
            "lateral_ratio": gx.abs().flatten(1).mean(1) / gz.abs().flatten(1).mean(1).clamp_min(1e-8)}


@torch.no_grad()
def gt_descriptors(seismic, rgt, valid, sigma_tensor=3.0):
    v = valid.float()
    m = erode(v)
    t = gaussian_blur(rgt.float(), 1.0)
    dz, dx = gradients(t)
    g = (dz ** 2 + dx ** 2).sqrt()
    overturn = (torch.relu(-dz) * m).flatten(1).sum(1) / ((dz.abs() * m).flatten(1).sum(1) + 1e-8)
    mean_dz = _wmean(dz, m)
    spacing_cv = _wmean((dz - mean_dz.view(-1, 1, 1, 1)) ** 2, m).sqrt() / mean_dz.abs().clamp_min(1e-8)
    disc = []
    for gi, mi in zip(g, m):  # median per section over its valid pixels
        vals = gi[mi > 0]
        med = vals.median() if vals.numel() else gi.new_tensor(0.0)
        disc.append(((gi > 3 * med) & (mi > 0)).float().sum() / mi.sum().clamp_min(1))
    dip, _ = DipScore(seismic, mask=v, sigma_tensor=sigma_tensor).terms(rgt)
    return {"gt_overturn": overturn, "gt_dip": dip, "gt_spacing_cv": spacing_cv, "gt_disc_frac": torch.stack(disc),
            "valid_frac": v.flatten(1).mean(1)}


def realization_tags(name):
    """Acquisition/geology settings encoded in a seismic dataset name, e.g.
    'Modelo2/Sism20Hz_30perVel_Ruido0p4_0p3_Aten11_7_Sal_Difra0p6_Modelo2_Compressivo_HighAng_SHmax20_e0p003.segy'."""
    base = name.split("/")[-1]

    def num(pattern, default=None):
        m = re.search(pattern, base)
        return float(m.group(1).replace("p", ".")) if m else default

    regime = next((r for r in ("Compressivo", "Extensional", "Transcorrente") if r in base), "unknown")
    return {"freq_hz": num(r"Sism(\d+)Hz"), "vel_pct": num(r"_(\d+)perVel"),
            "noise": num(r"Ruido(\d+(?:p\d+)?)", 0.0), "salt": int("_Sal_" in base),
            "diffraction": num(r"Difra(\d+(?:p\d+)?)", 0.0), "smooth": int("Smooth" in base), "regime": regime}


def section_hde(pred, target, valid, levels):
    """[B] horizon displacement error per section (same definition as RGTMetrics)."""
    from sld.eval.metrics import horizon_depths, horizon_presence

    pred, target, valid = pred.float().clamp(0, 1), target.float(), valid.float()
    lv = levels.to(pred.device)
    present = horizon_presence(target, valid, lv)
    disp = (horizon_depths(pred, valid, lv) - horizon_depths(target, valid, lv)).abs() * present
    return disp.flatten(1).sum(1) / present.flatten(1).sum(1).clamp_min(1)

