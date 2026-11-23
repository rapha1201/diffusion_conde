import math

import pytest
import torch

from sld.eval.descriptors import gt_descriptors, realization_tags, section_hde, seismic_descriptors
from sld.eval.metrics import DEFAULT_LEVELS, RGTMetrics


def layers(theta_deg, size=96, period=8.0):
    th = math.radians(theta_deg)
    z, x = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                          indexing="ij")
    phase = z * math.cos(th) + x * math.sin(th)
    return torch.sin(2 * math.pi * phase / period)[None, None], ((phase - phase.min()) / (phase.max() - phase.min()))[None, None]


def test_seismic_descriptors_measure_dip_coherence_and_frequency():
    s = torch.cat([layers(0)[0], layers(30)[0], torch.randn(1, 1, 96, 96)])
    d = seismic_descriptors(s)
    assert d["dip_deg"][0] < 2 and abs(d["dip_deg"][1] - 30) < 3
    assert d["coherence"][0] > 0.9 and d["coherence"][1] > 0.85 and d["coherence"][2] < 0.6
    assert d["chaotic_frac"][2] > d["chaotic_frac"][1]
    assert d["wavenumber"][0] == pytest.approx(1 / 8, abs=0.02)  # vertical period 8 px
    assert d["lateral_ratio"][0] < 0.05 and d["lateral_ratio"][1] > 0.4


def test_gt_descriptors_flag_overturning_and_discontinuities():
    s, tau = layers(0)
    valid = torch.ones_like(tau)
    fault = tau.clone()
    fault[..., 48:] = (fault[..., 48:] + 0.3).clamp(max=1.0)  # vertical fault: tau jumps across x = 48
    over = tau.clone()
    over[..., 40:56, :] = over[..., 40:56, :].flip(-2)  # a band where tau decreases with depth
    d = gt_descriptors(torch.cat([s, s, s]), torch.cat([tau, fault, over]), torch.cat([valid] * 3))
    assert d["gt_overturn"][0] < 1e-4 and d["gt_overturn"][2] > 0.05
    assert d["gt_disc_frac"][1] > d["gt_disc_frac"][0]
    assert d["gt_dip"][0] < 0.02 and d["valid_frac"].tolist() == [1.0, 1.0, 1.0]


def test_section_hde_matches_rgt_metrics_for_one_section():
    _, tau = layers(10, size=64)
    pred = (tau + 0.05 * torch.randn_like(tau)).clamp(0, 1)
    valid = torch.ones_like(tau)
    m = RGTMetrics()
    m.update(pred, tau, valid)
    assert section_hde(pred, tau, valid, DEFAULT_LEVELS).item() == pytest.approx(m.compute()["hde_px"], rel=1e-5)


def test_realization_tags_parse_file_names():
    t = realization_tags("Modelo2/Sism20Hz_30perVel_Ruido0p4_0p3_Aten11_7_Sal_Difra0p6_Modelo2_Compressivo_HighAng.segy")
    assert t == {"freq_hz": 20.0, "vel_pct": 30.0, "noise": 0.4, "salt": 1, "diffraction": 0.6, "smooth": 0,
                 "regime": "Compressivo"}
    t = realization_tags("Modelo8/Sism30Hz_100perVel_Ruido0p4_0p3_Aten18_1__Difra0p4_Modelo8_Extensional_HighAng.segy")
    assert t["salt"] == 0 and t["regime"] == "Extensional"
    assert realization_tags("Modelo3/seismic")["regime"] == "unknown"
