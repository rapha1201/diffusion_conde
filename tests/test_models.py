import pytest
import torch

from sld.eval.metrics import RGTMetrics, SeismicMetrics, horizon_depths
from sld.losses import ReconLoss, masked_grad_l1, masked_l1, masked_ssim_loss, monotonic_penalty, spectral_l1
from sld.models import LatentVAE, UNetRegressor, build_model, load_model

TINY = dict(block_out_channels=[32, 32, 32, 32], layers_per_block=1, norm_num_groups=8)


# ----------------------------------------------------------------------------- models

def test_vae_shapes_and_shared_downsampling():
    rgt_ae = LatentVAE(latent_channels=4, **TINY)
    seis_ae = LatentVAE(latent_channels=8, **TINY)
    x = torch.randn(2, 1, 128, 128)
    recon, post = rgt_ae(x)
    assert recon.shape == x.shape
    zy, zx = rgt_ae.encode(x), seis_ae.encode(x)
    assert zy.shape == (2, 4, 16, 16) and zx.shape == (2, 8, 16, 16)
    assert rgt_ae.downsample_factor == seis_ae.downsample_factor == 8
    assert torch.cat([zx, zy], dim=1).shape == (2, 12, 16, 16)


def test_vae_scale_factor_roundtrip_and_persistence(tmp_path):
    vae = LatentVAE(latent_channels=4, **TINY).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        ref = vae.vae.decode(vae.posterior(x).mode()).sample
        vae.scale_factor.fill_(0.37)
        z = vae.encode(x)
        assert torch.allclose(z, vae.posterior(x).mode() * 0.37)
        assert torch.allclose(vae.decode(z), ref, atol=1e-5)
    path = tmp_path / "vae.pt"
    torch.save({"kind": "vae", "model_cfg": vae.model_cfg, "model": vae.state_dict()}, path)
    loaded, _ = load_model(str(path))
    assert float(loaded.scale_factor) == pytest.approx(0.37)


def test_regressor_shapes():
    model = UNetRegressor(block_out_channels=[32, 32, 64, 64], layers_per_block=1, attention_levels=[3], norm_num_groups=8)
    assert model(torch.randn(2, 1, 128, 128)).shape == (2, 1, 128, 128)
    assert isinstance(build_model("regressor", model.model_cfg), UNetRegressor)
    with pytest.raises(ValueError):
        build_model("nope", {})


# ----------------------------------------------------------------------------- losses

def test_masked_losses_ignore_invalid_pixels():
    torch.manual_seed(0)
    target = torch.rand(2, 1, 32, 32)
    mask = torch.ones_like(target)
    mask[:, :, :8] = 0
    pred = target.clone()
    pred[:, :, :8] = 5.0  # garbage only where invalid
    assert masked_l1(pred, target, mask) == pytest.approx(0.0, abs=1e-7)
    assert masked_grad_l1(pred, target, mask) == pytest.approx(0.0, abs=1e-6)
    assert masked_l1(pred, target) > 0.5


def test_ssim_and_spectral_are_zero_at_identity():
    x = torch.rand(2, 1, 32, 32)
    assert masked_ssim_loss(x, x) == pytest.approx(0.0, abs=1e-4)
    assert spectral_l1(x, x) == pytest.approx(0.0, abs=1e-6)


def test_monotonic_penalty_only_where_target_increases():
    target = torch.linspace(0, 1, 16).view(1, 1, 16, 1).expand(1, 1, 16, 4).clone()
    flipped = target.flip(2)
    assert monotonic_penalty(target, target) == 0
    assert monotonic_penalty(flipped, target) > 0
    # Where the target itself decreases (e.g. a repeated section), a decreasing prediction is not penalized.
    assert monotonic_penalty(flipped, flipped) == 0


def test_recon_loss_rejects_wrong_terms_and_uses_kl():
    with pytest.raises(ValueError):
        ReconLoss("rgt", {"spectral": 1.0})
    vae = LatentVAE(latent_channels=4, **TINY)
    x = torch.rand(2, 1, 32, 32)
    recon, post = vae(x)
    total, parts = ReconLoss("rgt", {"l1": 1.0, "grad": 0.5, "ssim": 0.2, "kl": 1e-6, "monotonic": 0.0})(recon, x, None, post)
    assert set(parts) == {"l1", "grad", "ssim", "kl"}
    total.backward()


# ----------------------------------------------------------------------------- metrics

def _ramp(b=2, h=64, w=8):
    return torch.linspace(0, 1, h).view(1, 1, h, 1).expand(b, 1, h, w).clone()


def test_rgt_metrics_perfect_and_shifted():
    t = _ramp()
    valid = torch.ones_like(t)
    m = RGTMetrics()
    m.update(t, t, valid)
    out = m.compute()
    assert out["mae"] == pytest.approx(0.0) and out["hde_px"] == pytest.approx(0.0)
    # Shift every horizon down by 3 samples -> displacement of ~3 px.
    shifted = torch.cat([t[:, :, :1].expand(-1, -1, 3, -1), t[:, :, :-3]], dim=2)
    m = RGTMetrics()
    m.update(shifted, t, valid, groups=["A", "B"])
    out = m.compute()
    assert out["hde_px"] == pytest.approx(3.0, abs=0.5)
    assert set(m.groups()) == {"A", "B"} and m.compute("A")["n"] == 1


def test_rgt_metrics_respect_mask():
    t = _ramp()
    valid = torch.ones_like(t)
    valid[:, :, :10] = 0
    pred = t.clone()
    pred[:, :, :10] = 0.9
    m = RGTMetrics()
    m.update(pred, t, valid)
    assert m.compute()["mae"] == pytest.approx(0.0, abs=1e-7)


def test_horizon_depths_counts_samples_above_level():
    d = horizon_depths(_ramp(b=1, h=11, w=1), torch.ones(1, 1, 11, 1), torch.tensor([0.5]))
    assert d.item() == 5  # values 0.0..0.4 are below 0.5


def test_seismic_metrics():
    x = torch.randn(3, 1, 16, 16)
    m = SeismicMetrics()
    m.update(x, x)
    out = m.compute()
    assert out["mae"] == 0 and out["corr"] == pytest.approx(1.0, abs=1e-5)
