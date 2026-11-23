import math

import pytest
import torch

from sld.guidance import DipScore, gaussian_blur, structure_tensor
from sld.models import ConditionalLatentDiffusion

TINY = dict(block_out_channels=[32, 64], layers_per_block=1, attention_levels=[1], norm_num_groups=8, attention_head_dim=8)
THETA = math.radians(30)  # angle of the reflector normal from the depth axis


def layers(theta=THETA, size=64, period=8.0):
    """Plane dipping reflectors (seismic) and the RGT that follows them, depth = rows."""
    z, x = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                          indexing="ij")
    phase = z * math.cos(theta) + x * math.sin(theta)
    seismic = torch.sin(2 * math.pi * phase / period)[None, None]
    tau = (phase - phase.min()) / (phase.max() - phase.min())
    return seismic, tau[None, None], (z / (size - 1))[None, None]


def test_structure_tensor_recovers_reflector_normal():
    seismic, _, _ = layers()
    nz, nx, c = structure_tensor(seismic)
    inner = (slice(None), slice(None), slice(12, -12), slice(12, -12))
    align = (nz * math.cos(THETA) + nx * math.sin(THETA)).abs()[inner]
    assert align.min() > 0.99 and c[inner].min() > 0.8


def test_dip_score_separates_consistent_rotated_and_flipped_rgt():
    seismic, tau, flat = layers()
    score = DipScore(seismic)
    dip, mono = score.terms(tau)
    assert dip.item() < 0.02 and mono.item() == pytest.approx(0.0, abs=1e-6)  # residual: border padding
    inner = torch.zeros_like(tau)
    inner[..., 10:-10, 10:-10] = 1
    assert DipScore(seismic, mask=inner).terms(tau)[0].item() < 0.003
    dip_flat, _ = score.terms(flat)  # horizontal isolines on 30-degree reflectors: sin^2(30) = 0.25
    assert 0.18 < dip_flat.item() < 0.32
    dip_flip, mono_flip = score.terms(1 - tau)  # same orientation, but tau decreases with depth
    assert dip_flip.item() < 0.02 and mono_flip.item() > 0.95
    assert score(1 - tau).item() > score(tau).item() + 0.9


def test_dip_score_is_differentiable_and_mask_limits_it():
    seismic, tau, flat = layers()
    x = flat.clone().requires_grad_(True)
    DipScore(seismic)(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    mask = torch.zeros_like(tau)
    mask[..., :, :32] = 1
    bad = tau.clone()
    bad[..., :, 40:] = flat[..., :, 40:]  # wrong only outside the mask
    assert DipScore(seismic, mask=mask).terms(bad)[0].item() < 0.02
    assert DipScore(seismic).terms(bad)[0].item() > 0.05


def test_dip_score_rows_follow_repeat_and_index():
    seismic, tau, flat = layers()
    s2 = torch.cat([seismic, seismic.flip(-1)])
    score = DipScore(s2)
    rep = score.repeat_interleave(3)
    taus = torch.cat([tau, flat, tau, tau.flip(-1), flat, tau.flip(-1)])
    assert torch.allclose(rep(taus).view(2, 3), torch.stack([score(torch.cat([tau, tau.flip(-1)])),
                                                                 score(torch.cat([flat, flat])),
                                                                 score(torch.cat([tau, tau.flip(-1)]))], 1))
    assert torch.allclose(score.index(torch.tensor([1]))(tau.flip(-1)), score(torch.cat([tau, tau.flip(-1)]))[1:])


def test_gaussian_blur_preserves_constant():
    x = torch.full((2, 1, 20, 30), 3.0)
    assert torch.allclose(gaussian_blur(x, 2.0), x)


def make_ldm(**kw):
    torch.manual_seed(0)
    return ConditionalLatentDiffusion(cond_channels=8, latent_channels=4, **TINY, **kw).eval()


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_guidance_pulls_samples_toward_the_loss_minimum(prediction_type):
    model = make_ldm(prediction_type=prediction_type)
    z_x = torch.randn(3, 8, 16, 16)
    target = torch.full((3, 4, 16, 16), 2.0)
    calls = []

    def loss(z0):
        calls.append(z0.shape[0])
        return (z0 - target).pow(2).flatten(1).mean(1)

    def run(**kw):
        return model.sample(z_x, num_inference_steps=10, generator=torch.Generator().manual_seed(1), **kw)

    plain = run()
    assert torch.equal(run(guide_fn=loss, guide_scale=0.0), plain)  # off -> bit-identical, never called
    assert torch.equal(run(guide_fn=loss, guide_scale=0.5, guide_start=1.0), plain)
    assert not calls
    guided = run(guide_fn=loss, guide_scale=0.2, guide_start=0.3, guide_inner=2)
    assert len(calls) == 7 * 2  # steps 3..9 of 10, two inner steps each
    assert torch.isfinite(guided).all()

    # Oracle denoiser for data at 0: every step predicts z0 = 0, so unguided sampling ends at 0. The
    # (constant) normalized gradient of the loss moves z0 by +0.2 per inner step, and the last DDIM
    # step (alpha_prev = 1) returns the guided z0 itself: 0 + 2 * 0.2 = 0.4 everywhere.
    acp = model.scheduler.alphas_cumprod

    def oracle(z, t, _):
        a = acp[int(t)]
        return z / (1 - a).sqrt() if prediction_type == "epsilon" else a.sqrt() * z / (1 - a).sqrt()

    model.forward = oracle
    assert run().abs().max() < 1e-4
    assert torch.allclose(run(guide_fn=loss, guide_scale=0.2, guide_inner=2), torch.full_like(target, 0.4), atol=1e-4)


def test_evaluation_helpers():
    from evaluate_sampling import DipStats, pearson, section_mae
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4) * torch.tensor([0.5, 1.0]).view(2, 1, 1, 1)
    valid = torch.ones_like(pred)
    valid[1, ..., :2] = 0
    assert torch.allclose(section_mae(pred, target, valid), torch.tensor([0.5, 1.0]))
    assert pearson(torch.tensor([1.0, 2, 3]), torch.tensor([2.0, 4, 6])).item() == pytest.approx(1.0)
    stats = DipStats()
    stats.add("gt", torch.tensor([0.1, 0.3]), torch.tensor([0.0, 0.0]))
    stats.add_selection(torch.tensor([[1.0, 2, 3], [3.0, 2, 1]]), torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]))
    out = stats.compute()
    assert out["gt"]["dip"] == pytest.approx(0.2) and out["gt"]["n"] == 2
    assert out["selection"]["pearson_within_section"] == pytest.approx(1.0)
