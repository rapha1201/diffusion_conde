import pytest
import torch

from sld.eval.metrics import SpreadMetrics
from sld.models import ConditionalLatentDiffusion, UNetRegressor, build_model, load_model
from sld.utils.trainer import EMA

TINY = dict(block_out_channels=[32, 64], layers_per_block=1, attention_levels=[1], norm_num_groups=8, attention_head_dim=8)


def make_ldm(**kw):
    torch.manual_seed(0)
    return ConditionalLatentDiffusion(cond_channels=8, latent_channels=4, **TINY, **kw)


def capture_unet_inputs(model):
    seen = []
    model.unet.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    return seen


def test_training_loss_noises_only_z_y_and_backprops():
    model = make_ldm().train()
    seen = capture_unet_inputs(model)
    z_x, z_y = torch.randn(3, 8, 16, 16), torch.randn(3, 4, 16, 16)
    loss = model.training_loss(z_x, z_y)
    loss.backward()
    assert torch.isfinite(loss)
    assert seen[0].shape == (3, 12, 16, 16)
    assert torch.equal(seen[0][:, :8], z_x)            # condition passed clean
    assert not torch.allclose(seen[0][:, 8:], z_y)     # target noised
    assert all(p.grad is not None for p in model.unet.parameters() if p.requires_grad)


def test_condition_dropout_only_in_training():
    model = make_ldm(cond_drop_prob=1.0)
    seen = capture_unet_inputs(model)
    z_x, z_y = torch.randn(2, 8, 16, 16), torch.randn(2, 4, 16, 16)
    model.train().training_loss(z_x, z_y)
    assert seen[-1][:, :8].abs().sum() == 0
    model.eval().training_loss(z_x, z_y)
    assert torch.equal(seen[-1][:, :8], z_x)


def test_sampler_reanchors_clean_condition_every_step():
    model = make_ldm().eval()
    seen = capture_unet_inputs(model)
    z_x = torch.randn(2, 8, 16, 16)
    z = model.sample(z_x, num_inference_steps=5)
    assert z.shape == (2, 4, 16, 16) and torch.isfinite(z).all()
    assert len(seen) == 5
    assert all(torch.equal(x[:, :8], z_x) for x in seen)


def test_sampler_is_deterministic_given_generator():
    model = make_ldm().eval()
    z_x = torch.randn(2, 8, 16, 16)
    a = model.sample(z_x, num_inference_steps=4, generator=torch.Generator().manual_seed(1))
    b = model.sample(z_x, num_inference_steps=4, generator=torch.Generator().manual_seed(1))
    c = model.sample(z_x, num_inference_steps=4, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.allclose(a, c)
    # Stochastic DDIM (eta > 0) is also reproducible with the same generator.
    d = model.sample(z_x, num_inference_steps=4, eta=1.0, generator=torch.Generator().manual_seed(1))
    e = model.sample(z_x, num_inference_steps=4, eta=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(d, e)


def test_guidance_doubles_batch_with_null_condition():
    model = make_ldm(cond_drop_prob=0.1).eval()
    seen = capture_unet_inputs(model)
    z_x = torch.randn(2, 8, 16, 16)
    z = model.sample(z_x, num_inference_steps=3, guidance_scale=2.0)
    assert z.shape == (2, 4, 16, 16)
    assert seen[0].shape[0] == 4 and seen[0][2:, :8].abs().sum() == 0 and torch.equal(seen[0][:2, :8], z_x)


def test_v_prediction_and_rejects_unknown():
    model = make_ldm(prediction_type="v_prediction")
    assert torch.isfinite(model.training_loss(torch.randn(2, 8, 16, 16), torch.randn(2, 4, 16, 16)))
    assert model.eval().sample(torch.randn(1, 8, 16, 16), num_inference_steps=2).shape == (1, 4, 16, 16)
    with pytest.raises(ValueError):
        make_ldm(prediction_type="sample")


def test_ldm_checkpoint_roundtrip_with_ema(tmp_path):
    model = make_ldm()
    ema = EMA(model, 0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    path = tmp_path / "ldm.pt"
    torch.save({"kind": "ldm", "model_cfg": model.model_cfg, "model": model.state_dict(), "ema": ema.shadow}, path)
    raw, _ = load_model(str(path))
    averaged, _ = load_model(str(path), use_ema=True)
    k = "unet.conv_in.weight"
    assert torch.equal(raw.state_dict()[k], model.state_dict()[k])
    assert torch.allclose(averaged.state_dict()[k], ema.shadow[k])
    assert not torch.allclose(averaged.state_dict()[k], raw.state_dict()[k])
    assert isinstance(build_model("ldm", model.model_cfg), ConditionalLatentDiffusion)


def test_ema_swap_restores_raw_weights():
    model = make_ldm()
    ema = EMA(model, 0.9)
    raw = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.unet.conv_in.weight.add_(1.0)
    ema.update(model)
    trained = model.unet.conv_in.weight.clone()
    with ema.swapped_into(model):
        assert torch.allclose(model.unet.conv_in.weight, ema.shadow["unet.conv_in.weight"])
        assert not torch.allclose(model.unet.conv_in.weight, trained)
    assert torch.equal(model.unet.conv_in.weight, trained)
    assert not torch.equal(raw["unet.conv_in.weight"], trained)


def test_latent_regressor_shapes():
    model = UNetRegressor(in_channels=8, out_channels=4, **TINY)
    assert model(torch.randn(2, 8, 16, 16)).shape == (2, 4, 16, 16)


def test_spread_metrics():
    target = torch.rand(2, 1, 16, 16)
    valid = torch.ones_like(target)
    same = target[:, None].expand(2, 3, 1, 16, 16)
    m = SpreadMetrics()
    m.update(same, target, valid)
    assert m.compute()["spread"] == pytest.approx(0.0, abs=1e-7)
    # Samples that disagree exactly where the mean is wrong -> positive std/error correlation.
    noisy = same.clone()
    noisy[:, 0, :, :8] += 0.2
    m = SpreadMetrics()
    m.update(noisy.clamp(0, 1), target, valid)
    out = m.compute()
    assert out["spread"] > 0 and out["std_err_corr"] > 0.5
