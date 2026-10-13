"""End-to-end smoke tests: every training/eval script in --debug mode on the fake HDF5 (CPU, tiny models)."""
import json
import os
import subprocess
import sys

import pytest
import torch

from sld.models import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"{args[0]} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def trained(fake_h5, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    dirs = {}
    for name, script, cfg in [("ae_rgt", "train_ae.py", "configs/sld/ae_rgt.yaml"),
                              ("ae_seismic", "train_ae.py", "configs/sld/ae_seismic.yaml"),
                              ("b1", "train_baseline.py", "configs/sld/baseline_pixel.yaml")]:
        d = str(out / name)
        run(script, "--config", cfg, "--h5", fake_h5, "--out-dir", d, "--debug")
        dirs[name] = d
    return dirs


def test_training_writes_checkpoints_logs_and_figures(trained):
    for name, d in trained.items():
        for f in ("ckpt_latest.pt", "ckpt_best.pt", "ckpt_final.pt", "metrics.jsonl", "config.yaml", "split.json"):
            assert os.path.exists(os.path.join(d, f)), f"{name}: missing {f}"
        records = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        assert any(r["split"] == "train" for r in records) and any(r["split"] == "val" for r in records)
        assert all(r["loss"] == r["loss"] for r in records if r["split"] == "train"), "NaN loss"
        assert os.listdir(os.path.join(d, "vis"))
        ckpt = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu", weights_only=False)
        assert ckpt["step"] == 4


def test_resume_continues_from_latest(trained, fake_h5):
    d = trained["ae_rgt"]
    out = run("train_ae.py", "--config", "configs/sld/ae_rgt.yaml", "--h5", fake_h5, "--out-dir", d, "--debug",
              "--set", "optim.total_steps=6")
    assert "[resume]" in out and "at step 4" in out
    ckpt = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 6


def test_latent_scale_and_phase1_gate(trained, fake_h5):
    for name in ("ae_rgt", "ae_seismic"):
        out = run("compute_latent_scale.py", "--ckpt", os.path.join(trained[name], "ckpt_best.pt"),
                  "--h5", fake_h5, "--n-samples", "16", "--batch-size", "8")
        assert "scale_factor=" in out
        ckpt = torch.load(os.path.join(trained[name], "vae_scaled.pt"), map_location="cpu", weights_only=False)
        assert float(ckpt["model"]["scale_factor"]) == pytest.approx(1.0 / ckpt["latent_stats"]["std"], rel=1e-5)

    out = run("evaluate_phase1.py", "--rgt-ae", os.path.join(trained["ae_rgt"], "ckpt_best.pt"),
              "--baseline", os.path.join(trained["b1"], "ckpt_best.pt"), "--h5", fake_h5,
              "--max-per-source", "4", "--batch-size", "4")
    assert "AE/B1 ratios" in out
    report = json.load(open(os.path.join(trained["ae_rgt"], "phase1_test.json")))
    assert report["models"] == ["Modelo2"] and "gate" in report


def scaled_vaes(trained, fake_h5):
    paths = {}
    for name in ("ae_rgt", "ae_seismic"):
        paths[name] = os.path.join(trained[name], "vae_scaled.pt")
        if not os.path.exists(paths[name]):
            run("compute_latent_scale.py", "--ckpt", os.path.join(trained[name], "ckpt_best.pt"),
                "--h5", fake_h5, "--n-samples", "16", "--batch-size", "8")
    return [f"vae.rgt={paths['ae_rgt']}", f"vae.seismic={paths['ae_seismic']}"]


@pytest.mark.parametrize("cfg,prefix", [("configs/sld/ldm.yaml", "sample"), ("configs/sld/baseline_latent.yaml", "pred")])
def test_latent_models_train_validate_and_resume(trained, fake_h5, tmp_path, cfg, prefix):
    vaes = scaled_vaes(trained, fake_h5)
    d = str(tmp_path / "run")
    run("train_ldm.py", "--config", cfg, "--h5", fake_h5, "--out-dir", d, "--debug", "--set", *vaes)
    for f in ("ckpt_latest.pt", "ckpt_best.pt", "ckpt_final.pt", "metrics.jsonl", "config.yaml"):
        assert os.path.exists(os.path.join(d, f)), f"missing {f}"
    val = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if '"val"' in l]
    assert val and all(k in val[-1] for k in ("loss", "ae/mae", f"{prefix}/mae", f"{prefix}/hde_px"))
    if prefix == "sample":
        assert "mean2/mae" in val[-1] and "spread" in val[-1]
    assert len(os.listdir(os.path.join(d, "vis"))) == 2
    ckpt = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 4 and ckpt["ema"] is not None

    out = run("train_ldm.py", "--config", cfg, "--h5", fake_h5, "--out-dir", d, "--debug",
              "--set", *vaes, "optim.total_steps=6")
    assert "[resume]" in out and "at step 4" in out
    model, ckpt = load_model(os.path.join(d, "ckpt_final.pt"), use_ema=True)
    assert ckpt["step"] == 6 and ckpt["ema_num_updates"] == 6
