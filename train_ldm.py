"""Train the conditional latent diffusion model (or the B2 latent regressor) on a single GPU.

    python train_ldm.py --config configs/sld/ldm.yaml
    python train_ldm.py --config configs/sld/baseline_latent.yaml          # B2
    python train_ldm.py --config configs/sld/ldm.yaml --h5 fake.h5 --out-dir runs/debug_ldm --debug \\
        --set vae.rgt=runs/debug_ae_rgt/vae_scaled.pt vae.seismic=runs/debug_ae_seismic/vae_scaled.pt

Needs both autoencoders after compute_latent_scale.py (``vae_scaled.pt``); they stay frozen.
Each batch is augmented on the GPU and encoded on the fly: z_x = seismic posterior mode,
z_y = RGT posterior sample (LDM) or mode (B2), both scaled to unit variance.

kind: ldm               partial noising: only z_y is noised, the UNet sees cat(z_x, z_y_t) and
                        predicts the noise of z_y; sampling re-anchors the clean z_x at every
                        reverse step (sld/models/diffusion.py).
kind: latent_regressor  B2: deterministic UNet z_x -> z_y in the same latent space (MSE), decoded
                        by the same RGT decoder, so it has the same AE ceiling as the LDM.

Validation (EMA weights when optim.ema_decay is set) on the val models: denoising (or
regression) loss, the RGT-AE ceiling (``ae/*``, decode of the true z_y), one DDIM sample
(``sample/*``) and the mean of ``val.n_samples`` samples (``meanN/*``), all decoded to tau
and scored like Phase 1, plus the sample spread. Figures: <out_dir>/vis/step_XXXXXXX.png.
Resubmitting the same command resumes from <out_dir>/ckpt_latest.pt.
"""
import argparse
import os

import diffusers
import torch
import torch.nn.functional as F

from sld.data import PairTransform, load_split, write_split
from sld.eval.metrics import RGTMetrics, SpreadMetrics
from sld.eval.plots import save_ldm_grid, save_rgt_grid
from sld.models import ConditionalLatentDiffusion, UNetRegressor, load_model
from sld.utils.config import apply_overrides, load_config, save_config
from sld.utils.trainer import Trainer, apply_debug, make_eval_loader, make_train_loader, resolve_device, set_seed

TINY = {
    "ldm": dict(block_out_channels=[32, 64], layers_per_block=1, attention_levels=[1], norm_num_groups=8,
                attention_head_dim=8),
    "latent_regressor": dict(block_out_channels=[32, 64], layers_per_block=1, attention_levels=[1], norm_num_groups=8,
                             attention_head_dim=8),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--h5", default=None, help="override data.h5_path")
    p.add_argument("--out-dir", default=None, help="override out_dir")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--debug", action="store_true", help="tiny model, a few steps, no workers")
    p.add_argument("--allow-cpu", action="store_true", help="train on CPU if CUDA is unavailable (default: fail)")
    return p.parse_args()


def load_frozen_vae(path, modality, device):
    vae, ckpt = load_model(path)
    if ckpt["kind"] != "vae" or ckpt["config"].get("modality") != modality:
        raise ValueError(f"{path} is not the {modality} autoencoder")
    if "latent_stats" not in ckpt:
        print(f"[warn] {path} was not processed by compute_latent_scale.py; scale_factor="
              f"{float(vae.scale_factor):.4g}, latents may not have unit variance", flush=True)
    vae.requires_grad_(False)
    return vae.to(device).eval()


def decode(vae, z, chunk=256):
    return torch.cat([vae.decode(c) for c in z.split(chunk)]).float()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.h5:
        cfg["data"]["h5_path"] = args.h5
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    kind = cfg.get("kind", "ldm")
    if kind not in TINY:
        raise ValueError(f"kind must be one of {sorted(TINY)}, got {kind!r}")
    if args.debug:
        apply_debug(cfg, TINY[kind])
        cfg.setdefault("sampling", {})["steps"] = 4
        cfg.setdefault("val", {})["n_samples"] = 2
    apply_overrides(cfg, args.set)  # last, so explicit --set always wins

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    set_seed(cfg.get("seed", 0))
    device = resolve_device(require_gpu=not (args.debug or args.allow_cpu))
    print(f"device={device} torch={torch.__version__} diffusers={diffusers.__version__} kind={kind}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)} bf16={torch.cuda.is_bf16_supported()}", flush=True)

    split = load_split(cfg["data"].get("split"), h5_path=cfg["data"]["h5_path"])
    write_split(split, os.path.join(out_dir, "split.json"))

    vae_x = load_frozen_vae(cfg["vae"]["seismic"], "seismic", device)
    vae_y = load_frozen_vae(cfg["vae"]["rgt"], "rgt", device)
    if vae_x.downsample_factor != vae_y.downsample_factor:
        raise ValueError(f"Autoencoders downsample by {vae_x.downsample_factor} (seismic) and "
                         f"{vae_y.downsample_factor} (RGT); latents cannot be concatenated")
    cx, cy = vae_x.latent_channels, vae_y.latent_channels
    print(f"vae seismic: {cfg['vae']['seismic']} c={cx} scale={float(vae_x.scale_factor):.4g} | "
          f"rgt: {cfg['vae']['rgt']} c={cy} scale={float(vae_y.scale_factor):.4g} | f={vae_y.downsample_factor}",
          flush=True)

    if kind == "ldm":
        model = ConditionalLatentDiffusion(cond_channels=cx, latent_channels=cy, **cfg["model"],
                                           **cfg.get("diffusion", {}))
    else:
        model = UNetRegressor(in_channels=cx, out_channels=cy, **cfg["model"])
    model.to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    trainer = Trainer(cfg, "ldm" if kind == "ldm" else "regressor", model, out_dir, device)
    trainer.maybe_resume()

    train_loader, train_sources = make_train_loader(cfg["data"], split["train"], seed=cfg.get("seed", 0) + trainer.step)
    val_loader, val_ds = make_eval_loader(cfg["data"], split["val"])
    print(f"train sources={len(train_sources)} val sections={len(val_ds)}", flush=True)

    tcfg = cfg.get("transform", {})
    train_t = PairTransform(cfg.get("augment"), output_size=tcfg.get("output_size", 128), train=True,
                            min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    eval_t = PairTransform(output_size=tcfg.get("output_size", 128), train=False,
                           min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    scfg = cfg.get("sampling", {})
    n_samples = int(cfg.get("val", {}).get("n_samples", 4)) if kind == "ldm" else 1

    def encode(s, r):
        with torch.no_grad(), trainer.autocast():
            z_x = vae_x.encode(s)
            z_y = vae_y.encode(r, sample=(kind == "ldm"))
        return z_x.float(), z_y.float()

    def batches():
        # Crops with too little valid RGT were masked out of the RGT-AE loss, so their latents are
        # unreliable targets: drop them.
        while True:
            for seis, rgt, valid, _ in train_loader:
                s, r, _, keep = train_t(seis.to(device, non_blocking=True), rgt.to(device, non_blocking=True),
                                        valid.to(device, non_blocking=True))
                if keep.any():
                    yield s[keep], r[keep]

    def latent_loss(z_x, z_y, generator=None):
        with trainer.autocast():
            if kind == "ldm":
                return model.training_loss(z_x, z_y, generator=generator)
            return F.mse_loss(model(z_x).float(), z_y)

    def compute_loss(batch):
        return latent_loss(*encode(*batch)), {}

    def predict(z_x, generator):
        """tau predictions [B, n_samples, 1, H, W]."""
        with trainer.autocast():
            if kind == "ldm":
                z = model.sample(z_x.repeat_interleave(n_samples, dim=0), num_inference_steps=int(scfg.get("steps", 50)),
                                 eta=float(scfg.get("eta", 0.0)), guidance_scale=float(scfg.get("guidance_scale", 1.0)),
                                 generator=generator)
            else:
                z = model(z_x).float()
            tau = decode(vae_y, z)
        return tau.view(z_x.shape[0], n_samples, *tau.shape[1:])

    def validate():
        gen = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))  # same noise at every validation
        m_ae, m_one, m_mean, m_spread = RGTMetrics(), RGTMetrics(), RGTMetrics(), SpreadMetrics()
        loss_sum, count, vis = 0.0, 0, None
        for seis, rgt, valid, _ in val_loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            s, r, v = s[keep], r[keep], v[keep]
            if len(r) == 0:
                continue
            z_x, z_y = encode(s, r)
            loss_sum += float(latent_loss(z_x, z_y, generator=gen)) * len(r)
            count += len(r)
            tau = predict(z_x, gen)
            with trainer.autocast():
                m_ae.update(decode(vae_y, z_y), r, v)
            m_one.update(tau[:, 0], r, v)
            if n_samples > 1:
                m_mean.update(tau.mean(dim=1), r, v)
                m_spread.update(tau, r, v)
            if vis is None:
                vis = (s, r, tau, v)
        if vis is None:
            raise RuntimeError("No validation crop passed the validity filter.")

        out = {"loss": loss_sum / count, "n": count}
        groups = [("ae", m_ae), ("sample" if kind == "ldm" else "pred", m_one)]
        if n_samples > 1:
            groups.append((f"mean{n_samples}", m_mean))
            out.update({k: v for k, v in m_spread.compute().items() if k != "n"})
        for prefix, m in groups:
            out.update({f"{prefix}/{k}": v for k, v in m.compute().items() if k != "n"})
        path = os.path.join(out_dir, "vis", f"step_{trainer.step:07d}.png")
        if kind == "ldm":
            save_ldm_grid(*vis, path, n=cfg.get("n_vis", 8))
        else:
            s, r, tau, v = vis
            save_rgt_grid(s, r, tau[:, 0], v, path, n=cfg.get("n_vis", 8), titles=("GT", "B2"))
        return out

    val_key = cfg.get("val_key", "sample/mae" if kind == "ldm" else "pred/mae")
    final = trainer.fit(batches(), compute_loss, validate, val_key=val_key, batch_size=cfg["data"]["batch_size"])
    print(f"done: {final} (best {val_key}={trainer.best})", flush=True)


if __name__ == "__main__":
    main()
