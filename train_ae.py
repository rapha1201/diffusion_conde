"""Train one KL autoencoder (seismic or RGT) on a single GPU.

    python train_ae.py --config configs/sld/ae_rgt.yaml
    python train_ae.py --config configs/sld/ae_seismic.yaml --set optim.lr=2e-4
    python train_ae.py --config configs/sld/ae_rgt.yaml --h5 fake.h5 --out-dir runs/debug --debug

Resubmitting the same command resumes from <out_dir>/ckpt_latest.pt.
After training, run compute_latent_scale.py on ckpt_best.pt.
"""
import argparse
import os

import diffusers
import torch

from sld.data import PairTransform, load_split, write_split
from sld.eval.metrics import RGTMetrics, SeismicMetrics
from sld.eval.plots import save_rgt_grid, save_seismic_grid
from sld.losses import ReconLoss
from sld.models import LatentVAE
from sld.utils.config import apply_overrides, load_config, save_config
from sld.utils.trainer import (Trainer, apply_debug, group_names, make_eval_loader, make_train_loader,
                               resolve_device, set_seed)

TINY_VAE = dict(block_out_channels=[32, 32, 32, 32], layers_per_block=1, norm_num_groups=8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--h5", default=None, help="override data.h5_path")
    p.add_argument("--out-dir", default=None, help="override out_dir")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--debug", action="store_true", help="tiny model, a few steps, no workers")
    return p.parse_args()


def select(modality, seismic, rgt, valid, keep):
    """Input/target of the autoencoder and its loss mask."""
    if modality == "rgt":
        return rgt, valid * keep.view(-1, 1, 1, 1).to(valid.dtype)
    return seismic, None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.h5:
        cfg["data"]["h5_path"] = args.h5
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.debug:
        apply_debug(cfg, dict(TINY_VAE, latent_channels=cfg["model"]["latent_channels"]))
    apply_overrides(cfg, args.set)  # last, so explicit --set always wins
    modality = cfg["modality"]

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    set_seed(cfg.get("seed", 0))
    device = resolve_device()
    print(f"device={device} torch={torch.__version__} diffusers={diffusers.__version__} modality={modality}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)} bf16={torch.cuda.is_bf16_supported()}", flush=True)

    split = load_split(cfg["data"].get("split"), h5_path=cfg["data"]["h5_path"])
    write_split(split, os.path.join(out_dir, "split.json"))

    model = LatentVAE(in_channels=1, **cfg["model"]).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"downsample={model.downsample_factor} latent_channels={model.latent_channels}", flush=True)
    trainer = Trainer(cfg, "vae", model, out_dir, device)
    trainer.maybe_resume()

    # Offset the stream seed by the resume step so a resumed run does not replay the same order.
    train_loader, train_sources = make_train_loader(cfg["data"], split["train"], seed=cfg.get("seed", 0) + trainer.step)
    val_loader, val_ds = make_eval_loader(cfg["data"], split["val"])
    print(f"train sources={len(train_sources)} val sections={len(val_ds)}", flush=True)

    tcfg = cfg.get("transform", {})
    train_t = PairTransform(cfg.get("augment"), output_size=tcfg.get("output_size", 128), train=True,
                            min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    eval_t = PairTransform(output_size=tcfg.get("output_size", 128), train=False,
                           min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    loss_fn = ReconLoss(modality, cfg["loss"])

    def batches():
        while True:
            for seis, rgt, valid, _ in train_loader:
                yield train_t(seis.to(device, non_blocking=True), rgt.to(device, non_blocking=True),
                              valid.to(device, non_blocking=True))

    def compute_loss(batch):
        x, mask = select(modality, *batch)
        with trainer.autocast():
            recon, posterior = model(x)
        return loss_fn(recon, x, mask, posterior)

    def validate():
        metrics = RGTMetrics() if modality == "rgt" else SeismicMetrics()
        vis = None
        for seis, rgt, valid, meta in val_loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            x, mask = select(modality, s, r, v, keep)
            with trainer.autocast():
                recon = model.vae.decode(model.posterior(x).mode()).sample
            recon = recon.float()
            names = group_names(val_ds, meta)
            if modality == "rgt":
                k = keep.cpu()
                metrics.update(recon[keep], x[keep], v[keep], [n for n, kk in zip(names, k) if kk])
            else:
                metrics.update(recon, x, names)
            if vis is None:
                vis = (s, x, recon, v)
        out = metrics.compute()
        out.update({f"{g}/{k}": v for g in metrics.groups() for k, v in metrics.compute(g).items() if k != "n"})
        path = os.path.join(out_dir, "vis", f"step_{trainer.step:07d}.png")
        if modality == "rgt":
            save_rgt_grid(vis[0], vis[1], vis[2], vis[3], path, n=cfg.get("n_vis", 8), titles=("GT", "recon"))
        else:
            save_seismic_grid(vis[1], vis[2], path, n=cfg.get("n_vis", 8))
        return out

    final = trainer.fit(batches(), compute_loss, validate, val_key=cfg.get("val_key", "mae"),
                        batch_size=cfg["data"]["batch_size"])
    print(f"done: {final} (best {cfg.get('val_key', 'mae')}={trainer.best})", flush=True)


if __name__ == "__main__":
    main()
