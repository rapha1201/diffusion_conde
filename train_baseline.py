"""Train the deterministic B1 baseline (pixel-space UNet, seismic -> RGT) on a single GPU.

    python train_baseline.py --config configs/sld/baseline_pixel.yaml
    python train_baseline.py --config configs/sld/baseline_pixel.yaml --h5 fake.h5 --out-dir runs/debug_b1 --debug

Same masked RGT loss, metrics and figures as the RGT autoencoder, so its error sets
the reference for the AE acceptance gate (evaluate_phase1.py).
"""
import argparse
import os

import diffusers
import torch

from sld.data import PairTransform, load_split, write_split
from sld.eval.metrics import RGTMetrics
from sld.eval.plots import save_rgt_grid
from sld.losses import ReconLoss
from sld.models import UNetRegressor
from sld.utils.config import apply_overrides, load_config, save_config
from sld.utils.trainer import (Trainer, apply_debug, group_names, make_eval_loader, make_train_loader,
                               resolve_device, set_seed)

TINY_UNET = dict(block_out_channels=[32, 32, 64, 64], layers_per_block=1, attention_levels=[3], norm_num_groups=8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--h5", default=None, help="override data.h5_path")
    p.add_argument("--out-dir", default=None, help="override out_dir")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--debug", action="store_true", help="tiny model, a few steps, no workers")
    p.add_argument("--allow-cpu", action="store_true", help="train on CPU if CUDA is unavailable (default: fail)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.h5:
        cfg["data"]["h5_path"] = args.h5
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.debug:
        apply_debug(cfg, TINY_UNET)
    apply_overrides(cfg, args.set)  # last, so explicit --set always wins

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    set_seed(cfg.get("seed", 0))
    device = resolve_device(require_gpu=not (args.debug or args.allow_cpu))
    print(f"device={device} torch={torch.__version__} diffusers={diffusers.__version__}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)} bf16={torch.cuda.is_bf16_supported()}", flush=True)

    split = load_split(cfg["data"].get("split"), h5_path=cfg["data"]["h5_path"])
    write_split(split, os.path.join(out_dir, "split.json"))

    model = UNetRegressor(in_channels=1, out_channels=1, **cfg["model"]).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    trainer = Trainer(cfg, "regressor", model, out_dir, device)
    trainer.maybe_resume()

    train_loader, train_sources = make_train_loader(cfg["data"], split["train"], seed=cfg.get("seed", 0) + trainer.step)
    val_loader, val_ds = make_eval_loader(cfg["data"], split["val"])
    print(f"train sources={len(train_sources)} val sections={len(val_ds)}", flush=True)

    tcfg = cfg.get("transform", {})
    train_t = PairTransform(cfg.get("augment"), output_size=tcfg.get("output_size", 128), train=True,
                            min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    eval_t = PairTransform(output_size=tcfg.get("output_size", 128), train=False,
                           min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    loss_fn = ReconLoss("rgt", cfg["loss"])

    def batches():
        while True:
            for seis, rgt, valid, _ in train_loader:
                yield train_t(seis.to(device, non_blocking=True), rgt.to(device, non_blocking=True),
                              valid.to(device, non_blocking=True))

    def compute_loss(batch):
        s, r, v, keep = batch
        mask = v * keep.view(-1, 1, 1, 1).to(v.dtype)
        with trainer.autocast():
            pred = model(s)
        return loss_fn(pred, r, mask)

    def validate():
        metrics, vis = RGTMetrics(), None
        for seis, rgt, valid, meta in val_loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            with trainer.autocast():
                pred = model(s)
            pred = pred.float()
            names = [n for n, k in zip(group_names(val_ds, meta), keep.cpu()) if k]
            metrics.update(pred[keep], r[keep], v[keep], names)
            if vis is None:
                vis = (s, r, pred, v)
        out = metrics.compute()
        out.update({f"{g}/{k}": v for g in metrics.groups() for k, v in metrics.compute(g).items() if k != "n"})
        save_rgt_grid(*vis, os.path.join(out_dir, "vis", f"step_{trainer.step:07d}.png"),
                      n=cfg.get("n_vis", 8), titles=("GT", "B1"))
        return out

    final = trainer.fit(batches(), compute_loss, validate, val_key=cfg.get("val_key", "mae"),
                        batch_size=cfg["data"]["batch_size"])
    print(f"done: {final} (best {cfg.get('val_key', 'mae')}={trainer.best})", flush=True)


if __name__ == "__main__":
    main()
