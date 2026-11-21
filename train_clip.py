"""Phase A: CLIP-style pretraining of the seismic condition encoder on a single GPU.

    python train_clip.py --config configs/sld/clip.yaml
    python train_clip.py --config configs/sld/clip.yaml --h5 fake.h5 --out-dir runs/debug_clip --debug \\
        --set vae.rgt=runs/debug_ae_rgt/vae_scaled.pt vae.seismic=runs/debug_ae_seismic/vae_scaled.pt

The Stable Diffusion recipe, applied to seismic: the autoencoders stay frozen (reconstruction only),
a seismic encoder and an RGT encoder on their latents are trained with a contrastive loss only
(sld/models/clip.py), and Phase B (train_ldm.py with configs/sld/ldm_clip.yaml) loads the
seismic encoder as the frozen condition encoder of the LDM. The RGT encoder is discarded.

Each batch is augmented on the GPU (same pair transform as the other runs) and encoded on the fly:
z_x = seismic posterior mode, z_y = RGT posterior sample, as in the LDM.

Validation = retrieval on center crops, in fixed random groups of ``val.group_size`` sections
(chance = 1/group_size): for each seismic, is the most similar RGT its own (seismic->RGT), and vice
versa (RGT->seismic)? Dense mode scores every latent position. Realizations of the same slice share the
RGT and count as correct. Reported for the val models (``val_models/*``, the held-out volume) and for
a few sections of the train models (``train_models/*``), so the generalization gap is visible.
Checkpoint selection: lowest ``val_models/err`` = 1 - mean accuracy.
Resubmitting the same command resumes from <out_dir>/ckpt_latest.pt.
"""
import argparse
import os

import diffusers
import torch

from sld.data import PairTransform, load_split, write_split
from sld.models import LatentCLIP
from sld.utils.config import apply_overrides, load_config, save_config
from sld.utils.trainer import Trainer, apply_debug, make_eval_loader, make_train_loader, resolve_device, set_seed
from train_ldm import load_frozen_vae

TINY = dict(embed_dim=16, encoder_blocks=1, norm_num_groups=8, proj_dim=8)


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
        mode, temp = cfg["model"].get("mode", "dense"), cfg["model"].get("temperature", 0.07)
        apply_debug(cfg, dict(TINY, mode=mode, temperature=temp))
        cfg["data"]["train_eval_max_per_source"] = 2
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

    vae_x = load_frozen_vae(cfg["vae"]["seismic"], "seismic", device)
    vae_y = load_frozen_vae(cfg["vae"]["rgt"], "rgt", device)
    model = LatentCLIP(seismic_channels=vae_x.latent_channels, rgt_channels=vae_y.latent_channels,
                       **cfg["model"]).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M mode={model.mode}", flush=True)
    trainer = Trainer(cfg, "clip", model, out_dir, device)
    trainer.maybe_resume()

    train_loader, train_sources = make_train_loader(cfg["data"], split["train"], seed=cfg.get("seed", 0) + trainer.step)
    eval_sets = {"val_models": make_eval_loader(cfg["data"], split["val"])[0],
                 "train_models": make_eval_loader(cfg["data"], split["train"],
                                                  max_per_source=cfg["data"].get("train_eval_max_per_source", 32))[0]}
    print(f"train sources={len(train_sources)} " + " ".join(f"{k} sections={len(v.dataset)}"
                                                           for k, v in eval_sets.items()), flush=True)

    tcfg = cfg.get("transform", {})
    train_t = PairTransform(cfg.get("augment"), output_size=tcfg.get("output_size", 128), train=True,
                            min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    eval_t = PairTransform(output_size=tcfg.get("output_size", 128), train=False,
                           min_valid_frac=tcfg.get("min_valid_frac", 0.5)).to(device)
    group_size = int(cfg.get("val", {}).get("group_size", 64))

    def encode(s, r, sample_y):
        with torch.no_grad(), trainer.autocast():
            return vae_x.encode(s).float(), vae_y.encode(r, sample=sample_y).float()

    def batches():
        # Crops failing the RGT validity filter were masked out of the RGT-AE loss: their latents are unreliable.
        while True:
            for seis, rgt, valid, _ in train_loader:
                s, r, _, keep = train_t(seis.to(device, non_blocking=True), rgt.to(device, non_blocking=True),
                                        valid.to(device, non_blocking=True))
                if keep.sum() >= 2:
                    yield s[keep], r[keep]

    def compute_loss(batch):
        z_x, z_y = encode(*batch, sample_y=True)
        with trainer.autocast():
            loss, acc = model(z_x, z_y)
        return loss, {"acc": float(acc), "temp": model.temperature}

    def embed_all(loader):
        ex, ey, keys = [], [], []
        for seis, rgt, valid, meta in loader:
            s, r, _, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            if not keep.any():
                continue
            z_x, z_y = encode(s[keep], r[keep], sample_y=False)
            with trainer.autocast():
                ex.append(model.embed_seismic(z_x))
                ey.append(model.embed_rgt(z_y))
            keys.append(meta[keep.cpu()][:, [2, 1]])  # (model, slice): realizations of one slice share the RGT
        return torch.cat(ex), torch.cat(ey), torch.cat(keys).to(device)

    def validate():
        out = {}
        for name, loader in eval_sets.items():
            ex, ey, keys = embed_all(loader)
            perm = torch.randperm(len(ex), generator=torch.Generator().manual_seed(cfg.get("seed", 0))).to(device)
            sums, n = torch.zeros(3, device=device), 0
            for idx in perm.split(group_size):
                if len(idx) < 2:
                    continue
                sums += torch.stack(model.retrieval(ex[idx], ey[idx], keys[idx])) * len(idx)
                n += len(idx)
            s2r, r2s, loss = (sums / max(n, 1)).tolist()
            out.update({f"{name}/acc_s2r": s2r, f"{name}/acc_r2s": r2s, f"{name}/loss": loss,
                        f"{name}/err": 1.0 - 0.5 * (s2r + r2s), f"{name}/n": len(ex)})
        out["chance"] = 1.0 / group_size
        out["temp"] = model.temperature
        return out

    val_key = cfg.get("val_key", "val_models/err")
    final = trainer.fit(batches(), compute_loss, validate, val_key=val_key, batch_size=cfg["data"]["batch_size"])
    print(f"done: {final} (best {val_key}={trainer.best})", flush=True)


if __name__ == "__main__":
    main()
