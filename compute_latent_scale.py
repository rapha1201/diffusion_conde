"""Set a trained autoencoder's latent scale factor so scaled latents have unit variance.

    python compute_latent_scale.py --ckpt runs/ae_rgt/ckpt_best.pt
    -> writes runs/ae_rgt/vae_scaled.pt (same checkpoint + scale_factor + latent statistics)

Latents are posterior samples (LDM convention) of center-cropped training sections,
spread evenly through every training volume. For the RGT AE, crops failing the
validity filter are skipped.
"""
import argparse
import math
import os

import torch

from sld.data import PairTransform, index_pairs, load_split
from sld.models import load_model
from sld.utils.trainer import make_eval_loader, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="default: <ckpt dir>/vae_scaled.pt")
    p.add_argument("--h5", default=None, help="override the HDF5 path stored in the checkpoint config")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device()
    model, ckpt = load_model(args.ckpt, map_location="cpu")
    model.to(device).eval()
    if ckpt["kind"] != "vae":
        raise ValueError(f"{args.ckpt} is a {ckpt['kind']!r} checkpoint, not a VAE")
    cfg = ckpt["config"]
    modality = cfg["modality"]
    data_cfg = dict(cfg["data"])
    if args.h5:
        data_cfg["h5_path"] = args.h5
    split_path = os.path.join(os.path.dirname(args.ckpt), "split.json")
    split = load_split(split_path if os.path.exists(split_path) else data_cfg.get("split"), h5_path=data_cfg["h5_path"])

    n_sources = len(index_pairs(data_cfg["h5_path"], split["train"], include_reflectivity=data_cfg.get("include_reflectivity", False)))
    loader, ds = make_eval_loader(data_cfg, split["train"], max_per_source=math.ceil(args.n_samples / n_sources),
                                  batch_size=args.batch_size)
    eval_t = PairTransform(output_size=cfg.get("transform", {}).get("output_size", 128), train=False,
                           min_valid_frac=cfg.get("transform", {}).get("min_valid_frac", 0.5)).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    c = model.latent_channels
    s1, s2, n = torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64), 0
    with torch.no_grad():
        for seis, rgt, valid, _ in loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            x = r[keep] if modality == "rgt" else s
            if len(x) == 0:
                continue
            z = model.posterior(x).sample(generator=gen).double()
            s1 += z.sum(dim=(0, 2, 3)).cpu()
            s2 += (z ** 2).sum(dim=(0, 2, 3)).cpu()
            n += z.shape[0] * z.shape[2] * z.shape[3]
    if n == 0:
        raise RuntimeError("No latents collected; check the data path and validity filter.")

    mean_c = s1 / n
    std_c = (s2 / n - mean_c ** 2).clamp_min(0).sqrt()
    mean = s1.sum() / (n * c)
    std = float((s2.sum() / (n * c) - mean ** 2).clamp_min(0).sqrt())
    scale = 1.0 / std
    model.scale_factor.fill_(scale)

    ckpt["model"] = model.state_dict()
    ckpt["latent_stats"] = {"std": std, "mean": float(mean), "per_channel_std": std_c.tolist(),
                            "per_channel_mean": mean_c.tolist(), "n_latent_pixels": n, "n_sections": len(ds)}
    for k in ("optimizer", "scheduler"):
        ckpt.pop(k, None)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "vae_scaled.pt")
    torch.save(ckpt, out)
    print(f"modality={modality} sections={len(ds)} latent std={std:.4f} mean={float(mean):.4f} -> scale_factor={scale:.5f}")
    print(f"per-channel std={[round(x, 4) for x in std_c.tolist()]} mean={[round(x, 4) for x in mean_c.tolist()]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
