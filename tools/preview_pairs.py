"""Visual check of the training pipeline: RGT isolines drawn over the seismic.

Columns per row: center crop (eval transform) | augmented crop (train transform) |
RGT of the augmented crop | validity mask. If pairing and shared geometry are
right, the isolines follow the reflectors in both seismic columns.

    python tools/preview_pairs.py --h5 /path/to/fault_aware.h5 --models Modelo3 Modelo9 --n 8 --out preview.png
"""
import argparse
import os
import sys

# Allow `python tools/preview_pairs.py` from anywhere: put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from sld.data import PairedChunkStream, PairTransform, index_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--models", nargs="+", default=["Modelo3"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--levels", type=int, default=12, help="number of RGT isolines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="preview.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sources = index_pairs(args.h5, args.models)
    stream = PairedChunkStream(args.h5, sources, buffer_size=256, seed=args.seed, repeat=False)
    seis, rgt, valid, meta = next(iter(DataLoader(stream, batch_size=args.n)))

    eval_t = PairTransform(train=False, output_size=128)
    train_t = PairTransform(aug_kwargs=dict(aug_p=1.0, p_fault=0.5, p_rotation=0.5), train=True, output_size=128)
    se, re_, ve, _ = eval_t(seis, rgt, valid)
    sa, ra, va, keep = train_t(seis, rgt, valid)

    levels = torch.linspace(0.02, 0.98, args.levels).tolist()
    fig, axes = plt.subplots(args.n, 4, figsize=(13, 3.2 * args.n), squeeze=False)
    for i in range(args.n):
        src = sources[int(meta[i, 0])]
        for col, (s, r, v, title) in enumerate([(se[i, 0], re_[i, 0], ve[i, 0], "eval crop"),
                                                (sa[i, 0], ra[i, 0], va[i, 0], f"augmented (keep={bool(keep[i])})")]):
            ax = axes[i, col]
            ax.imshow(s, cmap="gray", vmin=-2.5, vmax=2.5)
            ax.contour(torch.where(v.bool(), r, torch.nan), levels=levels, colors="r", linewidths=0.7)
            ax.set_title(title, fontsize=8)
        axes[i, 0].set_ylabel(f"{src.model} #{int(meta[i, 1])}\n{src.seismic_key[:28]}", fontsize=7)
        axes[i, 2].imshow(ra[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[i, 2].set_title("RGT (augmented)", fontsize=8)
        axes[i, 3].imshow(va[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title("valid", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
