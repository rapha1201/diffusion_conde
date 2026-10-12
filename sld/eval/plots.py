"""Validation figures. RGT panels overlay iso-tau lines on the seismic, like tools/preview_pairs.py."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _prep(path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _overlay(ax, seismic, rgt, valid, levels, color):
    ax.imshow(seismic, cmap="gray", vmin=-2.5, vmax=2.5)
    ax.contour(torch.where(valid.bool(), rgt, torch.nan), levels=levels, colors=color, linewidths=0.7)


def save_rgt_grid(seismic, target, pred, valid, path, n=8, n_levels=12, titles=("GT", "prediction")):
    """Columns: seismic + GT isolines | seismic + predicted isolines | GT | prediction | |error|."""
    _prep(path)
    seismic, target, pred, valid = (t.detach().float().cpu() for t in (seismic, target, pred, valid))
    pred = pred.clamp(0, 1)
    n = min(n, len(target))
    levels = torch.linspace(0.02, 0.98, n_levels).tolist()
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n), squeeze=False)
    for i in range(n):
        s, t, p, v = seismic[i, 0], target[i, 0], pred[i, 0], valid[i, 0]
        _overlay(axes[i, 0], s, t, v, levels, "r")
        _overlay(axes[i, 1], s, p, v, levels, "c")
        axes[i, 2].imshow(t, cmap="turbo", vmin=0, vmax=1)
        axes[i, 3].imshow(p, cmap="turbo", vmin=0, vmax=1)
        axes[i, 4].imshow(((p - t).abs() * v), cmap="magma", vmin=0, vmax=0.1)
        for ax, title in zip(axes[i], (f"{titles[0]} isolines", f"{titles[1]} isolines", titles[0], titles[1],
                                       "|error| (0-0.1)")):
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def save_seismic_grid(target, pred, path, n=8):
    """Columns: input | reconstruction | difference."""
    _prep(path)
    target, pred = target.detach().float().cpu(), pred.detach().float().cpu()
    n = min(n, len(target))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n), squeeze=False)
    for i in range(n):
        for ax, img, title, cmap, lim in ((axes[i, 0], target[i, 0], "input", "gray", 2.5),
                                          (axes[i, 1], pred[i, 0], "reconstruction", "gray", 2.5),
                                          (axes[i, 2], pred[i, 0] - target[i, 0], "difference", "seismic", 1.0)):
            ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)
