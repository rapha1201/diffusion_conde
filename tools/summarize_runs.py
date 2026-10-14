"""Training curves and a metrics table from runs/*/metrics.jsonl (CPU only, no data access).

    python tools/summarize_runs.py runs/ae_rgt runs/ae_seismic runs/baseline_pixel runs/ldm --out figs

Per run: <out>/<run>_curves.png with the train loss and the validation metrics vs step.
Prints, per run, the best validation (by the run's val_key) and the latest one, with the
matching figure in <run>/vis/.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def run_info(run):
    cfg_path = os.path.join(run, "config.yaml")
    cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    kind = cfg.get("kind")
    mean = f"mean{cfg.get('val', {}).get('n_samples', 4)}"
    if cfg.get("modality") == "seismic":
        panels = [["mae", "rmse"], ["corr"]]
    elif kind == "ldm":
        panels = [["ae/mae", "sample/mae", f"{mean}/mae"], ["ae/hde_px", "sample/hde_px", f"{mean}/hde_px"],
                  ["spread", "std_err_corr"]]
    elif kind == "latent_regressor":
        panels = [["ae/mae", "pred/mae"], ["ae/hde_px", "pred/hde_px"]]
    else:  # RGT autoencoder or pixel baseline
        panels = [["mae", "rmse"], ["hde_px"]]
    val_key = cfg.get("val_key") or {"ldm": "sample/mae", "latent_regressor": "pred/mae"}.get(kind, "mae")
    return cfg, panels, val_key


def load(run):
    rows = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl")) if line.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def plot(run, name, train, val, panels, out):
    fig, axes = plt.subplots(1, 1 + len(panels), figsize=(4.5 * (1 + len(panels)), 3.6), squeeze=False)
    ax = axes[0, 0]
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"{name}: train loss", fontsize=9)
    for ax, keys in zip(axes[0, 1:], panels):
        for k in keys:
            pts = [(r["step"], r[k]) for r in val if k in r]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, lw=1, label=k)
        if all(r.get(k, 1) > 0 for r in val for k in keys) and not any("corr" in k for k in keys):
            ax.set_yscale("log")
        ax.set_title(f"{name}: validation", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[0]:
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, f"{name}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fmt(row, keys):
    return "  ".join(f"{k}={row[k]:.4g}" for k in keys if k in row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="figs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        if not os.path.exists(os.path.join(run, "metrics.jsonl")):
            print(f"[skip] {run}: no metrics.jsonl", file=sys.stderr)
            continue
        cfg, panels, val_key = run_info(run)
        train, val = load(run)
        print(f"\n== {name}  (steps logged: {train[-1]['step'] if train else 0}, validations: {len(val)})")
        print(f"   curves: {plot(run, name, train, val, panels, args.out)}")
        if not val:
            continue
        keys = [k for group in panels for k in group]
        best = min((r for r in val if val_key in r), key=lambda r: r[val_key], default=None)
        for label, row in (("best  ", best), ("latest", val[-1])):
            if row is None:
                continue
            vis = os.path.join(run, "vis", f"step_{row['step']:07d}.png")
            print(f"   {label} step {row['step']:>7d}: {fmt(row, keys)}")
            print(f"   {'':6s} figure: {vis if os.path.exists(vis) else '(none)'}")


if __name__ == "__main__":
    main()
