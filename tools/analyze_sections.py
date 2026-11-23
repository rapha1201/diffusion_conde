"""Which properties of a section make the sampling fail? Analysis of evaluate_sampling.py's sections.csv.

    python tools/analyze_sections.py runs/ldm_clip/sampling_eval_large/sections.csv --out figs/sections
    python tools/analyze_sections.py plain=runs/ldm_clip/eval_large/sections.csv \
        guided=runs/ldm_clip/eval_large_guided/sections.csv --out figs/sections      (paired: what guidance fixes)

For the first CSV, per split part (train / val / test) and over all sections:
  - error per volume and per seismic realization (frequency, noise, salt, diffraction, regime from the file name)
  - Spearman correlation of every descriptor with the error, grouped by what is available at inference
    (seismic-only, model uncertainty) and what needs the GT (explanation only)
  - error per quartile of the most correlated descriptors
  - scatter plots of the descriptors vs the error, coloured by part (<out>/scatter_<target>.png)
With a second CSV (same sections, e.g. with dip guidance): the per-section change of the error and its
correlation with the descriptors (which sections the change helps or hurts).
Everything is printed as markdown and written to <out>/analysis.md (CPU only, no data access).
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

INFERENCE = ("coherence", "chaotic_frac", "dip_deg", "dip_std_deg", "wavenumber", "lateral_ratio",
             "spread", "sample_score", "sample_dip")
GT_ONLY = ("gt_overturn", "gt_dip", "gt_spacing_cv", "gt_disc_frac", "valid_frac")
REALIZATION = ("freq_hz", "vel_pct", "noise", "salt", "diffraction", "smooth")
KEY = ("volume", "source", "slice")


def load(arg):
    label, _, path = arg.partition("=") if "=" in arg else ("", "", arg)
    rows = []
    with open(path, newline="") as fp:
        for r in csv.DictReader(fp):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(r)
    return label or os.path.basename(os.path.dirname(os.path.abspath(path))), rows


def col(rows, k):
    return np.array([r[k] if isinstance(r.get(k), float) else np.nan for r in rows], dtype=float)


def rank(x):
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    # average ranks of ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            out[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, header, rows):
        self("| " + " | ".join(header) + " |")
        self("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
        for r in rows:
            self("| " + " | ".join(r) + " |")
        self()


def f(x, fmt=".4f"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)


def group_table(out, rows, keys, target, extra=("mean_hde", "oracle_mae", "spread", "coherence")):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    body = []
    for g, rs in sorted(groups.items(), key=lambda kv: np.nanmean(col(kv[1], target))):
        body.append([" / ".join(str(x) if not isinstance(x, float) else f"{x:g}" for x in g), str(len(rs)),
                     f(np.nanmean(col(rs, target)))] + [f(np.nanmean(col(rs, k)), ".3f") for k in extra])
    out.table([" / ".join(keys), "n", target] + list(extra), body)


def correlations(out, rows, parts, target):
    out(f"Spearman correlation with `{target}` (positive: higher value, larger error)\n")
    header = ["descriptor", "available"] + parts
    body = []
    for names, avail in ((INFERENCE, "inference"), (GT_ONLY, "GT only"), (REALIZATION, "realization")):
        for k in names:
            if k not in rows[0]:
                continue
            vals = [spearman(col(subset, k), col(subset, target)) for subset in
                    [rows if p == "all" else [r for r in rows if r["part"] == p] for p in parts]]
            body.append([k, avail] + [f(v, "+.2f") for v in vals])
    out.table(header, body)


def quartiles(out, rows, target, descriptors):
    for k in descriptors:
        x, y = col(rows, k), col(rows, target)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or np.unique(x[ok]).size < 4:
            continue
        edges = np.quantile(x[ok], [0, 0.25, 0.5, 0.75, 1])
        bins = np.clip(np.searchsorted(edges, x[ok], side="right") - 1, 0, 3)
        body = [[f"Q{q + 1}", f"{edges[q]:.3g} - {edges[q + 1]:.3g}", str(int((bins == q).sum())),
                 f(y[ok][bins == q].mean())] for q in range(4)]
        out(f"`{target}` per quartile of `{k}`\n")
        out.table(["quartile", k, "n", target], body)


def scatter(rows, target, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in INFERENCE + GT_ONLY if k in rows[0]]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow), squeeze=False)
    y = col(rows, target)
    parts = sorted({r["part"] for r in rows})
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    for ax, k in zip(axes.flat, keys):
        x = col(rows, k)
        for p in parts:
            m = np.array([r["part"] == p for r in rows])
            ax.scatter(x[m], y[m], s=4, alpha=0.4, color=colors.get(p, "gray"),
                       label=f"{p} (rho {spearman(x[m], y[m]):+.2f})")
        ax.set_xlabel(k + ("" if k in INFERENCE else "  [GT]"), fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.legend(fontsize=6, markerscale=3)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def paired(out, a_label, a, b_label, b, target):
    idx = {tuple(r[k] for k in KEY): r for r in a}
    pairs = [(idx[tuple(r[k] for k in KEY)], r) for r in b if tuple(r[k] for k in KEY) in idx]
    if not pairs:
        out(f"No common sections between {a_label} and {b_label}.\n")
        return
    out(f"## Paired: `{b_label}` - `{a_label}` on `{target}` ({len(pairs)} common sections)\n")
    body = []
    for p in sorted({ra["part"] for ra, _ in pairs}):
        d = np.array([rb[target] - ra[target] for ra, rb in pairs if ra["part"] == p])
        body.append([p, str(len(d)), f(float(np.mean(d)), "+.4f"), f(float(np.median(d)), "+.4f"),
                     f"{(d < 0).mean():.0%}", f"{(d > 0).mean():.0%}"])
    out.table(["part", "n", "mean change", "median change", "improved", "worse"], body)
    base = [ra for ra, _ in pairs]
    delta = np.array([rb[target] - ra[target] for ra, rb in pairs])
    body = [[k, f(spearman(col(base, k), delta), "+.2f")] for k in INFERENCE + GT_ONLY + (target,) if k in base[0]]
    out(f"Spearman correlation of the change (negative: the change helps more where the descriptor is high)\n")
    out.table(["descriptor", "rho with change"], body)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", help="sections.csv, or label=sections.csv (a second one enables the paired analysis)")
    p.add_argument("--target", default="mean_mae", help="error column (mean_mae, sample_mae, mean_hde, sample_hde...)")
    p.add_argument("--out", default="figs/sections")
    p.add_argument("--top", type=int, default=4, help="quartile tables for the N most correlated descriptors")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    loaded = [load(a) for a in args.csvs]
    label, rows = loaded[0]
    out = Report()
    parts = [p for p in ("train", "val", "test") if any(r["part"] == p for r in rows)] + ["all"]

    out(f"# Section analysis: {label} ({len(rows)} sections, target `{args.target}`)\n")
    out("## Error per volume\n")
    group_table(out, rows, ("part", "volume"), args.target)
    out("## Error per seismic realization\n")
    group_table(out, rows, ("volume", "freq_hz", "noise", "salt", "diffraction", "regime"), args.target)
    out("## Error per tectonic regime\n")
    group_table(out, rows, ("regime",), args.target)
    out("## Descriptors\n")
    correlations(out, rows, parts, args.target)
    held_out = [r for r in rows if r["part"] in ("val", "test")] or rows
    strength = sorted(((abs(spearman(col(held_out, k), col(held_out, args.target))), k)
                       for k in INFERENCE + GT_ONLY if k in rows[0]), reverse=True)
    top = [k for s, k in strength if np.isfinite(s)][:args.top]
    out(f"## Quartiles (held-out sections) of the most correlated descriptors\n")
    quartiles(out, held_out, args.target, top)
    path = os.path.join(args.out, f"scatter_{args.target}.png")
    scatter(rows, args.target, path)
    out(f"Scatter plots: {path}\n")
    for b_label, b_rows in loaded[1:]:
        paired(out, label, rows, b_label, b_rows, args.target)
    with open(os.path.join(args.out, "analysis.md"), "w") as fp:
        fp.write("\n".join(out.lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'analysis.md')}")


if __name__ == "__main__":
    main()
