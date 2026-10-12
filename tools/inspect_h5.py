"""Inspect the layout and contents of a seismic/RGT HDF5 file.

Run on the cluster before training anything:

    python tools/inspect_h5.py /path/to/fault_aware.h5 --json h5_summary.json

Only h5py and numpy are required. Statistics are computed on a random subset of
slices per dataset so the script stays fast on large files.
"""
import argparse
import json
import re

import h5py
import numpy as np


ROLE_PATTERNS = {
    "rgt": re.compile(r"rgt", re.IGNORECASE),
    "fault": re.compile(r"fault|falha", re.IGNORECASE),
    "reflectivity": re.compile(r"refletividade|reflectivity", re.IGNORECASE),
}


def guess_role(name):
    for role, pattern in ROLE_PATTERNS.items():
        if pattern.search(name):
            return role
    return "seismic"


def sample_indices(n, k, rng):
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def monotonic_fraction(slices, axis):
    """Fraction of finite neighbouring pairs with non-negative difference along `axis`."""
    d = np.diff(slices, axis=axis)
    finite = np.isfinite(d)
    if not finite.any():
        return float("nan")
    return float((d[finite] >= 0).mean())


def dataset_stats(dset, k, rng):
    info = {
        "shape": list(dset.shape),
        "dtype": str(dset.dtype),
        "chunks": list(dset.chunks) if dset.chunks else None,
        "compression": dset.compression,
    }
    if dset.ndim < 2 or not np.issubdtype(dset.dtype, np.number):
        return info

    if dset.ndim == 2:
        data = np.asarray(dset[()], dtype=np.float64)[None]
    else:
        idx = sample_indices(dset.shape[0], k, rng)
        data = np.asarray(dset[idx], dtype=np.float64)

    finite = np.isfinite(data)
    info["n_sampled"] = int(data.shape[0])
    info["nan_fraction"] = float(1.0 - finite.mean())
    info["slices_with_nan"] = int((~finite).reshape(data.shape[0], -1).any(axis=1).sum())
    if finite.any():
        vals = data[finite]
        info.update({
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p0.5": float(np.percentile(vals, 0.5)),
            "p99.5": float(np.percentile(vals, 99.5)),
            "unique_values_sampled": int(min(np.unique(vals[:200000]).size, 10**6)),
        })
        per_slice_std = np.nanstd(np.where(finite, data, np.nan).reshape(data.shape[0], -1), axis=1)
        info["near_constant_slices"] = int((per_slice_std < 1e-6).sum())
    # Monotonicity along each in-slice axis tells which axis is depth for RGT.
    if data.ndim == 3:
        info["monotonic_frac_axis1"] = monotonic_fraction(data, axis=1)
        info["monotonic_frac_axis2"] = monotonic_fraction(data, axis=2)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_path")
    parser.add_argument("--max-slices", type=int, default=64, help="slices sampled per dataset for statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="optional path to write the full summary as JSON")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    summary = {"file": args.h5_path, "root_attrs": {}, "datasets": {}, "groups": {}}

    with h5py.File(args.h5_path, "r") as f:
        summary["root_attrs"] = {k: str(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                info = dataset_stats(obj, args.max_slices, rng)
                info["role_guess"] = guess_role(name.split("/")[-1])
                info["attrs"] = {k: str(v) for k, v in obj.attrs.items()}
                summary["datasets"][name] = info
            else:
                summary["groups"][name] = {"attrs": {k: str(v) for k, v in obj.attrs.items()}}

        f.visititems(visit)

    # Per top-level group (model / volume) overview.
    models = {}
    for name, info in summary["datasets"].items():
        top = name.split("/")[0] if "/" in name else "<root>"
        m = models.setdefault(top, {"seismic": [], "reflectivity": [], "rgt": [], "fault": [], "n_slices": set()})
        m[info["role_guess"]].append(name)
        if len(info["shape"]) >= 3:
            m["n_slices"].add(info["shape"][0])
    for m in models.values():
        m["aligned_slice_counts"] = len(m["n_slices"]) <= 1
        m["n_slices"] = sorted(m["n_slices"])
    summary["models"] = models

    # ------------------------------------------------------------------ report
    print(f"File: {args.h5_path}")
    print(f"Groups: {len(summary['groups'])}   Datasets: {len(summary['datasets'])}\n")
    for name, info in summary["datasets"].items():
        line = f"{name}\n    role={info['role_guess']}  shape={info['shape']}  dtype={info['dtype']}  chunks={info['chunks']}  compression={info['compression']}"
        if "min" in info:
            line += (f"\n    min={info['min']:.4g}  max={info['max']:.4g}  mean={info['mean']:.4g}  std={info['std']:.4g}"
                     f"  p0.5={info['p0.5']:.4g}  p99.5={info['p99.5']:.4g}")
            line += f"\n    nan_frac={info['nan_fraction']:.4f}  slices_with_nan={info['slices_with_nan']}/{info['n_sampled']}  near_constant={info['near_constant_slices']}"
            if "monotonic_frac_axis1" in info:
                line += f"  mono_ax1={info['monotonic_frac_axis1']:.3f}  mono_ax2={info['monotonic_frac_axis2']:.3f}"
        if info["attrs"]:
            line += f"\n    attrs={info['attrs']}"
        print(line)

    print("\n=== Per-model overview ===")
    for top, m in models.items():
        print(f"{top}: seismic={len(m['seismic'])} reflectivity={len(m['reflectivity'])} rgt={len(m['rgt'])} "
              f"fault={len(m['fault'])} n_slices={m['n_slices']} aligned={m['aligned_slice_counts']}")
    with_rgt = [t for t, m in models.items() if m["rgt"]]
    print(f"\nModels with RGT ({len(with_rgt)}): {with_rgt}")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
