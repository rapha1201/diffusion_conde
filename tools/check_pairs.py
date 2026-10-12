"""Second-pass checks on a seismic/RGT HDF5 (run after inspect_h5.py).

1) RGT validity: exact-zero fraction, whether zeros form a contiguous shallow
   band (top rows along the depth axis), and how many slices are empty/constant.
2) Pairing sanity: are the reflectors in each seismic dataset aligned with the
   isolines of the RGT in the same model? Reflector normals come from the
   seismic structure tensor; isoline normals from the RGT gradient. Scores are
   coherence-weighted |cos| between them (1 = perfectly aligned).
   Controls: the same seismic against the RGT of a different random slice of the
   same model, against a random RGT slice of a *different* model, and against
   its own RGT with the two in-slice axes swapped. A correctly paired dataset
   scores clearly above `other_model`; a seismic that actually belongs to
   another model scores about the same as `other_model`.

    python tools/check_pairs.py /path/to/fault_aware.h5 --depth-axis 1 --json pair_check.json

Requires h5py, numpy, scipy.
"""
import argparse
import json
import re

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


RGT_RE = re.compile(r"rgt", re.IGNORECASE)


def chunk_aligned_indices(n, chunk, n_chunks, rng):
    """Pick whole chunks so reads are cheap on gzip-chunked datasets."""
    starts = np.arange(0, n, chunk)
    pick = np.sort(rng.choice(starts, size=min(n_chunks, len(starts)), replace=False))
    return np.concatenate([np.arange(s, min(s + chunk, n)) for s in pick])


def read_rows(dset, idx):
    # h5py fancy indexing needs increasing indices; read contiguous runs.
    out = np.empty((len(idx),) + dset.shape[1:], dtype=np.float64)
    runs = np.split(np.arange(len(idx)), np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        out[r] = dset[idx[r[0]]: idx[r[-1]] + 1]
    return out


def unit_normals_from_seismic(s, sigma_grad=1.0, sigma_tensor=4.0):
    gz = gaussian_filter(s, sigma_grad, order=(1, 0))
    gx = gaussian_filter(s, sigma_grad, order=(0, 1))
    jzz = gaussian_filter(gz * gz, sigma_tensor)
    jxx = gaussian_filter(gx * gx, sigma_tensor)
    jzx = gaussian_filter(gz * gx, sigma_tensor)
    # Principal eigenvector of [[jzz, jzx], [jzx, jxx]].
    theta = 0.5 * np.arctan2(2 * jzx, jzz - jxx)
    nz, nx = np.cos(theta), np.sin(theta)
    tr = jzz + jxx
    disc = np.sqrt((jzz - jxx) ** 2 + 4 * jzx ** 2)
    coherence = np.where(tr > 1e-12, disc / np.maximum(tr, 1e-12), 0.0)
    return nz, nx, coherence


def alignment_score(seis, rgt, valid):
    nz, nx, coh = unit_normals_from_seismic(seis)
    rz = gaussian_filter(rgt, 2.0, order=(1, 0))
    rx = gaussian_filter(rgt, 2.0, order=(0, 1))
    mag = np.sqrt(rz ** 2 + rx ** 2)
    w = coh * valid * (mag > 1e-8)
    if w.sum() < 1e-6:
        return float("nan")
    cos = np.abs(nz * rz + nx * rx) / np.maximum(mag, 1e-12)
    return float((cos * w).sum() / w.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_path")
    parser.add_argument("--depth-axis", type=int, default=1, choices=[1, 2], help="depth axis of the (N, A, B) datasets")
    parser.add_argument("--chunks-per-dataset", type=int, default=8, help="whole HDF5 chunks sampled per dataset")
    parser.add_argument("--pairs-per-dataset", type=int, default=48, help="slices used for the alignment score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {}
    cache = {}  # model -> sampled RGT and validity, reused as the cross-model control

    with h5py.File(args.h5_path, "r") as f:
        models = [m for m in f.keys() if isinstance(f[m], h5py.Group) and any(RGT_RE.search(k) for k in f[m].keys())]

        # Pass 1: RGT statistics per model.
        for model in models:
            keys = list(f[model].keys())
            rgt_keys = [k for k in keys if RGT_RE.search(k)]
            rgt_ds = f[f"{model}/{rgt_keys[0]}"]
            n = rgt_ds.shape[0]
            chunk = rgt_ds.chunks[0] if rgt_ds.chunks else 32
            idx = chunk_aligned_indices(n, chunk, args.chunks_per_dataset, rng)

            rgt = read_rows(rgt_ds, idx)
            if args.depth_axis == 2:
                rgt = rgt.transpose(0, 2, 1)
            finite = np.isfinite(rgt)
            zero = (rgt == 0) & finite
            rng_per_slice = np.nanmax(np.where(finite, rgt, np.nan), axis=(1, 2)) - np.nanmin(np.where(finite, rgt, np.nan), axis=(1, 2))
            rng_per_slice = np.nan_to_num(rng_per_slice, nan=0.0)
            # "Near-constant" relative to the typical slice range, so it works for raw or normalized RGT.
            min_range = 1e-3 * max(float(np.median(rng_per_slice)), 1e-12)
            near_const = rng_per_slice < min_range
            # Rows (depth samples) that are entirely zero, per slice.
            zero_rows = zero.all(axis=2)
            n_zero_rows = zero_rows.sum(axis=1)
            # Contiguous from the top means the first k rows are all-zero rows and no others.
            leading = np.array([np.argmin(r) if not r.all() else r.size for r in zero_rows])
            top_band = (n_zero_rows == leading)
            zero_cols = zero.all(axis=1).sum(axis=1)

            m = {
                "rgt_key": rgt_keys[0],
                "n_slices": int(n),
                "sampled_slices": int(len(idx)),
                "exact_zero_fraction": float(zero.mean()),
                "slices_with_any_zero": int(zero.reshape(len(idx), -1).any(axis=1).sum()),
                "slices_all_zero": int(zero.reshape(len(idx), -1).all(axis=1).sum()),
                "slices_near_constant": int(near_const.sum()),
                "zero_rows_per_slice_mean": float(n_zero_rows.mean()),
                "zero_rows_per_slice_max": int(n_zero_rows.max()),
                "zero_rows_form_top_band_frac": float(top_band[n_zero_rows > 0].mean()) if (n_zero_rows > 0).any() else None,
                "all_zero_columns_per_slice_mean": float(zero_cols.mean()),
                "value_range_mean": float(np.nanmean(rng_per_slice)),
                "seismic": {},
            }
            report[model] = m
            cache[model] = {
                "idx": idx, "rgt": np.nan_to_num(rgt), "valid": (finite & ~zero).astype(np.float64),
                "near_const": near_const, "keys": [k for k in keys if k not in rgt_keys],
            }

        # Pass 2: alignment of every non-RGT dataset with its own RGT and with controls.
        for model in models:
            c = cache[model]
            idx, rgt, valid, near_const = c["idx"], c["rgt"], c["valid"], c["near_const"]
            others = [o for o in models if o != model and cache[o]["rgt"].shape[1:] == rgt.shape[1:]]
            usable = np.where(~near_const)[0]
            sel = rng.choice(usable, size=min(args.pairs_per_dataset, len(usable)), replace=False) if len(usable) else []
            for key in c["keys"]:
                s_all = read_rows(f[f"{model}/{key}"], idx)
                if args.depth_axis == 2:
                    s_all = s_all.transpose(0, 2, 1)
                seis_std = s_all.reshape(len(idx), -1).std(axis=1)
                matched, shuffled, other_model, swapped = [], [], [], []
                for j in sel:
                    if seis_std[j] < 1e-8:
                        continue
                    s = (s_all[j] - s_all[j].mean()) / seis_std[j]
                    matched.append(alignment_score(s, rgt[j], valid[j]))
                    k = rng.choice(np.setdiff1d(usable, [j])) if len(usable) > 1 else j
                    shuffled.append(alignment_score(s, rgt[k], valid[k]))
                    if others:
                        o = cache[others[rng.integers(len(others))]]
                        ok = np.where(~o["near_const"])[0]
                        if len(ok):
                            q = rng.choice(ok)
                            other_model.append(alignment_score(s, o["rgt"][q], o["valid"][q]))
                    if rgt.shape[1] == rgt.shape[2]:
                        swapped.append(alignment_score(s, rgt[j].T, valid[j].T))
                mean = lambda v: float(np.nanmean(v)) if v else None
                report[model]["seismic"][key] = {
                    "near_constant_slices": int((seis_std < 1e-8).sum()),
                    "align_matched": mean(matched),
                    "align_other_slice": mean(shuffled),
                    "align_other_model": mean(other_model),
                    "align_axes_swapped": mean(swapped),
                    "n_scored": len(matched),
                }

    for model, m in report.items():
        print(f"\n== {model}  ({m['rgt_key']}, {m['n_slices']} slices, {m['sampled_slices']} sampled)")
        print(f"   RGT zeros: frac={m['exact_zero_fraction']:.4f}  slices_with_zero={m['slices_with_any_zero']}  "
              f"all_zero={m['slices_all_zero']}  near_constant={m['slices_near_constant']}")
        print(f"   zero rows/slice: mean={m['zero_rows_per_slice_mean']:.1f} max={m['zero_rows_per_slice_max']}  "
              f"top_band_frac={m['zero_rows_form_top_band_frac']}  all-zero cols/slice={m['all_zero_columns_per_slice_mean']:.1f}  "
              f"mean range={m['value_range_mean']:.1f}")
        for key, s in m["seismic"].items():
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(f"   {key[:110]}\n      align matched={fmt(s['align_matched'])}  other_slice={fmt(s['align_other_slice'])}  "
                  f"other_model={fmt(s['align_other_model'])}  axes_swapped={fmt(s['align_axes_swapped'])}  "
                  f"n={s['n_scored']}  near_const={s['near_constant_slices']}")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
