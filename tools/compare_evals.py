"""Side-by-side table of several evaluate_sampling.py reports (CPU only, no data access).

    python tools/compare_evals.py ldm=runs/ldm/sampling_eval/report.json \
        ldm_clip=runs/ldm_clip/sampling_eval/report.json \
        ldm_clip+dip=runs/ldm_clip/guided_0.1/report.json --scope val test --csv figs/compare.csv

Each argument is label=path (or just a path: the label is then the run directory name). For every
scope (a volume such as Modelo7, or a split part: train / val / test) prints one markdown table with
MAE, RMSE, HDE and the dip-consistency columns for each (report, row), and the GT dip as reference.
"""
import argparse
import csv
import json
import os

DEFAULT_ROWS = ("sample", "mean", "best_dip", "oracle", "swap", "b1")


def load(arg):
    label, _, path = arg.partition("=") if "=" in arg else ("", "", arg)
    if not label:
        label = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    with open(path) as fp:
        return label, json.load(fp)


def row_key(row, report):
    return f"mean{report.get('n_samples', 4)}" if row == "mean" else row


def describe(report):
    g = report.get("dip_guidance") or {}
    guide = f"dip guidance {g['scale']:g} from {g.get('start', 0):g}" if g.get("scale") else "no dip guidance"
    return f"step {report.get('ldm_step')}, {report.get('steps')} DDIM steps, N={report.get('n_samples')}, {guide}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", help="label=path/to/report.json")
    p.add_argument("--scope", nargs="*", default=None, help="volumes or split parts (default: every scope found)")
    p.add_argument("--rows", nargs="*", default=list(DEFAULT_ROWS))
    p.add_argument("--csv", default=None, help="also write all numbers to this CSV")
    args = p.parse_args()
    reports = [load(a) for a in args.reports]
    scopes = args.scope or list(dict.fromkeys(s for _, r in reports for s in r["results"]))

    records = []
    for label, rep in reports:
        print(f"- {label}: {rep.get('ldm')} ({describe(rep)})")
    for scope in scopes:
        print(f"\n### {scope}\n")
        print("| model | row | MAE | RMSE | HDE px | dip | mono |")
        print("|---|---|---:|---:|---:|---:|---:|")
        gt_dip = None
        for label, rep in reports:
            res = rep["results"].get(scope)
            if not res:
                continue
            dip = res.get("dip", {})
            gt_dip = gt_dip or dip.get("gt")
            for row in args.rows:
                key = row_key(row, rep)
                if key not in res:
                    continue
                m, d = res[key], dip.get(key, {})
                rec = dict(scope=scope, model=label, row=key, mae=m["mae"], rmse=m["rmse"], hde_px=m["hde_px"],
                           dip=d.get("dip"), mono=d.get("mono"), n=m["n"])
                records.append(rec)
                fmt = lambda v, f: format(v, f) if v is not None else "-"
                print(f"| {label} | {key} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['hde_px']:.2f} | "
                      f"{fmt(rec['dip'], '.4f')} | {fmt(rec['mono'], '.4f')} |")
        if gt_dip:
            print(f"\nGT dip {gt_dip['dip']:.4f}, mono {gt_dip['mono']:.4f} (reference: what a perfect RGT scores)")
        for label, rep in reports:
            sel = rep["results"].get(scope, {}).get("dip", {}).get("selection")
            if sel:
                w = sel.get("pearson_within_section")
                print(f"{label}: dip score vs MAE pearson {sel['pearson_score_mae']:.3f}"
                      + (f", within section {w:.3f}" if w is not None else ""))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        keys = ["scope", "model", "row", "mae", "rmse", "hde_px", "dip", "mono", "n"]
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
