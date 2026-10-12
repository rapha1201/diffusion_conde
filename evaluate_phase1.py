"""Phase 1 acceptance gate: RGT-autoencoder reconstruction ceiling vs the B1 baseline.

    python evaluate_phase1.py --rgt-ae runs/ae_rgt/ckpt_best.pt --baseline runs/baseline_pixel/ckpt_best.pt

Both are evaluated on the same center-cropped sections of the held-out models
(test split by default). The gate passes when the AE reconstruction error is at
most ``--gate-ratio`` (default 1/3) of the B1 prediction error, for both MAE and
horizon displacement: otherwise the diffusion model would be capped near the
baseline before it starts. Writes a JSON report next to the AE checkpoint.
"""
import argparse
import json
import os

import torch

from sld.data import PairTransform, load_split
from sld.eval.metrics import RGTMetrics
from sld.eval.plots import save_rgt_grid
from sld.models import load_model
from sld.utils.trainer import group_names, make_eval_loader, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgt-ae", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--h5", default=None)
    p.add_argument("--part", default="test", choices=["val", "test"])
    p.add_argument("--max-per-source", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gate-ratio", type=float, default=1 / 3)
    p.add_argument("--out", default=None, help="default: <rgt-ae dir>/phase1_<part>.json")
    args = p.parse_args()

    device = resolve_device()
    ae, ae_ckpt = load_model(args.rgt_ae)
    ae.to(device).eval()
    if ae_ckpt["config"].get("modality") != "rgt":
        raise ValueError("--rgt-ae must be the RGT autoencoder")
    b1 = None
    if args.baseline:
        b1, _ = load_model(args.baseline)
        b1.to(device).eval()

    cfg = ae_ckpt["config"]
    data_cfg = dict(cfg["data"])
    if args.h5:
        data_cfg["h5_path"] = args.h5
    split_path = os.path.join(os.path.dirname(args.rgt_ae), "split.json")
    split = load_split(split_path if os.path.exists(split_path) else data_cfg.get("split"), h5_path=data_cfg["h5_path"])
    loader, ds = make_eval_loader(data_cfg, split[args.part], max_per_source=args.max_per_source,
                                  batch_size=args.batch_size)
    eval_t = PairTransform(output_size=cfg.get("transform", {}).get("output_size", 128), train=False,
                           min_valid_frac=cfg.get("transform", {}).get("min_valid_frac", 0.5)).to(device)

    m_ae, m_b1, vis = RGTMetrics(), RGTMetrics(), None
    with torch.no_grad():
        for seis, rgt, valid, meta in loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            s, r, v = s[keep], r[keep], v[keep]
            if len(r) == 0:
                continue
            names = [n for n, k in zip(group_names(ds, meta), keep.cpu()) if k]
            recon = ae.vae.decode(ae.posterior(r).mode()).sample
            m_ae.update(recon, r, v, names)
            if b1 is not None:
                pred = b1(s)
                m_b1.update(pred, r, v, names)
            if vis is None:
                vis = (s, r, recon, v, pred if b1 is not None else None)

    report = {"part": args.part, "models": split[args.part], "sections": m_ae.compute()["n"],
              "ae": {"all": m_ae.compute(), **{g: m_ae.compute(g) for g in m_ae.groups()}}}
    print(f"\n{args.part} models {split[args.part]}  sections={report['sections']}")
    print(f"{'':12s}{'MAE':>10s}{'RMSE':>10s}{'HDE px':>10s}")
    print(f"{'RGT-AE':12s}{report['ae']['all']['mae']:10.4f}{report['ae']['all']['rmse']:10.4f}{report['ae']['all']['hde_px']:10.3f}")
    if b1 is not None:
        report["b1"] = {"all": m_b1.compute(), **{g: m_b1.compute(g) for g in m_b1.groups()}}
        a, b = report["ae"]["all"], report["b1"]["all"]
        print(f"{'B1':12s}{b['mae']:10.4f}{b['rmse']:10.4f}{b['hde_px']:10.3f}")
        ratios = {k: a[k] / max(b[k], 1e-12) for k in ("mae", "hde_px")}
        passed = all(r <= args.gate_ratio for r in ratios.values())
        report["gate"] = {"ratio_threshold": args.gate_ratio, "ratios": ratios, "passed": passed}
        print(f"\nAE/B1 ratios: MAE={ratios['mae']:.3f}  HDE={ratios['hde_px']:.3f}  "
              f"(threshold {args.gate_ratio:.3f}) -> {'PASS' if passed else 'FAIL'}")

    out = args.out or os.path.join(os.path.dirname(args.rgt_ae), f"phase1_{args.part}.json")
    with open(out, "w") as fp:
        json.dump(report, fp, indent=2)
    if vis is not None:
        s, r, recon, v, pred = vis
        base = os.path.splitext(out)[0]
        save_rgt_grid(s, r, recon, v, base + "_ae.png", titles=("GT", "AE recon"))
        if pred is not None:
            save_rgt_grid(s, r, pred, v, base + "_b1.png", titles=("GT", "B1"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
