"""Sample the trained LDM (and optionally B1) per volume, to see how well the seismic conditions the RGT.

    python evaluate_sampling.py --ldm runs/ldm/ckpt_best.pt --baseline runs/baseline_pixel/ckpt_best.pt
    python evaluate_sampling.py --ldm runs/ldm/ckpt_best.pt --models Modelo3 Modelo7 --max-per-source 64

Default volumes: every train model plus the val model(s) of the run's split, so held-in (train)
and held-out (val) volumes can be compared side by side. Center-cropped sections, no augmentation.
Rows per volume (all scored against the section's own GT RGT):
    ae          decode of the true z_y: the RGT-AE ceiling
    sample      one DDIM sample conditioned on the section's seismic
    meanN       mean of N samples (spread = mean per-pixel std across them)
    swap        one sample conditioned on the seismic of ANOTHER section of the same volume
                (half the volume away). If it scores about as well as ``sample``, the seismic is ignored.
    partner_gt  the other section's true RGT: what ``swap`` should score if the model follows its condition
    b1          the pixel baseline, if --baseline is given
Outputs in --out (default <ldm dir>/sampling_eval): report.json and, per volume, figures for n_vis
sections spread over the volume: <model>_ldm.png (GT, samples, mean, std), <model>_swap.png, <model>_b1.png.
"""
import argparse
import json
import os
from collections import defaultdict

import torch

from sld.data import PairTransform, load_split
from sld.eval.metrics import RGTMetrics, SpreadMetrics
from sld.eval.plots import save_ldm_grid, save_rgt_grid
from sld.models import load_model
from sld.utils.trainer import make_eval_loader, resolve_device, resolve_precision
from train_ldm import decode, load_frozen_vae

ROWS = ("ae", "sample", "mean", "swap", "partner_gt", "b1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ldm", required=True, help="LDM checkpoint (EMA weights are used when present)")
    p.add_argument("--baseline", default=None, help="optional B1 checkpoint (runs/baseline_pixel/ckpt_best.pt)")
    p.add_argument("--models", nargs="*", default=None, help="volumes to evaluate (default: train + val of the split)")
    p.add_argument("--h5", default=None, help="override the HDF5 path stored in the checkpoint config")
    p.add_argument("--max-per-source", type=int, default=32, help="sections per seismic realization, spread over the volume")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=None, help="DDIM steps (default: sampling.steps of the run)")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--n-vis", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def swap_partners(local_idx):
    """For each section, the section about half the volume away (sorted by slice index). Realizations of
    the same volume share slice indices, so a plain roll of the batch could pair a section with its twin."""
    order = torch.argsort(local_idx, stable=True)
    n = len(order)
    partner = torch.empty_like(order)
    partner[order] = order[(torch.arange(n) + n // 2) % n]
    return partner


def main():
    args = parse_args()
    device = resolve_device()
    ldm, ckpt = load_model(args.ldm, use_ema=True)
    if ckpt["kind"] != "ldm":
        raise ValueError(f"{args.ldm} is a {ckpt['kind']!r} checkpoint, not the LDM")
    ldm.to(device).eval()
    cfg = ckpt["config"]
    dtype = resolve_precision(cfg.get("optim", {}).get("precision", "auto"), device)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)

    vae_x = load_frozen_vae(cfg["vae"]["seismic"], "seismic", device)
    vae_y = load_frozen_vae(cfg["vae"]["rgt"], "rgt", device)
    b1 = None
    if args.baseline:
        b1, _ = load_model(args.baseline)
        b1.to(device).eval()

    scfg = cfg.get("sampling", {})
    steps = args.steps or int(scfg.get("steps", 50))
    eta = args.eta if args.eta is not None else float(scfg.get("eta", 0.0))
    guidance = args.guidance_scale if args.guidance_scale is not None else float(scfg.get("guidance_scale", 1.0))
    if guidance != 1.0 and not ldm.cond_drop_prob:
        print(f"[warn] guidance_scale={guidance} but the model was trained without condition dropout", flush=True)
    n_samples = max(1, args.n_samples)

    data_cfg = dict(cfg["data"])
    if args.h5:
        data_cfg["h5_path"] = args.h5
    split_path = os.path.join(os.path.dirname(args.ldm), "split.json")
    split = load_split(split_path if os.path.exists(split_path) else data_cfg.get("split"), h5_path=data_cfg["h5_path"])
    models = args.models or split["train"] + split["val"]
    part_of = {m: part for part in ("train", "val", "test") for m in split[part]}
    eval_t = PairTransform(output_size=cfg.get("transform", {}).get("output_size", 128), train=False,
                           min_valid_frac=cfg.get("transform", {}).get("min_valid_frac", 0.5)).to(device)
    out_dir = args.out or os.path.join(os.path.dirname(args.ldm), "sampling_eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"ldm={args.ldm} (step {ckpt['step']}) steps={steps} eta={eta} guidance={guidance} N={n_samples} "
          f"models={models}", flush=True)

    def sample(z_x, gen):
        """[B, k, 1, H, W] tau samples, k = rows of z_x repeated."""
        with torch.no_grad(), autocast():
            z = ldm.sample(z_x, num_inference_steps=steps, eta=eta, guidance_scale=guidance, generator=gen)
            return decode(vae_y, z)

    # metrics[scope][row]: scope = model name and its split part ("train" / "val" / ...)
    metrics = defaultdict(lambda: {**{r: RGTMetrics() for r in ROWS}, "spread": SpreadMetrics()})
    for model in models:
        scopes = (model, part_of.get(model, "other"))
        loader, ds = make_eval_loader(data_cfg, [model], max_per_source=args.max_per_source, batch_size=args.batch_size)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        keep_all = defaultdict(list)
        for seis, rgt, valid, meta in loader:
            s, r, v, keep = eval_t(seis.to(device), rgt.to(device), valid.to(device))
            s, r, v, meta = s[keep], r[keep], v[keep], meta[keep.cpu()]
            if len(r) == 0:
                continue
            with torch.no_grad(), autocast():
                z_x, z_y = vae_x.encode(s).float(), vae_y.encode(r).float()
                ae = decode(vae_y, z_y)
                pred_b1 = b1(s).float() if b1 is not None else None
            tau = sample(z_x.repeat_interleave(n_samples, dim=0), gen).view(len(r), n_samples, *r.shape[1:])
            for scope in scopes:
                m = metrics[scope]
                m["ae"].update(ae, r, v)
                m["sample"].update(tau[:, 0], r, v)
                m["mean"].update(tau.mean(1), r, v)
                if n_samples > 1:
                    m["spread"].update(tau, r, v)
                if pred_b1 is not None:
                    m["b1"].update(pred_b1, r, v)
            for k, x in (("s", s), ("r", r), ("v", v), ("z_x", z_x), ("tau", tau)):
                keep_all[k].append(x.cpu())
            keep_all["local"].append(meta[:, 1])
            if pred_b1 is not None:
                keep_all["b1"].append(pred_b1.cpu())
        if not keep_all:
            print(f"[skip] {model}: no section passed the validity filter", flush=True)
            continue
        d = {k: torch.cat(v) for k, v in keep_all.items()}
        n = len(d["r"])

        # Swap test: condition each section on the seismic of its partner, score against its own GT.
        partner = swap_partners(d["local"])
        swap_tau = []
        for idx in torch.arange(n).split(args.batch_size * n_samples):
            swap_tau.append(sample(d["z_x"][partner[idx]].to(device), gen).cpu())
        d["swap"] = torch.cat(swap_tau)
        for i in range(0, n, args.batch_size):
            sl = slice(i, i + args.batch_size)
            r, v = d["r"][sl].to(device), d["v"][sl].to(device)
            r_p, v_p = d["r"][partner[sl]].to(device), d["v"][partner[sl]].to(device)
            for scope in scopes:
                metrics[scope]["swap"].update(d["swap"][sl].to(device), r, v)
                metrics[scope]["partner_gt"].update(r_p, r, v * v_p)

        # Figures for n_vis sections spread over the volume (slice order), not the first batch.
        order = torch.argsort(d["local"], stable=True)
        vis = order[torch.linspace(0, n - 1, min(args.n_vis, n)).round().long()]
        save_ldm_grid(d["s"][vis], d["r"][vis], d["tau"][vis], d["v"][vis],
                      os.path.join(out_dir, f"{model}_ldm.png"), n=len(vis))
        save_rgt_grid(d["s"][vis], d["r"][vis], d["swap"][vis], d["v"][vis], os.path.join(out_dir, f"{model}_swap.png"),
                      n=len(vis), titles=("GT", "swapped-seismic sample"))
        if b1 is not None:
            save_rgt_grid(d["s"][vis], d["r"][vis], d["b1"][vis], d["v"][vis], os.path.join(out_dir, f"{model}_b1.png"),
                          n=len(vis), titles=("GT", "B1"))
        print(f"{model} ({scopes[1]}): {n} sections done", flush=True)

    report = {"ldm": args.ldm, "ldm_step": ckpt["step"], "baseline": args.baseline, "steps": steps, "eta": eta,
              "guidance_scale": guidance, "n_samples": n_samples, "max_per_source": args.max_per_source,
              "split": split, "results": {}}
    labels = {"mean": f"mean{n_samples}"}
    scopes = [m for m in models if m in metrics] + [p for p in ("train", "val", "test", "other") if p in metrics]
    for scope in scopes:
        m = metrics[scope]
        res = {labels.get(r, r): m[r].compute() for r in ROWS if m[r].compute()["n"] > 0}
        if n_samples > 1:
            res["spread"] = m["spread"].compute()
        report["results"][scope] = res
        tag = f"{scope} ({part_of[scope]})" if scope in part_of else f"ALL {scope.upper()} VOLUMES"
        print(f"\n== {tag}   sections={res['ae']['n']}")
        print(f"   {'':12s}{'MAE':>9s}{'RMSE':>9s}{'HDE px':>9s}")
        for row, v in res.items():
            if row != "spread":
                print(f"   {row:12s}{v['mae']:9.4f}{v['rmse']:9.4f}{v['hde_px']:9.2f}")
        if "spread" in res:
            print(f"   spread={res['spread']['spread']:.4f}  std_err_corr={res['spread']['std_err_corr']:.3f}")

    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nwrote {path} and figures in {out_dir}")


if __name__ == "__main__":
    main()
