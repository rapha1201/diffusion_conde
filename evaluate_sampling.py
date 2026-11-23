"""Sample the trained LDM (and optionally B1) per volume, to see how well the seismic conditions the RGT.

    python evaluate_sampling.py --ldm runs/ldm/ckpt_best.pt --baseline runs/baseline_pixel/ckpt_best.pt
    python evaluate_sampling.py --ldm runs/ldm/ckpt_best.pt --models Modelo3 Modelo7 --max-per-source 64
    python evaluate_sampling.py --ldm runs/ldm_clip/ckpt_150k.pt --models Modelo7 --guide-scale 0.1   (dip guidance)

Default volumes: every train model plus the val model(s) of the run's split, so held-in (train)
and held-out (val) volumes can be compared side by side. Center-cropped sections, no augmentation.
Rows per volume (all scored against the section's own GT RGT):
    ae          decode of the true z_y: the RGT-AE ceiling
    sample      one DDIM sample conditioned on the section's seismic
    meanN       mean of N samples (spread = mean per-pixel std across them)
    best_dip    of the N samples, the most consistent with the seismic (lowest dip score, sld/guidance.py);
                selected without any GT (the score sees only the seismic, no validity mask)
    oracle      of the N samples, the one with the lowest MAE: upper bound for any selection rule
    swap        one sample conditioned on the seismic of ANOTHER section of the same volume
                (half the volume away). If it scores about as well as ``sample``, the seismic is ignored.
    partner_gt  the other section's true RGT: what ``swap`` should score if the model follows its condition
    b1          the pixel baseline, if --baseline is given
Dip consistency of every row (and of the GT) with the section's own seismic is reported too
(dip: 0 = isolines parallel to the reflectors, 0.5 = random; mono: fraction of tau decreasing with depth),
with the correlation between the score and the MAE over all samples (does the score find good samples?).

--guide-scale > 0 steers every sample with the dip score at sampling time (weights unchanged). With eta=0
the initial noise only depends on --seed and --sample-chunk, so runs that differ only by the guidance
flags sample from the same noise: their difference is the effect of the guidance alone.

Outputs in --out (default <ldm dir>/sampling_eval): report.json and, per volume, figures for n_vis
sections spread over the volume: <model>_ldm.png (GT, samples, mean, std), <model>_swap.png,
<model>_b1.png, <model>_dip.png (reflector coherence and per-pixel misfit of the GT and of a sample).
"""
import argparse
import json
import os
from collections import defaultdict

import torch

from sld.data import PairTransform, load_split
from sld.eval.metrics import RGTMetrics, SpreadMetrics
from sld.eval.plots import save_dip_grid, save_ldm_grid, save_rgt_grid
from sld.guidance import DipScore
from sld.models import load_model
from sld.utils.trainer import make_eval_loader, resolve_device, resolve_precision
from train_ldm import decode, load_frozen_vae

ROWS = ("ae", "sample", "mean", "best_dip", "oracle", "swap", "partner_gt", "b1")
DIP_ROWS = ("gt",) + ROWS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ldm", required=True, help="LDM checkpoint (EMA weights are used when present)")
    p.add_argument("--baseline", default=None, help="optional B1 checkpoint (runs/baseline_pixel/ckpt_best.pt)")
    p.add_argument("--models", nargs="*", default=None, help="volumes to evaluate (default: train + val of the split)")
    p.add_argument("--h5", default=None, help="override the HDF5 path stored in the checkpoint config")
    p.add_argument("--max-per-source", type=int, default=32, help="sections per seismic realization, spread over the volume")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-chunk", type=int, default=32, help="rows per sampler call (bounds guidance memory)")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=None, help="DDIM steps (default: sampling.steps of the run)")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None, help="classifier-free guidance (needs cond dropout)")
    p.add_argument("--no-swap", action="store_true", help="skip the swap test (faster, e.g. for guidance sweeps)")
    g = p.add_argument_group("dip score / guidance (sld/guidance.py)")
    g.add_argument("--guide-scale", type=float, default=0.0, help="dip guidance step (RMS change of z0 per step); 0 = off")
    g.add_argument("--guide-start", type=float, default=0.3, help="guide from this fraction of the DDIM steps on")
    g.add_argument("--guide-stop", type=float, default=1.0, help="... up to this fraction")
    g.add_argument("--guide-inner", type=int, default=1, help="gradient steps on z0 per guided DDIM step")
    g.add_argument("--mono-weight", type=float, default=1.0, help="weight of the depth-monotonicity term")
    g.add_argument("--sigma-tensor", type=float, default=3.0, help="structure-tensor smoothing (px)")
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


def section_mae(pred, target, valid):
    """[B] MAE per section over valid pixels."""
    err = (pred.float().clamp(0, 1) - target.float()).abs() * valid
    return err.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)


def pearson(a, b):
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


class DipStats:
    """Mean dip/mono per row, and how well the (mask-free) selection score ranks samples by MAE."""

    def __init__(self):
        self.sums = defaultdict(lambda: [0.0, 0.0, 0])
        self.pairs, self.within = [], []

    def add(self, row, dip, mono):
        s = self.sums[row]
        s[0] += float(dip.sum())
        s[1] += float(mono.sum())
        s[2] += dip.numel()

    def add_selection(self, score, mae):
        """score, mae: [B, N] for the N samples of each section."""
        self.pairs.append(torch.stack([score.flatten(), mae.flatten()]).cpu())
        if score.shape[1] >= 3:
            self.within.append(pearson(score, mae).cpu())

    def compute(self):
        out = {r: {"dip": s[0] / s[2], "mono": s[1] / s[2], "n": s[2]} for r, s in self.sums.items() if s[2]}
        if self.pairs:
            p = torch.cat(self.pairs, dim=1)
            out["selection"] = {"pearson_score_mae": float(pearson(p[0], p[1])),
                                "pearson_within_section": float(torch.cat(self.within).mean()) if self.within else None}
        return out


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
    dip_kw = dict(sigma_tensor=args.sigma_tensor, mono_weight=args.mono_weight)
    guide = dict(scale=args.guide_scale, start=args.guide_start, stop=args.guide_stop, inner=args.guide_inner, **dip_kw)

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
    print(f"ldm={args.ldm} (step {ckpt['step']}) steps={steps} eta={eta} cfg_guidance={guidance} N={n_samples} "
          f"dip_guidance={guide if args.guide_scale > 0 else 'off'} models={models}", flush=True)

    def sample(z_x, seis, gen):
        """tau [rows,1,H,W] for each row of z_x (seis: the matching seismic, used by the dip guidance only)."""
        out = []
        for idx in torch.arange(len(z_x)).split(args.sample_chunk):
            zx = z_x[idx].to(device)
            kw = {}
            if args.guide_scale > 0:
                score = DipScore(seis[idx].to(device), **dip_kw)  # seismic only: no GT, no validity mask
                kw = dict(guide_fn=lambda z0, score=score: score(decode(vae_y, z0)), guide_scale=args.guide_scale,
                          guide_start=args.guide_start, guide_stop=args.guide_stop, guide_inner=args.guide_inner)
            with torch.no_grad(), autocast():
                z = ldm.sample(zx, num_inference_steps=steps, eta=eta, guidance_scale=guidance, generator=gen, **kw)
                out.append(decode(vae_y, z))
        return torch.cat(out)

    # metrics[scope][row]: scope = model name and its split part ("train" / "val" / ...)
    metrics = defaultdict(lambda: {**{r: RGTMetrics() for r in ROWS}, "spread": SpreadMetrics(), "dip": DipStats()})
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
            b = len(r)
            with torch.no_grad(), autocast():
                z_x, z_y = vae_x.encode(s).float(), vae_y.encode(r).float()
                ae = decode(vae_y, z_y)
                pred_b1 = b1(s).float() if b1 is not None else None
            tau = sample(z_x.repeat_interleave(n_samples, dim=0), s.repeat_interleave(n_samples, dim=0), gen)
            tau = tau.view(b, n_samples, *r.shape[1:])

            # Selection among the N samples: mask-free dip score (what a real survey allows) vs the MAE oracle.
            rows = torch.arange(b, device=device)
            flat = tau.flatten(0, 1)
            sel = DipScore(s, **dip_kw).repeat_interleave(n_samples)(flat).view(b, n_samples)
            mae = section_mae(flat, r.repeat_interleave(n_samples, 0), v.repeat_interleave(n_samples, 0)).view(b, -1)
            best = tau[rows, sel.argmin(1)]
            oracle = tau[rows, mae.argmin(1)]

            # Reported dip consistency: against the section's own seismic, on its valid pixels.
            score = DipScore(s, mask=v, **dip_kw)
            dips = {"gt": score.terms(r), "ae": score.terms(ae.clamp(0, 1)),
                    "sample": score.repeat_interleave(n_samples).terms(flat.clamp(0, 1)),
                    "mean": score.terms(tau.mean(1).clamp(0, 1)), "best_dip": score.terms(best.clamp(0, 1)),
                    "oracle": score.terms(oracle.clamp(0, 1))}
            if pred_b1 is not None:
                dips["b1"] = score.terms(pred_b1.clamp(0, 1))
            for scope in scopes:
                m = metrics[scope]
                m["ae"].update(ae, r, v)
                m["sample"].update(tau[:, 0], r, v)
                m["mean"].update(tau.mean(1), r, v)
                if n_samples > 1:
                    m["spread"].update(tau, r, v)
                    m["best_dip"].update(best, r, v)
                    m["oracle"].update(oracle, r, v)
                    m["dip"].add_selection(sel, mae)
                if pred_b1 is not None:
                    m["b1"].update(pred_b1, r, v)
                for row, (dip, mono) in dips.items():
                    m["dip"].add(row, dip, mono)
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
        if not args.no_swap:
            partner = swap_partners(d["local"])
            d["swap"] = sample(d["z_x"][partner], d["s"][partner], gen).cpu()
            for i in range(0, n, args.batch_size):
                sl = slice(i, i + args.batch_size)
                s, r, v = d["s"][sl].to(device), d["r"][sl].to(device), d["v"][sl].to(device)
                r_p, v_p = d["r"][partner[sl]].to(device), d["v"][partner[sl]].to(device)
                sw = d["swap"][sl].to(device)
                for scope in scopes:
                    metrics[scope]["swap"].update(sw, r, v)
                    metrics[scope]["partner_gt"].update(r_p, r, v * v_p)
                    metrics[scope]["dip"].add("swap", *DipScore(s, mask=v, **dip_kw).terms(sw.clamp(0, 1)))
                    metrics[scope]["dip"].add("partner_gt", *DipScore(s, mask=v * v_p, **dip_kw).terms(r_p))

        # Figures for n_vis sections spread over the volume (slice order), not the first batch.
        order = torch.argsort(d["local"], stable=True)
        vis = order[torch.linspace(0, n - 1, min(args.n_vis, n)).round().long()]
        save_ldm_grid(d["s"][vis], d["r"][vis], d["tau"][vis], d["v"][vis],
                      os.path.join(out_dir, f"{model}_ldm.png"), n=len(vis))
        save_dip_grid(d["s"][vis], d["r"][vis], d["tau"][vis, 0], d["v"][vis], os.path.join(out_dir, f"{model}_dip.png"),
                      n=len(vis), **dip_kw)
        if "swap" in d:
            save_rgt_grid(d["s"][vis], d["r"][vis], d["swap"][vis], d["v"][vis],
                          os.path.join(out_dir, f"{model}_swap.png"), n=len(vis), titles=("GT", "swapped-seismic sample"))
        if b1 is not None:
            save_rgt_grid(d["s"][vis], d["r"][vis], d["b1"][vis], d["v"][vis], os.path.join(out_dir, f"{model}_b1.png"),
                          n=len(vis), titles=("GT", "B1"))
        print(f"{model} ({scopes[1]}): {n} sections done", flush=True)

    report = {"ldm": args.ldm, "ldm_step": ckpt["step"], "baseline": args.baseline, "steps": steps, "eta": eta,
              "guidance_scale": guidance, "dip_guidance": guide, "n_samples": n_samples, "seed": args.seed,
              "max_per_source": args.max_per_source, "split": split, "results": {}}
    labels = {"mean": f"mean{n_samples}"}
    scopes = [m for m in models if m in metrics] + [p for p in ("train", "val", "test", "other") if p in metrics]
    for scope in scopes:
        m = metrics[scope]
        res = {labels.get(r, r): m[r].compute() for r in ROWS if m[r].compute()["n"] > 0}
        if n_samples > 1:
            res["spread"] = m["spread"].compute()
        dip = m["dip"].compute()
        res["dip"] = {labels.get(r, r): v for r, v in dip.items()}
        report["results"][scope] = res
        tag = f"{scope} ({part_of[scope]})" if scope in part_of else f"ALL {scope.upper()} VOLUMES"
        print(f"\n== {tag}   sections={res['ae']['n']}")
        print(f"   {'':12s}{'MAE':>9s}{'RMSE':>9s}{'HDE px':>9s}{'dip':>8s}{'mono':>8s}")
        if "gt" in dip:
            print(f"   {'gt':12s}{'':27s}{dip['gt']['dip']:8.4f}{dip['gt']['mono']:8.4f}")
        for row in ROWS:
            key = labels.get(row, row)
            if key in res:
                v, dv = res[key], dip.get(row)
                tail = f"{dv['dip']:8.4f}{dv['mono']:8.4f}" if dv else ""
                print(f"   {key:12s}{v['mae']:9.4f}{v['rmse']:9.4f}{v['hde_px']:9.2f}{tail}")
        if "spread" in res:
            print(f"   spread={res['spread']['spread']:.4f}  std_err_corr={res['spread']['std_err_corr']:.3f}")
        if "selection" in dip:
            sel = dip["selection"]
            within = sel["pearson_within_section"]
            print(f"   dip score vs MAE: pearson={sel['pearson_score_mae']:.3f}"
                  + (f"  within-section={within:.3f}" if within is not None else ""))

    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nwrote {path} and figures in {out_dir}")


if __name__ == "__main__":
    main()
