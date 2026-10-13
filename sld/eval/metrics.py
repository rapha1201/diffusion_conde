"""Evaluation metrics, accumulated over batches and optionally per group (e.g. per model).

RGT (values in [0,1], depth = rows):
    mae, rmse          over valid pixels
    hde_px             horizon displacement error in pixels: for K iso-levels of tau,
                       the per-trace depth of the horizon is the number of valid samples
                       with tau < level; hde is |depth_pred - depth_true| averaged over
                       traces and levels present in the true trace.
Seismic:
    mae, rmse, corr    corr = mean per-sample Pearson correlation
"""
from collections import defaultdict

import torch


DEFAULT_LEVELS = torch.linspace(0.05, 0.95, 19)


@torch.no_grad()
def horizon_depths(rgt, valid, levels):
    """rgt, valid: [B,1,H,W]; levels: [K]. Returns depth [B,K,W] (float, pixels)."""
    r = rgt[:, 0]
    v = valid[:, 0].bool()
    below = (r[:, None] < levels.view(1, -1, 1, 1)) & v[:, None]
    return below.sum(dim=2).float()


@torch.no_grad()
def horizon_presence(rgt, valid, levels):
    """[B,K,W] bool: the level lies strictly inside the valid tau range of the trace."""
    r = rgt[:, 0]
    v = valid[:, 0].bool()
    big = torch.finfo(r.dtype).max
    tmin = torch.where(v, r, torch.full_like(r, big)).amin(dim=1)
    tmax = torch.where(v, r, torch.full_like(r, -big)).amax(dim=1)
    lv = levels.view(1, -1, 1)
    return (lv > tmin[:, None]) & (lv < tmax[:, None]) & v.any(dim=1)[:, None]


class _Accumulator:
    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))

    def _add_rows(self, stats, groups):
        """stats: dict key -> [B] tensor of per-sample sums. One device->host copy per batch."""
        keys = list(stats)
        rows = torch.stack([stats[k].float() for k in keys], dim=1).cpu().tolist()
        names = ["all"] * len(rows) if groups is None else list(groups)
        for row, g in zip(rows, names):
            for grp in {"all", g}:
                for k, v in zip(keys, row):
                    self.sums[grp][k] += v
                self.sums[grp]["n"] += 1

    def groups(self):
        return [g for g in self.sums if g != "all"]


class RGTMetrics(_Accumulator):
    def __init__(self, levels=DEFAULT_LEVELS):
        super().__init__()
        self.levels = levels

    @torch.no_grad()
    def update(self, pred, target, valid, groups=None):
        pred, target, valid = pred.float().clamp(0, 1), target.float(), valid.float()
        err = (pred - target).abs() * valid
        lv = self.levels.to(pred.device)
        present = horizon_presence(target, valid, lv)
        disp = (horizon_depths(pred, valid, lv) - horizon_depths(target, valid, lv)).abs() * present
        self._add_rows({
            "abs": err.flatten(1).sum(1),
            "sq": (err ** 2).flatten(1).sum(1),
            "px": valid.flatten(1).sum(1),
            "hde": disp.flatten(1).sum(1),
            "hde_n": present.flatten(1).sum(1),
        }, groups)

    def compute(self, group="all"):
        s = self.sums[group]
        px = max(s["px"], 1.0)
        return {"mae": s["abs"] / px, "rmse": (s["sq"] / px) ** 0.5,
                "hde_px": s["hde"] / max(s["hde_n"], 1.0), "n": int(s["n"])}


class SpreadMetrics(_Accumulator):
    """Diversity of N samples of tau for the same seismic:
    spread        mean per-pixel std across samples (valid pixels)
    std_err_corr  per-section Pearson correlation between that std and the |error| of the sample mean,
                  averaged over sections (> 0: the model is unsure where it is wrong)
    """

    @torch.no_grad()
    def update(self, samples, target, valid, groups=None):
        """samples: [B,N,1,H,W] with N >= 2."""
        samples = samples.float().clamp(0, 1)
        v = valid.float()
        std = samples.std(dim=1)
        err = (samples.mean(dim=1) - target.float()).abs()
        px = v.flatten(1).sum(1)

        def centered(x):
            m = (x * v).flatten(1).sum(1) / px.clamp_min(1)
            return ((x - m.view(-1, 1, 1, 1)) * v).flatten(1)

        a, b = centered(std), centered(err)
        corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
        self._add_rows({"spread": (std * v).flatten(1).sum(1), "px": px, "corr": corr}, groups)

    def compute(self, group="all"):
        s = self.sums[group]
        return {"spread": s["spread"] / max(s["px"], 1.0), "std_err_corr": s["corr"] / max(s["n"], 1.0),
                "n": int(s["n"])}


class SeismicMetrics(_Accumulator):
    @torch.no_grad()
    def update(self, pred, target, groups=None):
        pred, target = pred.float(), target.float()
        d = (pred - target).flatten(1)
        p = pred.flatten(1) - pred.flatten(1).mean(1, keepdim=True)
        t = target.flatten(1) - target.flatten(1).mean(1, keepdim=True)
        corr = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-8)
        self._add_rows({
            "abs": d.abs().sum(1),
            "sq": (d ** 2).sum(1),
            "px": torch.full_like(corr, d.shape[1]),
            "corr": corr,
        }, groups)

    def compute(self, group="all"):
        s = self.sums[group]
        px, n = max(s["px"], 1.0), max(s["n"], 1.0)
        return {"mae": s["abs"] / px, "rmse": (s["sq"] / px) ** 0.5, "corr": s["corr"] / n, "n": int(s["n"])}
