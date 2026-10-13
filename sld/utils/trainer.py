"""Single-GPU training harness shared by train_ae.py and train_baseline.py.

- step-based loop over the infinite PairedChunkStream
- bf16 autocast when the GPU supports it (``precision: auto``), fp32 otherwise
- AdamW + linear warmup + cosine decay, grad-norm clipping
- atomic checkpoints in ``out_dir``: ckpt_latest.pt (resume), ckpt_best.pt (best val), ckpt_final.pt
- automatic resume from ckpt_latest.pt, so resubmitting the slurm job continues the run
- optional EMA of the weights (``optim.ema_decay``): validation and ckpt_best use the EMA weights,
  stored under ``ema`` in the checkpoint (``load_model(..., use_ema=True)``)
- metrics.jsonl with train and val records
"""
import contextlib
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from sld.data import PairedChunkStream, PairedSliceDataset, index_pairs


# ----------------------------------------------------------------------------- setup helpers

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(require_gpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise SystemExit("CUDA is not available (torch.cuda.is_available() is False). Refusing to train on CPU; "
                         "run sbatch slurm/gpu_check.slurm to diagnose, or pass --allow-cpu.")
    return torch.device("cpu")


def resolve_precision(name, device):
    """Returns the autocast dtype, or None for fp32."""
    if device.type != "cuda" or name in (None, "fp32"):
        return None
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"precision must be auto, bf16 or fp32; got {name!r}")


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def reader_kwargs(data_cfg):
    return dict(clip_percentiles=tuple(data_cfg.get("clip_percentiles", (0.5, 99.5))),
                min_valid_frac=data_cfg.get("min_valid_frac_section", 0.25))


def make_train_loader(data_cfg, models, seed):
    sources = index_pairs(data_cfg["h5_path"], models, include_reflectivity=data_cfg.get("include_reflectivity", False))
    stream = PairedChunkStream(data_cfg["h5_path"], sources, buffer_size=data_cfg.get("buffer_size", 4096),
                               seed=seed, repeat=True, rank=0, world_size=1, **reader_kwargs(data_cfg))
    workers = int(data_cfg.get("num_workers", 8))
    extra = dict(persistent_workers=True, prefetch_factor=4) if workers > 0 else {}
    loader = DataLoader(stream, batch_size=data_cfg.get("batch_size", 32), num_workers=workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True, **extra)
    return loader, sources


def make_eval_loader(data_cfg, models, max_per_source=None, batch_size=None):
    sources = index_pairs(data_cfg["h5_path"], models, include_reflectivity=data_cfg.get("include_reflectivity", False))
    ds = PairedSliceDataset(data_cfg["h5_path"], sources, stride=data_cfg.get("val_stride", 1),
                            max_per_source=max_per_source or data_cfg.get("val_max_per_source", 256),
                            **reader_kwargs(data_cfg))
    workers = min(int(data_cfg.get("num_workers", 8)), 4)
    loader = DataLoader(ds, batch_size=batch_size or data_cfg.get("val_batch_size", 64), shuffle=False,
                        num_workers=workers, pin_memory=torch.cuda.is_available())
    return loader, ds


def group_names(dataset, meta):
    """Model name of each sample, from meta[:, 2]."""
    models = dataset.reader.models
    return [models[int(i)] for i in meta[:, 2]]


def apply_debug(cfg, tiny_model):
    """Tiny, fast settings for smoke tests on CPU with tools/make_fake_h5.py data."""
    cfg["data"].update(num_workers=0, batch_size=4, buffer_size=16, val_max_per_source=2, val_batch_size=4)
    cfg["optim"].update(total_steps=4, warmup_steps=1)
    cfg.update(log_every=1, val_every=2, ckpt_every=2, n_vis=2)
    cfg["model"] = dict(tiny_model)
    return cfg


# ----------------------------------------------------------------------------- trainer

class EMA:
    """Exponential moving average of the floating-point state, with the usual warmup
    ``decay_t = min(decay, (1 + t) / (10 + t))`` so early averages are not dominated by the init."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        state = model.state_dict()
        keys = list(self.shadow)
        torch._foreach_lerp_([self.shadow[k] for k in keys], [state[k].float() for k in keys], 1.0 - d)

    def load(self, shadow, num_updates):
        for k, v in shadow.items():
            self.shadow[k].copy_(v)
        self.num_updates = int(num_updates)

    @contextlib.contextmanager
    def swapped_into(self, model):
        """Temporarily load the EMA weights into ``model``."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup)


class Trainer:
    def __init__(self, cfg, kind, model, out_dir, device):
        self.cfg, self.kind, self.model, self.out_dir, self.device = cfg, kind, model, out_dir, device
        o = cfg["optim"]
        self.total_steps = int(o["total_steps"])
        self.grad_clip = o.get("grad_clip", 1.0)
        self.dtype = resolve_precision(o.get("precision", "auto"), device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]),
                                           weight_decay=float(o.get("weight_decay", 0.0)),
                                           betas=tuple(o.get("betas", (0.9, 0.999))))
        self.scheduler = cosine_with_warmup(self.optimizer, int(o.get("warmup_steps", 1000)), self.total_steps,
                                            float(o.get("min_lr_ratio", 0.0)))
        self.ema = EMA(model, o["ema_decay"]) if o.get("ema_decay") else None
        self.step = 0
        self.best = None
        self.log_path = os.path.join(out_dir, "metrics.jsonl")
        os.makedirs(out_dir, exist_ok=True)

    def autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def eval_weights(self):
        """Context in which validation runs: the EMA weights if enabled, else the raw ones."""
        return self.ema.swapped_into(self.model) if self.ema is not None else contextlib.nullcontext()

    # -- checkpoints
    def payload(self):
        return {"kind": self.kind, "model_cfg": self.model.model_cfg, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.shadow if self.ema is not None else None,
                "ema_num_updates": self.ema.num_updates if self.ema is not None else 0,
                "step": self.step, "best": self.best, "config": self.cfg}

    def save(self, name):
        path = os.path.join(self.out_dir, name)
        tmp = path + ".tmp"
        torch.save(self.payload(), tmp)
        os.replace(tmp, path)
        return path

    def maybe_resume(self):
        path = os.path.join(self.out_dir, "ckpt_latest.pt")
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.ema is not None and ckpt.get("ema"):
            self.ema.load(ckpt["ema"], ckpt.get("ema_num_updates", ckpt["step"]))
        self.step, self.best = int(ckpt["step"]), ckpt.get("best")
        print(f"[resume] {path} at step {self.step} (best={self.best})", flush=True)
        return True

    # -- logging
    def log(self, record):
        record = {"step": self.step, **record}
        with open(self.log_path, "a") as fp:
            fp.write(json.dumps(record) + "\n")
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)

    # -- loop
    def fit(self, batches, compute_loss, validate, val_key, batch_size):
        """compute_loss(batch) -> (loss, parts); validate() -> dict of metrics (lower ``val_key`` is better)."""
        c = self.cfg
        log_every, val_every, ckpt_every = int(c.get("log_every", 100)), int(c.get("val_every", 2500)), int(c.get("ckpt_every", 5000))
        running, t0, last_step = defaultdict(float), time.time(), self.step
        while self.step < self.total_steps:
            self.model.train()
            loss, parts = compute_loss(next(batches))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if not torch.isfinite(grad_norm):
                print(f"[warn] non-finite grad norm at step {self.step}; skipping update", flush=True)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
            self.scheduler.step()
            self.step += 1
            for k, v in {"loss": float(loss.detach()), "grad_norm": float(grad_norm), **parts}.items():
                running[k] += v

            if self.step % log_every == 0 or self.step == self.total_steps:
                n = self.step - last_step
                dt = time.time() - t0
                self.log({"split": "train", **{k: v / n for k, v in running.items()},
                          "lr": self.scheduler.get_last_lr()[0], "samples_per_s": n * batch_size / max(dt, 1e-9)})
                running, t0, last_step = defaultdict(float), time.time(), self.step

            if self.step % val_every == 0 or self.step == self.total_steps:
                self.model.eval()
                with torch.no_grad(), self.eval_weights():
                    metrics = validate()
                self.log({"split": "val", **metrics})
                if self.best is None or metrics[val_key] < self.best:
                    self.best = metrics[val_key]
                    self.save("ckpt_best.pt")
                    print(f"[best] {val_key}={self.best:.5g} at step {self.step}", flush=True)
                self.save("ckpt_latest.pt")
            elif self.step % ckpt_every == 0:
                self.save("ckpt_latest.pt")

        self.save("ckpt_latest.pt")
        return self.save("ckpt_final.pt")
