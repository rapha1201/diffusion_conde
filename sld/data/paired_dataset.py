"""Paired seismic / RGT sections read straight from the HDF5 file.

Layout (verified on fault_aware.h5 with tools/inspect_h5.py and tools/check_pairs.py):

    /<Model>/RGT_<Model>.segy          (N, 196, 196)  raw RGT, depth along axis 1 (rows)
    /<Model>/Refletividade_<Model>.segy
    /<Model>/Sism..._<Model>_....segy   one per seismic realization, slice-aligned with the RGT

RGT == 0 marks an undefined shallow band (always whole top rows) and, in Modelo9,
entire empty slices. Those pixels are invalid.

The training set is ~765k sections (~117 GB) stored gzip-compressed in chunks of
32 sections, so it cannot be preloaded and random single-section reads would
decompress 32 sections to use one. ``PairedChunkStream`` therefore reads whole
chunks in shuffled order and mixes them through a shuffle buffer.
``PairedSliceDataset`` is the deterministic, map-style counterpart for evaluation.

Both yield ``(seismic, rgt, valid, meta)``:
    seismic [1,H,W] float32, robustly normalized per section
    rgt     [1,H,W] float32, min-max normalized to [0,1] over valid pixels, 0 where invalid
    valid   [1,H,W] float32 in {0,1}
    meta    [3] int64: (source index, local slice index, model index)
"""
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass(frozen=True)
class PairSource:
    model: str
    seismic_key: str
    rgt_key: str
    n_slices: int
    chunk: int

    @property
    def seismic_path(self):
        return f"{self.model}/{self.seismic_key}"

    @property
    def rgt_path(self):
        return f"{self.model}/{self.rgt_key}"


def index_pairs(h5_path, model_names, rgt_token="rgt", reflectivity_token="refletividade",
                include_reflectivity=False, exclude_tokens=()):
    """One PairSource per (model, seismic realization)."""
    rgt_re = re.compile(rgt_token, re.IGNORECASE)
    sources = []
    with h5py.File(h5_path, "r") as f:
        for model in model_names:
            if model not in f:
                raise KeyError(f"Model '{model}' not found in {h5_path}")
            keys = sorted(f[model].keys())
            rgt_keys = [k for k in keys if rgt_re.search(k)]
            if len(rgt_keys) != 1:
                raise ValueError(f"Expected exactly one RGT dataset in '{model}', found {rgt_keys}")
            rgt_ds = f[f"{model}/{rgt_keys[0]}"]
            for key in keys:
                low = key.lower()
                if key in rgt_keys or any(t.lower() in low for t in exclude_tokens):
                    continue
                if reflectivity_token.lower() in low and not include_reflectivity:
                    continue
                ds = f[f"{model}/{key}"]
                if ds.shape != rgt_ds.shape:
                    raise ValueError(f"{model}/{key} has shape {ds.shape}, RGT has {rgt_ds.shape}")
                chunk = ds.chunks[0] if ds.chunks else 32
                sources.append(PairSource(model, key, rgt_keys[0], int(ds.shape[0]), int(chunk)))
    if not sources:
        raise ValueError(f"No seismic datasets found for models {model_names}")
    return sources


def robust_normalize(x, lo=0.5, hi=99.5, eps=1e-6):
    """Per-section percentile clipping followed by z-scoring. x: [n,H,W]."""
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(x.shape[0], -1)
    p_lo, p_hi = np.percentile(flat, [lo, hi], axis=1)
    x = np.clip(x, p_lo[:, None, None], p_hi[:, None, None])
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32), std.reshape(-1)


def rgt_to_unit(rgt, invalid_value=0.0, eps=1e-6):
    """Per-section min-max over valid pixels. Returns (rgt01, valid, value_range). rgt: [n,H,W]."""
    rgt = rgt.astype(np.float32, copy=False)
    valid = np.isfinite(rgt)
    if invalid_value is not None:
        valid &= rgt != invalid_value
    big = np.float32(np.finfo(np.float32).max)
    has_valid = valid.any(axis=(1, 2), keepdims=True)
    # Sentinels only matter for sections with no valid pixel; zero them before subtracting.
    vmin = np.where(has_valid, np.where(valid, rgt, big).min(axis=(1, 2), keepdims=True), 0.0)
    vmax = np.where(has_valid, np.where(valid, rgt, -big).max(axis=(1, 2), keepdims=True), 0.0)
    value_range = vmax - vmin
    out = (np.where(valid, rgt, vmin) - vmin) / np.maximum(value_range, eps)
    out = np.where(valid, out, 0.0).astype(np.float32)
    return out, valid, value_range.reshape(-1)


class _PairReader:
    """Reads and normalizes aligned seismic/RGT row ranges. Opens the file lazily (per worker)."""

    def __init__(self, h5_path, sources, clip_percentiles=(0.5, 99.5), rgt_invalid_value=0.0,
                 min_valid_frac=0.25, min_seismic_std=1e-6):
        self.h5_path = h5_path
        self.sources = list(sources)
        self.clip_percentiles = clip_percentiles
        self.rgt_invalid_value = rgt_invalid_value
        self.min_valid_frac = min_valid_frac
        self.min_seismic_std = min_seismic_std
        self.models = sorted({s.model for s in self.sources})
        self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # never ship an open h5py handle to workers
        return state

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def read(self, src_idx, start, stop):
        """Returns (seismic, rgt, valid, keep, local_indices) as numpy arrays for rows [start, stop)."""
        s = self.sources[src_idx]
        seis = self.file[s.seismic_path][start:stop]
        rgt = self.file[s.rgt_path][start:stop]
        seis, seis_std = robust_normalize(seis, *self.clip_percentiles)
        rgt, valid, value_range = rgt_to_unit(rgt, self.rgt_invalid_value)
        valid_frac = valid.reshape(valid.shape[0], -1).mean(axis=1)
        keep = (valid_frac >= self.min_valid_frac) & (value_range > 0) & (seis_std > self.min_seismic_std)
        return seis, rgt, valid.astype(np.float32), keep, np.arange(start, stop)

    def to_item(self, src_idx, seis, rgt, valid, local_idx):
        meta = torch.tensor([src_idx, local_idx, self.models.index(self.sources[src_idx].model)], dtype=torch.int64)
        return (torch.from_numpy(np.ascontiguousarray(seis[None])),
                torch.from_numpy(np.ascontiguousarray(rgt[None])),
                torch.from_numpy(np.ascontiguousarray(valid[None])),
                meta)


def _dist_info(rank, world_size):
    if rank is not None and world_size is not None:
        return rank, world_size
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class PairedChunkStream(IterableDataset):
    """Infinite (or single-pass) shuffled stream of training pairs.

    Units of work are whole HDF5 chunks. Each epoch the unit list is shuffled with
    ``seed + epoch``, truncated to a multiple of ``world_size`` and split across
    DDP ranks, then across DataLoader workers. Sections pass through a shuffle
    buffer so a batch mixes many chunks, volumes and realizations.

    With ``repeat=True`` the stream never ends; training loops are step-based,
    which also keeps DDP ranks in lockstep regardless of filtered sections.
    """

    def __init__(self, h5_path, sources, buffer_size=4096, seed=0, repeat=True, rank=None, world_size=None,
                 **reader_kwargs):
        super().__init__()
        self.reader = _PairReader(h5_path, sources, **reader_kwargs)
        self.buffer_size = int(buffer_size)
        self.seed = int(seed)
        self.repeat = repeat
        self.rank, self.world_size = _dist_info(rank, world_size)
        self.epoch = 0

    @property
    def sources(self):
        return self.reader.sources

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def units(self, epoch):
        units = [(i, start, min(start + s.chunk, s.n_slices))
                 for i, s in enumerate(self.sources) for start in range(0, s.n_slices, s.chunk)]
        random.Random(self.seed + epoch).shuffle(units)
        usable = len(units) - len(units) % self.world_size
        return units[self.rank:usable:self.world_size]

    def num_slices_per_rank(self):
        """Upper bound on sections per rank per pass (before validity filtering)."""
        return sum(stop - start for _, start, stop in self.units(0))

    def __iter__(self):
        info = get_worker_info()
        worker_id, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        rng = random.Random(hash((self.seed, self.rank, worker_id, self.epoch)))
        buffer = []
        epoch = self.epoch
        while True:
            for src_idx, start, stop in self.units(epoch)[worker_id::n_workers]:
                seis, rgt, valid, keep, local = self.reader.read(src_idx, start, stop)
                for j in np.flatnonzero(keep):
                    buffer.append(self.reader.to_item(src_idx, seis[j], rgt[j], valid[j], int(local[j])))
                    if len(buffer) >= self.buffer_size:
                        yield buffer.pop(rng.randrange(len(buffer)))
            if not self.repeat:
                break
            epoch += 1
        rng.shuffle(buffer)
        yield from buffer


class PairedSliceDataset(Dataset):
    """Deterministic map-style dataset for evaluation.

    Takes every ``stride``-th section of each source (optionally capped by
    ``max_per_source``) and drops sections that fail the validity filter. Index
    order is chunk-contiguous, and each worker keeps a small LRU cache of decoded
    chunks, so sequential reading touches every chunk once.
    """

    def __init__(self, h5_path, sources, stride=1, max_per_source=None, cache_chunks=4, **reader_kwargs):
        self.reader = _PairReader(h5_path, sources, **reader_kwargs)
        self.cache_chunks = cache_chunks
        self._cache = OrderedDict()
        self.index = []
        for src_idx, s in enumerate(self.sources):
            local = np.arange(0, s.n_slices, max(1, int(stride)))
            if max_per_source is not None and len(local) > max_per_source:
                pick = np.linspace(0, len(local) - 1, int(max_per_source)).round().astype(int)
                local = local[np.unique(pick)]
            for start in range(0, s.n_slices, s.chunk):
                rows = local[(local >= start) & (local < start + s.chunk)]
                if len(rows) == 0:
                    continue
                keep = self._chunk(src_idx, start)[3]
                self.index.extend((src_idx, int(r)) for r in rows if keep[r - start])
        self._cache.clear()
        self.reader._file = None

    @property
    def sources(self):
        return self.reader.sources

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _chunk(self, src_idx, start):
        key = (src_idx, start)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        s = self.sources[src_idx]
        data = self.reader.read(src_idx, start, min(start + s.chunk, s.n_slices))
        self._cache[key] = data
        if len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        src_idx, local = self.index[i]
        chunk = self.sources[src_idx].chunk
        start = (local // chunk) * chunk
        seis, rgt, valid, _, _ = self._chunk(src_idx, start)
        j = local - start
        return self.reader.to_item(src_idx, seis[j], rgt[j], valid[j], local)


def estimate_epoch_steps(stream, batch_size):
    """Approximate optimizer steps per pass for one rank (before validity filtering)."""
    return max(1, math.floor(stream.num_slices_per_rank() / batch_size))
