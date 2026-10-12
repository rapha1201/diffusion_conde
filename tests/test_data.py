import json

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sld.data import (DEFAULT_SPLIT, PairedChunkStream, PairedSliceDataset, PairTransform, center_crop,
                      discover_models, index_pairs, load_split, pad_to_multiple, renormalize_rgt, rgt_to_unit,
                      robust_normalize, unpad, validate_split)


# ----------------------------------------------------------------------------- splits

def test_discover_models_skips_groups_without_rgt(fake_h5):
    models = discover_models(fake_h5)
    assert "buzios" not in models
    assert sorted(models) == sorted(DEFAULT_SPLIT["train"] + DEFAULT_SPLIT["val"] + DEFAULT_SPLIT["test"])


def test_default_split_is_valid_on_file(fake_h5):
    split = load_split(None, h5_path=fake_h5)
    assert split["test"] == ["Modelo2"] and split["val"] == ["Modelo7"]


def test_split_rejects_overlap_and_unknown_models(fake_h5, tmp_path):
    with pytest.raises(ValueError, match="more than one split"):
        validate_split({"train": ["Modelo3"], "val": ["Modelo3"], "test": []})
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps({"train": ["Modelo3", "buzios"], "val": [], "test": []}))
    with pytest.raises(ValueError, match="without RGT"):
        load_split(str(bad), h5_path=fake_h5)


# ----------------------------------------------------------------------------- indexing / normalization

def test_index_pairs_excludes_reflectivity_by_default(fake_h5):
    sources = index_pairs(fake_h5, ["Modelo3", "Modelo4"])
    assert all("refletividade" not in s.seismic_key.lower() for s in sources)
    assert all(s.rgt_key.startswith("RGT_") for s in sources)
    with_refl = index_pairs(fake_h5, ["Modelo3", "Modelo4"], include_reflectivity=True)
    assert len(with_refl) == len(sources) + 2


def test_robust_normalize_is_scale_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 32, 32)).astype(np.float32)
    a, _ = robust_normalize(x)
    b, _ = robust_normalize(x * 37.0)
    np.testing.assert_allclose(a, b, atol=1e-4)
    np.testing.assert_allclose(a.mean(axis=(1, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(a.std(axis=(1, 2)), 1, atol=1e-3)


def test_rgt_to_unit_masks_zeros_and_nans():
    rgt = np.tile(np.linspace(1, 600, 16, dtype=np.float32)[:, None], (1, 1, 8))
    rgt[0, :3] = 0.0
    rgt[0, 5, 2] = np.nan
    out, valid, value_range = rgt_to_unit(rgt)
    assert not valid[0, :3].any() and not valid[0, 5, 2]
    assert out[0, :3].max() == 0.0
    assert out[valid].min() == pytest.approx(0.0) and out[valid].max() == pytest.approx(1.0)
    assert value_range[0] > 0


# ----------------------------------------------------------------------------- stream

def _collect(stream, n=None):
    items = []
    for item in stream:
        items.append(item)
        if n is not None and len(items) >= n:
            break
    return items


def test_stream_single_pass_shapes_and_filtering(fake_h5):
    sources = index_pairs(fake_h5, ["Modelo9"])
    stream = PairedChunkStream(fake_h5, sources, buffer_size=16, repeat=False)
    items = _collect(stream)
    with h5py.File(fake_h5, "r") as f:
        rgt = f[sources[0].rgt_path][()]
    empty = int((rgt.reshape(len(rgt), -1) == 0).all(axis=1).sum())
    assert empty > 0, "fixture should contain empty sections"
    # Every non-empty section appears exactly once; empty ones are dropped.
    keys = sorted((int(m[0]), int(m[1])) for *_, m in items)
    assert len(keys) == len(set(keys))
    assert len(items) <= len(sources) * (len(rgt) - empty)
    seis, r, v, meta = items[0]
    assert seis.shape == r.shape == v.shape == (1, 196, 196)
    assert seis.dtype == torch.float32 and meta.dtype == torch.int64
    assert torch.all(r[v == 0] == 0) and r.max() <= 1.0 and r.min() >= 0.0


def test_stream_ranks_partition_units(fake_h5):
    sources = index_pairs(fake_h5, DEFAULT_SPLIT["train"])
    r0 = PairedChunkStream(fake_h5, sources, rank=0, world_size=2, seed=3)
    r1 = PairedChunkStream(fake_h5, sources, rank=1, world_size=2, seed=3)
    u0, u1 = set(r0.units(0)), set(r1.units(0))
    assert u0.isdisjoint(u1)
    assert abs(len(u0) - len(u1)) == 0
    assert r0.units(0) != r0.units(1), "units must reshuffle every epoch"


def test_stream_repeat_is_infinite_and_works_with_workers(fake_h5):
    sources = index_pairs(fake_h5, ["Modelo4"])
    total = sum(s.n_slices for s in sources)
    stream = PairedChunkStream(fake_h5, sources, buffer_size=8, repeat=True)
    loader = DataLoader(stream, batch_size=16, num_workers=2)
    seen = 0
    for i, (seis, rgt, valid, meta) in enumerate(loader):
        assert seis.shape == (16, 1, 196, 196)
        seen += seis.shape[0]
        if seen > 2 * total:
            break
    assert seen > 2 * total


def test_slice_dataset_is_deterministic_and_matches_hdf5(fake_h5):
    sources = index_pairs(fake_h5, ["Modelo2"])
    ds = PairedSliceDataset(fake_h5, sources, stride=3, cache_chunks=2)
    assert len(ds) > 0
    a = ds[len(ds) // 2]
    b = ds[len(ds) // 2]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    src, local = int(a[3][0]), int(a[3][1])
    with h5py.File(fake_h5, "r") as f:
        raw = f[sources[src].seismic_path][local]
    expected, _ = robust_normalize(raw[None])
    np.testing.assert_allclose(a[0][0].numpy(), expected[0], atol=1e-5)
    ds_capped = PairedSliceDataset(fake_h5, sources, stride=1, max_per_source=5)
    assert len(ds_capped) <= 5 * len(sources)


# ----------------------------------------------------------------------------- transform

def _depth_ramp(b=2, size=196, band=0):
    rgt = torch.linspace(0, 1, size).view(1, 1, size, 1).expand(b, 1, size, size).clone()
    valid = torch.ones_like(rgt)
    if band:
        valid[:, :, :band] = 0
        rgt[:, :, :band] = 0
    seis = torch.randn(b, 1, size, size)
    return seis, rgt, valid


def test_eval_transform_center_crop_and_mask():
    seis, rgt, valid = _depth_ramp(band=50)
    t = PairTransform(train=False, output_size=128)
    s, r, v, keep = t(seis, rgt, valid)
    assert s.shape == r.shape == v.shape == (2, 1, 128, 128)
    # Crop starts at row 34, so rows 0..15 of the crop are inside the invalid band.
    assert torch.all(v[:, :, :16] == 0) and torch.all(v[:, :, 16:] == 1)
    assert torch.all(r[:, :, :16] == 0)
    rv = r[v.bool()]
    assert rv.min() == pytest.approx(0.0, abs=1e-6) and rv.max() == pytest.approx(1.0, abs=1e-6)
    assert keep.all()
    assert torch.allclose(s.mean(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-5)


def test_train_transform_keeps_rgt_monotonic_and_mask_consistent():
    torch.manual_seed(0)
    seis, rgt, valid = _depth_ramp(b=16, band=30)
    # Geometry without rotation/shear so monotonicity along rows must survive exactly.
    aug = dict(aug_p=1.0, p_rotation=0.0, p_shear=0.0, p_aniso=0.0, p_zoom=0.0, p_fault=0.0, p_statics=0.0,
               p_hflip=0.5)
    t = PairTransform(aug_kwargs=aug, train=True, output_size=128)
    s, r, v, keep = t(seis, rgt, valid)
    assert s.shape == (16, 1, 128, 128)
    d = r[:, :, 1:] - r[:, :, :-1]
    both_valid = (v[:, :, 1:] * v[:, :, :-1]).bool()
    assert (d[both_valid] >= -1e-5).float().mean() > 0.999
    assert torch.all(r[v == 0] == 0)
    assert r.max() <= 1.0 and r.min() >= 0.0


def test_train_transform_with_faults_and_rotation_runs():
    torch.manual_seed(1)
    seis, rgt, valid = _depth_ramp(b=8)
    t = PairTransform(aug_kwargs=dict(aug_p=1.0, p_fault=1.0, p_rotation=1.0), train=True, output_size=128)
    s, r, v, keep = t(seis, rgt, valid)
    assert torch.isfinite(s).all() and torch.isfinite(r).all()
    assert t.augment.p_vflip == 0.0


def test_keep_flags_degenerate_crops():
    seis, rgt, valid = _depth_ramp(b=2)
    valid[1] = 0  # no valid pixels
    _, _, _, keep = PairTransform(train=False)(seis, rgt, valid)
    assert keep.tolist() == [True, False]


def test_renormalize_rgt_handles_all_invalid():
    r, rng = renormalize_rgt(torch.rand(2, 1, 8, 8), torch.zeros(2, 1, 8, 8, dtype=torch.bool))
    assert torch.all(r == 0) and torch.all(rng == 0)


def test_pad_unpad_roundtrip():
    x = torch.randn(2, 1, 196, 196)
    p, pads = pad_to_multiple(x, 8)
    assert p.shape[-2:] == (200, 200) and pads == (4, 4)
    assert torch.equal(unpad(p, pads), x)
    assert torch.equal(center_crop(x, 196), x)
