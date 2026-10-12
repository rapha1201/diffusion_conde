"""Volume-level (model-level) train/val/test splits.

Sections of the same model are near-duplicates, so splitting by section leaks
information between sets. Every split here is a list of HDF5 model groups.
"""
import json
import os
import re

import h5py


# Agreed split for fault_aware.h5: one compressional model held out for test
# (M4 keeps compressional geology in train), one extensional model for val.
DEFAULT_SPLIT = {
    "train": ["Modelo3", "Modelo4", "Modelo5", "Modelo8", "Modelo9"],
    "val": ["Modelo7"],
    "test": ["Modelo2"],
}


def discover_models(h5_path, rgt_token="rgt"):
    """Top-level groups that contain at least one RGT dataset."""
    pattern = re.compile(rgt_token, re.IGNORECASE)
    with h5py.File(h5_path, "r") as f:
        return [name for name in f.keys()
                if isinstance(f[name], h5py.Group) and any(pattern.search(k) for k in f[name].keys())]


def validate_split(split, available_models=None):
    names = [m for part in ("train", "val", "test") for m in split.get(part, [])]
    duplicates = sorted({m for m in names if names.count(m) > 1})
    if duplicates:
        raise ValueError(f"Models appear in more than one split: {duplicates}")
    if not split.get("train"):
        raise ValueError("The train split is empty.")
    if available_models is not None:
        missing = sorted(set(names) - set(available_models))
        if missing:
            raise ValueError(f"Split references models without RGT in the HDF5 file: {missing}")
    return split


def load_split(split=None, h5_path=None):
    """Accepts None (default split), a dict, or a path to a JSON manifest."""
    if split is None:
        split = DEFAULT_SPLIT
    elif isinstance(split, str):
        with open(split) as fp:
            split = json.load(fp)
    split = {k: list(split.get(k, [])) for k in ("train", "val", "test")}
    available = discover_models(h5_path) if h5_path is not None else None
    return validate_split(split, available)


def write_split(split, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(split, fp, indent=2)
