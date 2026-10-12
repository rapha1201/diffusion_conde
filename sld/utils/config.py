"""YAML configs with dotted command-line overrides (``--set optim.lr=2e-4``)."""
import copy
import os

import yaml


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg, key, default=None):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def apply_overrides(cfg, overrides=()):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def load_config(path, overrides=()):
    with open(path) as fp:
        cfg = yaml.safe_load(fp) or {}
    return apply_overrides(cfg, overrides)


def save_config(cfg, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        yaml.safe_dump(copy.deepcopy(cfg), fp, sort_keys=False)
