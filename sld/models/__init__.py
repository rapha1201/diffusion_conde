import torch

from .regressor import UNetRegressor
from .vae import LatentVAE

MODEL_KINDS = {"vae": LatentVAE, "regressor": UNetRegressor}


def build_model(kind, model_cfg):
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {sorted(MODEL_KINDS)}")
    return MODEL_KINDS[kind](**model_cfg)


def load_model(path, map_location="cpu", use_ema=False):
    """Rebuild a model from a checkpoint written by sld.utils.trainer. Returns (model, checkpoint)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["kind"], ckpt["model_cfg"])
    model.load_state_dict(ckpt["ema" if use_ema and ckpt.get("ema") else "model"])
    return model, ckpt
