import json
import math
import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def mix_fourier_amplitudes(synthetic_fft, real_images, alpha, eps=1e-12):
    """Blend FFT amplitudes while retaining the synthetic complex phase.

    Parameters
    ----------
    synthetic_fft : torch.Tensor
        Batched one-sided spectra returned by ``torch.fft.rfft2``.
    real_images : torch.Tensor
        Batched real target images with the same image shape as the synthetic
        images used to create ``synthetic_fft``.
    alpha : torch.Tensor
        Per-image real-amplitude weights with shape ``[B]``.
    """
    if synthetic_fft.ndim != 4 or real_images.ndim != 4:
        raise ValueError("synthetic_fft and real_images must both be 4-D batches.")
    if synthetic_fft.shape[0] != real_images.shape[0]:
        raise ValueError("Synthetic and real batches must have the same batch size.")

    alpha = torch.as_tensor(alpha, device=synthetic_fft.device, dtype=synthetic_fft.real.dtype)
    if alpha.ndim != 1 or alpha.shape[0] != synthetic_fft.shape[0]:
        raise ValueError("alpha must have shape [B].")
    if not torch.all((alpha >= 0.0) & (alpha <= 1.0)):
        raise ValueError("alpha values must lie in [0, 1].")

    real_fft = torch.fft.rfft2(real_images)
    if real_fft.shape != synthetic_fft.shape:
        raise ValueError(
            f"Synthetic and real FFT shapes differ: {synthetic_fft.shape} vs {real_fft.shape}."
        )

    synthetic_amplitude = synthetic_fft.abs()
    real_amplitude = real_fft.abs()
    mixed_amplitude = torch.lerp(
        synthetic_amplitude,
        real_amplitude,
        alpha.view(-1, 1, 1, 1),
    )

    # Division avoids computing angle/exp and gives an exact unit phasor away
    # from zero-amplitude bins. The phase of a zero bin is conventionally zero.
    unit_phase = synthetic_fft / synthetic_amplitude.clamp_min(eps)
    unit_phase = torch.where(
        synthetic_amplitude > eps,
        unit_phase,
        torch.ones_like(unit_phase),
    )
    return mixed_amplitude * unit_phase


class H5FourierTargetPool:
    """Leakage-safe pool of real seismic images used only for FFT appearance.

    Direct child datasets of ``target_model`` are treated as indivisible source
    units. A deterministic fraction of those units is loaded into memory; the
    remaining unit paths are exposed for held-out evaluation.
    """

    def __init__(
        self,
        h5_path,
        target_model="buzios",
        fraction=0.05,
        seed=0,
        local_norm=True,
        min_held_out_units=2,
        pin_memory=True,
    ):
        self.h5_path = os.fspath(h5_path)
        self.requested_target_model = str(target_model)
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.local_norm = bool(local_norm)
        self.min_held_out_units = int(min_held_out_units)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())

        if not 0.0 < self.fraction < 1.0:
            raise ValueError("fraction must be strictly between 0 and 1.")
        if self.min_held_out_units < 1:
            raise ValueError("min_held_out_units must be at least 1.")

        with h5py.File(self.h5_path, "r") as h5_file:
            self.target_model = self._resolve_target_model(h5_file)
            group = h5_file[self.target_model]
            unit_paths = sorted(
                f"{self.target_model}/{name}"
                for name in group.keys()
                if isinstance(group[name], h5py.Dataset) and "rgt" not in name.lower()
            )

        if not unit_paths:
            raise ValueError(
                f"No non-RGT child datasets were found under '{self.target_model}'."
            )

        selected_count = max(1, int(math.ceil(len(unit_paths) * self.fraction)))
        if len(unit_paths) - selected_count < self.min_held_out_units:
            raise ValueError(
                "A leakage-safe Fourier target split is impossible: "
                f"{len(unit_paths)} target unit(s), {selected_count} selected, and "
                f"at least {self.min_held_out_units} held-out unit(s) required."
            )

        split_rng = random.Random(self.seed)
        self.selected_dataset_names = sorted(split_rng.sample(unit_paths, selected_count))
        selected_set = set(self.selected_dataset_names)
        self.held_out_dataset_names = [p for p in unit_paths if p not in selected_set]

        self._unit_tensors = []
        self._unit_sample_counts = []
        flat_unit_ids = []
        flat_sample_ids = []

        with h5py.File(self.h5_path, "r") as h5_file:
            for unit_id, dataset_name in enumerate(self.selected_dataset_names):
                images = self._load_unit(h5_file[dataset_name], dataset_name)
                tensor = torch.from_numpy(images)
                if self.pin_memory:
                    tensor = tensor.pin_memory()
                self._unit_tensors.append(tensor)
                self._unit_sample_counts.append(len(tensor))
                flat_unit_ids.extend([unit_id] * len(tensor))
                flat_sample_ids.extend(range(len(tensor)))

        if not flat_unit_ids:
            raise ValueError("The selected Fourier target units contain no seismic images.")

        self._flat_unit_ids = torch.tensor(flat_unit_ids, dtype=torch.long)
        self._flat_sample_ids = torch.tensor(flat_sample_ids, dtype=torch.long)

    def _resolve_target_model(self, h5_file):
        matches = [
            name
            for name in h5_file.keys()
            if name.lower() == self.requested_target_model.lower()
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one '{self.requested_target_model}' group; found {matches}."
            )
        if not isinstance(h5_file[matches[0]], h5py.Group):
            raise TypeError(f"'{matches[0]}' must be an HDF5 group.")
        return matches[0]

    def _load_unit(self, dataset, dataset_name):
        images = np.asarray(dataset[:], dtype=np.float32)
        images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

        if images.ndim == 2:
            images = images[None, None, :, :]
        elif images.ndim == 3:
            images = images[:, None, :, :]
        elif images.ndim != 4:
            raise ValueError(
                f"Target dataset '{dataset_name}' must have shape [N,H,W] or [N,C,H,W], "
                f"but received {images.shape}."
            )

        if len(images) == 0:
            raise ValueError(f"Target dataset '{dataset_name}' is empty.")

        if self.local_norm:
            mean = images.mean(axis=(1, 2, 3), keepdims=True)
            std = images.std(axis=(1, 2, 3), keepdims=True)
            images = (images - mean) / (std + 1e-6)
        return np.ascontiguousarray(images, dtype=np.float32)

    def __len__(self):
        return int(self._flat_unit_ids.numel())

    def fetch(self, flat_indices, image_shape, device, dtype):
        """Fetch and resize a target batch in the order given by flat_indices."""
        channels, height, width = map(int, image_shape)
        device = torch.device(device)
        flat_indices = torch.as_tensor(flat_indices, dtype=torch.long).detach().cpu()

        if flat_indices.ndim != 1:
            raise ValueError("flat_indices must be one-dimensional.")
        if flat_indices.numel() == 0:
            return torch.empty((0, channels, height, width), device=device, dtype=dtype)
        if int(flat_indices.min()) < 0 or int(flat_indices.max()) >= len(self):
            raise IndexError("Fourier target index is outside the pool.")

        unit_ids = self._flat_unit_ids[flat_indices]
        sample_ids = self._flat_sample_ids[flat_indices]
        output = torch.empty(
            (flat_indices.numel(), channels, height, width),
            device=device,
            dtype=dtype,
        )

        for unit_id in torch.unique(unit_ids).tolist():
            positions = (unit_ids == unit_id).nonzero(as_tuple=False).squeeze(1)
            unit_sample_ids = sample_ids[positions]
            batch = self._unit_tensors[unit_id][unit_sample_ids]
            batch = batch.to(
                device=device,
                dtype=dtype,
                non_blocking=self.pin_memory and device.type == "cuda",
            )

            if batch.shape[1] == 1 and channels != 1:
                batch = batch.expand(-1, channels, -1, -1)
            elif batch.shape[1] != channels:
                raise ValueError(
                    f"Target channel count {batch.shape[1]} does not match source channel count {channels}."
                )

            if batch.shape[-2:] != (height, width):
                batch = F.interpolate(
                    batch,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            if self.local_norm:
                # Interpolation changes the target variance, so normalize
                # immediately before its FFT. This is harmless for already
                # matching image sizes.
                mean = batch.mean(dim=(1, 2, 3), keepdim=True)
                std = batch.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
                batch = (batch - mean) / (std + 1e-6)
            output[positions.to(device)] = batch

        return output

    def manifest(self):
        return {
            "h5_path": self.h5_path,
            "target_model": self.target_model,
            "fraction": self.fraction,
            "seed": self.seed,
            "selected_dataset_names": list(self.selected_dataset_names),
            "held_out_dataset_names": list(self.held_out_dataset_names),
            "selected_unit_count": len(self.selected_dataset_names),
            "held_out_unit_count": len(self.held_out_dataset_names),
            "selected_sample_count": len(self),
            "selected_unit_sample_counts": dict(
                zip(self.selected_dataset_names, self._unit_sample_counts)
            ),
        }

    def write_manifest(self, path, **extra):
        payload = self.manifest()
        payload.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2, sort_keys=True)
        return payload
