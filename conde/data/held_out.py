import random

from .seismic_dataset import ResidualCondeDataset


class HeldOutResidualCondeDataset(ResidualCondeDataset):
    """``ResidualCondeDataset`` that never uses the Fourier-pool slice positions.

    Anchors, positives and negatives are all drawn from slice positions that
    are NOT in ``excluded_indices`` (``H5FourierTargetPool.excluded_indices``).
    The exclusion applies to every volume, because slice positions are aligned
    across the volumes of a model.

    Kept in its own module: ``fourier_mix`` is imported by ``augmentation``,
    which is imported by ``seismic_dataset``, so ``fourier_mix`` must not
    import ``seismic_dataset``.
    """

    def __init__(self, *args, excluded_indices=(), **kwargs):
        self.excluded_indices = frozenset(int(i) for i in excluded_indices)
        super().__init__(*args, **kwargs)

        self.allowed_indices = {}
        for path in self.dataset_names:
            allowed = [
                i for i in range(len(self.data[path])) if i not in self.excluded_indices
            ]
            if len(allowed) < 2:
                raise ValueError(
                    f"Dataset '{path}' has {len(allowed)} slice(s) left after the "
                    "Fourier-pool exclusion; at least 2 are required."
                )
            self.allowed_indices[path] = allowed

        n_before = len(self.samples)
        self.samples = [
            (path, local_idx)
            for path, local_idx in self.samples
            if local_idx not in self.excluded_indices
        ]
        print(
            f"Excluded {n_before - len(self.samples)} Fourier-pool slice(s); "
            f"{len(self.samples)} evaluation anchors remain."
        )

    def get_random_from_res(self, seismic_path, exclude=None):
        allowed = self.allowed_indices[seismic_path]
        while True:
            local_idx = allowed[random.randrange(len(allowed))]
            if exclude is None or local_idx != exclude:
                return self.data[seismic_path][local_idx], local_idx