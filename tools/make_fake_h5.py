"""Write a small synthetic HDF5 that mimics fault_aware.h5.

Mirrors what tools/inspect_h5.py and tools/check_pairs.py reported for the real file:

    /<Model>/RGT_<Model>.segy                 (N, S, S) raw RGT (~0-650), increasing with depth = axis 1 (rows)
    /<Model>/Refletividade_<Model>.segy       (N, S, S) reflectivity in [-1, 1]
    /<Model>/Sism<f>Hz_..._<Model>_....segy   (N, S, S) seismic realizations sharing the RGT, varied amplitude scale
    /buzios/train, /buzios/test               seismic only, no RGT

RGT == 0 is an undefined shallow band of whole top rows in some sections; one
model ("Modelo9") also has entirely empty sections. Gzip, chunks of 32 sections.

    python tools/make_fake_h5.py --out fake.h5
"""
import argparse
import os

import h5py
import numpy as np


MODELS = ("Modelo2", "Modelo3", "Modelo4", "Modelo5", "Modelo7", "Modelo8", "Modelo9")

# (name template, Ricker peak frequency in cycles/sample, relative noise, amplitude scale)
REALIZATIONS = [
    ("Sism20Hz_100perVel_Ruido0_0____{m}_Fake_HighAng.segy", 0.045, 0.0, 0.30),
    ("Sism30Hz_40perVel_Ruido0p5_0p5_Aten7_{m}_Fake_HighAng.segy", 0.065, 0.4, 1.00),
    ("Sism35Hz_70perVel_Ruido0p4_0p5_Smooth5_{m}_Fake_HighAng.segy", 0.085, 0.3, 0.05),
]


def ricker(f, length=61):
    t = np.arange(length) - length // 2
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_rgt(n_slices, size, rng):
    """Unit-range RGT (N,S,S) with rows = depth: dip + folding + one planar fault per section."""
    z = np.linspace(0.0, 1.0, size)[:, None]   # depth (rows)
    x = np.linspace(0.0, 1.0, size)[None, :]   # lateral (cols)
    out = np.empty((n_slices, size, size), dtype=np.float64)
    dip0, amp, k, phi0 = rng.uniform(-0.15, 0.15), rng.uniform(0.02, 0.07), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi)
    x0, slope, throw = rng.uniform(0.3, 0.7), rng.uniform(-0.5, 0.5), rng.uniform(0.03, 0.10) * rng.choice([-1, 1])
    for i in range(n_slices):
        s = i / max(1, n_slices - 1)
        # |du/dz| stays well below 1, so the RGT is monotonic in depth away from the fault.
        u = (dip0 + 0.05 * np.sin(2 * np.pi * s)) * x + amp * np.sin(2 * np.pi * k * x + phi0 + 1.5 * s) * (0.5 + 0.5 * z)
        u = u + throw * (x >= x0 + 0.1 * s + slope * (z - 0.5))
        out[i] = z - u
    return (out - out.min()) / (out.max() - out.min())


def make_reflectivity(rgt01, rng, n_layers=80):
    levels = np.sort(rng.uniform(-0.05, 1.05, n_layers))
    log_imp = np.cumsum(rng.normal(0.0, 0.2, n_layers + 1))
    li = log_imp[np.searchsorted(levels, rgt01)]
    refl = np.diff(li, axis=1, prepend=li[:, :1, :])
    return (refl / max(np.abs(refl).max(), 1e-12)).astype(np.float32)


def make_seismic(refl, f, noise, scale, rng):
    w = ricker(f)
    seis = np.apply_along_axis(lambda tr: np.convolve(tr, w, mode="same"), 1, refl.astype(np.float64))
    if noise > 0:
        seis = seis + rng.normal(0.0, noise * seis.std(), seis.shape)
    return (scale * seis / max(seis.std(), 1e-12)).astype(np.float32)


def make_fake_h5(path, models=MODELS, n_slices=64, size=196, chunk=32, seed=0, empty_model="Modelo9",
                 band_prob=0.3, empty_prob=0.15, with_buzios=True):
    rng = np.random.default_rng(seed)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(dtype="float32", compression="gzip", chunks=(min(chunk, n_slices), size, size))

    with h5py.File(path, "w") as f:
        for mi, model in enumerate(models):
            g = f.create_group(model)
            rgt01 = make_rgt(n_slices, size, rng)
            refl = make_reflectivity(rgt01, rng)

            rgt = (1.0 + 600.0 * rgt01).astype(np.float32)
            # Undefined shallow band: whole top rows set to 0.
            for i in np.flatnonzero(rng.random(n_slices) < band_prob):
                rgt[i, : rng.integers(5, 70)] = 0.0
            if model == empty_model:
                empty = rng.random(n_slices) < empty_prob
                rgt[empty] = 0.0
                refl[empty] = 0.0
            g.create_dataset(f"RGT_{model}.segy", data=rgt, **kw)
            g.create_dataset(f"Refletividade_{model}.segy", data=refl, **kw)
            # 1-3 realizations per model, like the real file.
            for name, freq, noise, scale in REALIZATIONS[: 1 + mi % 3]:
                g.create_dataset(name.format(m=model), data=make_seismic(refl, freq, noise, scale, rng), **kw)

        if with_buzios:
            g = f.create_group("buzios")
            for split, n in (("train", n_slices), ("test", max(1, n_slices // 4))):
                data = np.clip(rng.normal(0.0, 0.9, (n, size, size)), -3, 3).astype(np.float32)
                g.create_dataset(split, data=data, dtype="float32", compression="gzip",
                                 chunks=(min(chunk, n), size, size))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="fake.h5")
    parser.add_argument("--n-slices", type=int, default=64)
    parser.add_argument("--size", type=int, default=196)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_fake_h5(args.out, n_slices=args.n_slices, size=args.size, chunk=args.chunk, seed=args.seed)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
