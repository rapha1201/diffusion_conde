#!/bin/bash
#SBATCH --job-name=sld_latent_scale
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Latent scale factor for both trained autoencoders (a few minutes). Submit from the repo root:
#     sbatch slurm/latent_scale.slurm
#     sbatch slurm/latent_scale.slurm --n-samples 20000     (extra args go to compute_latent_scale.py)
# Writes runs/ae_rgt/vae_scaled.pt and runs/ae_seismic/vae_scaled.pt (used by configs/sld/ldm.yaml).

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

for ae in ae_rgt ae_seismic; do
    srun singularity exec --nv \
        -B/pgeoprj,/tatu \
        /tatu/container_images/go-deep.sif \
        bash slurm/in_container.sh python compute_latent_scale.py --ckpt runs/$ae/ckpt_best.pt "$@" || exit 1
done
