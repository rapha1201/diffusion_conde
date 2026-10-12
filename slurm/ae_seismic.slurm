#!/bin/bash
#SBATCH --job-name=sld_ae_seismic
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Seismic autoencoder on one GPU. Submit from the repo root:
#     sbatch slurm/ae_seismic.slurm
# Resubmitting resumes from runs/ae_seismic/ckpt_latest.pt.

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity run --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    python train_ae.py --config configs/sld/ae_seismic.yaml "$@"
