#!/bin/bash
#SBATCH --job-name=sld_ae_rgt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# RGT autoencoder on one GPU. Submit from the repo root:
#     sbatch slurm/ae_rgt.slurm
#     sbatch slurm/ae_rgt.slurm --set optim.lr=2e-4     (extra args go to train_ae.py)
# Resubmitting resumes from runs/ae_rgt/ckpt_latest.pt.

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity run --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    python train_ae.py --config configs/sld/ae_rgt.yaml "$@"
