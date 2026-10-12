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

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python train_ae.py --config configs/sld/ae_rgt.yaml "$@"
