#!/bin/bash
#SBATCH --job-name=sld_baseline_pixel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# B1 baseline (pixel-space UNet, seismic -> RGT) on one GPU. Submit from the repo root:
#     sbatch slurm/baseline_pixel.slurm
# Resubmitting resumes from runs/baseline_pixel/ckpt_latest.pt.

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python train_baseline.py --config configs/sld/baseline_pixel.yaml "$@"
