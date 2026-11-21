#!/bin/bash
#SBATCH --job-name=sld_clip
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Phase A: CLIP pretraining of the seismic condition encoder (configs/sld/clip.yaml). Submit from the repo root:
#     sbatch slurm/clip.slurm
#     sbatch slurm/clip.slurm --set model.mode=global out_dir=runs/clip_global
# Then Phase B: sbatch slurm/ldm_clip.slurm. Resubmitting resumes from <out_dir>/ckpt_latest.pt.

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python train_clip.py --config configs/sld/clip.yaml "$@"
