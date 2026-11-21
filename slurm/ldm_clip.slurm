#!/bin/bash
#SBATCH --job-name=sld_ldm_clip
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Phase B: LDM conditioned on the frozen CLIP seismic encoder (configs/sld/ldm_clip.yaml).
# Needs Phase A first (runs/clip/ckpt_best.pt, slurm/clip.slurm). Submit from the repo root:
#     sbatch slurm/ldm_clip.slurm
#     sbatch slurm/ldm_clip.slurm --set condition.encoder_ckpt=null out_dir=runs/ldm_aug experiment=ldm_aug   (ablation)
# Resubmitting the same command resumes from <out_dir>/ckpt_latest.pt.

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python train_ldm.py --config configs/sld/ldm_clip.yaml "$@"
