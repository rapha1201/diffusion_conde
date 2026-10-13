#!/bin/bash
#SBATCH --job-name=sld_ldm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Conditional latent diffusion on one GPU. Needs slurm/latent_scale.slurm first. Submit from the repo root:
#     sbatch slurm/ldm.slurm
#     sbatch slurm/ldm.slurm --set diffusion.prediction_type=v_prediction out_dir=runs/ldm_v
# Resubmitting resumes from runs/ldm/ckpt_latest.pt.

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python train_ldm.py --config configs/sld/ldm.yaml "$@"
