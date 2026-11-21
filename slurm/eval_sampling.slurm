#!/bin/bash
#SBATCH --job-name=sld_eval_sampling
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Per-volume sampling test of the LDM (train volumes vs val), with the swap test and B1 alongside.
# Submit from the repo root:
#     sbatch slurm/eval_sampling.slurm
#     sbatch slurm/eval_sampling.slurm --models Modelo3 Modelo7 --max-per-source 64   (extra args go to the script)
# Writes runs/ldm/sampling_eval/report.json and per-volume figures.

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh python evaluate_sampling.py \
        --ldm runs/ldm/ckpt_best.pt --baseline runs/baseline_pixel/ckpt_best.pt "$@"
