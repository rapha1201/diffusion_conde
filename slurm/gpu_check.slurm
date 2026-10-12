#!/bin/bash
#SBATCH --job-name=sld_gpu_check
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Same resources as the training jobs; takes ~1 minute. Submit from the repo root:
#     sbatch slurm/gpu_check.slurm
# Compares the host driver with what the container sees, with the ~/.local torch
# and with the container's own torch (PYTHONNOUSERSITE=1).

SIF=/tatu/container_images/go-deep.sif
PY='import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "from", torch.__file__);
print("torch cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count());
torch.cuda.is_available() and print("gpu", torch.cuda.get_device_name(0), "| bf16", torch.cuda.is_bf16_supported())'

echo "== host: $(hostname)  partition=$SLURM_JOB_PARTITION  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cat /proc/driver/nvidia/version 2>/dev/null | head -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1

echo; echo "== container: nvidia-smi"
singularity exec --nv -B/pgeoprj,/tatu "$SIF" nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1

echo; echo "== container: torch with ~/.local packages (what training uses)"
singularity exec --nv -B/pgeoprj,/tatu "$SIF" python -c "$PY" 2>&1 | grep -v FutureWarning

echo; echo "== container: container's own torch (PYTHONNOUSERSITE=1)"
PYTHONNOUSERSITE=1 singularity exec --nv -B/pgeoprj,/tatu "$SIF" python -c "$PY" 2>&1 | grep -v FutureWarning
