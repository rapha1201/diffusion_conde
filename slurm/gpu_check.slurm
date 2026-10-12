#!/bin/bash
#SBATCH --job-name=sld_gpu_check
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Same resources as the training jobs; takes ~1 minute. Submit from the repo root:
#     sbatch slurm/gpu_check.slurm
# Compares the host driver with what the container sees, and tests CUDA with and
# without the container's CUDA forward-compat libraries (see slurm/in_container.sh).

SIF=/tatu/container_images/go-deep.sif
RUN="singularity exec --nv -B/pgeoprj,/tatu $SIF"
PY='import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "from", torch.__file__);
print("torch cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count());
torch.cuda.is_available() and print("gpu", torch.cuda.get_device_name(0), "| bf16", torch.cuda.is_bf16_supported(), "| matmul ok", float((torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda"))[0, 0]) == 64.0)'
LIBCUDA='import ctypes, re; ctypes.CDLL("libcuda.so.1"); print("libcuda loaded from:", [l.split()[-1] for l in open("/proc/self/maps") if re.search(r"libcuda\.so", l)][:1])'

echo "== host: $(hostname)  partition=$SLURM_JOB_PARTITION  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cat /proc/driver/nvidia/version 2>/dev/null | head -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1

echo; echo "== container: nvidia-smi"
$RUN nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1

echo; echo "== container: LD_LIBRARY_PATH and compat libs"
$RUN bash -c 'echo "$LD_LIBRARY_PATH" | tr ":" "\n"; ls -l /usr/local/cuda/compat/lib* 2>&1 | head -8'

echo; echo "== container, as-is: libcuda + torch (~/.local)"
$RUN python -c "$LIBCUDA" 2>&1
$RUN python -c "$PY" 2>&1 | grep -v Warning | grep -v "return torch"

echo; echo "== container, compat libs removed (slurm/in_container.sh): libcuda + torch (~/.local)"
$RUN bash slurm/in_container.sh python -c "$LIBCUDA" 2>&1
$RUN bash slurm/in_container.sh python -c "$PY" 2>&1 | grep -v FutureWarning

echo; echo "== container, compat libs removed: container's own torch (PYTHONNOUSERSITE=1)"
PYTHONNOUSERSITE=1 $RUN bash slurm/in_container.sh python -c "$PY" 2>&1 | grep -v FutureWarning
