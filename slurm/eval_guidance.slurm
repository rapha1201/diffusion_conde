#!/bin/bash
#SBATCH --job-name=sld_eval_guidance
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j.out

# Dip-guidance sweep at sampling time on the validation volume (Modelo7), no retraining.
# Submit from the repo root:   sbatch slurm/eval_guidance.slurm <ldm ckpt> [guide scales...]
#     sbatch slurm/eval_guidance.slurm runs/ldm_clip/ckpt_eval.pt
#     sbatch slurm/eval_guidance.slurm runs/ldm/ckpt_best.pt 0 0.1 0.2        (custom scales)
# One evaluate_sampling.py run per scale, same seed, so every run starts from the same noise
# (scale 0 = unguided reference). Writes <ckpt dir>/guide_sweep/scale_<s>/report.json, then prints
# the comparison table (tools/compare_evals.py) and writes <ckpt dir>/guide_sweep/compare.csv.
# Extra evaluate_sampling.py arguments: EXTRA="--guide-start 0.5 --guide-inner 2" sbatch ...

LDM=${1:?usage: sbatch slurm/eval_guidance.slurm <ldm ckpt> [scales...]}
shift
SCALES=${*:-0 0.05 0.1 0.2 0.5 1}
MODELS=${MODELS:-Modelo7}
OUT=${OUT:-$(dirname "$LDM")/guide_sweep}

# in_container.sh drops the image's CUDA forward-compat libs, which break CUDA (error 803)
# on nodes whose driver is newer than the container's CUDA 12.6 (e.g. driver 595 on gpu_batch).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

srun singularity exec --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    bash slurm/in_container.sh bash -c '
set -e
reports=()
for s in '"$SCALES"'; do
    python evaluate_sampling.py --ldm '"$LDM"' --models '"$MODELS"' --max-per-source 32 --no-swap \
        --guide-scale $s --out '"$OUT"'/scale_$s '"${EXTRA:-}"'
    reports+=("scale_$s='"$OUT"'/scale_$s/report.json")
done
python tools/compare_evals.py "${reports[@]}" --rows sample mean best_dip oracle --csv '"$OUT"'/compare.csv
'
