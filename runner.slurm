#!/bin/bash

 
#SBATCH --job-name=conde_experiments
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=experiments_log.out

# ---------- APPENDING CUSTOM MODULE PATH TO PYTHONPATH ---------- #

check_pythonpath() {
    local target_path="$1"
    local paths=("$PYTHONPATH")

    # Iterate over each path in PYTHONPATH
    for path in "${paths[@]}"; do
        # Check if the target path is a substring of the current path
        if [[ "$target_path" == "$path"* ]]; then
            return 0  # true
        fi
    done

    # Return false if the target path was not found in PYTHONPATH
    return 1
}

custom_module_path="$(pwd)"

if [ -z "$PYTHONPATH" ]; then
    echo "PYTHONPATH environment variable not found... Creating one"
    export PYTHONPATH="$custom_module_path"
else
    echo "PYTHONPATH environment variable found"

    if ! check_pythonpath "$custom_module_path"; then
        echo "Custom module path not found in PYTHONPATH environment variable"
        export PYTHONPATH="$PYTHONPATH:$custom_module_path"
        echo "Successfully added custom module path to PYTHONPATH"
    else
        echo "Custom module path already in PYTHONPATH environment variable"
    fi
fi

echo "Setting PYTHONPATH = $PYTHONPATH"

# ---------- DEFAULT PROCEDURE TO DEPLOY PYTHON SCRIPT WITH SLURM ---------- #

nodes=("$(scontrol show hostnames "$SLURM_JOB_NODELIST")")

for i in "${nodes[@]}"
do
    echo "$i"
done

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)
echo Rank 0: Node "${nodes[0]}" , IP: "$head_node_ip"

GPUS=4
NODES=$SLURM_JOB_NUM_NODES

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=^mlx5_2:1

srun singularity run --nv \
    -B/pgeoprj,/tatu \
    /tatu/container_images/go-deep.sif \
    torchrun \
    --nnodes "${NODES}" \
    --nproc_per_node "${GPUS}" \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint "${head_node_ip}:27027" \
    ddp_train_df_conde.py
