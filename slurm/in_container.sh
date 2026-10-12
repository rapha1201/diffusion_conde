#!/bin/bash
# Runs a command inside the container without the CUDA forward-compatibility libraries.
#
# The NGC image puts /usr/local/cuda/compat/lib on LD_LIBRARY_PATH. Those libs are
# for hosts whose driver is OLDER than the container's CUDA; on a host with a newer
# driver they shadow the host libcuda mounted by `singularity --nv` and CUDA fails
# with "error 803: unsupported display driver / cuda driver combination".
#
# Usage (inside the container): bash slurm/in_container.sh python train_ae.py ...
export LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/compat' | paste -sd: -)"
exec "$@"
