#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /SSD_DISK/users/wuruihan/wam/external/robotwin/bin/python /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/native-v9-residual-async-20261101/scripts/bounded_feedback_manual_entry.py "$@"
