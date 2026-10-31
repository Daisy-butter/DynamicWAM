#!/usr/bin/env bash
set -euo pipefail
ROOT=/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031
PY=/SSD_DISK/users/wuruihan/wam/external/robotwin/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
if [[ "${1:-}" == "--check" && $# -eq 1 ]]; then
  exec "$PY" -B scripts/continuous_v8_final_eval.py --spec configs/continuous_v8_final_eval.json --check
fi
if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/launch_continuous_v8_eval.sh [--check]' >&2; exit 2; fi
exec "$PY" -B scripts/continuous_v8_final_eval.py --spec configs/continuous_v8_final_eval.json --detach
