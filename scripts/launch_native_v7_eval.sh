#!/usr/bin/env bash
set -euo pipefail
ROOT=/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v7-native-eval-20261031
PY=/SSD_DISK/users/wuruihan/wam/external/robotwin/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
if [[ "${1:-}" == "--check" ]]; then
  exec "$PY" -B scripts/native_v7_posttrain_eval.py --spec configs/native_v7_final_eval.json --check
fi
if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/launch_native_v7_eval.sh [--check]' >&2; exit 2; fi
exec "$PY" -B scripts/native_v7_posttrain_eval.py --spec configs/native_v7_final_eval.json --detach
