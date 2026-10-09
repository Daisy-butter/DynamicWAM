#!/usr/bin/env bash
set -euo pipefail
umask 077
repo=/home/wuruihan/DynamicWAM
run_root=${1:-/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/predictive_feedback_v1_20261009}
stop_after=${2:-all}
export TMPDIR=/SSD_DISK/users/wuruihan/DynamicWAM/outputs/runtime_tmp
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/SSD_DISK/users/wuruihan/DynamicWAM/outputs/runtime_cache
export __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
mkdir -p "$TMPDIR" "$__GL_SHADER_DISK_CACHE_PATH"
if [[ -e "$run_root" ]]; then
  printf '%s\n' "Refusing to overwrite or restart existing run: $run_root" >&2
  exit 2
fi
cd "$repo"
runtime="$repo/external/robotwin/bin/python3.10"
exec "$runtime" scripts/run_predictive_feedback_pipeline.py \
  --project-root "$repo" --run-root "$run_root" --detach --stop-after "$stop_after" \
  --python "$runtime" --sim-python "$runtime" --gpus auto \
  --episodes "$repo/data/packed/domino_absolute_motion_v2/episodes.jsonl" \
  --raw-root /SSD_DISK/users/wuruihan/DynamicWAM/data/raw \
  --official-seed-manifest /SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/native_graph_eval_20261005/episode_seed_manifest.json \
  --base-checkpoint "$repo/external/checkpoints/DynamicWAM_full.pt" \
  --init-plan-feedback /SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/plan_feedback_v1_20261005/stage3/exports/feedback_step_2000.pt \
  --packet /SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/feedback_preflight_20261003/synthetic_packet.npz \
  --accepted-run /SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/world_context_rt_v1_20261008
