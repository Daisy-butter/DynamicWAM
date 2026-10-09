#!/usr/bin/env bash
# Automated V1 joint retrain on 7 GPUs, then latency-focused DOMINO probes.
# Warmup r2 is assumed already exported. Does not overwrite existing probe dirs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
ACC="$ROOT/.venv/bin/accelerate"
SSD="/SSD_DISK/users/wuruihan/DynamicWAM"
TRAIN_CFG="$ROOT/configs/world_action_multirate_v1_joint_r2_7gpu.yaml"
CKPT="$SSD/outputs/training/world_action_multirate_joint_r2_7gpu/exports/multirate_step_1000.pt"
WARMUP="$SSD/outputs/training/world_action_multirate_fast_warmup_r2/exports/multirate_step_1000.pt"
PIPE_ROOT="$SSD/outputs/evaluation/multirate_joint_r2_7gpu_pipeline"
TRAIN_LOG="$SSD/outputs/training/world_action_multirate_joint_r2_7gpu/pipeline_train.log"
PIPE_LOG="$PIPE_ROOT/pipeline.log"
PORT="${MASTER_PORT:-36151}"

mkdir -p "$SSD/outputs/training/world_action_multirate_joint_r2_7gpu" "$PIPE_ROOT"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "=== pipeline start $(date -Is) ==="
echo "root=$ROOT ckpt=$CKPT"

if [[ ! -f "$WARMUP" ]]; then
  echo "missing warmup checkpoint: $WARMUP" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "=== joint train 7gpu $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  PYTHONUNBUFFERED=1 \
  "$ACC" launch \
    --multi_gpu \
    --num_processes 7 \
    --mixed_precision bf16 \
    --main_process_port "$PORT" \
    "$ROOT/scripts/train_multirate.py" \
    --config "$TRAIN_CFG" \
    2>&1 | tee "$TRAIN_LOG"
  if ! grep -q "Exported multirate checkpoint" "$TRAIN_LOG"; then
    echo "joint train did not export a checkpoint" >&2
    exit 2
  fi
else
  echo "joint checkpoint already present, skip train: $CKPT"
fi

echo "=== wait for stable export $(date -Is) ==="
"$PY" - <<'PY'
import time
from pathlib import Path
path = Path("/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/world_action_multirate_joint_r2_7gpu/exports/multirate_step_1000.pt")
deadline = time.monotonic() + 600
last = -1
stable = None
while time.monotonic() < deadline:
    if path.is_file():
        size = path.stat().st_size
        if size > 0 and size == last:
            if stable is None:
                stable = time.monotonic()
            elif time.monotonic() - stable >= 10:
                print(f"ready bytes={size}")
                raise SystemExit(0)
        else:
            stable = None
            last = size
    time.sleep(2)
raise SystemExit("checkpoint not stable")
PY

# Probe A: matched H=4, same backbone. Latency + success on 6 GPUs.
PROBE_A="$PIPE_ROOT/probe_matched_h4"
if [[ ! -d "$PROBE_A" ]] || [[ -z "$(ls -A "$PROBE_A" 2>/dev/null || true)" ]]; then
  echo "=== probe matched H=4 $(date -Is) ==="
  mkdir -p "$PROBE_A"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/eval_multirate_probe.py" \
    --run-root "$PROBE_A" \
    --checkpoint "$CKPT" \
    --episodes 2 \
    --sync-horizon 4 \
    --scheduling independent_worker \
    --sync-gpus 0,1,2 \
    --multirate-gpus 3,4,5 \
    2>&1 | tee "$PIPE_ROOT/probe_matched_h4.log"
else
  echo "probe A already populated, skip: $PROBE_A"
fi

# Probe B: official long-chunk sync H=16 vs multirate H=4. Amortized latency reference.
PROBE_B="$PIPE_ROOT/probe_official_h16"
if [[ ! -d "$PROBE_B" ]] || [[ -z "$(ls -A "$PROBE_B" 2>/dev/null || true)" ]]; then
  echo "=== probe official sync H=16 $(date -Is) ==="
  mkdir -p "$PROBE_B"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/eval_multirate_probe.py" \
    --run-root "$PROBE_B" \
    --checkpoint "$CKPT" \
    --episodes 2 \
    --sync-horizon 16 \
    --official-sync \
    --scheduling independent_worker \
    --sync-gpus 0,1,2 \
    --multirate-gpus 3,4,5 \
    2>&1 | tee "$PIPE_ROOT/probe_official_h16.log"
else
  echo "probe B already populated, skip: $PROBE_B"
fi

# Probe C: paced H=4 on one GPU (serial). Wall-clock overlap closer to 100ms/action.
PROBE_C="$PIPE_ROOT/probe_paced_h4_serial"
if [[ ! -d "$PROBE_C" ]] || [[ -z "$(ls -A "$PROBE_C" 2>/dev/null || true)" ]]; then
  echo "=== probe paced serial H=4 $(date -Is) ==="
  mkdir -p "$PROBE_C"
  CUDA_VISIBLE_DEVICES=6 \
  PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/eval_multirate_probe.py" \
    --run-root "$PROBE_C" \
    --checkpoint "$CKPT" \
    --episodes 2 \
    --sync-horizon 4 \
    --scheduling independent_worker \
    --pace-action-interval \
    --serial \
    --sync-gpus 6,6,6 \
    --multirate-gpus 6,6,6 \
    2>&1 | tee "$PIPE_ROOT/probe_paced_h4_serial.log"
else
  echo "probe C already populated, skip: $PROBE_C"
fi

echo "=== summarize $(date -Is) ==="
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/multirate_joint_r2_7gpu_pipeline")
summary = {}
for name in ("probe_matched_h4", "probe_official_h16", "probe_paced_h4_serial"):
    report = root / name / "probe_report.json"
    if not report.is_file():
        summary[name] = {"missing": True}
        continue
    rows = json.loads(report.read_text())
    slim = []
    for row in rows:
        slim.append({
            "arm": row.get("arm"),
            "task": row.get("task"),
            "success_count": row.get("success_count"),
            "reported_episodes": row.get("reported_episodes"),
            "ms_per_executed_action": row.get("ms_per_executed_action"),
            "query_ms_mean": row.get("query_ms_mean"),
            "execute_horizon": row.get("execute_horizon"),
            "scheduling": row.get("scheduling"),
            "bridge_modes": row.get("bridge_modes"),
            "return_code": row.get("return_code"),
        })
    summary[name] = slim
out = root / "pipeline_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"wrote {out}")
PY

echo "=== pipeline done $(date -Is) ==="
