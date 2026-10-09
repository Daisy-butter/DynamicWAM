#!/usr/bin/env bash
# Official-anchored temporal compact student: stage2 -> stage3 -> latency probes.
# Keeps official motion normalization, full-chunk fast loss, fixed native teacher,
# seconds TTL, and cross-query noise alignment at eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
ACC="$ROOT/.venv/bin/accelerate"
SSD="/SSD_DISK/users/wuruihan/DynamicWAM"
PIPE="$SSD/outputs/evaluation/official_temporal_7gpu_pipeline_20260930"
STAGE2_CFG="$ROOT/configs/official_temporal_stage2_7gpu.yaml"
STAGE3_CFG="$ROOT/configs/official_temporal_stage3_7gpu.yaml"
STAGE2_CKPT="$SSD/outputs/training/official_temporal_stage2_7gpu_20260930/exports/multirate_step_7000.pt"
STAGE3_CKPT="$SSD/outputs/training/official_temporal_stage3_7gpu_20260930/exports/multirate_step_3500.pt"
PORT2="${MASTER_PORT_STAGE2:-36311}"
PORT3="${MASTER_PORT_STAGE3:-36321}"

mkdir -p "$PIPE" \
  "$SSD/outputs/training/official_temporal_stage2_7gpu_20260930" \
  "$SSD/outputs/training/official_temporal_stage3_7gpu_20260930"
exec > >(tee -a "$PIPE/pipeline.log") 2>&1

echo "=== pipeline start $(date -Is) ==="

run_train() {
  local stage="$1" cfg="$2" port="$3" ckpt="$4" steps_marker="$5"
  local log="$SSD/outputs/training/official_temporal_stage${stage}_7gpu_20260930/pipeline_train.log"
  if [[ -f "$ckpt" ]]; then
    echo "stage${stage} checkpoint present, skip: $ckpt"
    return 0
  fi
  echo "=== stage${stage} train $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" \
  PYTHONUNBUFFERED=1 \
  "$ACC" launch \
    --multi_gpu \
    --num_processes 7 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$port" \
    "$ROOT/scripts/train_multirate.py" \
    --config "$cfg" \
    2>&1 | tee "$log"
  if ! grep -q "Exported multirate checkpoint" "$log"; then
    echo "stage${stage} did not export" >&2
    exit 2
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "missing expected export $ckpt" >&2
    exit 3
  fi
}

run_train 2 "$STAGE2_CFG" "$PORT2" "$STAGE2_CKPT" 7000
run_train 3 "$STAGE3_CFG" "$PORT3" "$STAGE3_CKPT" 3500

wait_ckpt() {
  "$PY" - "$1" <<'PY'
import sys, time
from pathlib import Path
path = Path(sys.argv[1])
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
                print(f"ready {path} bytes={size}")
                raise SystemExit(0)
        else:
            stable = None
            last = size
    time.sleep(2)
raise SystemExit(f"not stable: {path}")
PY
}

wait_ckpt "$STAGE3_CKPT"

# Verify motion provenance stayed on the official scale.
"$PY" - "$STAGE3_CKPT" <<'PY'
import sys, torch
obj = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
prov = obj["config"].get("initialization_provenance") or {}
motion = obj["config"]["model"]["absolute_motion"]["statistics_sha256"]
official = "7a2109f0d93418e9bc3b5c1c6c9961ac7afbda61f672a153b999e8df385a01b4"
assert motion == official, (motion, official)
assert prov.get("motion_statistics_sha256") == official or (
    (prov.get("parent_provenance") or {}).get("motion_statistics_sha256") == official
), prov
print("provenance ok", {k: prov.get(k) for k in ("path", "motion_statistics_sha256")})
print("parent source", (prov.get("source_dataset_identity") or {}).get("motion_statistics_sha256"))
PY

# Probe A: matched H=4, R=2, independent worker, noise reuse, same backbone.
PROBE_A="$PIPE/probe_matched_h4_r2"
if [[ ! -d "$PROBE_A" ]] || [[ -z "$(ls -A "$PROBE_A" 2>/dev/null || true)" ]]; then
  echo "=== probe matched H=4 R=2 $(date -Is) ==="
  mkdir -p "$PROBE_A"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/eval_multirate_probe.py" \
    --run-root "$PROBE_A" \
    --checkpoint "$STAGE3_CKPT" \
    --episodes 2 \
    --sync-horizon 4 \
    --bridge-refresh-ratio 2 \
    --scheduling independent_worker \
    --reuse-action-noise \
    --sync-gpus 0,1,2 \
    --multirate-gpus 3,4,5 \
    2>&1 | tee "$PIPE/probe_matched_h4_r2.log"
fi

# Probe B: official sync H=16 vs temporal H=4 R=2 (amortized latency reference).
PROBE_B="$PIPE/probe_official_h16"
if [[ ! -d "$PROBE_B" ]] || [[ -z "$(ls -A "$PROBE_B" 2>/dev/null || true)" ]]; then
  echo "=== probe official sync H=16 $(date -Is) ==="
  mkdir -p "$PROBE_B"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/eval_multirate_probe.py" \
    --run-root "$PROBE_B" \
    --checkpoint "$STAGE3_CKPT" \
    --episodes 2 \
    --sync-horizon 16 \
    --official-sync \
    --bridge-refresh-ratio 2 \
    --scheduling independent_worker \
    --reuse-action-noise \
    --sync-gpus 0,1,2 \
    --multirate-gpus 3,4,5 \
    2>&1 | tee "$PIPE/probe_official_h16.log"
fi

echo "=== summarize $(date -Is) ==="
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/official_temporal_7gpu_pipeline_20260930")
summary = {}
for name in ("probe_matched_h4_r2", "probe_official_h16"):
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
