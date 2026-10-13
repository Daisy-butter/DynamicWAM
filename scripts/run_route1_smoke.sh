#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
export WAM_SSD_ROOT=/SSD_DISK/users/wuruihan
export TMPDIR="$WAM_SSD_ROOT/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WAM_SSD_ROOT/caches/route1"
export HF_HOME="$WAM_SSD_ROOT/hf_cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
export PYTHONPATH="$PWD/src:$PWD/external/flash_attn_runtime:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="$PWD/external/vulkan-icd/nvidia_icd.json"
export SAPIEN_VULKAN_LIBRARY_PATH="$PWD/external/vulkan-conda/lib/libvulkan.so.1"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export SAPIEN_RENDER=egl
PY="$PWD/external/robotwin/bin/python"
SAPIEN_LIB="$($PY -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("sapien").origin).parent.parent/"sapien.libs")')"
export LD_LIBRARY_PATH="$SAPIEN_LIB:${LD_LIBRARY_PATH:-}"
GPU="${ROUTE1_GPU_UUID:-GPU-a60c2940-0863-4ad5-531a-425b5db644f3}"
OUT="${1:?pass a fresh SSD output directory}"
TASKS="${2:-adjust_bottle,place_can_basket,click_bell}"
EPISODES="${3:-2}"
$PY scripts/harness_preflight.py --output "$OUT" --gpu "$GPU"
cp harness.md "$OUT/harness.md"
tar --exclude='__pycache__' -czf "$OUT/source_snapshot.tar.gz" src/dynamicwam/runtime/domino_continuous scripts/eval_domino_continuous_sync.py scripts/eval_domino_continuous_sync_suite.py scripts/video_artifacts.py scripts/summarize_domino_continuous_sync.py scripts/harness_preflight.py scripts/run_route1_smoke.sh docs/route1_port_source.json docs/domino_continuous_sync_timing_20261009.json harness.md AGENTS.md
$PY scripts/eval_domino_continuous_sync_suite.py --checkpoint external/checkpoints/DynamicWAM_full.pt --output-dir "$OUT" --gpus "$GPU" --episodes "$EPISODES" --tasks "$TASKS" --timing-contract docs/domino_continuous_sync_timing_20261009.json --source-generation route1_continuous_port_20261013 --max-infra-retries 1
