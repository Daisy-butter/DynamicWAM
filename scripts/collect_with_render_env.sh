#!/usr/bin/env bash
# Headless SAPIEN/Vulkan env for this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH=/SSD_DISK_1/users/wuruihan/DynamicWAM/external/vulkan-conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/SSD_DISK_1/users/wuruihan/DynamicWAM/external/vulkan-icd/nvidia_icd.json
export VK_DRIVER_FILES=$VK_ICD_FILENAMES
export VK_LOADER_LAYERS_DISABLE=~implicit~
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1
nvidia-modprobe -m >/dev/null 2>&1 || true
cmd=$(printf '%q ' "$@")
exec sg video -c "$cmd"
