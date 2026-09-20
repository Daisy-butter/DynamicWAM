"""Headless NVIDIA rendering for SAPIEN on machines with Display disabled."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_EGL_LIBRARY = "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
DEFAULT_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def nvidia_egl_icd_path() -> Path:
    return (
        Path(__file__).resolve().parents[3] / "configs" / "nvidia_icd_egl.json"
    )


def apply_headless_nvidia_egl(env: dict[str, str] | None = None) -> dict[str, str]:
    """Point Vulkan at libEGL_nvidia instead of libGLX_nvidia.

    On this cluster Display is disabled, so the distro ICD
    (``libGLX_nvidia.so.0``) fails. SAPIEN renders once the ICD library is EGL.
    """
    target = os.environ if env is None else env
    icd = nvidia_egl_icd_path()
    if icd.is_file() and not target.get("VK_ICD_FILENAMES"):
        target["VK_ICD_FILENAMES"] = str(icd)
    if Path(DEFAULT_EGL_VENDOR).is_file() and not target.get(
        "__EGL_VENDOR_LIBRARY_FILENAMES"
    ):
        target["__EGL_VENDOR_LIBRARY_FILENAMES"] = DEFAULT_EGL_VENDOR
    if Path(DEFAULT_EGL_LIBRARY).is_file():
        target.setdefault("SAPIEN_RENDER", "egl")
    return target
