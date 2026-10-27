"""Pure file checks and process environment for the isolated SAPIEN runtime.

This module never imports SAPIEN, CUDA, or a dynamic library. The supervisor
validates and applies this before spawning a deployment worker; training is
unaffected. The worker revalidates the same files before its first GPU import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_renderer_runtime(config):
    binding = config.get("renderer_runtime")
    if not isinstance(binding, dict) or binding.get("schema") != "sapien3_explicit_runtime_v1":
        raise ValueError("deployment requires an explicit hash-bound renderer_runtime")
    hashes = binding.get("file_sha256", {})
    if not hashes:
        raise ValueError("renderer runtime library hashes are absent")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() or str(path.resolve(strict=True)) != name:
            raise ValueError("renderer runtime hashes require canonical absolute paths")
        if _sha256(path) != expected:
            raise ValueError("renderer runtime file changed: " + name)

    def bound_file(name):
        path = Path(name)
        if not path.is_absolute() or str(path.resolve(strict=True)) not in hashes:
            raise ValueError("unbound renderer runtime file: " + str(name))
        return path

    loader = bound_file(binding["vulkan_loader"])
    for directory_name in ("sapien_library_dir", "oidn_library_dir"):
        directory = Path(binding[directory_name])
        if not directory.is_absolute() or not directory.is_dir():
            raise ValueError("renderer library directory is absent")
        libraries = list(directory.glob("*.so*"))
        if not libraries:
            raise ValueError("empty renderer library directory")
        for library in libraries:
            bound_file(library)
    # Loader SONAME and explicitly selected loader must resolve to the same bytes.
    if (loader.parent / "libvulkan.so.1").resolve(strict=True) != loader.resolve(strict=True):
        raise ValueError("renderer loader SONAME points to different bytes")
    vendors = binding.get("vendor_libraries", {})
    for key in ("vulkan_icd", "egl_icd"):
        icd = bound_file(binding[key])
        library_name = json.loads(icd.read_text())["ICD"]["library_path"]
        if library_name not in vendors:
            raise ValueError("renderer ICD vendor library is not bound: " + library_name)
        library = bound_file(vendors[library_name])
        if library.name != library_name:
            raise ValueError("renderer ICD vendor SONAME differs from bound library")
    return binding


def renderer_environment(config, environment):
    """Return deployment overrides; safe to apply repeatedly to the same env."""
    if config.get("run_kind") != "domino_deployment_smoke":
        return {}
    binding = validate_renderer_runtime(config)
    directories = [str(Path(binding["vulkan_loader"]).parent),
                   binding["sapien_library_dir"], binding["oidn_library_dir"]]
    inherited = [p for p in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                 if p and p not in directories]
    return {
        "VK_ICD_FILENAMES": binding["vulkan_icd"],
        "SAPIEN_VULKAN_LIBRARY_PATH": binding["vulkan_loader"],
        "__EGL_VENDOR_LIBRARY_FILENAMES": binding["egl_icd"],
        "LD_LIBRARY_PATH": os.pathsep.join(directories + inherited),
        "SAPIEN_RENDER": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }


def verify_renderer_environment(config, environment):
    expected = renderer_environment(config, environment)
    for key, value in expected.items():
        if environment.get(key) != value:
            raise ValueError("renderer worker environment differs from binding: " + key)
    return expected
