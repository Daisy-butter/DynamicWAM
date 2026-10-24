import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from dynamicwam.dualrate.renderer_runtime import (
    renderer_environment, validate_renderer_runtime, verify_renderer_environment,
)


def runtime_fixture(tmp_path):
    files = {}

    def file(name, data=b"library"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return str(path)

    loader = file("vulkan/libvulkan.so.1")
    glx = file("vendor/libGLX_nvidia.so.0")
    egl = file("vendor/libEGL_nvidia.so.0")
    icd = file("nvidia_icd.json", json.dumps({"ICD": {"library_path": "libGLX_nvidia.so.0"}}).encode())
    egl_icd = file("egl.json", json.dumps({"ICD": {"library_path": "libEGL_nvidia.so.0"}}).encode())
    file("sapien.libs/libsapien.so")
    file("oidn_library/libOpenImageDenoise.so.2")
    return {"run_kind": "domino_deployment_smoke", "renderer_runtime": {
        "schema": "sapien3_explicit_runtime_v1", "vulkan_loader": loader,
        "vulkan_icd": icd, "egl_icd": egl_icd,
        "sapien_library_dir": str(tmp_path / "sapien.libs"),
        "oidn_library_dir": str(tmp_path / "oidn_library"),
        "vendor_libraries": {"libGLX_nvidia.so.0": glx, "libEGL_nvidia.so.0": egl},
        "file_sha256": files,
    }}


def test_renderer_env_bound_before_spawn_and_idempotent(tmp_path):
    config = runtime_fixture(tmp_path)
    inherited = {"LD_LIBRARY_PATH": "/old/path", "SAPIEN_VULKAN_LIBRARY_PATH": "/wrong"}
    overrides = renderer_environment(config, inherited)
    assert inherited["SAPIEN_VULKAN_LIBRARY_PATH"] == "/wrong"
    env = dict(inherited, **overrides)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(tmp_path / "vulkan"),
        str(tmp_path / "sapien.libs"), str(tmp_path / "oidn_library"), "/old/path"]
    assert env["PYTHONFAULTHANDLER"] == env["PYTHONUNBUFFERED"] == "1"
    assert verify_renderer_environment(config, env) == overrides
    assert renderer_environment(config, env) == overrides
    env["VK_ICD_FILENAMES"] = "/wrong"
    with pytest.raises(ValueError, match="environment differs"):
        verify_renderer_environment(config, env)


def test_changed_runtime_or_unbound_vendor_rejected(tmp_path):
    config = runtime_fixture(tmp_path)
    binding = config["renderer_runtime"]
    binding["vendor_libraries"].pop("libGLX_nvidia.so.0")
    with pytest.raises(ValueError, match="vendor library is not bound"):
        validate_renderer_runtime(config)
    config = runtime_fixture(tmp_path)
    Path(config["renderer_runtime"]["vulkan_loader"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        validate_renderer_runtime(config)


def test_unrecorded_library_and_missing_binding_rejected(tmp_path):
    config = runtime_fixture(tmp_path)
    (tmp_path / "sapien.libs/unknown.so").write_bytes(b"new")
    with pytest.raises(ValueError, match="unbound"):
        validate_renderer_runtime(config)
    with pytest.raises(ValueError, match="renderer_runtime"):
        renderer_environment({"run_kind": "domino_deployment_smoke"}, {})


def test_guard_applies_renderer_only_to_deployment(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/guarded_dualrate_smoke.py"
    spec = importlib.util.spec_from_file_location("renderer_test_guard", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setenv("SAPIEN_VULKAN_LIBRARY_PATH", "/inherited")
    training = guard.worker_environment(tmp_path, tmp_path / "train", "GPU-x",
        config={"run_kind": "distributed_training_smoke"})
    assert training["SAPIEN_VULKAN_LIBRARY_PATH"] == "/inherited"
    config = runtime_fixture(tmp_path / "runtime")
    deployed = guard.worker_environment(tmp_path, tmp_path / "eval", "GPU-x", config=config)
    assert deployed["SAPIEN_VULKAN_LIBRARY_PATH"] == config["renderer_runtime"]["vulkan_loader"]
