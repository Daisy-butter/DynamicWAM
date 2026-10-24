"""Hash-bound optional dependency overlay; no base-environment installation."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dependency_overlay(config):
    """Pure filesystem validation, safe before importing torch or any extension."""
    value, declared = config.get("python_dependency_overlay"), config.get("python_dependency_files")
    versions = config.get("python_dependency_versions", {})
    if value is None:
        if declared not in (None, {}):
            raise ValueError("dependency files require an explicit overlay path")
        return None
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("dependency overlay must be an absolute SSD path")
    storage = Path(config["storage_root"]).resolve(strict=True)
    root = Path(value).resolve(strict=True)
    if (storage.name != "wuruihan" or not root.is_dir() or root == storage or
            not root.is_relative_to(storage) or root.stat().st_dev != storage.stat().st_dev):
        raise ValueError("dependency overlay must stay within the verified artifact filesystem")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("overlay requires a nonempty complete .py/.so SHA256 mapping")
    if not isinstance(versions, dict) or any(k not in ("torch", "flash_attn") or not isinstance(v, str) or not v for k,v in versions.items()):
        raise ValueError("dependency versions may bind nonempty torch/flash_attn version strings")
    actual = {}
    for relative, expected in declared.items():
        if not isinstance(relative, str) or "\\" in relative or ":" in relative or "\0" in relative:
            raise ValueError("dependency entries must be canonical relative POSIX paths")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative or pure.suffix not in (".py", ".so"):
            raise ValueError("dependency entries must be relative .py/.so paths without traversal")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("dependency entries require lowercase SHA256 values")
        path = root / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not path.is_file():
            raise ValueError("dependency entry escapes overlay or is not a file")
        digest = _sha(path)
        if digest != expected:
            raise ValueError(f"dependency file hash changed: {relative}")
        actual[relative] = digest
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("overlay symlinks are not permitted; bind real copied files")
        if path.is_file():
            if path.suffix in (".pyc", ".pth", ".pyd"):
                raise ValueError("unbound bytecode/startup hooks are not allowed in dependency overlay")
            if path.suffix in (".py", ".so"):
                found.add(path.relative_to(root).as_posix())
    if found != set(actual):
        raise ValueError("overlay contains undeclared or missing .py/.so dependencies")
    encoded = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    return dict(root=str(root), files=actual, files_sha256=hashlib.sha256(encoded).hexdigest(),
                expected_versions=versions)


def activate_dependency_overlay(binding):
    """Only this process changes search paths; no site-packages or pip mutation."""
    if binding is not None:
        root = binding["root"]
        if root not in sys.path:
            sys.path.insert(0, root)


def probe_dependency_imports(config, *, require_flash_attention):
    """Import actual CPU modules/extensions while rejecting Torch CUDA init.

    Guard invokes this in a CPU subprocess with CUDA_VISIBLE_DEVICES empty;
    training repeats it inside the authorized worker before model allocation.
    """
    binding = validate_dependency_overlay(config)
    activate_dependency_overlay(binding)
    torch = importlib.import_module("torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("dependency import probe entered with initialized CUDA")
    original = torch.cuda._lazy_init
    def reject(*_args, **_kwargs):
        raise RuntimeError("CPU dependency probe attempted CUDA initialization")
    torch.cuda._lazy_init = reject
    try:
        versions = {"torch": str(torch.__version__)}
        modules = {"torch": str(Path(torch.__file__).resolve())}
        if require_flash_attention:
            flash = importlib.import_module("flash_attn")
            versions["flash_attn"] = str(flash.__version__)
            loaded = {name: str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items())
                      if name.startswith("flash_attn") and getattr(module, "__file__", None)
                      and Path(module.__file__).suffix in (".py", ".so")}
            if not any(Path(path).suffix == ".so" for path in loaded.values()):
                raise RuntimeError("FlashAttention import did not load its native extension")
            if binding is not None:
                root = Path(binding["root"])
                for name,path in loaded.items():
                    file = Path(path)
                    if not file.is_relative_to(root) or file.relative_to(root).as_posix() not in binding["files"]:
                        raise RuntimeError(f"imported {name} outside the bound overlay")
            modules.update(loaded)
        expected = config.get("python_dependency_versions", {})
        for name,value in expected.items():
            if name in versions and versions[name] != value:
                raise RuntimeError(f"{name} version differs from configuration: {versions[name]} != {value}")
        if torch.cuda.is_initialized():
            raise RuntimeError("dependency import unexpectedly initialized CUDA")
        return dict(versions=versions, module_paths=modules, cuda_initialized=False,
                    flash_attention_required=bool(require_flash_attention), overlay=binding)
    finally:
        torch.cuda._lazy_init = original
