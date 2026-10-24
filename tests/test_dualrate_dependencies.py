"""Pure CPU/file fixtures: no native extension or GPU is loaded here."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/dynamicwam/dualrate/dependencies.py"
spec = importlib.util.spec_from_file_location("dependency_fixture", SOURCE)
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)


def fixture(tmp_path):
    storage = tmp_path / "wuruihan"
    overlay = storage / "dependencies/flash"
    package = overlay / "flash_attn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__='fixture'\n")
    (overlay / "flash_attn_2_cuda.so").write_bytes(b"explicit non-loadable test fixture")
    files = {p.relative_to(overlay).as_posix(): deps._sha(p) for p in overlay.rglob("*") if p.is_file()}
    return dict(storage_root=str(storage), python_dependency_overlay=str(overlay),
        python_dependency_files=files, python_dependency_versions={"torch": "fixture-torch", "flash_attn": "fixture"})


def test_complete_overlay_binding_and_post_run_mutation(tmp_path):
    config = fixture(tmp_path)
    first = deps.validate_dependency_overlay(config)
    assert first["files"] == config["python_dependency_files"]
    assert deps.validate_dependency_overlay(config) == first
    (Path(first["root"])/"flash_attn/__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        deps.validate_dependency_overlay(config)


@pytest.mark.parametrize("relative", ["../escape.py", "/absolute.py", "a\\escape.py", "./flash_attn/__init__.py"])
def test_reject_path_traversal_or_noncanonical_paths(tmp_path, relative):
    config = fixture(tmp_path)
    config["python_dependency_files"] = {relative: "0" * 64}
    with pytest.raises(ValueError):
        deps.validate_dependency_overlay(config)


def test_undeclared_source_is_not_silently_loaded(tmp_path):
    config = fixture(tmp_path)
    (Path(config["python_dependency_overlay"])/"surprise.py").write_text("pass\n")
    with pytest.raises(ValueError, match="undeclared"):
        deps.validate_dependency_overlay(config)


def test_overlay_must_stay_in_storage_root(tmp_path):
    config = fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    config["python_dependency_overlay"] = str(outside)
    with pytest.raises(ValueError, match="artifact filesystem"):
        deps.validate_dependency_overlay(config)


def test_cpu_probe_imports_actual_bound_module_paths_without_cuda(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    root = Path(config["python_dependency_overlay"])
    init = lambda: None
    torch = SimpleNamespace(__version__="fixture-torch", __file__=str(tmp_path/"torch.py"),
                            cuda=SimpleNamespace(is_initialized=lambda: False, _lazy_init=init))
    flash = SimpleNamespace(__version__="fixture", __file__=str(root/"flash_attn/__init__.py"))
    native = SimpleNamespace(__file__=str(root/"flash_attn_2_cuda.so"))
    for name in list(sys.modules):
        if name.startswith("flash_attn"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "flash_attn", flash)
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", native)
    def imported(name):
        if name == "torch":
            return torch
        assert name == "flash_attn"
        with pytest.raises(RuntimeError, match="attempted CUDA"):
            torch.cuda._lazy_init()
        return flash
    monkeypatch.setattr(deps.importlib, "import_module", imported)
    monkeypatch.setattr(sys, "path", list(sys.path))
    result = deps.probe_dependency_imports(config, require_flash_attention=True)
    assert result["versions"] == config["python_dependency_versions"]
    assert not result["cuda_initialized"]
    assert result["module_paths"]["flash_attn_2_cuda"] == str(root/"flash_attn_2_cuda.so")
    assert torch.cuda._lazy_init is init


def test_text_probe_does_not_require_flash_attention(tmp_path, monkeypatch):
    fake = SimpleNamespace(__version__="fixture-torch", __file__=str(tmp_path/"torch.py"),
                           cuda=SimpleNamespace(is_initialized=lambda: False, _lazy_init=lambda: None))
    def imported(name):
        assert name == "torch", "text preparation must not require FlashAttention"
        return fake
    monkeypatch.setattr(deps.importlib, "import_module", imported)
    result = deps.probe_dependency_imports({}, require_flash_attention=False)
    assert result["versions"] == {"torch": "fixture-torch"}
    assert not result["flash_attention_required"]
