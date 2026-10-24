"""CPU-only supervisor tests; not evidence of a real GPU model/driver exit."""
import ast
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/guarded_dualrate_smoke.py"
spec = importlib.util.spec_from_file_location("guarded_dualrate_smoke", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def xml(uuid="GPU-a", pci="00000000:4f:00.0", used=0, util=0, processes="", recovery="None"):
    return f"""<nvidia_smi_log><gpu><uuid>{uuid}</uuid><product_name>A6000</product_name>
      <pci><pci_bus_id>{pci}</pci_bus_id></pci><fb_memory_usage><used>{used} MiB</used>
      <free>48540 MiB</free></fb_memory_usage><utilization><gpu_util>{util} %</gpu_util></utilization>
      <gpu_recovery_action>{recovery}</gpu_recovery_action><processes>{processes}</processes></gpu></nvidia_smi_log>"""


EXPECTED = {"GPU-a": {"pci": "0000:4f:00.0", "minor": 0, "device_path": "/dev/nvidia0"}}


def test_pci_domain_width_and_idle():
    rows = runner.parse_telemetry(xml(), EXPECTED)
    runner.require_idle(rows, "GPU-a")
    assert rows[0]["free_mib"] == 48540
    assert runner.pci_identity("00000000:4f:00.0") == runner.pci_identity("0000:4f:00.0")


@pytest.mark.parametrize("kwargs,match", [({"pci": "0000:50:00.0"}, "PCI"),
                                        ({"uuid": "GPU-b"}, "UUID"), ({"recovery": "Reset"}, "recovery")])
def test_bad_telemetry_fails_closed(kwargs, match):
    with pytest.raises(RuntimeError, match=match):
        runner.parse_telemetry(xml(**kwargs), EXPECTED)


@pytest.mark.parametrize("kwargs", [{"used": 513}, {"util": 1},
    {"processes": "<process_info><pid>123</pid><type>C</type><used_memory>1 MiB</used_memory></process_info>"}])
def test_selected_busy_fails_without_killing(kwargs):
    with pytest.raises(RuntimeError, match="occupied"):
        runner.require_idle(runner.parse_telemetry(xml(**kwargs), EXPECTED), "GPU-a")


def test_unselected_activity_is_reported_not_rejected():
    rows = runner.parse_telemetry(xml(), EXPECTED)
    rows.append({"uuid": "GPU-b", "used_mib": 40000, "gpu_util_percent": 100, "processes": [{"pid": 77}]})
    runner.require_idle(rows, "GPU-a")


def test_missing_process_inventory_and_unknown_util_fail():
    with pytest.raises(RuntimeError, match="process inventory"):
        runner.parse_telemetry(xml().replace("<processes></processes>", ""), EXPECTED)
    with pytest.raises(RuntimeError, match="telemetry quantity"):
        runner.parse_telemetry(xml().replace("0 %", "N/A"), EXPECTED)


def test_proc_inventory_checks_distinct_minors(tmp_path):
    for pci, uuid in (("0000:01:00.0", "GPU-a"), ("0000:02:00.0", "GPU-b")):
        p = tmp_path / pci / "information"
        p.parent.mkdir()
        p.write_text(f"GPU UUID: {uuid}\nDevice Minor: 0\n")
    with pytest.raises(RuntimeError, match="eight-adapter"):
        runner.inventory(tmp_path, expected_count=2)


def test_source_and_identity_exactly_match_training_implementation(tmp_path):
    # Execute only pure hash functions extracted from train.py, no torch import.
    tree = ast.parse((SCRIPT.parents[1] / "src/dynamicwam/dualrate/train.py").read_text(encoding="utf-8"))
    pure = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
            {"file_sha256", "canonical_hash", "source_identity"}]
    namespace = {"Path": Path, "hashlib": runner.hashlib, "json": json, "Mapping": dict}
    exec(compile(ast.Module(body=pure, type_ignores=[]), "train-hash-functions", "exec"), namespace)
    (tmp_path / "a.py").write_text("a=1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.py").write_text("b=2\n")
    assert runner.source_identity(tmp_path) == namespace["source_identity"](tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = {"manifest": str(manifest), "init_checkpoint_sha256": "f" * 64, "steps": 2}
    identity = runner.build_identity(config, {"source_sha256": runner.source_identity(tmp_path)}, "2.4.1+cu121")
    assert identity == {"source_sha256": namespace["source_identity"](tmp_path),
                        "config_sha256": namespace["canonical_hash"](config),
                        "manifest_sha256": namespace["file_sha256"](manifest),
                        "init_checkpoint_sha256": "f" * 64, "device_type": "cuda", "torch_version": "2.4.1+cu121"}


def test_only_fixed_worker_and_runner_owned_arguments(tmp_path):
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/prepare_dualrate_text.py"
    script.write_text("pass\n")
    other = tmp_path / "scripts/other.py"
    other.write_text("pass\n")
    arguments = tmp_path / "args.json"
    arguments.write_text('["--label", "a; $not_a_shell"]')
    mode, command, receipt = runner.fixed_worker(tmp_path, script, arguments)
    assert mode == "data_preparation" and command[-1] == "a; $not_a_shell"
    assert receipt["worker_sha256"] == runner.file_sha256(script)
    with pytest.raises(ValueError, match="only the audited"):
        runner.fixed_worker(tmp_path, other, arguments)
    arguments.write_text('["--authorization=forged"]')
    with pytest.raises(ValueError, match="runner owns"):
        runner.fixed_worker(tmp_path, script, arguments)


def test_cleanup_only_owned_live_identity_and_pid_reuse(monkeypatch):
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    leader = {"pid": 100, "start_ticks": 77, "session": 100, "pgrp": 100, "state": "S"}
    child = {"pid": 101, "start_ticks": 78, "session": 100, "pgrp": 101, "state": "S"}
    current = {100: leader, 101: child}
    sent = []
    def send(pid, sig):
        sent.append((pid, sig))
        current.pop(pid)
    proc = SimpleNamespace(pid=100, poll=lambda: None)
    remaining = runner.cleanup_owned(proc, 77, observe=lambda _: list(current.values()), identify=current.__getitem__, send=send)
    assert remaining == [] and {p for p, _ in sent} == {100, 101}
    with pytest.raises(RuntimeError, match="PID reused"):
        runner.cleanup_owned(proc, 76, observe=lambda _: [leader], identify=lambda _: leader, send=send)


def test_cleanup_unreapable_is_explicit_not_success(monkeypatch):
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    row = {"pid": 100, "start_ticks": 77, "session": 100, "pgrp": 100, "state": "D"}
    sent = []
    result = runner.cleanup_owned(SimpleNamespace(pid=100, poll=lambda: None), 77, term_seconds=0, kill_seconds=0,
                                  observe=lambda _: [row], identify=lambda _: row, send=lambda *args: sent.append(args))
    assert result == [row] and len(sent) == 2


def test_telemetry_guard_and_worker_launch_order_and_no_shell():
    # Static boundary proof complements logic tests, never claims GPU acceptance.
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.index("guard.open()") < text.index('rows = telemetry(devices, out, "preflight")') < text.index("proc = subprocess.Popen(command")
    assert text.index("remaining = cleanup_owned") < text.index('post = telemetry(devices, out, "postflight")') < text.index("guard.close()")
    assert "shell=True" not in text and "/cmdline" not in text and "/environ" not in text
    assert '"cleanup_pending"' in text


def test_dependency_overlay_search_path_and_receipt_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    overlay = tmp_path / "dependencies"
    env = runner.worker_environment(tmp_path, tmp_path/"out", "GPU-test", {"root": str(overlay)})
    assert env["PYTHONPATH"].split(runner.os.pathsep) == [str(tmp_path/"src"), str(overlay), "existing-path"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.index('persist("dependencies_before.json"') < text.index("proc = subprocess.Popen(command")
    assert text.index('persist("dependencies_after.json"') > text.index('persist("owned_exit.json"')
    assert '"python_dependencies": dependency_binding' in text
    assert "require_flash_attention=" in text
