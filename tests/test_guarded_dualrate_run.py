"""CPU contracts for the eight-rank supervisor; real launch acceptance is separate."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import guarded_dualrate_run as runner


def uuids():
    return [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(8)]


def test_requires_eight_exact_distinct_training_devices():
    ids = uuids()
    inventory = dict.fromkeys(ids)
    assert runner.selected_uuids(",".join(ids), inventory, "distributed_training_smoke") == ids
    for bad in (ids[:1], ids[:-1], ids[:-1] + [ids[0]], ids + [ids[0]]):
        with pytest.raises(ValueError):
            runner.selected_uuids(",".join(bad), inventory, "distributed_training_smoke")
    assert runner.selected_uuids(",".join(ids[:2]), inventory, "domino_deployment_smoke") == ids[:2]
    assert runner.selected_uuids(",".join(ids[:3]), inventory, "domino_deployment_smoke") == ids[:3]
    with pytest.raises(ValueError):
        runner.selected_uuids(",".join(ids[:4]), inventory, "domino_deployment_smoke")


def test_only_fixed_workers_and_explicit_resume(tmp_path):
    command = runner.worker_command(tmp_path, {"run_kind":"distributed_training_smoke", "world_size":8},
                                    tmp_path/"config.json", tmp_path/"auth.json")
    assert command[1] == str(tmp_path/"scripts/dualrate_rank_launcher.py")
    assert command[-4:] == ["--device", "cuda", "--authorization", str(tmp_path/"auth.json")]
    with pytest.raises(ValueError):
        runner.worker_command(tmp_path, {"run_kind":"arbitrary_script"}, None, None)
    resumed = runner.worker_command(tmp_path,{"run_kind":"distributed_training_smoke","world_size":8},
        tmp_path/"config.json",tmp_path/"auth.json",tmp_path/"last.pt")
    assert resumed[-2:] == ["--resume",str(tmp_path/"last.pt")]


def test_phase_acceptance_binds_semantic_profile_and_evidence(tmp_path):
    config = dict(run_kind="formal_distributed_training",phase="warmup",steps=1000,output="new",
                  batch_size=1,gradient_accumulation_steps=2,world_size=8,fast_lr=.0001)
    source = dict(source_sha256="source",distributed_supervisor_sha256="supervisor",
                  guard_sha256="guard",supervisor_sha256="base",scripts_sha256="scripts")
    evidence = tmp_path/"rank0.json"
    evidence.write_text('{"passed":true}')
    receipt = tmp_path/"acceptance.json"
    value = dict(passed=True,source_sha256="source",supervisor_sha256="supervisor",world_size=8,
                 source_contract=runner.source_contract(source),
                 profiles={"warmup":runner.profile_sha256(config)},
                 evidence=[dict(path=str(evidence),sha256=runner.base.file_sha256(evidence))])
    receipt.write_text(json.dumps(value))
    config.update(acceptance_receipt=str(receipt),acceptance_receipt_sha256=runner.base.file_sha256(receipt))
    assert runner.validate_acceptance(config, source)["passed"]
    # Budget/output changes are explicit administrative fields, not training math.
    assert runner.profile_sha256(dict(config,steps=2000,output="other")) == runner.profile_sha256(config)
    for key,value in (("fast_lr",.001),("batch_size",2),("gradient_accumulation_steps",4)):
        with pytest.raises(ValueError,match="configuration"):
            runner.validate_acceptance(dict(config,**{key:value}),source)
    evidence.write_text('{"passed":false}')
    with pytest.raises(ValueError,match="evidence changed"):
        runner.validate_acceptance(config,source)


def test_all_ranks_must_consume_same_receipt(tmp_path):
    auth = tmp_path/"authorization.json"
    auth.write_text("{}")
    for rank in range(7):
        auth.with_name(auth.name+f".rank_{rank}.consumed").write_text(
            json.dumps(dict(receipt_sha256=runner.base.file_sha256(auth))))
    with pytest.raises(FileNotFoundError):
        runner.verify_consumed(auth,"distributed_training_smoke")
    auth.with_name(auth.name+".rank_7.consumed").write_text(json.dumps(dict(receipt_sha256=runner.base.file_sha256(auth))))
    runner.verify_consumed(auth,"distributed_training_smoke")


def test_lifetime_order_in_fixed_supervisor():
    source = (SCRIPTS/"guarded_dualrate_run.py").read_text()
    assert source.index("guard.open()") < source.index('base.telemetry(devices, out, "preflight")') < source.index("proc = subprocess.Popen(command")
    assert source.index("remaining = base.cleanup_owned") < source.index('postflight_idle(devices,out,selected)') < source.index("guard.close()")
    assert "shell=True" not in source and "nvidia-smi -r" not in source


def test_postflight_drains_utilization_without_relaxing_idle(monkeypatch,tmp_path):
    clock=[0.];samples=iter([1,1,0]);calls=[]
    def telemetry(*args):
        u=next(samples);calls.append(u)
        return [dict(uuid='gpu',processes=[],used_mib=1,gpu_util_percent=u)]
    monkeypatch.setattr(runner.base,'telemetry',telemetry)
    runner.postflight_idle({},tmp_path,['gpu'],clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    assert calls==[1,1,0] and json.loads((tmp_path/'postflight_idle.json').read_text())['strict_idle']


def test_postflight_foreign_process_and_timeout_fail(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.base,'telemetry',lambda *a:[dict(uuid='gpu',processes=[dict(pid=42)],used_mib=1,gpu_util_percent=1)])
    with pytest.raises(RuntimeError):runner.postflight_idle({},tmp_path,['gpu'])
    clock=[0.]
    monkeypatch.setattr(runner.base,'telemetry',lambda *a:[dict(uuid='gpu',processes=[],used_mib=1,gpu_util_percent=1)])
    with pytest.raises(RuntimeError,match='did not drain'):
        runner.postflight_idle({},tmp_path,['gpu'],timeout=1.,clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
