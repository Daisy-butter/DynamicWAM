import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import dualrate_training_sequence as sequence


def test_completed_phase_requires_all_eight_strict_restores(tmp_path):
    checkpoint = tmp_path/"last.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint-no-model")
    config = dict(phase="warmup",steps=4,run_kind="distributed_training_smoke")
    result = dict(result_kind="completed_phase_not_policy_quality_acceptance",completed_steps=4,
        configured_steps=4,phase="warmup",run_kind=config["run_kind"],world_size=8,
        checkpoint_sha256=sequence.file_sha256(checkpoint))
    reload = dict(passed=True,world_size=8,ranks=[dict(rank=i,passed=True,phase_step=4) for i in range(8)])
    (tmp_path/"result.json").write_text(json.dumps(result))
    (tmp_path/"reload.json").write_text(json.dumps(reload))
    assert sequence.phase_completed(tmp_path,config)[0] == checkpoint
    reload["ranks"][7]["passed"] = False
    (tmp_path/"reload.json").write_text(json.dumps(reload))
    with pytest.raises(RuntimeError,match="all eight"):
        sequence.phase_completed(tmp_path,config)


def test_sequence_rejects_arbitrary_stage_order_and_smoke_init(tmp_path):
    phases = []
    for phase in sequence.PHASES:
        path = tmp_path/(phase+".json")
        path.write_text(json.dumps(dict(phase=phase,world_size=8,run_kind="formal_distributed_training")))
        phases.append(dict(phase=phase,config=str(path),config_sha256=sequence.file_sha256(path),max_runtime_seconds=3600))
    spec = tmp_path/"sequence.json"
    spec.write_text(json.dumps(dict(sequence_id="unit-test",phases=phases)))
    assert len(sequence.load_sequence(spec)[1]) == 3
    path = tmp_path/"small_age_joint.json"
    value = json.loads(path.read_text())
    value["stage_init_checkpoint"] = "smoke.pt"
    path.write_text(json.dumps(value))
    phases[1]["config_sha256"] = sequence.file_sha256(path)
    spec.write_text(json.dumps(dict(sequence_id="unit-test",phases=phases)))
    with pytest.raises(ValueError,match="bind predecessors"):
        sequence.load_sequence(spec)
