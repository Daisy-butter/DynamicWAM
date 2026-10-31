import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import continuous_v8_final_eval as entry
import dualrate_eval_worker as worker


@pytest.mark.parametrize('fault',[None,'native','fixture','recipe','reserve','smoothing','runtime','linear','sequence','gripper','flow'])
def test_final_eval_contract_rejects_train_deploy_drift(fault):
    spec=json.loads((ROOT/'configs/continuous_v8_final_eval.json').read_text())
    c=json.loads((ROOT/'configs/continuous_v8_final_smoke.json').read_text())
    train=copy.deepcopy(c)
    # The real formal receipt gate is exercised by training_ready tests and live
    # --check. This fixture tests physical labels and deployment drift in memory.
    train['run_kind']='distributed_training_smoke'
    manifest=dict(measured_runtime=copy.deepcopy(c['runtime']),time_contract=c['time_contract'],
        continuous_gripper_sampling='interval_start_previous_event',continuous_flow_sampling='raw_frames_v1',
        continuous_rtc_contract=dict(version='continuous_rtc_v8',clock='physical_seconds',
            action_semantics='fixed_interval_endpoint_arm_start_gripper',
            action_interval_seconds=c['action_interval_seconds'],history_endpoint_seconds=.72,
            source='sampled_slow_prediction',labels='same_absolute_physical_interval',
            prefix='actual_committed_values_and_intervals'))
    if fault=='native':c['native_v7']={'enabled':True}
    elif fault=='fixture':c['engineering_initialization_only']=True
    elif fault=='recipe':c['training_recipe'][-1]['steps']=20000
    elif fault=='reserve':c['deployment_min_query_budget_seconds']=.4
    elif fault=='smoothing':c['deployment_arm_smoothing']={'kind':'ema'}
    elif fault=='runtime':manifest['measured_runtime']['protect_steps']=3
    elif fault=='linear':c['benchmark_contract']['controller']='TOPP'
    elif fault=='sequence':c['sequence_id']='other_final'
    elif fault=='gripper':manifest['continuous_gripper_sampling']='endpoint'
    elif fault=='flow':manifest['continuous_flow_sampling']='cached_old'
    if fault is None:entry.validate_contract(spec,c,train,manifest)
    else:
        with pytest.raises(ValueError):entry.validate_contract(spec,c,train,manifest)


def test_pending_checkpoint_is_preflight_only(monkeypatch,tmp_path):
    c=json.loads((ROOT/'configs/continuous_v8_final_smoke.json').read_text())
    c.update(storage_root=str(tmp_path/'wuruihan'),output=str(tmp_path/'wuruihan/out'))
    Path(c['storage_root']).mkdir()
    monkeypatch.setattr(worker,'validate_source_compatibility',lambda c:None)
    def digest(path):
        if path==c['checkpoint']:raise FileNotFoundError('final weight not written')
        return c['profile_sha256'] if path==c['profile'] else c['init_checkpoint_sha256']
    monkeypatch.setattr(worker,'sha',digest)
    worker.validate_config(c,checkpoint_pending=True)
    with pytest.raises(FileNotFoundError):worker.validate_config(c)
    c['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):worker.validate_config(c,checkpoint_pending=True)
