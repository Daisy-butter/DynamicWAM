import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from native_v9_smoke import audit


def events():
    return [dict(kind='slow_request',version=1,source_index=0,source_time=0.,committed_rows=0),
        dict(kind='slow_publish',version=1,source_index=0,source_time=0.,complete_physical=0.,consumed_at_physical=0.),
        dict(kind='fast_query',command_index=0,query_physical_time=0.,source_command_index=0,reference_age=0.,prefix_rows=0,
            slow_inflight_at_fast_start=False,reference_reused=False,wait_reason='bootstrap'),
        dict(kind='native_topp_execute',command_index=0,physical_start=0.,physical_end=1.9,execute_start_wall=100.,execute_end_wall=101.),
        dict(kind='fast_query',command_index=1,query_physical_time=1.9,source_command_index=0,reference_age=1.9,prefix_rows=0,
            slow_inflight_at_fast_start=False,reference_reused=True,wait_reason=None)]


def test_native_duration_not_wall_duration_or_point18():
    result=audit(events());assert result['commands']==1 and result['reused']==1


@pytest.mark.parametrize('field,value',[('command_index',2),('query_physical_time',.18),('source_command_index',2),('prefix_rows',1)])
def test_reject_shifted_clock_or_event(field,value):
    e=events();e[-1][field]=value
    with pytest.raises(ValueError):audit(e)


def test_actual_model_overlap_not_merely_inflight_flag():
    e=events();e.insert(3,dict(kind='slow_request',version=2,source_index=0,source_time=0.,committed_rows=1))
    e.insert(4,dict(kind='slow_publish',version=2,source_index=0,source_time=0.,complete_physical=.5,consumed_at_physical=1.9,
        model_start_wall=100.2,model_end_wall=100.7))
    assert audit(e)['model_execution_overlap']==1


def test_reject_partial_topp_source():
    e=events();e[0]["source_time"]=.3
    with pytest.raises(ValueError,match="command boundary"):audit(e)


def test_replay_manifest_accepts_json_string_paths_and_checks_hash(tmp_path):
    import json,hashlib
    from dynamicwam.dualrate.native_v9_data import build_manifest
    expert=tmp_path/'expert.json';expert.write_text(json.dumps(dict(motion_sampling='raw_frames_v1')))
    shard=tmp_path/'row.pt';shard.write_bytes(b'hash-bound native sample')
    proof=dict(passed=True,samples=[dict(path=str(shard),sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),task='x',seed=3,command_index=4)])
    (tmp_path/'replay_acceptance.json').write_text(json.dumps(proof))
    result=build_manifest(expert,[tmp_path],tmp_path/'out/train.json')
    assert result['samples']==1 and result['tasks']==['x']
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError,match='shard mismatch'):build_manifest(expert,[tmp_path],tmp_path/'changed/train.json')


def test_parallel_guards_share_start_and_drain_barriers(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from parallel_guard_cohort import ParallelGuardCohort
    (tmp_path/'manifest.json').write_text(json.dumps(dict(members=['0','1'])))
    gates=[ParallelGuardCohort(dict(directory=str(tmp_path),member=str(i)),tmp_path) for i in range(2)]
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(g.barrier,'ready',timeout=3.) for g in gates]
        for f in futures:f.result(timeout=5)
        gates[0].abort('owned worker failed')
        with pytest.raises(RuntimeError,match='aborted'):gates[1].barrier('ready',timeout=1.)
        futures=[pool.submit(g.barrier,'drained',timeout=3.,honor_abort=False) for g in gates]
        for f in futures:f.result(timeout=5)
    assert len(list((tmp_path/'drained').glob('*.json')))==2


def test_native_only_rejects_exact_infeasible_grasp_candidate():
    from dynamicwam.runtime.domino_rt.domino_session import _accept_expert_seed
    class Env:
        plan_success=True
        def __init__(self,message):self.message=message;self.seeds=[];self.closed=0
        def setup_demo(self,**kw):self.seeds.append(kw['seed'])
        def play_once(self):
            if len(self.seeds)==1:raise AssertionError(self.message)
            return dict(info='native expert')
        def close_env(self):self.closed+=1
        def check_success(self):return True
    e=Env('target_pose cannot be None for move action.')
    assert _accept_expert_seed(e,dict(render_freq=1),7,LookupError)==8
    assert e.seeds==[7,8] and e.closed==2
    e=Env('unrelated assertion')
    with pytest.raises(AssertionError,match='unrelated'):_accept_expert_seed(e,dict(render_freq=1),7,LookupError)
    assert e.seeds==[7] and e.closed==1
