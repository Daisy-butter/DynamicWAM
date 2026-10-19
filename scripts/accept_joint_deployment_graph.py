#!/usr/bin/env python3
"""Convert actual guarded CUDA evidence to joint-v3 deployment acceptance.

This CPU-only reader runs no model or GPU operation. Missing or mismatched
parity, source/weight bindings, concurrency or final guard closure is fatal.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from run_motionkv_joint_eval_pipeline import GRAPH_KIND,verify_support_guard
from run_motionkv_scaled_eval_pipeline import read,require,sha,utc
from supervise_motionkv_training import source_files_and_hash

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_CASES={
    'none_to_d0','committed_prefix','prefix_values_changed','current_state_changed',
    'current_motion_changed','world_kv_replaced','combined_values_changed',
    'return_to_none_clears_previous_prefix','return_to_d0','source_clean_plan_changed',
    'source_motion_changed','source_rgb_cache_refreshed','source_plan_sigma_changed',
    'observation_delay_changed','physical_action_target_offsets_changed',
    'source_physical_offsets_changed','previous_proposal_changed',
    'source_publication_age_changed','source_completed_K10_plan_changed',
    'in_place_prefix_buffer_update',
}


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def current_source_sha():
    return source_files_and_hash(ROOT)[1]


def flag_value(command,flag):
    require(command.count(flag)==1 and command.index(flag)+1<len(command),f'guard command lacks unique {flag}')
    return command[command.index(flag)+1]


def accept(*,raw_path,guard_dir,checkpoint,config):
    raw_path,guard_dir,checkpoint,config=map(lambda p:Path(p).resolve(strict=True),
        (raw_path,guard_dir,checkpoint,config))
    raw=read(raw_path);manifest=read(guard_dir/'manifest.json')
    require(raw.get('kind')=='motionkv_rtc_native_graph' and raw.get('passed') is True,
        'actual graph test did not pass')
    require(not raw.get('error') and not raw.get('input_validation_error') and raw.get('inputs_unchanged') is True,
        'graph test errored or lacks unchanged-input evidence')
    require(raw.get('conditioner_interface_version')==3 and raw.get('trained_fast_steps')==10
        and raw.get('tested_fast_steps')==10 and raw.get('engine_only_override') is False,
        'acceptance requires the trained joint-v3 K10 without solver override')
    require(raw.get('trained_prefix_max')==4 and raw.get('tested_prefix')==4,
        'acceptance requires actual H4 prefix coverage')
    bindings=raw.get('input_bindings',{})
    hashes=dict(checkpoint_sha256=sha(checkpoint),config_sha256=sha(config),source_sha256=current_source_sha())
    for key,path in [('checkpoint',checkpoint),('config',config)]:
        expected=dict(path=str(path),bytes=path.stat().st_size,sha256=hashes[key+'_sha256'])
        require(bindings.get(key)==expected and raw.get(key)==str(path)
            and raw.get(key+'_sha256')==expected['sha256'],f'graph {key} binding mismatch')
    require(bindings.get('source_sha256')==raw.get('source_sha256')==hashes['source_sha256']
        ==manifest.get('source_sha256'),'graph/guard/current source hash mismatch')
    command=manifest.get('command',[])
    require(isinstance(command,list) and all(isinstance(x,str) for x in command),'malformed guard command')
    require(any(Path(x).name=='check_motionkv_graph.py' for x in command),'guard did not run the actual graph checker')
    for flag,path in [('--checkpoint',checkpoint),('--deploy-config',config),('--output',raw_path)]:
        require(Path(flag_value(command,flag)).resolve()==path,f'guard command {flag} differs')
    require('--concurrent-slow' in command and '--fast-steps-override' not in command,
        'guard command lacks genuine background test or changes solver steps')
    cases=raw.get('cases',[])
    require(isinstance(cases,list) and cases,'graph has no parity cases')
    names=[case.get('name') for case in cases]
    require(len(set(names))==len(names) and REQUIRED_CASES<=set(names),'incomplete joint-v3 graph input coverage')
    errors=[case.get('max_abs_error') for case in cases]
    require(all(finite(error) and error>=0 for error in errors),'invalid graph error metrics')
    require(raw.get('initial_eager_finite') is True and raw.get('initial_graph_finite') is True
        and raw.get('caller_inputs_unchanged') is True and raw.get('rejects_prefix_beyond_training_range') is True,
        'graph finite/prefix/input ownership checks missing')
    stats=raw.get('graph_stats',{})
    require(stats.get('enabled') is True and stats.get('captures')==1 and stats.get('replays',0)>0,
        'no actual fixed-signature CUDA graph capture/replay')
    concurrent=raw.get('concurrent_slow',{})
    before=concurrent.get('fast_calls_before_slow_publication')
    require(concurrent.get('passed') is True and type(before) is int and before>0,
        'actual slow/fast concurrency did not pass')
    for key in ('producer_completion_event_ready','completed_cache_consumed','producer_stream_distinct','worker_closed'):
        require(concurrent.get(key) is True,'concurrent slow lifecycle missing '+key)
    require(concurrent.get('fast_captures_before')==concurrent.get('fast_captures_after')==1,
        'background refresh recaptured graph')
    begin,end=concurrent.get('slow_started_monotonic'),concurrent.get('slow_completed_monotonic')
    require(finite(begin) and finite(end) and end>begin,'invalid actual slow interval')
    calls=concurrent.get('calls',[])
    require(isinstance(calls,list) and calls,'missing actual fast calls during slow probe')
    require(sum(call.get('slow_unpublished_at_start') is True for call in calls)==before,
        'background overlap count differs from actual calls')
    require(any(call.get('slow_unpublished_at_start') is True and finite(call.get('started_monotonic'))
        and begin<=call['started_monotonic']<end for call in calls),'no fast call began while slow actually ran')
    tested,uuids=manifest.get('worker_gpu_uuids',[]),manifest.get('guard_gpu_uuids',[])
    require(tested and len(set(tested))==len(tested) and set(tested)<=set(uuids),'invalid tested GPU identities')
    require(len(uuids)==len(set(uuids))==8 and all(isinstance(x,str) and x.startswith('GPU-') for x in uuids),
        'guard must cover all eight adapter UUIDs')
    auth_path=Path(manifest['authorization']).resolve(strict=True);auth=read(auth_path)
    verify_support_guard(guard_dir,preflight=False,tested=tested,uuids=uuids,auth=auth,
        auth_sha=sha(auth_path),source_sha=hashes['source_sha256'])
    driver=manifest['driver']
    require(all(isinstance(driver.get(key),str) and driver[key] for key in ('host','boot_id')),
        'guard has no actual host/boot identity')
    evidence=[raw_path,guard_dir/'manifest.json',guard_dir/'status.json',
        guard_dir/'driver_postexit.json',guard_dir/'device_guard_events.json',auth_path]
    return dict(schema_version=1,kind=GRAPH_KIND,passed=True,created_at=utc(),
        host=driver['host'],boot_id=driver['boot_id'],**hashes,
        gpu_uuids_tested=tested,guard_gpu_uuids=uuids,
        graph_parity=dict(passed=True,cases=len(cases),max_abs_error=max(errors)),
        background_concurrency=dict(passed=True,fast_consumed_while_slow_running=True,
            fast_calls_before_slow_publication=before),
        evidence_files=[dict(path=str(path),sha256=sha(path)) for path in evidence],
        limitations=['actual graph parity and producer/consumer overlap only; no task quality or continuous 1x acceptance',
            'overlapping host intervals do not prove simultaneous GPU kernel execution'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','guard-dir','checkpoint','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'refusing to overwrite an acceptance receipt')
    receipt=accept(raw_path=args.raw,guard_dir=args.guard_dir,checkpoint=args.checkpoint,config=args.config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=True,output=str(args.output.resolve()),cases=receipt['graph_parity']['cases'])))

if __name__=='__main__':main()
