#!/usr/bin/env python3
"""Eight independent native DOMINO tasks. Fail fast, never retry a worker."""
from pathlib import Path
import argparse, collections, hashlib, json, os, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from supervise_motionkv_training import atomic_json, file_digest, utc
from supervise_motionkv_eval import proc_gpu_inventory
POLICY='dynamicwam.runtime.ciwam.adapters.domino.deploy_policy_native_causal_spatial'
SMOKE=['place_object_basket','place_container_plate','handover_block','hanging_mug','put_bottles_dustbin','shake_bottle','place_empty_cup','place_fan']

def validate_task(run,task,count,decode=True):
    paths=list((run/'eval_result'/task).glob('*/*/*/*/_episodes_detail.json'))
    if len(paths)!=1:raise RuntimeError(f'{task}: require one native report, found {len(paths)}')
    path=paths[0];episodes=json.loads(path.read_text())
    if not isinstance(episodes,list) or len(episodes)!=count:raise RuntimeError(f'{task}: incomplete native report')
    for name in ['_metrics.json','_metrics_report.txt','_result.txt']:
        if not (path.parent/name).is_file():raise RuntimeError('missing native artifact '+name)
    videos=sorted(path.parent.glob('episode*.mp4'))
    if len(videos)!=count:raise RuntimeError(f'{task}: expected {count} native MP4s, found {len(videos)}')
    receipts=[]
    for video in videos:
        if decode:
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=600)
        receipts.append(dict(path=str(video),bytes=video.stat().st_size,sha256=file_digest(video),full_decode=decode))
    from dynamicwam.evaluation.run_domino_parallel_eval import _compact_episodes, _policy_summaries
    compact=_compact_episodes(episodes,telemetry=_policy_summaries(run/'logs'/f'{task}.log'))
    if len(compact)!=count:raise RuntimeError('native metric record mismatch')
    if not (run/'receipts'/task/'closed.json').is_file():raise RuntimeError('policy did not close')
    return dict(task=task,count=count,report=str(path),episodes=compact,videos=receipts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--episodes',type=int)
    a=p.parse_args();run=Path(a.run);run.mkdir(parents=True,exist_ok=False)
    from dynamicwam.inference.causal_spatial_config import load_deployment
    _,profile,_=load_deployment(ROOT/'configs/causal_spatial/deploy.yaml')
    tasks=SMOKE if a.smoke else profile.benchmark_config()['tasks'];count=a.episodes or (2 if a.smoke else 20)
    if len(tasks)!= (8 if a.smoke else 35):raise RuntimeError('task count mismatch')
    ids=a.gpus.split(',');inventory={r['uuid']:r for r in proc_gpu_inventory()}
    if len(ids)!=8 or len(set(ids))!=8:raise RuntimeError('eight distinct GPUs required')
    for directory in ['configs','logs','receipts']:(run/directory).mkdir()
    (run/'manifest.json').write_text(json.dumps(dict(protocol='native_domino_sync_causal_h2r2_v1',continuous=False,wall_rtc=False,
        H=2,R=2,K=10,chunk=16,nominal_action_dt=.16,actual_action_time='original_TOPP_variable',slow_refresh_actions=16,
        task_list=tasks,episodes_per_task=count,start_seed=1610026 if a.smoke else 100000,
        checkpoint=a.checkpoint,checkpoint_sha256=file_digest(a.checkpoint),gpu_uuids=ids,
        seed_selection='original_DOMINO_expert_feasibility_filter',created_at=utc()),indent=2))
    pending=collections.deque(tasks);free=collections.deque(ids);running={};completed=[]
    status=dict(state='running',started_at=utc(),tasks=tasks,episodes_per_task=count,completed=[],running={})
    def save():
        status.update(updated_at=utc(),completed=completed,running={t:dict(pid=r['p'].pid,gpu=r['gpu']) for t,r in running.items()})
        atomic_json(run/'status.json',status)
    def stop(sig,frame):raise RuntimeError('suite interrupted '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or running:
            while pending and free:
                task=pending.popleft();gpu=free.popleft()
                config=dict(task_name=task,task_config='demo_clean_dynamic',policy_name=POLICY,instruction_type='unseen',
                    seed=0,start_seed=1610026 if a.smoke else 100000,episode_num=count,eval_video_log=True,
                    ckpt_setting='causal-v4-final20k-native-h2r2',eval_output_root=str(run/'eval_result'),
                    causal_spatial_deploy_config=str(ROOT/'configs/causal_spatial/deploy.yaml'),causal_spatial_checkpoint=a.checkpoint,
                    causal_spatial_cuda_graph=True,policy_sampling_seed=20261026,
                    domino_root=str(ROOT/'external/DOMINO'),receipt_dir=str(run/'receipts'/task))
                cfg=run/'configs'/f'{task}.json';atomic_json(cfg,config)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,DYNAMICWAM_GPU_PCI=inventory[gpu]['pci_bus_id'],
                    DYNAMICWAM_SIM_CUDA='0',DYNAMICWAM_INFER_CUDA='0',WAM_NATIVE_GUARD_PID=str(os.getppid()),
                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                    PYTHONDONTWRITEBYTECODE='1')
                env['PYTHONPATH']=':'.join([str(ROOT/'src'),str(ROOT/'src/dynamicwam/runtime'),str(ROOT/'external/flash_attn_runtime'),str(ROOT/'external/curobo/src'),str(ROOT/'external/DOMINO/script')])
                log=(run/'logs'/f'{task}.log').open('x')
                cmd=[str(ROOT/'external/robotwin/bin/python'),'-u',str(ROOT/'scripts/eval_causal_native_task.py'),'--config',str(cfg)]
                # Inherit the suite SID: the guarded parent owns every descendant.
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[task]=dict(p=child,gpu=gpu,log=log,started=time.monotonic());save()
                time.sleep(.25)
            for task,record in list(running.items()):
                code=record['p'].poll()
                if code is None:
                    if time.monotonic()-record['started']>86400:raise TimeoutError(task)
                    continue
                record['log'].close()
                if code:raise RuntimeError(f'{task} failed exit {code}; no retry')
                audit=validate_task(run,task,count);atomic_json(run/'receipts'/task/'artifact_audit.json',audit)
                completed.append(task);free.append(record['gpu']);del running[task];save()
            if pending or running:time.sleep(2)
        status.update(state='completed',passed=True,finished_at=utc());save()
        rows=[]
        for task in tasks:rows.extend(json.loads((run/'receipts'/task/'artifact_audit.json').read_text())['episodes'])
        if any(r['manipulation_score'] is None for r in rows):raise RuntimeError('missing native MS')
        atomic_json(run/'aggregate.json',dict(complete=True,tasks=len(tasks),episodes=rows,total=len(rows),
            successes=sum(r['success'] for r in rows),sr_percent=100*sum(r['success'] for r in rows)/len(rows),
            ms_mean=sum(r['manipulation_score'] for r in rows)/len(rows)))
    except BaseException:
        status.update(state='failed_no_retry',passed=False,error=traceback.format_exc());save();raise
    finally:
        for record in running.values():
            if record['p'].poll() is None:record['p'].terminate()
        for record in running.values():
            try:record['p'].wait(timeout=120)
            except subprocess.TimeoutExpired:pass  # Guarded supervisor performs final owned-tree cleanup.
            record['log'].close()

if __name__=='__main__':main()
