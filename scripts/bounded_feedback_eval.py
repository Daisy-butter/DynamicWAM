"""Frozen native 35x20 evaluation contract and final artifact acceptance."""
from pathlib import Path
import guarded_dualrate_run as runner
from guarded_dualrate_smoke import atomic,file_sha256
from dualrate_eval_worker import load_frozen_task


def prepared_contract(path,deploy):
    old=runner.read_json(path);root=Path(old['prepared'])
    if file_sha256(root/'frozen_manifest.json')!=old['prepared_sha256']:raise ValueError('frozen scenes changed')
    suite=runner.read_json(root/'suite_started.json')
    tasks=old['tasks']
    if len(tasks)!=35 or len(set(tasks))!=35 or set(tasks)!=set(suite['tasks']) or old['episodes']!=20:raise ValueError('complete native 35x20 required')
    for key in ('task_config','instruction_type'):
        if old[key]!=deploy['deployment'][key]:raise ValueError('native evaluation setting changed: '+key)
    contract=dict(prepared=str(root),prepared_sha256=old['prepared_sha256'],seed=suite['seed'],episodes=20,tasks=tasks)
    for task in tasks:load_frozen_task(contract,task)
    return contract


def aggregate(root,groups,contract,checkpoint_sha256):
    from video_artifacts import validate_video_artifacts
    reports=[];tasks={};evidence=[]
    for name,c in groups:
        guard=runner.read_json(root/(name+'_guard')/'status.json')
        if guard['state']!='completed' or guard.get('errors') or not guard['guard_closed']:raise ValueError('evaluation guard failed')
        expected={(task,ep) for task in c['deployment']['tasks'] for ep in range(20)}
        found=set()
        for task in c['deployment']['tasks']:
            refs=load_frozen_task(contract,task)
            for ep,(descriptor,_,_) in enumerate(refs):
                d=Path(c['output'])/task/f'episode_{ep:04d}';p=d/'report.json';r=runner.read_json(p)
                identity=(r['task'],r['episode_index'])
                if identity!=(task,ep) or identity in found or r['seed']!=descriptor['seed']:raise ValueError('scored scene identity mismatch')
                if r['checkpoint_sha256']!=checkpoint_sha256 or r['purpose']!='formal_native_evaluation':raise ValueError('diagnostic/mixed checkpoint result')
                if r['protocol']!='native_topp_dualrate_bounded_feedback_v10' or not r['clock_audit']['passed'] or not r['native_motion_unmodified']:raise ValueError('non-native result')
                if runner.read_json(d/'manifest.json')!=descriptor or r['action_limit']!=descriptor['native_action_limit']:raise ValueError('scene or native budget changed')
                if not validate_video_artifacts(r):raise ValueError('native MP4 acceptance failed')
                metric=runner.read_json(d/'metrics.json')
                if bool(metric['success'])!=bool(r['success']):raise ValueError('metric/report success mismatch')
                found.add(identity);reports.append(r)
                for f in (p,d/'metrics.json',d/'manifest.json',Path(r['video']),Path(r['video_timestamps'])):
                    evidence.append(dict(path=str(f),sha256=file_sha256(f)))
            rows=[r for r in reports if r['task']==task]
            tasks[task]=dict(completed=len(rows),successes=sum(bool(r['success']) for r in rows))
        if found!=expected:raise ValueError('incomplete or duplicate native task group')
    if set(tasks)!=set(contract['tasks']) or len(reports)!=700 or len({(r['task'],r['episode_index']) for r in reports})!=700:raise ValueError('700 unique complete native episodes required')
    result=dict(passed=True,state='completed',protocol='native_topp_dualrate_bounded_feedback_v10',
        planned=700,completed=700,successes=sum(bool(r['success']) for r in reports),tasks=tasks,
        checkpoint_sha256=checkpoint_sha256,prepared_sha256=contract['prepared_sha256'],evidence=evidence,
        native_mp4_fully_decoded=True,smoke_excluded=True)
    result['success_rate']=result['successes']/700
    atomic(root/'formal_eval_result.json',result)
    return result
