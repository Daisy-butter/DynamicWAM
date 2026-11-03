"""Revalidate completed preflight evidence after a checkpoint-loader-only repair."""
import ast
import hashlib
import subprocess
from pathlib import Path
import guarded_dualrate_run as runner
from guarded_dualrate_smoke import file_sha256,atomic,live_owned_members
from dualrate_training_sequence import phase_completed,PHASES


def inactive_loader_equivalent(before,after):
    def normalized(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='load_trained_state']
        child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='child')
        matches=[n for n in ast.walk(child) if isinstance(n,ast.If) and ast.unparse(n.test)=="config.get('checkpoint')"]
        if len(matches)!=1:raise ValueError('exact trained-checkpoint branch required')
        matches[0].body=[ast.Pass()]
        return ast.dump(tree,include_attributes=False)
    return normalized(before)==normalized(after)


def validate_reuse(repo,prior,base,root):
    prior=Path(prior).resolve(strict=True);root=Path(root)
    if not prior.is_relative_to(Path(base['storage_root']).resolve()) or prior==root:
        raise ValueError('reuse requires a separate immutable SSD run')
    seq=runner.read_json(prior/'smoke_sequence/status.json');old=seq['source'];now=runner.frozen_source(repo)
    if seq['state']!='completed' or seq.get('errors') or old['source_sha256']!=now['source_sha256']:
        raise ValueError('numerical model/trainer source changed or prior smoke incomplete')
    if live_owned_members(seq['controller']['pid']):raise ValueError('prior smoke still owns live workers')
    commit=old['commit'];paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'scripts'],cwd=repo,text=True).splitlines()
    control={'scripts/bounded_feedback_pipeline.py','scripts/bounded_feedback_manual_entry.py'}
    protected=[]
    for name in paths:
        if not name.endswith(('.py','.sh')) or name in control:continue
        before=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo)
        after=(repo/name).read_bytes()
        if name=='scripts/native_v9_slow_service.py':
            if not inactive_loader_equivalent(before.decode(),after.decode()):raise ValueError('non-loader slow behavior changed')
        elif before!=after:raise ValueError('protected preflight script changed: '+name)
        protected.append(dict(path=name,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest()))
    evidence=[]
    def bind(path):
        evidence.append(dict(path=str(path),sha256=file_sha256(path)))
    def clean_guard(g):
        status=runner.read_json(g/'status.json')
        if status['state']!='completed' or status.get('errors') or not status.get('guard_closed'):raise ValueError('prior guard not clean')
        if live_owned_members(status['supervisor']['pid']):raise ValueError('prior guard still active')
        events=runner.read_json(g/'guard_events.json')
        if sum(e['event']=='after_close' for e in events)!=8:raise ValueError('prior GPU descriptors not closed')
        exited=runner.read_json(g/'owned_exit.json');kernel=runner.read_json(g/'kernel_after.json')
        if exited['returncode']!=0 or exited['remaining'] or live_owned_members(exited['owned_pgid']):raise ValueError('prior workers not cleanly exited')
        if not kernel['unchanged'] or not kernel['same_boot'] or kernel['added']:raise ValueError('prior kernel changed')
        if not runner.read_json(g/'postflight_idle.json')['passed']:raise ValueError('prior postflight failed')
        for name in ('status.json','guard_events.json','source.json','config.frozen.json','owned_exit.json','kernel_after.json','postflight_idle.json'):bind(g/name)
    tasks=set();count=0
    for i in range(4):
        d=prior/f'collect_{i}';c=runner.read_json(prior/f'collect_{i}.json')
        if c.get('checkpoint') or c['purpose']!='native_training_collection':raise ValueError('collection must use inactive trained loader')
        clean_guard(prior/f'collect_{i}_guard')
        if not runner.read_json(d/'deployment_acceptance.json')['passed'] or not runner.read_json(d/'replay_acceptance.json')['passed']:
            raise ValueError('incomplete native collection')
        for name in ('deployment_acceptance.json','replay_acceptance.json','model_acceptance.json'):bind(d/name)
        reports=list(d.glob('*/episode_*/report.json'))
        if len(reports)!=len(c['deployment']['tasks']):raise ValueError('collection report count mismatch')
        for p in reports:
            r=runner.read_json(p)
            if not r['clock_audit']['passed'] or r['protocol']!='native_topp_dualrate_bounded_feedback_v10':raise ValueError('wrong native protocol')
            if file_sha256(Path(r['video']))!=r['video_sha256'] or file_sha256(Path(r['video_timestamps']))!=r['video_timestamps_sha256']:raise ValueError('native video changed')
            tasks.add(r['task']);count+=1;bind(p);bind(Path(r['video']));bind(Path(r['video_timestamps']))
    if len(tasks)!=35 or count!=35:raise ValueError('all 35 training tasks required')
    manifest=prior/'dataset/train.json';data=runner.read_json(manifest);accept=runner.read_json(prior/'dataset/acceptance.json')
    if not accept['passed'] or file_sha256(manifest)!=accept['sha256'] or data['expert_manifest']!=base['manifest']:
        raise ValueError('native replay/expert identity changed')
    if file_sha256(Path(base['manifest']))!=data['expert_manifest_sha256']:raise ValueError('expert subset changed')
    for row in data['native_replay_samples']:
        if file_sha256(Path(row['path']))!=row['sha256']:raise ValueError('native replay bytes changed')
    for p in (manifest,prior/'dataset/acceptance.json',prior/'smoke_sequence/status.json'):bind(p)
    configs=[]
    for phase,domains in zip(PHASES,(['expert_bootstrap'],['native_replay'],['expert_bootstrap','native_replay'])):
        c=runner.read_json(prior/f'smoke_sequence_{phase}.json')
        expected=dict(base,manifest=str(manifest),run_kind='distributed_training_smoke',phase=phase,
            sequence_id=c['sequence_id'],output=c['output'],steps=4,query_age_seconds=base['query_age_seconds'],
            training_domains=domains,log_every=1,
            data_distribution='expert bootstrap + actual native Full-anchored feedback collection replay, no retimed labels')
        for key in ('stage_init_checkpoint','stage_init_checkpoint_sha256','acceptance_receipt','acceptance_receipt_sha256','resume'):expected.pop(key,None)
        if runner.profile_sha256(c)!=runner.profile_sha256(expected):raise ValueError('accepted training recipe changed')
        clean_guard(prior/'smoke_sequence'/phase)
        _,_,result,reload=phase_completed(Path(c['output']),c)
        if result['identity']['source_sha256']!=now['source_sha256'] or result['identity']['manifest_sha256']!=file_sha256(manifest):raise ValueError('trained model/data differ')
        if not reload['equal_model'] or not reload['equal_optimizer']:raise ValueError('non-exact restore')
        for p in (Path(c['output'])/'result.json',Path(c['output'])/'reload.json',prior/f'smoke_sequence_{phase}.json'):bind(p)
        configs.append(c)
    proof=dict(passed=True,prior_run=str(prior),prior_source=old,current_source=now,protected_scripts=protected,
        numerical_source_byte_identical=True,collection_loader_inactive=True,native_tasks=35,
        native_replay_samples=len(data['native_replay_samples']),strict_eight_rank_phases=3,evidence=evidence,
        fresh_trained_deployment_required=True,formal_smoke_weights_forbidden=True)
    path=root/'reused_preflight.json';atomic(path,proof);bind(path)
    return configs,manifest,evidence,prior
