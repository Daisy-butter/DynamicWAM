"""CPU-only, one-shot not-before gate. Never claims CPU checks are GPU checks."""
from datetime import datetime
import json
from pathlib import Path
import time
from guarded_dualrate_smoke import atomic,file_sha256 as _path_sha256
import guarded_dualrate_run as runner


def file_sha256(path):
    return _path_sha256(Path(path))


def verify(receipt,repo,core,deploy):
    if receipt.get('passed') is not True or receipt.get('review_iterations',0)<3:
        raise ValueError('three reviewed CPU iterations are required')
    if receipt.get('formal_training_ready') is not False:
        raise ValueError('CPU evidence must not masquerade as GPU training acceptance')
    if receipt.get('source_contract')!=runner.source_contract(runner.frozen_source(repo)):
        raise ValueError('source changed after reviewed acceptance')
    for key,path in (('core_sha256',core),('deployment_sha256',deploy)):
        if receipt.get(key)!=file_sha256(path):raise ValueError('frozen configuration changed')
    config=json.loads(Path(core).read_text())
    if receipt.get('data_manifest_sha256')!=file_sha256(config['manifest']):raise ValueError('expert manifest changed')
    manifest=json.loads(Path(config['manifest']).read_text())
    index=Path(config['manifest']).parent/manifest['rows_file']
    if file_sha256(index)!=manifest['rows_sha256']:raise ValueError('expert episode/window index changed')
    if receipt.get('native_profile_sha256')!=file_sha256(config['native_profile']):raise ValueError('Full native profile changed')
    for item in receipt.get('evidence',[]):
        if file_sha256(item['path'])!=item['sha256']:raise ValueError('review evidence changed')
    if len(receipt.get('evidence',[]))<3:raise ValueError('missing review evidence')
    reviews=receipt.get('reviews',[])
    if len(reviews)!=3 or [r.get('iteration') for r in reviews]!=[1,2,3] or not all(r.get('passed') is True for r in reviews):
        raise ValueError('three explicit passing iteration records required')


def wait_and_validate(args,repo,root,core,deploy,*,clock=time.time,sleep=time.sleep):
    target=datetime.fromisoformat(args.not_before)
    if target.tzinfo is None:raise ValueError('not-before must specify Asia/Shanghai UTC offset')
    if target.isoformat()!='2026-11-03T02:00:00+08:00':raise ValueError('unauthorized scheduled start')
    receipt=json.loads(Path(args.cpu_acceptance).read_text())
    verify(receipt,repo,args.core,args.deployment)
    if core.get('bounded_feedback')!=deploy.get('bounded_feedback') or not core.get('bounded_feedback'):
        raise ValueError('train and deploy feedback contract mismatch')
    deadline=target.timestamp()
    atomic(root/'status.json',dict(state='waiting_not_before',not_before=target.isoformat(),
        formal_training_started=False,gpu_context_created=False,no_retry=True))
    while clock()<deadline:sleep(min(30.,deadline-clock()))
    try:
        verify(receipt,repo,args.core,args.deployment)
        atomic(root/'status.json',dict(state='gpu_acceptance_pending',not_before=target.isoformat(),
            formal_training_started=False,gpu_context_created=False,no_retry=True,
            next='existing guarded pipeline checks idle GPUs then actual deployment/DDP/reload'))
    except BaseException as exc:
        atomic(root/'status.json',dict(state='blocked',error=repr(exc),formal_training_started=False,no_retry=True))
        raise
