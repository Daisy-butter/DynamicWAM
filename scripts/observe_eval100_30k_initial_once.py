"""Bounded initial acceptance only; exits on stable formal startup or any failure.

No GPU/kernel queries, retries, job restarts, model edits or post-acceptance polling.
The independent evaluation supervisor retains necessary lifecycle cleanup.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

root=Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
run=root/'outputs/joint30k_35x20_01_scaled025'
out=root/'launch30k/initial_observer_status.json'
deadline=time.monotonic()+7200
read=lambda p:json.loads(p.read_text())

def save(state,**extra):
    value=dict(state=state,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=os.getpid(),bounded_seconds=7200,no_gpu_queries=True,no_retry=True,
        no_post_acceptance_polling=True,**extra)
    temporary=out.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(out)

save('waiting_for_smoke_and_formal_start')
try:
    while time.monotonic()<deadline:
        controller=read(root/'launch30k/status.json')
        p=run/'pipeline/pipeline_status.json'
        pipeline=read(p) if p.exists() else {}
        if controller.get('state')=='failed' or pipeline.get('state') in ('failed','interrupted'):
            save('stopped_after_failure',controller_state=controller.get('state'),
                pipeline_state=pipeline.get('state'),error=controller.get('error') or pipeline.get('error'))
            break
        if pipeline.get('state')=='completed':
            save('evaluation_already_terminal',pipeline_state='completed')
            break
        if pipeline.get('state')=='formal_running':
            guard=read(run/'formal_guard/status.json')
            reports=list((run/'formal35x20').glob('*/episode_0000/report.json'))
            if guard.get('state')=='running' and guard.get('startup_health_window_closed') and len(reports)==8:
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
                command=[str(root/'repo/external/robotwin/bin/python'),str(root/'tools/accept_eval100_30k_initial.py')]
                with (root/'launch30k/initial_acceptance.log').open('x') as log:
                    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                save('accepted_and_stopped' if result.returncode==0 else 'acceptance_failed_and_stopped',
                    returncode=result.returncode,receipt=str(root/'launch30k/initial_stability.json'))
                break
        time.sleep(min(30,max(0,deadline-time.monotonic())))
    else:
        save('time_limit_reached_and_stopped',evaluation_left_to_its_own_supervisor=True)
except BaseException as exc:
    save('observer_failed_and_stopped',error=repr(exc),evaluation_left_to_its_own_supervisor=True)
    raise
