"""Manual one-shot CPU preflight and detached guarded pipeline; no timer."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
repo=Path('/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/native-v9-residual-async-20261101')
base=Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/bounded_feedback_v10_20261102')
launcher=Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/launchers/bounded_feedback_v10_20261102')
sys.path[:0]=[str(repo/'scripts'),str(repo/'src')]
from guarded_dualrate_smoke import atomic,file_sha256,proc_identity
from bounded_feedback_launch import verify


def check():
    receipt=base/'prep/cpu_acceptance_recover03.json';r=json.loads(receipt.read_text())
    core=repo/'configs/bounded_feedback_core.json';deploy=repo/'configs/bounded_feedback_deploy.json'
    verify(r,repo,core,deploy)
    c=json.loads(core.read_text());d=json.loads(deploy.read_text())
    differences={k for k in c.keys()&d.keys() if c[k]!=d[k]}
    if differences!={'run_kind','output'}:raise RuntimeError('unexpected train/deploy configuration divergence: '+str(differences))
    checkpoint=Path(c['init_checkpoint']).resolve(strict=True)
    if checkpoint.name!='DynamicWAM_full.pt' or file_sha256(checkpoint)!=c['init_checkpoint_sha256']:
        raise RuntimeError('official Full initialization identity mismatch')
    if any(c.get(k) for k in ('resume','stage_init_checkpoint','checkpoint')):
        raise RuntimeError('formal warmup must not inherit smoke weights or resume')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if commit!=r['git_commit']:raise RuntimeError('manual launcher commit changed')
    proof=dict(passed=True,start_mode='manual',timer_present=False,formal_training_started=False,
        git_commit=commit,source_contract=r['source_contract'],init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=c['init_checkpoint_sha256'],train_deploy_shared_settings_equal=True,
        allowed_differences=sorted(differences),fast_lr=c['fast_lr'],slow_lr=c['slow_lr'],
        global_batch=c['global_batch_size'],steps=[500,1500,3000],manifest=c['manifest'],
        fast_steps=c['runtime']['sample_steps'],native_topp=True,fast_commands=1,slow_commands=4,
        gpu_acceptance_pending=True,cpu_acceptance_sha256=file_sha256(receipt))
    atomic(launcher/'last_preflight.json',proof)
    return proof,receipt,core,deploy


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    p.add_argument('--run-id',default='manual03')
    p.add_argument('--reuse-preflight')
    g.add_argument('--check',action='store_true');g.add_argument('--start',action='store_true');g.add_argument('--status',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'manual[0-9]{2}',a.run_id):raise ValueError('explicit manualNN run identity required')
    root=base/a.run_id;run_launcher=launcher/a.run_id
    if a.status:
        path=root/'status.json';print(path.read_text() if path.exists() else json.dumps(dict(state='not_started',timer_present=False)));return
    proof,receipt,core,deploy=check()
    if a.check:print(json.dumps(proof));return
    if root.exists():raise RuntimeError('manual01 already exists; no overwrite or automatic retry')
    gpus=['GPU-dd833f16-f4e8-f4b1-6a97-3f59083d487b','GPU-69b57014-610e-23cd-89e5-f87ff01350c9',
        'GPU-b7e8a23e-7e71-5cb2-7305-15fc28a22473','GPU-0da59dce-7375-6853-16d5-4b2c2f6b8398',
        'GPU-b6c9020a-ae6b-6b50-abc7-11724d2d055f','GPU-985f4c3a-9584-4425-f00f-689f33203feb',
        'GPU-37d3f015-545d-6b50-5679-4deba63249fe','GPU-e212b019-c5c9-be83-28ea-d4aa23c248ba']
    cmd=[sys.executable,str(repo/'scripts/bounded_feedback_pipeline.py'),'--manual-start','--root',str(root),
        '--core',str(core),'--deployment',str(deploy),'--gpus',','.join(gpus),'--cpu-acceptance',str(receipt),
        '--prepared-eval','/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/native_v7_final_eval_20261031/run01/formal.json']
    if a.reuse_preflight:cmd.extend(['--reuse-preflight',str(Path(a.reuse_preflight).resolve(strict=True))])
    # O_EXCL is the one-shot reservation; never recover/retry a partial launch.
    run_launcher.mkdir(exist_ok=False)
    with (run_launcher/'launch.json').open('x') as identity:
        with (run_launcher/'controller.log').open('x') as log:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src'))
            child=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        state=dict(controller=proc_identity(child.pid),argv=cmd,preflight=proof,no_retry=True,start_mode='manual')
        json.dump(state,identity,indent=2)
    deadline=time.monotonic()+30
    while not (root/'status.json').exists() and time.monotonic()<deadline:
        if child.poll() is not None:raise RuntimeError('pipeline exited before initial status; inspect controller.log, no retry')
        time.sleep(.25)
    print(json.dumps(dict(controller_pid=child.pid,status_file=str(root/'status.json'),log=str(run_launcher/'controller.log'),
        formal_training_started=False,next='GPU idle/native replay/three-phase restore gates precede formal training')))


if __name__=='__main__':main()
