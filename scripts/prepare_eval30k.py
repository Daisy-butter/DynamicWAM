"""CPU-only transfer/audit; does not access source training processes or GPUs."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess

ROOT = Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
REPO = ROOT / 'repo'
OUT = ROOT / 'checkpoint30k'
SOURCE = '/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/joint_motionkv_v33_stage3_mb8_20261018/exports'
GRID_SOURCE = '/SSD_DISK/users/wuruihan/DynamicWAM/outputs/data/joint_motionkv_physical_grid_20261015'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

old = json.loads((ROOT/'outputs/joint20k_35x20_04_scaled025/formal_guard/status.json').read_text())
assert all(old.get(k) is True for k in ('guard_workers_exited','guard_all_handles_closed','driver_exit_clean'))
assert shutil.disk_usage(ROOT).free > 20 * 1024**3
assert not OUT.exists(), 'Preserve previous attempts'
OUT.mkdir()
grid = OUT/'physical_grid_identity'
(grid/'future_latents').mkdir(parents=True)
files = [(f'{SOURCE}/{n}', OUT/n) for n in ('multirate_step_30000.pt','multirate_fast_step_30000.pt')]
files += [(f'{GRID_SOURCE}/manifest.json', grid/'manifest.json'),
          (f'{GRID_SOURCE}/future_latents/manifest.json', grid/'future_latents/manifest.json')]
receipts=[]
for source, target in files:
    before=subprocess.check_output(['ssh','-o','BatchMode=yes','wuruihan@10.12.208.50',
        'nice -n 15 ionice -c 3 sha256sum '+source],text=True).split()[0]
    subprocess.run(['rsync','-a','--bwlimit=80000','--rsync-path=nice -n 15 ionice -c 3 rsync',
        '-e','ssh -o BatchMode=yes -o ConnectTimeout=12',
        'wuruihan@10.12.208.50:'+source,str(target)],check=True)
    assert sha(target)==before, 'Source/destination hash mismatch'
    receipts.append(dict(source=source,destination=str(target),sha256=before,bytes=target.stat().st_size))
    print('verified',target.name,flush=True)
(OUT/'transfer_receipt.json').write_text(json.dumps(dict(source_host='10.12.208.50',destination_host='10.12.208.100',
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=receipts,source_training_unchanged=True),indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=[str(REPO/'external/robotwin/bin/python'),str(REPO/'scripts/audit_stage3_eval_checkpoint.py'),
    '--full',str(OUT/'multirate_step_30000.pt'),'--fast',str(OUT/'multirate_fast_step_30000.pt'),
    '--profile',str(REPO/'configs/multirate/joint_motionkv_v33_stage3_mb8_20261018.yaml'),
    '--physical-grid',str(grid),'--output',str(OUT),'--step','30000']
with (OUT/'audit.log').open('x') as log:
    result=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0, 'See checkpoint30k/audit.log; no GPU launch'
new=json.loads((OUT/'checkpoint_audit.json').read_text())
prior=json.loads((ROOT/'checkpoint/checkpoint_audit.json').read_text())
for k in ('contract','action_stats_sha256','motion_statistics_sha256'):
    assert new[k]==prior[k], 'Checkpoint contract changed: '+k
print(json.dumps(dict(cpu_audit_passed=True,step=new['step'],full_sha256=new['full']['sha256'],
    strict_full_reload=new['strict_full_reload'],strict_fast_reload=new['strict_fast_reload'],
    same_20k_contract_and_normalization=True)),flush=True)
