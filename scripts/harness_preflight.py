#!/usr/bin/env python3
"""Read-only host/GPU checks and SSD storage gate; no device recovery operations."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
SSD=pathlib.Path(os.environ.get('WAM_SSD_ROOT','/SSD_DISK/users/wuruihan')).resolve()

def require_ssd(path):
    p=pathlib.Path(path).expanduser().resolve()
    if not SSD.is_dir() or SSD.name!='wuruihan' or not p.is_relative_to(SSD):
        raise ValueError(f'artifact must resolve inside verified SSD user directory {SSD}: {p}')
    anchor=p
    while not anchor.exists(): anchor=anchor.parent
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(anchor),'-o','TARGET,SOURCE,FSTYPE'],text=True))['filesystems'][0]
    if mount['target']=='/' or anchor.stat().st_dev != SSD.stat().st_dev:
        raise ValueError(f'not a separate SSD filesystem: {mount}')
    if shutil.disk_usage(SSD).free < 10*1024**3:
        raise RuntimeError('less than 10 GiB free; refuse starting new output')
    return str(p)

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--gpu',required=True); a=p.parse_args()
    require_ssd(a.output)
    if pathlib.Path(a.output).exists() and any(pathlib.Path(a.output).iterdir()):
        raise FileExistsError('launcher needs a fresh run directory; existing evidence is immutable')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    devices=[r.split(', ') for r in rows]
    if len(devices)!=8: raise RuntimeError('expected 8 GPUs on Route1 host; stop on missing devices')
    selected=[r for r in devices if r[1]==a.gpu]
    if len(selected)!=1 or int(selected[0][3])>100: raise RuntimeError('selected UUID missing or GPU already occupied')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if a.gpu in apps: raise RuntimeError('selected GPU has an active compute process')
    for pth in [ROOT/'external/checkpoints/DynamicWAM_full.pt',ROOT/'outputs']: require_ssd(pth)
    if not (ROOT/'harness.md').is_file(): raise RuntimeError('harness.md missing')
    for key in ('TMPDIR','TMP','TEMP','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME'):
        if not os.environ.get(key): raise RuntimeError(f'{key} must point to SSD')
        require_ssd(os.environ[key])
    model_dirs=['src/dynamicwam/models','src/dynamicwam/inference','src/dynamicwam/runtime/ciwam']
    policy={str(f.relative_to(ROOT)):sha(f) for d in model_dirs for f in (ROOT/d).rglob('*.py')}
    record={'time_unix_s':time.time(),'hostname':os.uname().nodename,'harness_sha256':sha(ROOT/'harness.md'),'ssd_root':str(SSD),'ssd_free_bytes':shutil.disk_usage(SSD).free,'ssd_free_inodes':os.statvfs(SSD).f_favail,'gpus':devices,'selected_uuid':a.gpu,'compute_apps':apps,'policy_source_sha256':policy,'head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True),'gpu_operations':'read-only'}
    out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'gpu':a.gpu,'ssd_root':str(SSD),'harness_sha256':record['harness_sha256']}))
if __name__=='__main__': main()
