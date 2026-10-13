#!/usr/bin/env python3
"""Bounded application-level CUDA probe. No reset, recovery, or driver changes.

One fresh child per selected UUID; fail on any new driver message, timeout,
numerical failure or missing GPU. Results describe this test, not driver repair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

PATTERN = re.compile(r"NVRM.*(?:assert|error|Xid|RmInitAdapter|IOVAS|iovaspace)|Xid\s*\(|fallen off the bus|GPU.*(?:fault|error)", re.I)

def driver():
    log=subprocess.check_output(['dmesg','--color=never'],text=True,timeout=15)
    lines=[s.strip() for s in log.splitlines() if PATTERN.search(s)]
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        latest_driver_anomaly_sha256=sha(lines[-1]) if lines else None,
        driver_anomalies_sha256=sha('\n'.join(lines)),anomaly_count=len(lines),
        latest_driver_anomaly=lines[-1] if lines else None)

def child():
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.device_count()==1, 'probe must see exactly one CUDA device'
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.02,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261014)
    a=torch.randn(256,256)/16
    b=torch.randn(256,256)/16
    ga,gb=a.cuda(),b.cuda()
    actual=ga@gb
    torch.testing.assert_close(actual.cpu(),a@b,atol=2e-6,rtol=2e-4)
    low=ga.bfloat16()@gb.bfloat16()
    reference=a.bfloat16().float()@b.bfloat16().float()
    torch.testing.assert_close(low.float().cpu(),reference,atol=.002,rtol=.02)
    ga.requires_grad_(True)
    loss=(ga@gb).square().mean()
    loss.backward()
    assert torch.isfinite(ga.grad).all().item()
    torch.cuda.synchronize()
    allocated=torch.cuda.max_memory_allocated()
    assert allocated < 64*1024**2, 'tensor allocation exceeded diagnostic bound'
    print(json.dumps(dict(passed=True,torch=torch.__version__,device=torch.cuda.get_device_name(0),
        fp32_max_error=float((actual.cpu()-a@b).abs().max()),loss=float(loss.detach()),
        max_tensor_allocated_bytes=allocated,max_reserved_bytes=torch.cuda.max_memory_reserved(),
        note='64MiB tensor bound excludes CUDA context and library memory')))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuids',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,default=90)
    p.add_argument('--child',action='store_true')
    args=p.parse_args()
    if args.child:
        child();return
    root=Path('/SSD_DISK/users/wuruihan/DynamicWAM')
    out=args.output.resolve()
    if not out.is_relative_to(root) or out.exists():
        raise ValueError('fresh output inside project SSD required')
    uuids=args.gpu_uuids.split(',')
    if not uuids or len(set(uuids))!=len(uuids) or any(not u.startswith('GPU-') for u in uuids):
        raise ValueError('unique full GPU UUIDs required')
    out.parent.mkdir(parents=True,exist_ok=True)
    before=driver()
    report=dict(passed=False,host=before['host'],boot_id=before['boot_id'],gpu_uuids=uuids,
        started_at=datetime.now(timezone.utc).isoformat(),before=before,cases=[],driver_resolved=False,
        scope='user-authorized application CUDA diagnostic; no driver repair or health guarantee')
    try:
        for uuid in uuids:
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid,PYTORCH_NVML_BASED_CUDA_CHECK='1',
                PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                OMP_NUM_THREADS='2',CUDA_MODULE_LOADING='LAZY')
            for key,path in dict(TMPDIR=root/'tmp',TMP=root/'tmp',TEMP=root/'tmp',
                XDG_CACHE_HOME=root/'caches/motionkv/xdg',TORCH_HOME=root/'caches/motionkv/torch',
                CUDA_CACHE_PATH=root/'caches/motionkv/cuda',TRITON_CACHE_DIR=root/'caches/motionkv/triton',
                TORCHINDUCTOR_CACHE_DIR=root/'caches/motionkv/inductor').items():
                path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
            print(json.dumps(dict(stage='probe_start',uuid=uuid)),flush=True)
            result=subprocess.run([sys.executable,__file__,'--child','--gpu-uuids',uuid,'--output',str(out)],
                env=env,text=True,capture_output=True,timeout=args.timeout_seconds)
            item=dict(uuid=uuid,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
            report['cases'].append(item)
            if result.returncode:
                raise RuntimeError('CUDA diagnostic child failed')
            item['result']=json.loads(result.stdout.strip().splitlines()[-1])
            after=driver()
            report['after']=after
            if any(before[k]!=after[k] for k in ('boot_id','driver_anomalies_sha256','latest_driver_anomaly_sha256')):
                raise RuntimeError('new driver anomaly: stop, no further GPU probes')
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=30).splitlines()
            if len(rows)!=8 or any(u not in rows for u in uuids):
                raise RuntimeError('GPU visibility changed during diagnostic')
            print(json.dumps(dict(stage='probe_passed',uuid=uuid)),flush=True)
        report.update(passed=True,latest_driver_anomaly_sha256=before['latest_driver_anomaly_sha256'],
            driver_anomalies_sha256=before['driver_anomalies_sha256'])
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=report['passed'],output=str(out))),flush=True)

if __name__=='__main__':main()
