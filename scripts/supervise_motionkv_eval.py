#!/usr/bin/env python3
"""Bounded-startup, durable evaluator supervisor; no driver/device recovery."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import traceback

from supervise_motionkv_training import (atomic_json, driver_evidence, driver_identity,
    file_digest, finish_device_guard, gpu_preflight, inside_ssd, load_device_guard,
    register_run_processes, source_files_and_hash, utc, verify_ssd)

ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIFETIME_MODE = 'fds_before_telemetry_v1'


def record_phase(run, phases, name, **values):
    phases.append(dict(phase=name, at=utc(), monotonic=time.monotonic(), **values))
    atomic_json(run / 'driver_phases.json', phases)


def driver_fence(run, phases, status, phase, baseline):
    """Fail closed against the original authorization; never advance its baseline."""
    status['launch_phase'] = phase
    current = driver_evidence()
    before, after = driver_identity(baseline), driver_identity(current)
    keys = ('host', 'boot_id', 'anomaly_count', 'driver_anomalies_sha256',
            'latest_driver_anomaly_sha256')
    record_phase(run, phases, phase, driver=current, baseline_identity=before,
        delta=dict(changed_identity_fields=[key for key, old, new in zip(keys, before, after) if old != new],
            anomaly_count=current['anomaly_count'] - baseline['anomaly_count'],
            new_dmesg_tail_lines=[line for line in current.get('dmesg_tail', [])
                if line not in baseline.get('dmesg_tail', [])]), passed=after == before)
    if after != before:
        raise RuntimeError('driver evidence changed at launch fence: ' + phase)


def proc_gpu_inventory():
    """Validate UUID/device-node mapping without NVML, CUDA or device opens."""
    devices, minors = {}, set()
    for info in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        uuid, minor = fields.get('GPU UUID', '').strip(), fields.get('Device Minor', '').strip()
        if not uuid.startswith('GPU-') or uuid in devices or not minor.isdecimal() or int(minor) in minors:
            raise RuntimeError('ambiguous proc UUID/minor inventory')
        node = Path('/dev') / ('nvidia' + minor)
        value = node.stat()
        if (not stat.S_ISCHR(value.st_mode) or os.major(value.st_rdev) != 195
                or os.minor(value.st_rdev) != int(minor)):
            raise RuntimeError('unexpected NVIDIA character device in proc inventory')
        minors.add(int(minor))
        devices[uuid] = dict(uuid=uuid, pci_bus_id=info.parent.name, minor=int(minor), path=str(node))
    if len(devices) != 8:
        raise RuntimeError('proc GPU inventory must contain exactly eight devices')
    return list(devices.values())


def environment(run):
    env = os.environ.copy()
    ssd = Path('/SSD_DISK/users/wuruihan')
    cache = ssd / 'DynamicWAM/caches/motionkv_eval'
    # All SAPIEN workers MUST share this lock directory.
    env.update(WAM_SSD_ROOT=str(ssd), TMPDIR=str(ssd / 'tmp'), TMP=str(ssd / 'tmp'), TEMP=str(ssd / 'tmp'),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(ssd / 'hf_cache'), TORCH_HOME=str(cache / 'torch'),
        TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
        CUDA_VISIBLE_DEVICES='', WAM_TRAINING_RUN_DIR=str(run),
        PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'external/flash_attn_runtime'),
        VK_ICD_FILENAMES=str(ROOT / 'external/vulkan-icd/nvidia_icd.json'),
        SAPIEN_VULKAN_LIBRARY_PATH=str(ROOT / 'external/vulkan-conda/lib/libvulkan.so.1'),
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json', SAPIEN_RENDER='egl')
    lib = subprocess.check_output([str(ROOT / 'external/robotwin/bin/python'), '-c',
        'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("sapien").origin).parent.parent/"sapien.libs")'], text=True).strip()
    env['LD_LIBRARY_PATH'] = lib + ':' + env.get('LD_LIBRARY_PATH', '')
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR',
                'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--authorization', type=Path, required=True)
    p.add_argument('--guard-gpus', help='Explicit descriptor coverage; simulator stages require all visible devices')
    p.add_argument('--stage', choices=('graph', 'prepare', 'smoke', 'formal'), required=True)
    p.add_argument('--single-worker', action='store_true')
    p.add_argument('--preflight-only', action='store_true', help='Guarded telemetry and full exit check, without any worker')
    p.add_argument('--startup-seconds', type=float, default=300)
    p.add_argument('--max-runtime-seconds', type=float, default=0)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command and not a.preflight_only:
        p.error('a command is required unless --preflight-only is used')
    if command and a.preflight_only:
        p.error('--preflight-only cannot accept a worker command')
    ssd, storage = verify_ssd(Path('/SSD_DISK/users/wuruihan/DynamicWAM'), 30)
    run = inside_ssd(a.run_dir, ssd)
    run.mkdir(parents=True, exist_ok=True)
    if any((run / name).exists() for name in ('manifest.json', 'status.json', 'driver_phases.json')):
        raise FileExistsError('each supervisor launch needs a fresh run directory')
    ids = a.gpus.split(',')
    if len(ids) != len(set(ids)) or any(not v.startswith('GPU-') for v in ids):
        raise ValueError('explicit distinct UUIDs are required')
    if a.stage == 'formal' and len(ids) != 8:
        raise ValueError('formal evaluation requires eight GPUs')
    baseline = driver_evidence()
    authorization = json.loads(a.authorization.read_text())
    if driver_identity(baseline) != driver_identity(authorization['driver']):
        raise RuntimeError('driver evidence changed since current user-authorized preflight')
    if not set(ids).issubset(authorization['gpu_uuids']):
        raise RuntimeError('GPU not covered by current launch evidence')
    guard_ids = a.guard_gpus.split(',') if a.guard_gpus else ids
    if len(guard_ids) != len(set(guard_ids)) or not set(ids).issubset(guard_ids):
        raise ValueError('guard UUIDs must be distinct and include worker UUIDs')
    if not set(guard_ids).issubset(authorization['gpu_uuids']):
        raise RuntimeError('guard coverage exceeds current evidence')
    guard = None
    process = None
    owned = {}
    phases = []
    status = dict(state='starting', supervisor_pid=os.getpid(), stage=a.stage,
        started_at=utc(), stop_reason=None, worker_started=False, preflight_only=a.preflight_only,
        device_lifetime_mode=DEVICE_LIFETIME_MODE)
    started = time.monotonic()
    code = 1
    def terminate(signum, _frame):
        raise RuntimeError('supervisor received signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        status['launch_phase'] = 'guard_preflight'
        guard_path = ROOT / 'scripts/device_lifetime_guard.py'
        guard_sha = file_digest(guard_path)
        if guard_sha != authorization['guard_sha256']:
            raise RuntimeError('guard helper changed')
        if authorization.get('device_lifetime_mode') != DEVICE_LIFETIME_MODE:
            raise RuntimeError('authorization must bind FD-before-telemetry lifetime mode')
        inventory = proc_gpu_inventory()
        visible = {row['uuid'] for row in inventory}
        if not set(guard_ids).issubset(visible):
            raise RuntimeError('guard UUID missing from proc inventory')
        if a.stage != 'graph' and set(guard_ids) != visible:
            raise RuntimeError('SAPIEN/Vulkan enumeration requires temporary guard coverage of every visible GPU')
        if a.single_worker and len(ids) != 1:
            raise ValueError('single-worker requires one UUID')
        sources, source_sha = source_files_and_hash(ROOT)
        manifest = dict(created_at=utc(), stage=a.stage, command=command, storage=storage, gpu=None,
            proc_gpu_inventory=inventory, preflight_only=a.preflight_only, device_lifetime_mode=DEVICE_LIFETIME_MODE,
            driver=baseline, authorization=str(a.authorization.resolve()), authorization_sha256=file_digest(a.authorization),
            device_guard_sha256=guard_sha, worker_gpu_uuids=ids, guard_gpu_uuids=guard_ids, git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True), source_sha256=source_sha,
            startup_health_seconds=a.startup_seconds, postwindow_polling=False, retry_policy='none', environment={})
        atomic_json(run/'manifest.json',manifest)
        atomic_json(run/'source_files.json',sources)
        guard = load_device_guard(guard_path,guard_sha,guard_ids,run)
        driver_fence(run, phases, status, 'before_guard_open', baseline)
        status['launch_phase'] = 'guard_open'
        guard.open()
        driver_fence(run, phases, status, 'after_guard_open_before_telemetry', baseline)
        status['launch_phase'] = 'gpu_preflight'
        gpu = gpu_preflight(guard_ids, 8, 70, 100)
        manifest['gpu'] = gpu
        atomic_json(run/'manifest.json',manifest)
        driver_fence(run, phases, status, 'after_gpu_preflight', baseline)
        if visible != {row['uuid'] for row in gpu['devices']}:
            raise RuntimeError('NVML UUID inventory differs from verified proc inventory')
        if a.preflight_only:
            code = 0
            status.update(state='preflight_complete', exit_code=0)
        else:
            env = environment(run)
            if a.single_worker:
                env['CUDA_VISIBLE_DEVICES'] = ids[0]
                device = next(x for x in gpu['devices'] if x['uuid'] == ids[0])
                env.update(DYNAMICWAM_SIM_CUDA='0', DYNAMICWAM_INFER_CUDA='0', DYNAMICWAM_GPU_PCI=device['pci.bus_id'])
            manifest['environment'] = {k:v for k,v in env.items() if k in {'TMPDIR','TMP','TEMP','CUDA_VISIBLE_DEVICES','HF_HOME','PYTHONPATH','LD_LIBRARY_PATH','VK_ICD_FILENAMES','SAPIEN_VULKAN_LIBRARY_PATH'} }
            atomic_json(run/'manifest.json',manifest)
            driver_fence(run, phases, status, 'before_worker_launch', baseline)
            with (run/'run.log').open('w') as log:
                status['launch_phase'] = 'worker_launch'
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status.update(state='running',worker_pid=process.pid,worker_started=True)
                record_phase(run, phases, 'worker_started', worker_pid=process.pid)
                next_health = time.monotonic() + 30
                while process.poll() is None:
                    register_run_processes(run,process,owned)
                    elapsed = time.monotonic() - started
                    if a.max_runtime_seconds and elapsed > a.max_runtime_seconds:
                        raise TimeoutError('bounded run timeout')
                    if elapsed < a.startup_seconds and time.monotonic() >= next_health:
                        current = driver_evidence()
                        atomic_json(run/'startup_driver.json',current)
                        if driver_identity(current) != driver_identity(baseline):
                            raise RuntimeError('new driver anomaly during startup')
                        next_health = time.monotonic() + 30
                    status.update(timestamp=utc(), runtime_seconds=elapsed, startup_health_window_closed=elapsed>=a.startup_seconds)
                    atomic_json(run/'status.json',status)
                    time.sleep(2)
                code=process.returncode
                status.update(state='process_exited',exit_code=code)
    except BaseException as exc:
        status.update(state='failed',stop_reason=str(exc),traceback=traceback.format_exc(),
            blocked_before_worker=process is None, failure_phase=status.get('launch_phase'))
        code=1
    finally:
        try:
            try:
                finish_device_guard(guard,run,status,process,owned)
            except BaseException as exc:
                code=1
                status.update(state='failed', stop_reason=status.get('stop_reason') or str(exc),
                    cleanup_traceback=traceback.format_exc())
            post=driver_evidence()
            atomic_json(run/'driver_postexit.json',post)
            clean=driver_identity(post)==driver_identity(baseline)
            status['driver_exit_clean']=clean
            if not clean or status.get('stop_reason'):
                code=1
                status.update(state='failed', stop_reason=status.get('stop_reason') or 'driver evidence changed after final guard close')
        except BaseException as exc:
            code=1
            status.update(state='failed', stop_reason=status.get('stop_reason') or str(exc),
                final_check_traceback=traceback.format_exc())
        finally:
            status.update(final_exit_code=code,ended_at=utc())
            atomic_json(run/'status.json',status)
    return code


if __name__=='__main__':
    raise SystemExit(main())
