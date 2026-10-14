#!/usr/bin/env python3
"""One bounded CUDA/NCCL test, with read-only GPU/driver checks and clean teardown.

The parent never imports torch. Historical driver errors are recorded, not erased
or declared resolved. A new error makes this case fail; no other case is started.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/SSD_DISK/users/wuruihan/DynamicWAM')
PATTERN = re.compile(r'NVRM.*(?:assert|error|Xid|RmInitAdapter|IOVAS|iovaspace)|Xid\s*\(|fallen off the bus|GPU.*(?:fault|error)|AER.*\bFatal\b|PCIe Bus Error:.*\bFatal\b', re.I)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def command(args):
    return subprocess.check_output(args, text=True, timeout=15)


def driver():
    kernel = command(['dmesg', '--color=never'])
    lines = [line.strip() for line in kernel.splitlines() if PATTERN.search(line)]
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return dict(host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        anomaly_count=len(lines), driver_anomalies_sha256=digest('\n'.join(lines)),
        latest_driver_anomaly_sha256=digest(lines[-1]) if lines else None,
        latest_driver_anomaly=lines[-1] if lines else None, dmesg_tail=kernel.splitlines()[-150:])


def driver_identity(value):
    return tuple(value[key] for key in ('host', 'boot_id', 'anomaly_count',
        'driver_anomalies_sha256', 'latest_driver_anomaly_sha256'))


def gpu_snapshot(uuids, *, idle):
    fields = 'uuid,pci.bus_id,temperature.gpu,memory.used'
    rows = list(csv.reader(io.StringIO(command(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits']))))
    devices = [dict(zip(fields.split(','), [value.strip() for value in row])) for row in rows]
    by_uuid = {row['uuid']: row for row in devices}
    if len(devices) != 8 or len(by_uuid) != 8 or not set(uuids).issubset(by_uuid):
        raise RuntimeError('expected all 8 GPUs and every selected full UUID to remain visible')
    processes = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).splitlines()
    for uuid in uuids:
        if float(by_uuid[uuid]['temperature.gpu']) >= 70:
            raise RuntimeError('selected GPU temperature exceeded 70C diagnostic stop threshold')
        if idle and (float(by_uuid[uuid]['memory.used']) > 64 or any(line.split(',')[0].strip() == uuid for line in processes)):
            raise RuntimeError('selected GPU not idle; no existing process will be stopped')
    return dict(devices=devices, compute_processes=processes)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), starttime=int(fields[19]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def register_children(run, root_pid, owned):
    """Only accept a worker whose live ancestry reaches this still-owned root."""
    for path in run.glob('rank_*.identity.json'):
        item = json.loads(path.read_text())
        if item['pid'] in owned:
            continue
        live = process_identity(item['pid'])
        if live is None or live['starttime'] != item['starttime']:
            continue
        ancestor, seen = live, set()
        while ancestor is not None and ancestor['pid'] not in seen:
            if ancestor['pid'] == root_pid:
                if live['pgid'] not in (live['pid'], root_pid):
                    raise RuntimeError('worker has an unexpected process group')
                owned[live['pid']] = live
                break
            seen.add(ancestor['pid'])
            ancestor = process_identity(ancestor['ppid']) if ancestor['ppid'] > 1 else None


def stop_owned(process, owned):
    """Signals target only PID/starttime-verified groups created for this case."""
    for sig, wait in ((signal.SIGTERM, 6), (signal.SIGKILL, 2)):
        groups = set()
        for pid, expected in owned.items():
            live = process_identity(pid)
            if expected and live and live['state'] != 'Z' and live['starttime'] == expected['starttime'] and live['pgid'] == expected['pgid']:
                if live['pgid'] == os.getpgrp():
                    raise RuntimeError('refusing to signal parent process group')
                groups.add(live['pgid'])
        for pgid in groups:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass


def collect_rank_results(run, expected):
    results = []
    for rank in range(expected):
        path = run / f'rank_{rank}.result.json'
        if path.is_file():
            try:
                results.append(json.loads(path.read_text()))
            except (ValueError, OSError) as exc:
                results.append(dict(rank=rank, passed=False, result_read_error=repr(exc)))
        else:
            results.append(dict(rank=rank, passed=False, result_missing=True))
    return results


def independent_launcher(args):
    """Launch concurrent single-visible children without importing torch."""
    assert args.case == 'independent' and args.rank_visibility == 'isolated'
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    uuids = args.gpu_uuids.split(',')
    assert len(uuids) == 2
    run = args.output.parent / (args.output.stem + '_logs')
    children, owned = [], {}
    def interrupted(signum, _frame):
        raise RuntimeError(f'independent launcher interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        for rank in range(len(uuids)):
            env = os.environ.copy()
            # Pass an already isolated UUID as well as explicit rank metadata.
            # No parent imports torch and no distributed rendezvous is created.
            env.update(CUDA_VISIBLE_DEVICES=uuids[rank], RANK=str(rank), WORLD_SIZE=str(len(uuids)),
                LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(len(uuids)))
            argv = [sys.executable, str(Path(__file__).resolve()), '--child', '--case', 'independent',
                '--rank-visibility', 'isolated', '--gpu-uuids', args.gpu_uuids,
                '--p2p', args.p2p, '--nccl-alloc', args.nccl_alloc, '--output', str(args.output)]
            process = subprocess.Popen(argv, env=env, start_new_session=True)
            children.append(process)
            owned[process.pid] = process_identity(process.pid)
        write_json(run / 'independent_launcher.json', dict(pid=os.getpid(),
            launcher_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            torch_imported='torch' in sys.modules, children=list(owned.values())))
        deadline = time.monotonic() + args.timeout_seconds
        while any(process.poll() is None for process in children):
            if any(process.returncode not in (None, 0) for process in children):
                raise RuntimeError('one independent CUDA child failed; stopping this case')
            if time.monotonic() >= deadline:
                raise TimeoutError('independent launcher deadline exceeded')
            time.sleep(.1)
        codes = [process.returncode for process in children]
        write_json(run / 'independent_exit_codes.json', dict(exit_codes=codes))
        if any(codes):
            raise RuntimeError('independent child exit failure')
    except BaseException:
        # Outer supervisor also owns these exact registered children; either
        # layer can stop its own process tree if the other layer is interrupted.
        for process in children:
            stop_owned(process, owned)
        raise


def child(args):
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    physical_local_rank = local_rank
    if args.rank_visibility == 'isolated':
        assert (args.case == 'ddp' and args.p2p == 'disable') or args.case == 'independent'
        uuids = args.gpu_uuids.split(',')
        assert rank == physical_local_rank and len(uuids) == int(os.environ['WORLD_SIZE'])
        os.environ.update(CUDA_VISIBLE_DEVICES=uuids[physical_local_rank], LOCAL_RANK='0',
            LOCAL_WORLD_SIZE='1', GROUP_RANK=str(rank), GROUP_WORLD_SIZE=os.environ['WORLD_SIZE'])
        local_rank = 0
    run = args.output.parent / (args.output.stem + '_logs')
    write_json(run / f'rank_{rank}.identity.json', process_identity(os.getpid()))
    import torch
    if args.case == 'ddp':
        import torch.distributed as dist
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == (1 if args.rank_visibility == 'isolated' else len(args.gpu_uuids.split(',')))
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(.04, local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261014)
    context_ready_at = None
    if args.case == 'independent':
        # File synchronization, not a CUDA or distributed communication path.
        context_marker = torch.zeros(1, device=local_rank)
        torch.cuda.synchronize(local_rank)
        context_ready_at = time.monotonic()
        write_json(run / f'rank_{rank}.context_ready.json', dict(rank=rank, pid=os.getpid(),
            ready_monotonic=context_ready_at, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
        deadline = time.monotonic() + 45
        while not all((run / f'rank_{other}.context_ready.json').is_file() for other in range(2)):
            if time.monotonic() >= deadline:
                raise TimeoutError('independent CUDA context overlap barrier timed out')
            time.sleep(.02)
    if args.case == 'ddp':
        dist.init_process_group('nccl', timeout=timedelta(seconds=45), device_id=torch.device('cuda', local_rank))
    try:
        cpu = torch.nn.Linear(64, 32, bias=False)
        reference_weight = cpu.weight.detach().clone()
        layer = cpu.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(layer, device_ids=[local_rank]) if args.case == 'ddp' else layer
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        value = torch.randn(8, 64)
        data = value.to(local_rank)
        # All ranks deliberately use the same deterministic batch, making the
        # synchronized DDP gradient directly comparable to a CPU reference.
        prediction = model(data)
        torch.testing.assert_close(prediction.detach().cpu(), value @ reference_weight.T, rtol=1e-4, atol=1e-5)
        loss = prediction.square().mean()
        loss.backward()
        gradient = layer.weight.grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        expected = 2 * (value @ reference_weight.T).T @ value / (8 * 32)
        torch.testing.assert_close(gradient.cpu(), expected, rtol=1e-4, atol=1e-5)
        optimizer.step()
        assert bool(torch.isfinite(layer.weight).all())
        if args.case == 'ddp':
            reduced = torch.tensor([rank + 1.0], device=local_rank)
            dist.all_reduce(reduced)
            assert reduced.item() == dist.get_world_size() * (dist.get_world_size() + 1) / 2
            dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize(local_rank)
        allocated = torch.cuda.max_memory_allocated(local_rank)
        assert allocated < 128 * 1024**2, 'diagnostic tensor bound exceeded'
        result = dict(passed=True, rank=rank, local_rank=local_rank, physical_local_rank=physical_local_rank,
            visible_uuids=os.environ['CUDA_VISIBLE_DEVICES'], device=torch.cuda.get_device_name(local_rank),
            visible_device_count=torch.cuda.device_count(), torch=torch.__version__, loss=float(loss.detach()),
            gradient_norm=float(gradient.norm()), max_tensor_allocated_bytes=allocated,
            context_ready_monotonic=context_ready_at, compute_finished_monotonic=time.monotonic(),
            distributed_initialized=args.case == 'ddp',
            environment={key: value for key, value in os.environ.items() if key.startswith(('NCCL_', 'TORCH_NCCL_')) or key in (
                'CUDA_VISIBLE_DEVICES', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'PYTORCH_NVML_BASED_CUDA_CHECK')},
            note='tensor allocation bound excludes CUDA/NCCL/library context memory')
    finally:
        if args.case == 'ddp' and dist.is_initialized():
            dist.destroy_process_group()
    torch.cuda.synchronize(local_rank)
    result['distributed_teardown_returned'] = args.case == 'ddp'
    result['cuda_final_synchronize_returned'] = True
    write_json(run / f'rank_{rank}.result.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['cuda-visible', 'ddp', 'independent'], required=True)
    p.add_argument('--gpu-uuids', required=True)
    p.add_argument('--p2p', choices=['default', 'disable'], default='default')
    p.add_argument('--nccl-alloc', choices=['auto', 'legacy'], default='auto',
        help='legacy disables NCCL cuMem allocation in DDP only; changes no transport settings')
    p.add_argument('--nccl-link', choices=['default', 'socket'], default='default')
    p.add_argument('--rank-visibility', choices=['all', 'isolated'], default='all')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--timeout-seconds', type=int, default=120)
    p.add_argument('--child', action='store_true')
    p.add_argument('--independent-launcher', action='store_true')
    args = p.parse_args()
    if args.child:
        child(args)
        return
    if args.independent_launcher:
        independent_launcher(args)
        return
    if not 30 <= args.timeout_seconds <= 120:
        raise ValueError('diagnostic timeout must be within 30..120 seconds')
    uuids = args.gpu_uuids.split(',')
    if not uuids or len(set(uuids)) != len(uuids) or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', value) for value in uuids):
        raise ValueError('unique full GPU UUIDs are required')
    if args.case == 'ddp' and len(uuids) < 2:
        raise ValueError('DDP diagnostic requires at least 2 devices')
    if args.case == 'independent' and (len(uuids) != 2 or args.rank_visibility != 'isolated'):
        raise ValueError('independent diagnostic requires exactly 2 UUIDs and isolated visibility')
    if args.nccl_alloc == 'legacy' and args.case != 'ddp':
        raise ValueError('legacy NCCL allocation diagnostic applies only to DDP')
    if args.nccl_link == 'socket' and (args.case != 'ddp' or args.p2p != 'disable' or args.nccl_alloc != 'legacy'):
        raise ValueError('socket contrast requires DDP, disabled P2P and legacy allocation')
    root, out = ROOT.resolve(strict=True), args.output.resolve()
    if root.stat().st_dev == Path('/').stat().st_dev or not out.is_relative_to(root) or out.exists():
        raise ValueError('fresh output under established non-system SSD root required')
    run = out.parent / (out.stem + '_logs')
    run.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, driver_resolved=False, started_at=utc(), case=args.case,
        p2p=args.p2p, nccl_alloc=args.nccl_alloc, nccl_link=args.nccl_link, rank_visibility=args.rank_visibility, gpu_uuids=uuids,
        log_directory=str(run), timeout_seconds=args.timeout_seconds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    expected = len(uuids) if args.case in ('ddp', 'independent') else 1
    process, owned = None, {}
    def interrupted(signum, _frame):
        raise RuntimeError(f'diagnostic parent received signal {signum}; stop owned children')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        report['before'] = baseline = driver()
        report['preflight'] = gpu_snapshot(uuids, idle=True)
        env = os.environ.copy()
        # Prevent an inherited rank or transport setting from changing the case.
        for name in tuple(env):
            if name.startswith('NCCL_') or name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name)
        env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuids, CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTORCH_NVML_BASED_CUDA_CHECK='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2', CUDA_MODULE_LOADING='LAZY', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_DEBUG='WARN')
        # torchrun itself must not create a multi-device CUDA context. Each
        # isolated worker restores its one UUID from the explicit case args
        # before importing torch; the stdlib supervisor and launcher see none.
        if args.rank_visibility == 'isolated':
            env['CUDA_VISIBLE_DEVICES'] = ''
        report['launcher_cuda_visible_devices'] = env['CUDA_VISIBLE_DEVICES']
        if args.p2p == 'disable':
            env['NCCL_P2P_DISABLE'] = '1'
        if args.nccl_alloc == 'legacy':
            env.update(NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0',
                NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,ENV,ALLOC,SHM')
        if args.nccl_link == 'socket':
            env.update(NCCL_SHM_DISABLE='1', NCCL_NET='Socket',
                NCCL_DEBUG_SUBSYS='INIT,ENV,ALLOC,SHM,NET')
        report['launcher_environment'] = {key: value for key, value in env.items()
            if key.startswith(('NCCL_', 'TORCH_NCCL_')) or key in ('CUDA_VISIBLE_DEVICES', 'PYTORCH_NVML_BASED_CUDA_CHECK')}
        for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH',
                     'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'HF_HOME'):
            folder = run / 'cache' / name.lower()
            folder.mkdir(parents=True, exist_ok=True)
            env[name] = str(folder)
        script_args = [str(Path(__file__).resolve()), '--child', '--case', args.case,
            '--gpu-uuids', args.gpu_uuids, '--p2p', args.p2p, '--nccl-alloc', args.nccl_alloc,
            '--nccl-link', args.nccl_link, '--rank-visibility', args.rank_visibility, '--output', str(out)]
        argv = [sys.executable] + script_args
        if args.case == 'ddp':
            argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                f'--nproc-per-node={len(uuids)}', '--max-restarts=0'] + script_args
        elif args.case == 'independent':
            argv = [sys.executable] + [part for part in script_args if part != '--child'] + ['--independent-launcher',
                '--timeout-seconds', str(args.timeout_seconds)]
        report['command'] = argv
        with (run / 'worker.log').open('w') as handle:
            process = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            owned[process.pid] = process_identity(process.pid)
            report['launcher'] = owned[process.pid]
            write_json(out, report)
            deadline, next_health = time.monotonic() + args.timeout_seconds, time.monotonic() + 10
            while process.poll() is None:
                register_children(run, process.pid, owned)
                if time.monotonic() >= deadline:
                    raise TimeoutError('bounded diagnostic deadline exceeded')
                if time.monotonic() >= next_health:
                    current = driver()
                    report['last_running_driver'] = current
                    if driver_identity(current) != driver_identity(baseline):
                        raise RuntimeError('new driver evidence during diagnostic; stop this case')
                    gpu_snapshot(uuids, idle=False)
                    next_health = time.monotonic() + 10
                time.sleep(.2)
        report['exit_code'] = process.returncode
        report['ranks'] = collect_rank_results(run, expected)
        # Catch errors emitted only when CUDA/NCCL contexts leave the process.
        time.sleep(2)
        report['after'] = after = driver()
        report['postexit_gpu'] = gpu_snapshot(uuids, idle=True)
        if process.returncode:
            raise RuntimeError('diagnostic child failed; inspect worker.log')
        if driver_identity(after) != driver_identity(baseline):
            raise RuntimeError('new driver evidence during/after teardown; no further cases authorized by this result')
        if any(row.get('passed') is not True for row in report['ranks']):
            raise RuntimeError('missing successful rank computation/teardown')
        report.update(passed=True, host=after['host'], boot_id=after['boot_id'],
            driver_anomalies_sha256=after['driver_anomalies_sha256'],
            latest_driver_anomaly_sha256=after['latest_driver_anomaly_sha256'])
    except BaseException as exc:
        report['error'] = repr(exc)
        if process is not None:
            register_children(run, process.pid, owned)
            stop_owned(process, owned)
        try:
            report['after'] = driver()
            report['postfailure_gpu'] = gpu_snapshot(uuids, idle=False)
        except Exception as health_exc:
            report['postfailure_read_error'] = repr(health_exc)
    finally:
        report['finished_at'] = utc()
        report['owned_processes'] = list(owned.values())
        report['ranks'] = collect_rank_results(run, expected)
        for name in ('independent_launcher', 'independent_exit_codes'):
            path = run / f'{name}.json'
            if path.is_file():
                report[name] = json.loads(path.read_text())
        write_json(out, report)
        print(json.dumps(dict(passed=report['passed'], output=str(out), error=report.get('error'))), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
