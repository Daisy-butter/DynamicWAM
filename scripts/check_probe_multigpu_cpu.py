"""CPU-only bounded probe checks; imports neither torch nor a CUDA library."""
import ast
import ctypes
from datetime import timedelta
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

source = Path(__file__).with_name('probe_multigpu_teardown.py')
ast.parse(source.read_text())
module = runpy.run_path(str(source), run_name='probe_cpu_check')

class Function:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)

def runtime(fail_copy=False):
    calls, buffers = [], {}
    def alloc(pointer, size, flags):
        assert size == 4096 and flags == 2
        buffer = ctypes.create_string_buffer(size)
        address = ctypes.addressof(buffer)
        buffers[address] = buffer
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
        calls.append('alloc')
        return 0
    def device(pointer, host, flags):
        assert host.value in buffers and flags == 0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = host.value
        calls.append('device_pointer')
        return 0
    def copy(target, source, size, kind):
        assert kind == 2 and size == 4096
        calls.append('copy')
        if fail_copy:
            return 11
        ctypes.memmove(target, source, size)
        return 0
    def free(pointer):
        calls.append('free')
        del buffers[pointer.value]
        return 0
    def sync():
        calls.append('synchronize')
        return 0
    api = SimpleNamespace(**{name: Function(callback) for name, callback in {
        'cudaHostAlloc': alloc, 'cudaHostGetDevicePointer': device,
        'cudaMemcpy': copy, 'cudaFreeHost': free, 'cudaDeviceSynchronize': sync,
    }.items()})
    return api, calls, buffers

for failure in (False, True):
    api, calls, buffers = runtime(failure)
    stages = []
    try:
        result = module['mapped_host_probe'](api, stages.append, lambda: calls.append('overlap'), 1)
    except RuntimeError as exc:
        assert failure and 'cudaMemcpy' in str(exc)
    else:
        assert not failure and result['roundtrip_passed']
        assert result['allocated_monotonic'] <= result['freed_monotonic']
    assert not buffers and calls.count('free') == 1
    assert calls.index('alloc') < calls.index('overlap') < calls.index('copy') < calls.index('free')
    assert 'after_mapped_host_free' in stages

args = SimpleNamespace(case='independent', gpu_uuids='a,b', p2p='default', nccl_alloc='auto',
    nccl_link='default', rank_visibility='isolated', init_mode='lazy', mapped_host=True,
    release_before_destroy=False, backend='nccl', launcher='direct',
    output=Path('cpu-only-output.json'), timeout_seconds=30)
assert '--mapped-host' in module['child_arguments'](args, 'output')
args.mapped_host, args.release_before_destroy = False, True
assert '--release-before-destroy' in module['child_arguments'](args, 'output')
assert '--mapped-host' not in module['child_arguments'](args, 'output')

# Lazy initialization must omit device_id entirely (not merely set it to None).
device = object()
assert module['distributed_init_kwargs']('lazy', device) == {'timeout': timedelta(seconds=45)}
assert module['distributed_init_kwargs']('eager', device) == {
    'timeout': timedelta(seconds=45), 'device_id': device}
try:
    module['distributed_init_kwargs']('unknown', device)
except ValueError:
    pass
else:
    raise AssertionError('unknown initialization mode was accepted')

# Exercise actual direct-launch construction with two stub child processes:
# parameter propagation, rank isolation and DDP rendezvous must all survive.
class Reservation:
    def __enter__(self):
        return self
    def __exit__(self, *unused):
        pass
    def bind(self, address):
        assert address == ('127.0.0.1', 0)
    def getsockname(self):
        return ('127.0.0.1', 29577)

class Process:
    returncode = 0
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        return self.returncode

for case, backend in [('independent', 'nccl'), ('ddp', 'nccl'), ('ddp', 'gloo')]:
    args.case, args.backend = case, backend
    args.p2p = 'disable' if case == 'ddp' else 'default'
    args.release_before_destroy = False
    calls, receipts = [], {}
    def popen(argv, *, env, start_new_session):
        assert start_new_session
        calls.append((argv, env))
        return Process(1000 + len(calls))
    def record(path, value):
        receipts[path.name] = value
    launcher_globals = module['independent_launcher'].__globals__
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=True), \
            patch.object(module['socket'], 'socket', return_value=Reservation()), \
            patch.object(module['signal'], 'signal'), \
            patch.object(module['signal'], 'SIGHUP', 1, create=True), \
            patch.object(module['subprocess'], 'Popen', side_effect=popen), \
            patch.dict(launcher_globals, {'write_json': record,
                'process_identity': lambda pid: {'pid': pid}}):
        module['independent_launcher'](args)
    assert len(calls) == 2
    for rank, (argv, env) in enumerate(calls):
        assert argv[0] == sys.executable
        for option, expected in [('--case', case), ('--backend', backend),
                ('--launcher', 'direct'), ('--init-mode', 'lazy')]:
            assert argv[argv.index(option) + 1] == expected
        assert env['CUDA_VISIBLE_DEVICES'] == ['a', 'b'][rank]
        assert env['RANK'] == env['LOCAL_RANK'] == str(rank)
        assert env['WORLD_SIZE'] == env['LOCAL_WORLD_SIZE'] == '2'
        if case == 'ddp':
            assert env['MASTER_ADDR'] == '127.0.0.1' and env['MASTER_PORT'] == '29577'
        else:
            assert 'MASTER_ADDR' not in env and 'MASTER_PORT' not in env
    assert receipts['independent_exit_codes.json']['exit_codes'] == [0, 0]
    assert not receipts['independent_launcher.json']['torch_imported']

# Keep launcher PID metadata backwards-compatible and preserve selected mode.
tree = ast.parse(source.read_text())
report_builders = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'report' for target in node.targets)
    and isinstance(node.value, ast.Call)]
assert len(report_builders) == 1
keywords = {item.arg: item.value for item in report_builders[0].value.keywords}
mode = keywords['launcher_mode']
assert isinstance(mode, ast.Attribute) and mode.attr == 'launcher'
assert isinstance(mode.value, ast.Name) and mode.value.id == 'args'
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                    and target.value.id == 'report' and isinstance(target.slice, ast.Constant):
                assert target.slice.value != 'launcher_mode', 'launcher mode overwritten after initialization'
assert 'torch' not in sys.modules
print('PASS: AST; mocked mapped-host cleanup; eager/lazy kwargs; direct NCCL/Gloo '
      'launch rank/rendezvous/option propagation; launcher mode preserved; no CUDA loaded')
