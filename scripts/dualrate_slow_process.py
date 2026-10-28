"""Deployment-only slow prediction isolation, with exact bootstrap/RNG parity.

No simulator, GPU recovery, model mutation or training change. Importing this
module uses only the standard library; CUDA is initialized only by construction.
"""
from __future__ import annotations

import ctypes
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import threading
import time
import traceback

_MAX_MESSAGE = 128 << 20


def _timeout(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('positive finite IPC timeout required')
    return float(value)


def _remaining(deadline):
    left = deadline-time.monotonic()
    if left <= 0:
        raise TimeoutError('slow inference IPC deadline exceeded')
    return left


def _send(sock, value, deadline):
    data = pickle.dumps(value, protocol=5)
    if len(data) > _MAX_MESSAGE:
        raise ValueError('slow inference message exceeds bounded protocol')
    sock.settimeout(_remaining(deadline))
    sock.sendall(struct.pack('!Q', len(data))+data)


def _receive(sock, deadline):
    def exactly(count):
        parts=[]
        while count:
            sock.settimeout(_remaining(deadline))
            data=sock.recv(min(count, 1 << 20))
            if not data:
                raise EOFError('owned slow inference process closed its IPC socket')
            parts.append(data);count-=len(data)
        return b''.join(parts)
    length=struct.unpack('!Q',exactly(8))[0]
    if length > _MAX_MESSAGE:
        raise ValueError('slow inference response exceeds bounded protocol')
    # A private socketpair connects only this parent and its spawned child.
    return pickle.loads(exactly(length))


def _cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().to('cpu').contiguous().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError('unsupported inference IPC value: '+type(value).__name__)


def _assert_exact(actual, expected, path='world'):
    import torch
    if isinstance(actual,torch.Tensor) and isinstance(expected,torch.Tensor):
        if (actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                   expected.contiguous().reshape(-1).view(torch.uint8))):
            raise ValueError('slow process exact parity mismatch: '+path)
    elif isinstance(actual,dict) and isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError('slow parity keys differ: '+path)
        for key in actual:_assert_exact(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(actual,(tuple,list)) and type(actual) is type(expected):
        if len(actual)!=len(expected):raise ValueError('slow parity length differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):_assert_exact(a,b,path+f'[{i}]')
    elif type(actual) is not type(expected) or actual!=expected:
        raise ValueError('slow process exact parity mismatch: '+path)


def _load_system(config,device):
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay, activate_dependency_overlay
    activate_dependency_overlay(validate_dependency_overlay(config))
    import torch
    from dynamicwam.dualrate.factory import build_system,place_inference_devices
    from dynamicwam.dualrate.train import _validate_normalization
    from dualrate_eval_worker import validate_checkpoint_payload,validate_source_compatibility,sha
    if sha(config['checkpoint'])!=config['checkpoint_sha256']:
        raise ValueError('slow child final checkpoint hash mismatch')
    manifest=json.loads(Path(config['manifest']).read_text())
    _validate_normalization(config,manifest)
    payload=torch.load(config['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    contract=validate_checkpoint_payload(config,payload,manifest)
    validate_source_compatibility(config,contract)
    torch.cuda.set_device(device)
    system=build_system(config,device=device).eval().requires_grad_(False)
    system.load_state_dict(payload['model'],strict=True)
    del payload
    # Parent retains its VAE. Child only calls slow prediction and trained readout.
    place_inference_devices(system,fast_device='cpu',slow_device=device)
    return system


def _child(sock,config,device,parent_pid,parent_session,loader):
    try:
        if os.name=='posix':
            if os.getsid(0)!=parent_session:raise RuntimeError('slow child escaped owned session')
            if hasattr(signal,'SIGTERM') and os.path.exists('/proc/self'):
                if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:
                    raise OSError(ctypes.get_errno(),'could not bind slow child parent lifetime')
        if os.getppid()!=parent_pid:raise RuntimeError('slow parent exited during spawn')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        system=loader(config,torch.device(device))
        _send(sock,dict(kind='ready',pid=os.getpid(),parent_pid=os.getppid(),
            session=os.getsid(0) if os.name=='posix' else None,device=str(device)),time.monotonic()+60.)
        while True:
            # Parent may be between long episodes; this is idle only, not compute.
            request=_receive(sock,time.monotonic()+86400.)
            if request['kind']=='close':
                _send(sock,dict(kind='closed'),time.monotonic()+10.);return
            if request['kind']!='predict':raise ValueError('unexpected slow process request')
            generator=torch.Generator(device=device)
            generator.set_state(request['rng'])
            batch={k:v.to(device) for k,v in request['batch'].items()}
            started=time.perf_counter()
            with torch.no_grad():
                world=system.world_readout(system.predict_reference(batch,generator=generator))
                world=_cpu_tree(world)
            response=dict(kind='result',request_id=request['request_id'],world=world,
                rng=generator.get_state().cpu().clone(),compute_wall_seconds=time.perf_counter()-started)
            _send(sock,response,time.monotonic()+request['timeout'])
    except BaseException as exc:
        try:_send(sock,dict(kind='error',error=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc()[-16000:]),time.monotonic()+5.)
        except BaseException:pass
        raise
    finally:sock.close()


class _World:
    def __init__(self,world):self.world=world


class SlowProcessSystem:
    """Facade forwarding all fast/native attributes to the untouched parent model."""
    _private=frozenset(('_model','_socket','_process','_lock','_timeout','_receipt_path',
        '_stats','_closed','_failed','_parity','_sequence'))

    def __init__(self,system,config,*,slow_device='cuda:2',startup_timeout=300.,
                 call_timeout=60.,receipt_path=None,_loader=None,_test_fixture=False):
        startup_timeout=_timeout(startup_timeout);call_timeout=_timeout(call_timeout)
        if _loader is not None and not _test_fixture:
            raise ValueError('custom slow loader is reserved for explicit CPU tests')
        if str(slow_device).startswith('cpu') and not _test_fixture:
            raise ValueError('production slow inference requires its reserved CUDA device')
        if system.training:raise ValueError('slow process requires the frozen eval model')
        self._model=system;self._timeout=call_timeout;self._lock=threading.Lock()
        self._receipt_path=Path(receipt_path) if receipt_path else None
        self._closed=False;self._failed=False;self._parity=False;self._sequence=0
        self._stats=dict(mode='dedicated_slow_process_exact_checkpoint',calls=0,
            bootstrap_exact_parity_passed=False,rng_roundtrip=True,closed=False,
            max_rpc_wall_seconds=0.,max_child_compute_wall_seconds=0.)
        parent,child=socket.socketpair();self._socket=parent
        self._process=multiprocessing.get_context('spawn').Process(target=_child,
            args=(child,config,str(slow_device),os.getpid(),os.getsid(0) if os.name=='posix' else None,
                  _loader or _load_system),name='dualrate-slow-inference',daemon=False)
        try:
            self._process.start();child.close()
            reply=_receive(parent,time.monotonic()+startup_timeout)
            if reply.get('kind')!='ready':raise RuntimeError('slow child startup failed: '+str(reply))
            if reply['pid']!=self._process.pid or reply['parent_pid']!=os.getpid():
                raise RuntimeError('slow process identity differs from owned child')
            self._stats.update(reply);self._stats.pop('kind');self._save()
        except BaseException:
            child.close();self._failed=True;self.close(suppress_errors=True);raise

    def __getattr__(self,name):return getattr(self._model,name)

    def __setattr__(self,name,value):
        if name in self._private:object.__setattr__(self,name,value)
        else:setattr(self._model,name,value)

    def _save(self):
        if self._receipt_path is None:return
        self._receipt_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self._receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.receipt(),indent=2)+'\n')
        temporary.replace(self._receipt_path)

    def receipt(self):return dict(self._stats,failed=self._failed)

    def predict_reference(self,batch,*,generator=None):
        import torch
        if generator is None:raise ValueError('explicit slow RNG is required')
        if self._closed or self._failed:raise RuntimeError('slow inference process is unavailable')
        if not self._lock.acquire(blocking=False):raise RuntimeError('only one slow query may be in flight')
        before=generator.get_state().cpu().clone()
        try:
            started=time.perf_counter();cpu_batch=_cpu_tree(batch)
            local=local_rng=None
            if not self._parity:
                # First call is inactive bootstrap. Original input/RNG are untouched.
                with torch.no_grad():
                    local=_cpu_tree(self._model.world_readout(
                        self._model.predict_reference(batch,generator=generator)))
                local_rng=generator.get_state().cpu().clone()
                generator.set_state(before)
            self._sequence+=1
            deadline=time.monotonic()+self._timeout
            _send(self._socket,dict(kind='predict',request_id=self._sequence,batch=cpu_batch,
                rng=before,timeout=self._timeout),deadline)
            result=_receive(self._socket,deadline)
            if result.get('kind')!='result' or result.get('request_id')!=self._sequence:
                raise RuntimeError('slow inference child failed: '+str(result))
            if not self._parity:
                _assert_exact(result['world'],local)
                _assert_exact(result['rng'],local_rng,'rng')
                self._parity=True;self._stats['bootstrap_exact_parity_passed']=True
            generator.set_state(result['rng'])
            self._stats['calls']+=1
            self._stats['max_rpc_wall_seconds']=max(self._stats['max_rpc_wall_seconds'],time.perf_counter()-started)
            self._stats['max_child_compute_wall_seconds']=max(self._stats['max_child_compute_wall_seconds'],result['compute_wall_seconds'])
            if self._stats['calls']==1:self._save()
            return _World(result['world'])
        except BaseException:
            generator.set_state(before);self._failed=True;self._save();raise
        finally:self._lock.release()

    @staticmethod
    def world_readout(reference):
        if not isinstance(reference,_World):raise TypeError('slow RPC result envelope required')
        return reference.world

    def close(self,*,suppress_errors=False):
        if self._closed:return
        self._closed=True;failure=None;forced=False
        try:
            if self._failed and self._process.pid is not None and self._process.is_alive():
                raise RuntimeError('failed slow RPC requires owned-child cleanup')
            if self._process.pid is not None and self._process.is_alive() and not self._failed:
                deadline=time.monotonic()+10.
                _send(self._socket,dict(kind='close'),deadline)
                if _receive(self._socket,deadline).get('kind')!='closed':
                    raise RuntimeError('slow child did not acknowledge shutdown')
            if self._process.pid is not None:self._process.join(10.)
            if self._process.pid is not None and self._process.is_alive():
                raise TimeoutError('owned slow child did not exit')
            if self._process.exitcode not in (0,None):
                raise RuntimeError('slow child failed with exit '+str(self._process.exitcode))
        except BaseException as exc:
            failure=exc;self._failed=True
            if self._process.pid is not None and self._process.is_alive():
                forced=True;self._process.terminate();self._process.join(10.)
                if self._process.is_alive():self._process.kill();self._process.join(10.)
        finally:
            self._socket.close()
            self._stats.update(closed=not self._process.is_alive(),exitcode=self._process.exitcode,
                forced_owned_child_cleanup=forced)
            self._save()
        if failure is not None and not suppress_errors:raise failure
