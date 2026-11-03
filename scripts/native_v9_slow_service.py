"""One owned slow process and one in-flight request; no simulator in the child."""
import concurrent.futures
import ctypes
import multiprocessing
import os
import signal
import socket
import time
from dualrate_slow_process import _send,_receive,_cpu_tree


def load_trained_state(system,config):
    """Use one verified Path for hashing and strict trained-state loading."""
    from pathlib import Path
    import torch
    from dynamicwam.dualrate.data import sha256_file
    path=Path(config['checkpoint'])
    if sha256_file(path)!=config['checkpoint_sha256']:raise ValueError('checkpoint changed')
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    system.load_state_dict(payload['model'],strict=True)


def child(sock,config,parent_pid,parent_session):
    try:
        if os.getsid(0)!=parent_session:raise RuntimeError('slow child session mismatch')
        if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError('parent death binding failed')
        if os.getppid()!=parent_pid:raise RuntimeError('parent disappeared')
        from dynamicwam.dualrate.dependencies import activate_dependency_overlay,validate_dependency_overlay
        activate_dependency_overlay(validate_dependency_overlay(config))
        import torch
        from dynamicwam.dualrate.factory import build_system
        from dynamicwam.dualrate.online import OnlineBatchBuilder
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(1)
        system=build_system(config,'cuda:1').eval().requires_grad_(False)
        if config.get('checkpoint'):
            load_trained_state(system,config)
        _send(sock,dict(kind='ready',pid=os.getpid()),time.monotonic()+60)
        while True:
            request=_receive(sock,time.monotonic()+86400)
            if request['kind']=='close':
                _send(sock,dict(kind='closed'),time.monotonic()+10);return
            if request['kind']!='predict':raise ValueError('unexpected request')
            started=time.monotonic();m=request['manifest']
            builder=OnlineBatchBuilder(m,action_mean=m['action_statistics']['mean'],action_scale=m['action_statistics']['std'],
                language=request['language'],device='cuda:1')
            source=builder.source(request['frames'],source_time=request['source_time'],action_start=request['source_time'],
                vae=system.slow.native.compact_wan.video_model.vae)
            source['native_source_index']=torch.tensor([request['index']],device='cuda:1')
            source['source_known_actions']=source['source_initial_state'].new_zeros(1,16,14)
            source['source_known_mask']=torch.zeros((1,16),device='cuda:1',dtype=torch.bool)
            if request['committed'] is not None:
                source['source_known_actions'][0,0]=torch.as_tensor(request['committed'],device='cuda:1')
                source['source_known_mask'][0,0]=True
            source['source_has_committed_waypoint']=torch.tensor([request['committed'] is not None],device='cuda:1')
            source['frozen_source_seed']=torch.tensor([request['seed']],device='cuda:1')
            if config.get('bounded_feedback'):
                source['feedback_noise_seed']=source['frozen_source_seed'].clone()
            with torch.no_grad():
                model_start=time.monotonic()
                reference=system.predict_reference(source)
                torch.cuda.synchronize(1);model_end=time.monotonic()
                # V9 never consumes native video K/V; do not transfer megabytes of unused caches.
                reference['native_video_cache']={}
                result=_cpu_tree(reference);batch=_cpu_tree(source)
            _send(sock,dict(kind='result',request_id=request['request_id'],reference=result,source=batch,
                compute_wall_seconds=time.monotonic()-started,model_start_wall=model_start,model_end_wall=model_end),time.monotonic()+60)
    except BaseException as exc:
        import traceback
        try:_send(sock,dict(kind='error',error=repr(exc),traceback=traceback.format_exc()),time.monotonic()+5)
        except BaseException:pass
        raise
    finally:sock.close()


class NativeSlowService:
    def __init__(self,config):
        self.socket,other=socket.socketpair()
        self.process=multiprocessing.get_context('spawn').Process(target=child,
            args=(other,config,os.getpid(),os.getsid(0)),name='native-v9-slow',daemon=False)
        self.process.start();other.close()
        reply=_receive(self.socket,time.monotonic()+300)
        if reply.get('kind')!='ready':raise RuntimeError(reply)
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future=None;self.counter=0;self.closed=False

    def submit(self,request,clock):
        if self.future is not None:raise RuntimeError('slow queue must contain at most one request')
        self.counter+=1;request=dict(request,kind='predict',request_id=self.counter)
        def call():
            _send(self.socket,request,time.monotonic()+60)
            result=_receive(self.socket,time.monotonic()+60)
            if result.get('kind')!='result' or result['request_id']!=request['request_id']:raise RuntimeError(result)
            # Read-only thread-safe clock; the child never touches an environment.
            result['complete_physical']=float(clock.snapshot().time_seconds)
            result['complete_wall']=time.monotonic()
            return result
        self.future=self.pool.submit(call)

    def poll(self,wait=False):
        if self.future is None:return None
        if not wait and not self.future.done():return None
        result=self.future.result(timeout=65);self.future=None
        return result

    def close(self):
        if self.closed:return
        try:
            self.poll(wait=True)
            _send(self.socket,dict(kind='close'),time.monotonic()+10)
            if _receive(self.socket,time.monotonic()+10).get('kind')!='closed':raise RuntimeError('slow close failed')
            self.process.join(15)
            if self.process.exitcode!=0:raise RuntimeError('slow process did not exit cleanly')
        finally:
            self.closed=True;self.socket.close();self.pool.shutdown(wait=True)
            if self.process.is_alive():self.process.terminate();self.process.join(10)
