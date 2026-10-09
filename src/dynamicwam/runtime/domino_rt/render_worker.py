"""Independent replica renderer; only its process touches its SAPIEN scene.

Physics publishes immutable pose snapshots. There is one in-flight render and
at most one queued latest snapshot on the owner side. Completed images retain
the capture timestamp, never the ready/receive timestamp.
"""
from __future__ import annotations
import copy
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
import numpy as np


def _serve(config, requests, results, ready):
    env=None
    try:
        if os.name=='posix':
            import ctypes
            import signal
            ctypes.CDLL(None).prctl(1,signal.SIGKILL)
            if os.getppid()==1: raise RuntimeError('physics owner has exited')
        from .preparation import prepare_fixed_scene
        from .scene_snapshot import SceneSnapshotBridge
        env, _, _=prepare_fixed_scene(Path(config['domino_root']),task=config['task'],
            task_config=config['task_config'],seed=config['seed'],
            instruction_type=config['instruction_type'],record_path=Path(config['accepted_record']))
        bridge=SceneSnapshotBridge(env)
        # Renderer warmup precedes the timed episode; it never steps physics.
        initial=bridge.capture();bridge.apply(initial)
        env.scene.update_render();env.cameras.update_picture();env.cameras.get_rgb()
        ready.put(dict(ready=True,identity=bridge.identity,pid=os.getpid()))
        while True:
            request=requests.get()
            if request is None: return
            started=time.perf_counter()
            try:
                if request['packet']['source_physics_time_seconds'] != request['snapshot']['source_sim_time_seconds']:
                    raise RuntimeError('camera packet and scene snapshot clock identity mismatch')
                if request['packet']['snapshot_sequence'] != request['snapshot']['sequence']:
                    raise RuntimeError('camera packet and scene snapshot sequence mismatch')
                oracle=bridge.apply(request['snapshot'])
                before=env._scene_step_clock.snapshot().time_seconds
                # Snapshot already contains source wrist poses and light state.
                # get_obs would resample robot state/random light from the replica.
                env.scene.update_render();env.cameras.update_picture()
                rgb=env.cameras.get_rgb()
                if env._scene_step_clock.snapshot().time_seconds != before:
                    raise RuntimeError('renderer replica advanced physical time')
                images={name:np.ascontiguousarray(rgb[name]['rgb']).copy()
                        for name in ('head_camera','left_camera','right_camera')}
                results.put(dict(capture_id=request['capture_id'],images=images,
                    packet=request['packet'],renderer_started_wall=started,
                    renderer_finished_wall=time.perf_counter(),snapshot_oracle=oracle))
            except BaseException:
                results.put(dict(capture_id=request['capture_id'],error=traceback.format_exc()))
    except BaseException:
        ready.put(dict(ready=False,error=traceback.format_exc()))
    finally:
        if env is not None:
            try: env.close_env();env.release_episode_resources()
            except Exception: pass


class ProcessRenderWorker:
    def __init__(self, config: dict, *, startup_timeout: float=900.):
        ctx=mp.get_context('spawn')
        self.requests=ctx.Queue(maxsize=1);self.results=ctx.Queue(maxsize=1);ready=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=_serve,args=(copy.deepcopy(config),self.requests,self.results,ready),
            name='continuous-renderer',daemon=True)
        self.busy=False;self.latest=None;self.counter=0;self.superseded=0
        self.process.start()
        try:
            message=ready.get(timeout=startup_timeout)
            if not message['ready']: raise RuntimeError(message['error'])
            self.identity=message['identity'];self.pid=message['pid']
        except BaseException:
            self.close();raise
        finally: ready.close()

    def submit(self, snapshot: dict, packet: dict) -> int:
        if not self.process.is_alive(): raise RuntimeError('renderer process exited')
        self.counter+=1
        request=dict(capture_id=self.counter,snapshot=copy.deepcopy(snapshot),packet=copy.deepcopy(packet))
        if self.busy:
            self.superseded+=int(self.latest is not None)
            self.latest=request
        else:
            self.requests.put_nowait(request);self.busy=True
        return self.counter

    def poll(self):
        try: result=self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive(): raise RuntimeError(f'renderer exited with {self.process.exitcode}')
            return None
        self.busy=False
        if 'error' in result: raise RuntimeError(result['error'])
        if self.latest is not None:
            request=self.latest;self.latest=None
            self.requests.put_nowait(request);self.busy=True
        return result

    def close(self):
        self.latest=None
        if self.process.is_alive():
            try:self.requests.put_nowait(None)
            except queue.Full:pass
            self.process.join(timeout=2.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5.)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=5.)
        self.requests.cancel_join_thread();self.results.cancel_join_thread()
        self.requests.close();self.results.close()
