from types import SimpleNamespace as S
import numpy as np
import pytest
import torch

from dynamicwam.dualrate.native_history import NativePhysicalHistory, capture_rendered_rgb
from dynamicwam.dualrate.online import history_endpoints, OnlineBatchBuilder
from dynamicwam.dualrate.producer import CAMERAS, flow_history_selection
from dynamicwam.dualrate.timing import TimingError


def obs(now=0.):
    return dict(observation={c: dict(rgb=np.full((8,8,3), int(now*10)%255, np.uint8)) for c in CAMERAS},
                joint_action=dict(vector=np.full(14, now, np.float32)))


class Env:
    def __init__(self, dt=.004):
        self.time=0.;self.renders=0;self.steps=0;self.count=0;self.captures=[]
        self.dt=dt
        self.scene=S(get_timestep=lambda:dt)
        self._scene_step_clock=S(snapshot=lambda:S(time_seconds=self.time))
    def _update_render(self):
        self.renders+=1
    def action(self, steps, render=True):
        self.count+=1
        self._update_render()
        for _ in range(steps):
            self.time+=self.dt;self.steps+=1
            if render:self._update_render()
    def capture(self):
        self.captures.append(self.time)
        return obs(self.time)


def run(steps, extra_queries=0, dt=.004):
    e=Env(dt);h=NativePhysicalHistory(capture=lambda env:env.capture())
    h.query_frames(e,obs())
    for n in steps:
        for _ in range(extra_queries):h.query_frames(e,obs(e.time))
        with h.during_action(e):e.action(n)
        frames=h.query_frames(e,obs(e.time))
    return e,h,frames


def test_same_physical_trace_independent_of_action_and_query_boundaries():
    a,ha,fa=run([124]*8, extra_queries=80)
    b,hb,fb=run([31]*32)
    assert a.steps==b.steps==992 and a.time==b.time
    assert a.captures==b.captures and a.renders==a.steps+8 and b.renders==b.steps+32
    assert [x.exposure for x in ha.history.frames]==[x.exposure for x in hb.history.frames]
    selected,valid=history_endpoints(fa)
    assert all(valid)
    # Query can lie between 60ms sensor ticks; preserve its actual exposure.
    assert np.diff([x.exposure for x in selected])==pytest.approx([.72]*5,abs=.0341)
    assert np.array_equal(selected[-1].drive_target,obs(a.time)['joint_action']['vector'])
    assert not selected[0].images[0].flags.writeable


@pytest.mark.parametrize('dt',[.004,float(np.float32(.004)),.007])
def test_training_selector_matches_exact_same_capture_log(dt):
    e,h,frames=run([23,124,3,800,10,101],dt=dt)
    times=np.array([x.exposure for x in frames])
    indices,mask=flow_history_selection(times,len(times)-1,interval_seconds=.72)
    selected,online_mask=history_endpoints(frames,action_dt=.18)
    assert list(online_mask)==mask
    assert [x.frame_id for x in selected]==[frames[i].frame_id for i in indices]
    assert all(mask)
    for event in h.capture_events:
        assert -1e-6<=event['exposure']-event['requested_exposure']<=dt+1e-6


def test_startup_and_missing_endpoints_remain_masked():
    e=Env();h=NativePhysicalHistory(capture=lambda env:env.capture())
    f=h.query_frames(e,obs())
    assert history_endpoints(f)[1]==(False,False,False,False,False,True)
    with h.during_action(e):e.action(180)
    f=h.query_frames(e,obs(e.time))
    assert history_endpoints(f)[1]==(False,False,False,False,True,True)


def test_observation_does_not_step_or_repeat_render_or_lighting():
    e=Env();e.robot=S(get_left_arm_jointState=lambda:list(range(7)),get_right_arm_jointState=lambda:list(range(7,14)))
    calls=[]
    e.cameras=S(update_picture=lambda:calls.append('picture'),get_rgb=lambda:obs()['observation'])
    result=capture_rendered_rgb(e)
    assert calls==['picture'] and e.steps==0 and e.renders==0
    assert result['joint_action']['vector'].tolist()==list(range(14))


def test_exception_restores_method_and_missing_step_render_fails():
    e=Env();h=NativePhysicalHistory(capture=lambda env:env.capture());h.query_frames(e,obs())
    original=e._update_render
    with pytest.raises(RuntimeError,match='injected'):
        with h.during_action(e):raise RuntimeError('injected')
    assert e._update_render==original and '_update_render' not in vars(e) and not h.active
    with pytest.raises(TimingError,match='without its due'):
        with h.during_action(e):e.action(30,render=False)
    assert e._update_render==original


def test_render_jump_cannot_fabricate_missed_captures_and_reset_is_rejected():
    e=Env();h=NativePhysicalHistory(capture=lambda env:env.capture());h.query_frames(e,obs())
    with pytest.raises(TimingError,match='missed'):
        with h.during_action(e):
            e.time=.2;e._update_render()
    assert not h.capture_events
    with pytest.raises(TimingError,match='new episode'):
        h.query_frames(Env(),obs())


def test_native_builder_selects_physical_history_for_both_current_and_source(monkeypatch):
    from dynamicwam.dualrate import online
    from test_dualrate_online_preprocessing_parity import setup_source,RecordingVAE
    builder,frames,_,_=setup_source(monkeypatch)
    manifest=dict(builder.manifest,time_contract='native_event_single_v6')
    native=OnlineBatchBuilder(manifest,action_mean=builder.mean,action_scale=builder.scale,language=builder.language)
    assert not native.native_ordinal and native.history_sampling=='physical_time'
    original=online.history_features;calls=[]
    def record(*args,**kw):
        calls.append(kw['native_ordinal']);return original(*args,**kw)
    monkeypatch.setattr(online,'history_features',record)
    t=frames[-1].available
    native.current(frames,query_time=t,action_start=t)
    native.source(frames,source_time=t,action_start=t,vae=RecordingVAE(torch.float32))
    assert calls==[False,False]


def test_source_derivation_preserves_frozen_model_and_rejects_wrong_checkpoint():
    import hashlib,subprocess
    from pathlib import Path
    from dynamicwam.dualrate.native_v7_source_compat import validate_deployment_source,TRAINING_COMMIT
    root=Path(__file__).resolve().parents[1]
    prefix='src/dynamicwam/'
    names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',TRAINING_COMMIT,prefix],text=True).splitlines()
    h=hashlib.sha256()
    for name in sorted(x for x in names if x.endswith('.py')):
        data=subprocess.check_output(['git','-C',str(root),'show',TRAINING_COMMIT+':'+name])
        h.update(name[len(prefix):].encode()+b'\0');h.update(hashlib.sha256(data).digest())
    receipt=validate_deployment_source(h.hexdigest())
    assert receipt['protected_model_and_training_equal']
    assert 'dualrate/online.py' in receipt['deployment_changes']
    with pytest.raises(ValueError,match='exact frozen'):
        validate_deployment_source('0'*64)


def test_v7_policy_connects_tick_capture_without_changing_action_ledger(monkeypatch):
    from dynamicwam.dualrate.native_v7_deployment import NativeDualRatePolicy
    from dynamicwam.dualrate.actuators import ActuatorProjection
    class Builder:
        device=slow_device=torch.device('cpu')
        manifest={'time_contract':'native_event_single_v6'}
        native_ordinal=False;history_sampling='physical_time';action_dt=.18
        mean=torch.zeros(14);scale=torch.ones(14)
        def source(self,frames,*,source_time,**kw):
            return dict(source_initial_state=torch.zeros(1,14),source_time=torch.tensor([source_time],dtype=torch.float64))
        def current(self,frames,*,query_time,**kw):
            return dict(state=torch.zeros(1,14),query_time=torch.tensor([query_time],dtype=torch.float64))
    class Model:
        native_v7={'prior_maximum':.15}
        slow=S(native=S(compact_wan=S(video_model=S(vae=None))))
        _fast_graph_sampler=S(receipt=lambda:{'passed':True})
        def eval(self):return self
        def predict_reference(self,batch,**kw):return dict(source_time=batch['source_time'])
        def world_readout(self,r):return r
        def sample_fast(self,*args,**kw):return {'actions':torch.zeros(1,16,14)}
    bounds=ActuatorProjection([[-10,10]]*6+[[0,1]]+[[-10,10]]*6+[[0,1]],provenance='unit fixture')
    monkeypatch.setattr(ActuatorProjection,'from_robot',classmethod(lambda cls,robot:bounds))
    class PolicyEnv(Env):
        take_action_cnt=0
        def take_action(self,a,action_type):
            assert action_type=='qpos'
            self.action([120,128][self.take_action_cnt%2])
            self.take_action_cnt+=1
    e=PolicyEnv();e.robot=None
    p=NativeDualRatePolicy(Model(),Builder());p.sensors.capture=lambda env:env.capture()
    for _ in range(10):p.step(e,obs(e.time))
    queries=[x for x in p.records if 'query_physical_time' in x]
    execution=[x for x in p.records if x.get('kind')=='native_topp_execute']
    assert [x['command_index'] for x in queries]==[0,2,4,6,8]
    assert len(execution)==e.take_action_cnt==10 and e.steps==1240
    assert all(queries[-1]['history_endpoint_valid'])
    assert np.diff(queries[-1]['history_exposure_times'])==pytest.approx([.72]*5,abs=.0341)
    assert all(x['prefix_rows']==0 for x in queries)
    assert e._update_render.__func__ is Env._update_render
