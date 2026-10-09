"""Bounded, in-memory behavior trajectories; persistence only after physics ends.

This records observations already returned by the camera and commands actually
passed to step_command. It never renders, invokes a teacher, reconstructs a
missing action, or exposes a future target to the policy. Image times are the
source snapshot times, not renderer completion times.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

FORMAT='dynamicwam_continuous_trajectory'
VERSION=1
CONTROLLER_VECTORS=('q','v','start_q','start_v','goal','end_v','vmax','amax')
POLICY_ARRAYS=('frame','flow_frames','motion_features','motion_interval_valid_mask',
               'motion_acceleration_valid_mask','previous_head_rgb')


class TrajectoryRecordingError(RuntimeError):
    pass


class ExecutedCommandRing:
    """Exactly the past two physical seconds; never a predicted command path."""
    def __init__(self,*,seconds=2.,dt=.004):
        if seconds!=2. or dt!=.004:raise ValueError('causal command prefix requires the fixed 2s/4ms bound')
        self.dt=dt;self.capacity=round(seconds/dt);self.count=0
        self.q=np.empty((self.capacity,14),np.float64);self.velocity=np.empty_like(self.q)

    def append(self,q,velocity,*,tick_index):
        if tick_index!=self.count:raise TrajectoryRecordingError('executed prefix has a missing or duplicate tick')
        slot=tick_index%self.capacity
        self.q[slot]=_finite(q,(14,));self.velocity[slot]=_finite(velocity,(14,));self.count+=1

    def prefix(self,source_time,dispatch_time):
        if not np.isfinite([source_time,dispatch_time]).all():raise TrajectoryRecordingError('nonfinite prefix clock')
        start,end=round(source_time/self.dt),round(dispatch_time/self.dt)
        if (abs(source_time-start*self.dt)>1e-8 or abs(dispatch_time-end*self.dt)>1e-8 or
            start<0 or start>end or end!=self.count or end-start>self.capacity):
            raise TrajectoryRecordingError('source-to-dispatch commands exceed 2s or do not cover the exact executed grid')
        ticks=np.arange(start,end,dtype=np.int64);indices=ticks%self.capacity
        return dict(start_sim_time=ticks*self.dt,end_sim_time=(ticks+1)*self.dt,
            q_command=self.q[indices].copy(),velocity_command=self.velocity[indices].copy())


def _json(value):
    def plain(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=plain)


def _finite(value,shape=None):
    array=np.asarray(value)
    if (shape is not None and array.shape!=shape) or not np.isfinite(array).all():
        raise TrajectoryRecordingError('nonfinite or wrong-shaped trajectory value')
    return array


def _load_view_encoder():
    # Import before the timed loop. This is exactly the policy crop/letterbox,
    # inverted back to byte RGB without a second resizing implementation.
    from dynamicwam.observations.multiview import fast_views_from_composite,IMAGENET_MEAN,IMAGENET_STD
    def encode(frame):
        normalized,mask=fast_views_from_composite(frame)
        rgb=normalized.transpose(0,2,3,1)*IMAGENET_STD+IMAGENET_MEAN
        return np.rint(rgb*255).clip(0,255).astype(np.uint8),mask
    return encode


def controller_snapshot(controller,*,source_physics_time=None,future_knots=()):
    """JSON-only exact controller state at the image's source physical time."""
    return dict(**{name:_finite(getattr(controller,name),(14,)).tolist() for name in CONTROLLER_VECTORS},
                elapsed=float(controller.elapsed),period=float(controller.period),
                ticks=int(controller.ticks),limited_ticks=int(controller.limited_ticks),initialized=True,
                source_physics_time=float(controller.ticks*.004 if source_physics_time is None else source_physics_time),
                future_knots=[dict(end_slot=int(row['end_slot']),q=_finite(row['q'],(14,)).tolist()) for row in future_knots])


class TrajectoryRecorder:
    def __init__(self,*,max_seconds=30.,physics_dt=.004,control_period=.1,query_slots=2,
                 memory_limit_bytes=512*1024**2,event_limit_bytes=16*1024**2):
        if not 0<max_seconds<=30 or physics_dt!=.004 or control_period!=.1 or query_slots<1:
            raise ValueError('recording requires the bounded <=30s, 4ms/100ms controller protocol')
        self.max_seconds=float(max_seconds);self.dt=float(physics_dt);self.period=float(control_period)
        self.query_slots=int(query_slots);self.tick_capacity=math.ceil(max_seconds/physics_dt)
        # A stale ready frame may schedule a fresh capture on the next 100ms
        # slot, even with H2 queries. Bound that actual maximum, not nominal H2.
        self.frame_capacity=math.ceil(max_seconds/control_period)+2
        self.memory_limit=int(memory_limit_bytes);self.event_limit=int(event_limit_bytes)
        if self.event_limit<=0 or self.memory_limit<=self.event_limit:raise ValueError('invalid recording memory bound')
        self.allocated=0;self.events=[];self.event_bytes=0;self.tick_count=self.frame_count=0
        self.state='new';self.metadata={};self.measured={};self.measured_names={};self.measurement_errors=[]
        self.policy_arrays={};self.capture_indices={}
        self.frames={
            'rgb':self._allocate((self.frame_capacity,3,160,160,3),np.uint8),
            'rgb_valid_mask':self._allocate((self.frame_capacity,3,160,160),np.bool_),
            'drive_targets':self._allocate((self.frame_capacity,14),np.float64),
            'policy_state':self._allocate((self.frame_capacity,14),np.float32)}
        for key in ('observation_sim_time','source_physics_time','capture_roundoff_s','received_sim_time',
                    'camera_ready_wall','history_dt','policy_submitted_wall','controller_elapsed','controller_period',
                    'controller_source_physics_time','dispatch_sim_time'):
            self.frames[key]=self._allocate((self.frame_capacity,),np.float64)
        for key in ('capture_id','snapshot_sequence','command_index','policy_sampling_seed','policy_request_id',
                    'controller_ticks','controller_limited_ticks','controller_future_knot_count'):
            self.frames[key]=self._allocate((self.frame_capacity,),np.int64)
        for key in ('history_valid','policy_submitted','policy_reset','controller_initialized'):
            self.frames[key]=self._allocate((self.frame_capacity,),np.bool_)
        for key in CONTROLLER_VECTORS:
            self.frames['controller_'+key]=self._allocate((self.frame_capacity,14),np.float64)
        self.frames['controller_future_knot_q']=self._allocate((self.frame_capacity,32,14),np.float64)
        self.frames['controller_future_knot_end_slot']=self._allocate((self.frame_capacity,32),np.int64)
        for key in ('policy_sampling_seed','policy_request_id','policy_submitted_wall'):
            self.frames[key].fill(-1)
        self.commands={'q_command':self._allocate((self.tick_capacity,14),np.float64),
            'velocity_command':self._allocate((self.tick_capacity,14),np.float64),
            'hold':self._allocate((self.tick_capacity,),np.bool_)}
        for key in ('start_sim_time','end_sim_time','engine_start_time','engine_end_time',
                    'start_roundoff_s','end_roundoff_s','observation_sim_time','action_origin_sim_time'):
            self.commands[key]=self._allocate((self.tick_capacity,),np.float64)
        for key in ('tick_index','request_id','returned_row_index'):
            self.commands[key]=self._allocate((self.tick_capacity,),np.int64)

    def _allocate(self,shape,dtype):
        size=math.prod(shape)*np.dtype(dtype).itemsize
        if self.allocated+size+self.event_limit>self.memory_limit:
            raise TrajectoryRecordingError('trajectory memory capacity exceeded; downsampling is forbidden')
        self.allocated+=size
        return np.zeros(shape,dtype=dtype)

    def begin(self,*,task,seed,origin_sim_time,initial_drive_targets,physics_timeout):
        if self.state!='new' or physics_timeout>self.max_seconds+1e-9:
            raise TrajectoryRecordingError('recorder cannot be reused or exceed its physical-time budget')
        self.encoder=_load_view_encoder()
        self.metadata=dict(task=str(task),seed=int(seed),origin_sim_time=float(origin_sim_time),
            initial_drive_targets=_finite(initial_drive_targets,(14,)).tolist(),physics_timeout=float(physics_timeout))
        self.state='recording'

    def _recording(self):
        if self.state!='recording':raise TrajectoryRecordingError('recorder is not in an active physical episode')

    def event(self,event):
        self._recording();encoded=_json(event)
        size=len(encoded.encode('utf-8'))+1
        if self.event_bytes+size>self.event_limit:
            raise TrajectoryRecordingError('event memory capacity exceeded; dropping events is forbidden')
        if event.get('event')=='submit':
            i=self.capture_indices.get(event.get('capture_id'))
            if i is None or self.frames['policy_submitted'][i]:
                raise TrajectoryRecordingError('submission must refer to one recorded, previously unsubmitted camera frame')
            seed=event.get('sampling_seed');request=event.get('request_id')
            if type(seed) is not int or not 0<=seed<2**63 or type(request) is not int or request<0:
                raise TrajectoryRecordingError('invalid actual policy query identity')
            if self.frames['dispatch_sim_time'][i]!=event.get('dispatch_sim_time'):
                raise TrajectoryRecordingError('query event differs from the recorded actual dispatch time')
            self.frames['policy_submitted'][i]=True;self.frames['policy_sampling_seed'][i]=seed
            self.frames['policy_request_id'][i]=request;self.frames['policy_submitted_wall'][i]=event['submitted_wall']
            self.frames['policy_reset'][i]=bool(event['initial'])
        self.events.append(encoded);self.event_bytes+=size

    def mark_query(self,packet):
        """Bind an actual dispatch to its already captured frame before IPC."""
        self._recording();i=self.capture_indices.get(packet.get('capture_id'))
        if i is None or self.frames['policy_submitted'][i]:
            raise TrajectoryRecordingError('dispatch must refer to an unsubmitted recorded frame')
        source=self.frames['observation_sim_time'][i];dispatch=float(packet['dispatch_sim_time'])
        state=packet['controller_state'];actual=state['dispatch_controller'];prefix=state['executed_prefix']
        start,end=round(source/self.dt),round(dispatch/self.dt)
        if (not np.isfinite(dispatch) or dispatch<source or dispatch!=end*self.dt or end!=self.tick_count or
            actual['source_physics_time']!=dispatch or end-start>500):
            raise TrajectoryRecordingError('dispatch state or prefix clock does not match completed physics ticks')
        for given,saved in (('start_sim_time','start_sim_time'),('end_sim_time','end_sim_time'),
                            ('q_command','q_command'),('velocity_command','velocity_command')):
            if not np.array_equal(np.asarray(prefix[given]),self.commands[saved][start:end]):
                raise TrajectoryRecordingError('dispatch prefix differs from the commands actually sent to physics')
        self.frames['dispatch_sim_time'][i]=dispatch

    def observation(self,packet,*,received_sim_time):
        self._recording();i=self.frame_count
        if i>=self.frame_capacity:raise TrajectoryRecordingError('camera memory capacity exceeded; dropping frames is forbidden')
        stamp=float(packet['observation_sim_time']);source=float(packet['source_physics_time_seconds'])
        if not np.isfinite([stamp,source,received_sim_time]).all() or not 0<=stamp<=self.max_seconds+1e-7:
            raise TrajectoryRecordingError('invalid observation physical timestamp')
        if i and stamp<=self.frames['observation_sim_time'][i-1]:
            raise TrajectoryRecordingError('duplicate/nonmonotonic camera source timestamp')
        if received_sim_time+1e-7<stamp or abs(source-stamp)>max(1e-7,abs(stamp)*1e-7):
            raise TrajectoryRecordingError('camera snapshot retimestamping or negative camera age')
        oracle=packet.get('snapshot_oracle',{})
        if oracle.get('passed') is not True or oracle.get('physics_steps')!=0:
            raise TrajectoryRecordingError('camera frame lacks zero-step restore evidence')
        rgb,mask=self.encoder(packet['frame'])
        if rgb.shape!=(3,160,160,3) or rgb.dtype!=np.uint8 or mask.shape!=(3,160,160) or mask.dtype!=np.bool_:
            raise TrajectoryRecordingError('shared fast-view preprocessing contract changed')
        self.frames['rgb'][i]=rgb;self.frames['rgb_valid_mask'][i]=mask
        self.frames['drive_targets'][i]=_finite(packet['_trajectory_drive_targets'],(14,))
        self.frames['policy_state'][i]=_finite(packet['state'],(14,))
        instruction=packet['instruction']
        if not isinstance(instruction,str) or len(instruction)>16384:
            raise TrajectoryRecordingError('invalid or unbounded task instruction')
        if i and instruction!=self.metadata['instruction']:
            raise TrajectoryRecordingError('episode instruction changed')
        self.metadata['instruction']=instruction
        for name in POLICY_ARRAYS:
            value=np.asarray(packet[name])
            if value.ndim<1 or value.size==0 or value.dtype.kind not in 'bufi' or not np.isfinite(value).all():
                raise TrajectoryRecordingError('invalid causal policy array: '+name)
            if name in ('frame','flow_frames','previous_head_rgb') and (value.dtype!=np.uint8 or value.shape[-1]!=3):
                raise TrajectoryRecordingError('policy RGB must retain exact uint8 bytes: '+name)
            if name.endswith('_mask') and value.dtype!=np.bool_:
                raise TrajectoryRecordingError('motion validity masks must remain bool')
            key='policy_frame' if name=='frame' else name
            if key not in self.policy_arrays:self.policy_arrays[key]=self._allocate((self.frame_capacity,*value.shape),value.dtype)
            destination=self.policy_arrays[key]
            if destination.shape[1:]!=value.shape or destination.dtype!=value.dtype:
                raise TrajectoryRecordingError('causal policy array shape/dtype changed: '+name)
            destination[i]=value
        history_dt=float(packet['history_dt']);history_valid=packet['history_valid']
        if not np.isfinite(history_dt) or history_dt<0 or type(history_valid) not in (bool,np.bool_) or (history_valid and history_dt<=0):
            raise TrajectoryRecordingError('invalid causal history metadata')
        self.frames['history_dt'][i]=history_dt;self.frames['history_valid'][i]=history_valid
        controller=packet['controller_state']
        for key in CONTROLLER_VECTORS:self.frames['controller_'+key][i]=_finite(controller[key],(14,))
        for key in ('elapsed','period','ticks','limited_ticks','initialized','source_physics_time'):
            value=controller[key]
            if not np.isfinite(value):raise TrajectoryRecordingError('nonfinite source controller state')
            self.frames['controller_'+key][i]=value
        if (controller['initialized'] is not True or controller['period']!=self.period or
            not 0<=controller['elapsed']<=self.period+1e-8 or controller['ticks']<0 or
            not 0<=controller['limited_ticks']<=controller['ticks']):
            raise TrajectoryRecordingError('invalid source controller state')
        knots=controller['future_knots']
        if len(knots)>32 or controller['source_physics_time']!=stamp:
            raise TrajectoryRecordingError('unbounded future queue or mismatched controller snapshot time')
        end_slots=[k['end_slot'] for k in knots]
        if end_slots!=sorted(set(end_slots)) or any(type(k)is not int or k*self.period<=stamp for k in end_slots):
            raise TrajectoryRecordingError('future controller knots must be unique ordered causal future slots')
        self.frames['controller_future_knot_count'][i]=len(knots)
        for j,knot in enumerate(knots):
            self.frames['controller_future_knot_q'][i,j]=_finite(knot['q'],(14,))
            self.frames['controller_future_knot_end_slot'][i,j]=knot['end_slot']
        for key,value in dict(observation_sim_time=stamp,source_physics_time=source,capture_roundoff_s=source-stamp,
            received_sim_time=received_sim_time,camera_ready_wall=packet['camera_ready_wall'],capture_id=packet['capture_id'],
            snapshot_sequence=packet['snapshot_sequence'],command_index=packet['command_index'],dispatch_sim_time=stamp).items():
            if not np.isfinite(value):raise TrajectoryRecordingError('nonfinite camera metadata')
            self.frames[key][i]=value
        measured=packet.get('_trajectory_measured_joint_state',{})
        for side in ('left','right'):
            item=measured.get(side,{})
            if item.get('error'):
                self.measurement_errors.append(dict(frame_index=i,side=side,error=str(item['error'])[:256]))
            for kind in ('qpos','qvel'):
                value=item.get(kind)
                if value is None:continue
                value=_finite(value)
                if value.ndim!=1 or not 0<len(value)<=256:raise TrajectoryRecordingError('unbounded measured articulation layout')
                key=side+'_'+kind;names=list(item.get('joint_names',[]))
                if len(names)>256 or any(not isinstance(n,str) or len(n)>128 for n in names):
                    raise TrajectoryRecordingError('invalid measured joint name mapping')
                if key not in self.measured:
                    self.measured[key]=self._allocate((self.frame_capacity,len(value)),np.float64)
                    self.measured[key+'_valid']=self._allocate((self.frame_capacity,),np.bool_)
                    self.measured_names[key]=names
                if value.shape!=self.measured[key].shape[1:] or self.measured_names[key]!=names:
                    raise TrajectoryRecordingError('measured articulation layout changed during episode')
                self.measured[key][i]=value;self.measured[key+'_valid'][i]=True
        capture_id=int(packet['capture_id'])
        if capture_id in self.capture_indices:raise TrajectoryRecordingError('duplicate camera capture identity')
        self.capture_indices[capture_id]=i;self.frame_count+=1

    def tick(self,q,velocity,*,tick_index,start_sim_time,end_sim_time,engine_start_time,engine_end_time,hold,record):
        self._recording();i=self.tick_count
        if i>=self.tick_capacity or tick_index!=i:raise TrajectoryRecordingError('tick capacity/continuity violation')
        if abs(start_sim_time-i*self.dt)>1e-8 or abs(end_sim_time-(i+1)*self.dt)>1e-8:
            raise TrajectoryRecordingError('command intervals must cover the contiguous physical grid')
        origin=self.metadata['origin_sim_time'];before=engine_start_time-origin;after=engine_end_time-origin
        if (not np.isfinite([before,after]).all() or abs(before-start_sim_time)>self.dt*.1 or
                abs(after-end_sim_time)>self.dt*.1):raise TrajectoryRecordingError('actual physics clock does not cover applied command')
        self.commands['q_command'][i]=_finite(q,(14,));self.commands['velocity_command'][i]=_finite(velocity,(14,))
        values=dict(tick_index=i,start_sim_time=start_sim_time,end_sim_time=end_sim_time,
            engine_start_time=engine_start_time,engine_end_time=engine_end_time,
            start_roundoff_s=before-start_sim_time,end_roundoff_s=after-end_sim_time,hold=bool(hold),
            request_id=-1 if record is None else int(record['request_id']),
            observation_sim_time=-1. if record is None else float(record['observation_sim_time']),
            action_origin_sim_time=-1. if record is None else float(record['action_origin_sim_time']),
            returned_row_index=-1 if record is None else round(start_sim_time/self.period)-round(record['action_origin_sim_time']/self.period))
        for key,value in values.items():self.commands[key][i]=value
        self.tick_count+=1

    def finish(self,report,*,pending_camera_at_end=None):
        self._recording()
        if report['physics_ticks']!=self.tick_count:
            self.abort('applied command count differs from owner physics ticks')
            raise TrajectoryRecordingError(self.metadata['error'])
        valid=report.get('realtime_valid') is True and self.frame_count>0 and self.tick_count>0
        self.state='complete' if valid else 'incomplete'
        self.metadata.update(stop_reason=report.get('stop_reason'),success=report.get('success'),
            realtime_valid=report.get('realtime_valid'),physical_end_time=report['physics_time'],
            terminal_inflight=report.get('terminal_inflight'),pending_camera_at_end=pending_camera_at_end)
        return self.summary()

    def abort(self,error):
        self.state='incomplete';self.metadata['error']=str(error)[:2048]

    def summary(self):
        return dict(format=FORMAT,version=VERSION,status=self.state,frames=self.frame_count,ticks=self.tick_count,
            allocated_array_bytes=self.allocated,event_bytes=self.event_bytes,memory_limit_bytes=self.memory_limit,
            frame_capacity=self.frame_capacity,tick_capacity=self.tick_capacity,storage='memory_until_episode_end')

    def flush_events(self,sink):
        if self.state not in ('complete','incomplete'):raise TrajectoryRecordingError('cannot flush during physics')
        for event in self.events:sink(json.loads(event))

    def write(self,directory,*,provenance):
        """Call after the loop returns; completion.json is the sole commit marker."""
        if self.state not in ('complete','incomplete'):raise TrajectoryRecordingError('cannot persist during physics')
        for key in ('controller_protocol_sha256','accepted_record_sha256','policy_config_sha256'):
            value=provenance.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise TrajectoryRecordingError('missing trajectory provenance: '+key)
        if not provenance.get('source_sha256'):raise TrajectoryRecordingError('missing trajectory source identities')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        artifacts={}
        for name,arrays,count in [('observations',dict(self.frames,**self.measured,**self.policy_arrays),self.frame_count),
                                   ('commands',self.commands,self.tick_count)]:
            path=directory/(name+'.npz')
            with path.open('xb') as stream:np.savez(stream,**{key:value[:count] for key,value in arrays.items()})
            artifacts[name]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        path=directory/'events.jsonl'
        with path.open('x',encoding='utf-8') as stream:stream.write('\n'.join(self.events)+'\n')
        artifacts['events']=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        manifest=dict(**self.summary(),metadata=self.metadata,provenance=provenance,artifacts=artifacts,
            physics_dt=self.dt,control_period=self.period,query_slots=self.query_slots,
            view_order=['head','left_wrist','right_wrist'],rgb_layout='frame,view,height,width,channel; uint8 RGB',
            rgb_preprocessing='shared_fast_views_from_composite_inverse_imagenet_round_uint8_v1',
            drive_state_semantics='commanded_joint_drive_targets; not measured q',
            measured_state_semantics='raw articulation qpos/qvel at source snapshot; explicit per-field validity masks',
            measured_joint_names=self.measured_names,measurement_errors=self.measurement_errors,
            causal_policy_inputs='exact camera-ready composite/flow/motion/masks/previous_head arrays; no future inputs',
            policy_query_identity='only actual submissions populate policy_sampling_seed/request_id; -1 for unqueried frames',
            dispatch_time_semantics='actual policy send physical time when policy_submitted; defaults to source time for unqueried frames',
            controller_state_semantics='LimitedHermiteController immediately after interval begin, at camera source time; distinct from measured joints',
            command_semantics='exact q/velocity arguments passed to step_command over [start_sim_time,end_sim_time); includes holds and old-plan commands',
            gripper_velocity_semantics='controller velocity reference only; step_command passes q to gripper API, no gripper velocity argument',
            missing_future='terminal and skipped camera timestamps are missing/censored; never forward-fill as a target')
        path=directory/'manifest.json'
        with path.open('x',encoding='utf-8') as stream:stream.write(_json(manifest)+'\n')
        completion=dict(format=FORMAT+'_completion',version=VERSION,status=self.state,
            manifest=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with (directory/'completion.json').open('x',encoding='utf-8') as stream:stream.write(_json(completion)+'\n')
        return completion
