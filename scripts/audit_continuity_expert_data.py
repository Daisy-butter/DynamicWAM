"""CPU-only real-label, physical-time and online/offline history audit."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
import h5py
from dynamicwam.dualrate import producer as p
from dynamicwam.dualrate.online import OnlineFrame,history_features
from dynamicwam.dualrate.system import DualRateSystem

torch.set_num_threads(2)
root=Path(sys.argv[1]); output=Path(sys.argv[2]); examples=[];histories=[];subjects={}
def task(ep):return ep['task']
for split in ('train','validation'):
    data=p.DominoPhysicalDataset(root/(split+'.json'))
    subject={k:v for k,v in data.manifest.items() if k not in ('continuous_acceptance_file','continuous_acceptance_sha256')}
    subjects[split]=hashlib.sha256(json.dumps(subject,sort_keys=True).encode()).hexdigest()
    assert data.manifest['continuous_flow_sampling']=='raw_frames_v1'
    assert data.exposure_anchored and data.manifest['continuous_tail_policy']=='valid_prefix_and_two_free_rows'
    selected={}
    for idx,(epi,frame,template) in enumerate(data.rows):selected.setdefault(task(data.episodes[int(epi)]),[]).append(idx)
    assert len(selected)==35
    indices=[]
    for name,rows in selected.items():
        indices.extend([rows[len(rows)//2],rows[-1]] if split=='train' else [rows[-1]])
    partial=0;source_partial=0
    for idx in indices:
        item=data[idx];e=float(item['source_time']);q=float(item['query_time']);dt=data.dt
        known=item['fast_known_mask'];valid=item['fast_action_valid']
        assert bool(valid[known].all()) and int((valid&~known).sum())>=2
        partial+=int(not bool(valid.all()))
        assert data.manifest.get('residual_v10')
        origin=item['action_times'][int(known.sum())]
        expected=origin+torch.tensor([.1,.2,.4,.8]*3,dtype=torch.float64)
        torch.testing.assert_close(item['future_times'],expected,atol=1e-7,rtol=0)
        exposure=float(item['state_time']);teacher=item['teacher_action_intervals']
        assert exposure==float(item['rgb_times'][0])
        assert float(teacher[0,0])<=exposure+1e-7<float(teacher[0,1])+1e-7
        tm=item['teacher_known_mask'];assert 0<int(tm.sum())<16
        assert abs(float(teacher[int(tm.sum())-1,1])-float(origin))<1e-7
        from dynamicwam.dualrate.residual_v10 import current_teacher_batch
        batch={k:v[None] for k,v in item.items() if isinstance(v,torch.Tensor)}
        current=current_teacher_batch(batch)
        assert float(current['source_time'])==exposure
        match=(teacher[:,None]-item['action_intervals'][None]).abs().amax(-1)<1e-7
        for ti,fi in match.nonzero().tolist():
            if known[fi]:assert torch.equal(current['source_known_actions'][0,ti],item['fast_known_actions'][fi])

        assert float(item['source_rgb_actual_times'])==e
        assert float(item['source_action_times'][0])==e
        assert torch.allclose((item['action_times']-e)/dt,((item['action_times']-e)/dt).round(),atol=1e-8,rtol=0)
        assert float(item['action_times'][0])<=q+1e-6<float(item['action_times'][0])+dt+1e-6
        for stem in ('source','fast'):
            mask=item[stem+'_known_mask'];target=item[stem+'_action'];values=item[stem+'_known_actions']
            assert torch.equal(target[mask],values[mask]) and torch.count_nonzero(values[~mask])==0
        rgb_valid=item.get('source_future_rgb_valid',torch.ones_like(item['future_rgb_time_errors'],dtype=torch.bool))
        if rgb_valid.any():assert float(item['future_rgb_time_errors'][rgb_valid].abs().max())<=.030001
        if not item['source_action_valid'].all():
            source_partial+=1
            assert data.manifest['continuous_source_tail_policy']=='known_only_frozen_slow_v11'
            assert item['source_action_valid'][item['source_known_mask']].all()
            assert not rgb_valid.all()
        for stem in ('source_motion','motion'):
            mv=item[stem+'_valid'];times=item[stem+'_times']
            assert torch.allclose(item[stem][mv,4],(times[mv,1]-times[mv,0]).float(),atol=1e-6,rtol=0)
            assert not mv.any() or float(times[mv,1].max())<=float(item['source_time'] if stem.startswith('source') else item['rgb_times'][0])+1e-6
        clocks={k:v[None] for k,v in item.items() if k in ('source_time','completed_time','query_time','source_request_time','source_available_time','query_available_time')}
        DualRateSystem._validate_clocks(clocks)
        epi,frame,_=map(int,data.rows[idx]);ep=data.episodes[epi]
        records,actions,_,_=data._episode(epi);raw_times=records.timestamps.numpy()
        # Independently check the mixed actuator semantics on real command logs:
        # arm endpoint and interval-start previous-event gripper, including flips.
        for stem,time_key in [('source','source_action_times'),('fast','action_times')]:
            starts=item[time_key].numpy()
            endpoint,ev,_=p.resample_actions(raw_times,actions.numpy(),starts+dt)
            unique=p.unique_last_indices(raw_times);ut=raw_times[unique]
            ids=np.searchsorted(ut,starts+1e-8,side='right')-1
            expected=endpoint.copy();expected[:,[6,13]]=actions.numpy()[unique[np.maximum(ids,0)]][:,[6,13]]
            mask=item[stem+'_action_valid'].numpy()
            actual=(item[stem+'_action']*data.action_std+data.action_mean).numpy()
            np.testing.assert_allclose(actual[mask],expected[mask],atol=2e-6,rtol=1e-6)
        examples.append(dict(split=split,index=idx,task=task(ep),source=e,query=q,known=int(known.sum()),valid_rows=int(valid.sum())))
        # One real history per task, including actual optical estimation. Both
        # paths receive identical real frames and timestamps, no mocked flow.
        if split=='train' and idx==selected[task(ep)][len(selected[task(ep)])//2]:
            records,actions,cache,_=data._episode(epi);times=records.timestamps.numpy()
            anchor=p.select_frame(times,float(item['rgb_times'][0]),past_only=True,tolerance=1e-6)
            ids,matched=p.flow_history_selection(times,anchor,interval_seconds=4*dt)
            ids=sorted(set(int(i) for i,ok in zip(ids,matched) if ok))
            with h5py.File(ep['raw_path'],'r') as handle:
                frames=[OnlineFrame((p._rgb(handle,i,p.CAMERAS[0]),)*3,np.zeros(14,np.float32),float(times[i]),float(times[i]),i) for i in ids]
                offline=p._flow_features(handle,times,anchor,cache,data.manifest['flow_contract'],interval_seconds=4*dt,return_flow_rgb=True,use_numeric_cache=False)
            online=history_features(frames,data.manifest['flow_contract'],action_dt=dt)
            numeric_errors={}
            for a,key in zip(offline[:-1],('motion','motion_valid','motion_acceleration_valid','motion_times','spatial_flow','spatial_flow_valid','spatial_flow_times')):
                numeric=key in ('motion','spatial_flow')
                torch.testing.assert_close(a,online[key],rtol=1e-6 if numeric else 0,atol=1e-6 if numeric else 0)
                if numeric:numeric_errors[key]=float((a-online[key]).abs().max())
            np.testing.assert_array_equal(offline[-1],online['flow_rgb'])
            histories.append(dict(task=task(ep),frame_ids=ids,times=[float(times[i]) for i in ids],time_masks_and_rgb_bitwise_equal=True,numeric_max_abs=numeric_errors,numeric_atol=1e-6,numeric_rtol=1e-6))
    assert partial>0,'real tail coverage is absent from audit samples'
    assert source_partial>0,'frozen-source terminal windows are absent'
assert len(histories)==35 and not torch.cuda.is_initialized()
output.write_text(json.dumps(dict(passed=True,cuda_used=False,examples=examples,history_parity=histories,audited_subjects=subjects,
    interpretation='actual expert labels; not counterfactual supervision for new policy actions'),indent=2))
print(json.dumps(dict(passed=True,examples=len(examples),history_tasks=len(histories))))
