import pytest
import torch
from types import SimpleNamespace
from dynamicwam.dualrate.handoff_v12 import reconcile,local_mask,CONTRACT
from dynamicwam.dualrate.continuity_v11 import defaults,ARMS

def case():
    s=SimpleNamespace(handoff_v12=dict(local_rows=4,handoff_step_rad=.06,same_step_rad=.02),
        residual_v10=dict(action_dt=.18,bounds=[[-10,10]]*14),action_scale_v10=torch.ones(14),action_mean_v10=torch.zeros(14))
    base=torch.zeros(1,16,14);known=torch.arange(16)[None]<2
    b=defaults(dict(fast_known_actions=base.clone(),fast_known_mask=known,query_time=torch.tensor([.36],dtype=torch.float64)))
    b['previous_valid'][:]=True;b['previous_actions'][:]=.04;b['previous_actions'][:,:,[6,13]]=0
    return s,b,base,torch.ones(1,16,dtype=torch.bool)

def test_cross_reference_zero_residual_no_longer_allows_point46_jump():
    s,b,base,valid=case();base[:,:,ARMS]=.5;original=base.clone()
    out,effective,m=reconcile(s,b,base,base,valid)
    assert torch.equal(base,original) and torch.equal(out,effective)
    assert m['active'].all() and m['raw_change_rad'].item()>.45
    assert m['achieved_change_rad'].item()<=.060001
    assert torch.equal(out[b['fast_known_mask']],b['fast_known_actions'][b['fast_known_mask']])
    torch.testing.assert_close(out[:,2:,ARMS],torch.full_like(out[:,2:,ARMS],.10))

def test_whole_future_plan_is_bounded_not_only_the_first_executed_row():
    s,b,base,valid=case();base[:,2:,ARMS]=.05;base[:,-1,ARMS]=1.
    out,_,m=reconcile(s,b,base,base,valid)
    assert m['achieved_change_rad'].item()<=.060001
    assert out[:,-1,ARMS].max()<=.100001
    # A common scalar retains coordinated arm direction.
    ratio=(out[:,2:,ARMS]-.04)/(base[:,2:,ARMS]-.04)
    torch.testing.assert_close(ratio,torch.full_like(ratio,m['alpha'].item()),atol=1e-6,rtol=0)

def test_same_reference_budget_and_physical_elapsed_time():
    s,b,base,valid=case();b['previous_reference_same'][:]=True
    b['previous_query_time']=b['query_time']-.09
    out,_,m=reconcile(s,b,base,base,valid)
    assert m['budget_rad'].item()==pytest.approx(.01)
    assert m['achieved_change_rad'].item()<=.010001
    b['previous_handoff']=torch.tensor([True])
    assert reconcile(s,b,base,base,valid)[2]['budget_rad'].item()==pytest.approx(.03)

def test_missing_tail_connects_and_repeated_queries_converge():
    s,b,base,valid=case();b['previous_valid'][:,12:]=False;base[:,:,ARMS]=.5
    previous=.04
    for _ in range(12):
        out,_,m=reconcile(s,b,base,base,valid)
        assert m['achieved_change_rad'].item()<=.060001
        assert out[:,2:,ARMS].min()>=previous-1e-6
        previous=out[:,2:,ARMS].min().item()
        b['previous_actions']=out.detach();b['previous_valid'][:]=True
        b['previous_handoff']=m['active'];b['previous_reference_same'][:]=True
    assert not m['active'].any()
    torch.testing.assert_close(out[:,2:,ARMS],base[:,2:,ARMS])

def test_grippers_are_full_events_not_fractional_interpolation():
    s,b,base,valid=case();base[:,:]=.5;base[:,:,[6,13]]=1
    out,eff,m=reconcile(s,b,base,base,valid)
    assert not out[:,:,[6,13]].any()
    assert torch.equal(out,eff)

def test_no_history_uses_committed_endpoint_and_local_mask_excludes_tail():
    s,b,base,valid=case();b['previous_valid'][:]=False;base[:,:,ARMS]=.5
    out,_,m=reconcile(s,b,base,base,valid)
    assert m['enabled'].all() and out[:,2:,ARMS].max()<=.060001
    assert local_mask(s,b,valid).sum()==4
    b['fast_known_mask'][:]=False
    out,_,m=reconcile(s,b,base,base,valid)
    assert not m['enabled'].any() and torch.equal(out,base)

def test_finite_gradient_and_local_feedback_zero_preservation():
    from test_residual_v10 import fixture,generator
    from dynamicwam.dualrate.continuity_v11 import configure,CONTRACT as V11
    s,b,w=fixture()
    configure(s,dict(contract=V11,reference_hold_steps=4,seam_weight=.1,revision_weight=.1,out_of_radius_weight=.2,revision_radius_fraction=1/3,
        handoff=dict(contract=CONTRACT,local_rows=4,handoff_step_rad=.06,same_step_rad=.02)))
    b['feedback_noise_seed']=torch.tensor([17]);b['action_grid_origin']=b['source_time'].clone()
    s.residual_v10['executor']='phase_c2';b['teacher_motion']=b['teacher_motion']+10.;s.eval()
    out=s.sample_fast(b,w,steps=1)
    torch.testing.assert_close(out['actions'],out['baseline_actions'],atol=1e-6,rtol=0)
    assert not out['learned_correction_actions'].any()
    losses=s.training_loss(b,generator=generator())
    assert torch.isfinite(losses['loss']) and 'teacher_all_first_over_radius' in losses
    losses['loss'].backward()
    assert torch.isfinite(s.fast.action_out.weight.grad).all() and s.fast.action_out.weight.grad.abs().max()>0
    assert all(p.grad is None for p in s.slow.parameters())

def test_limit_violation_fails_instead_of_independent_joint_clipping():
    s,b,base,valid=case();b['previous_actions'][:]=20.
    with pytest.raises(ValueError,match='joint limits'):reconcile(s,b,base,base,valid)

def test_logged_composition_and_violation_gate():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from dualrate_continuous_acceptance import validate_handoff_events
    s,b,base,valid=case();base[:,:,ARMS]=.5
    proposal=base.clone();proposal[:,2:6,ARMS]+=.01
    result,effective,m=reconcile(s,b,base,proposal,valid)
    r=dict(event='residual_action_audit',query_id=1,reference_version=2,baseline_raw=effective[0].tolist(),proposed_raw=result[0].tolist())
    h=dict(event='whole_plan_handoff_audit',query_id=1,reference_version=2,raw_full_baseline=base[0].tolist(),
        handoff_offset=(effective-base)[0].tolist(),learned_correction=(proposal-base)[0].tolist(),
        active=True,enabled=True,alpha=m['alpha'].item(),budget_rad=m['budget_rad'].item(),raw_revision_rad=m['raw_change_rad'].item(),revision_rad=m['achieved_change_rad'].item())
    cfg=dict(residual_v10=dict(arm_radius=.06,continuity=dict(handoff=s.handoff_v12)))
    assert validate_handoff_events([r,h,dict(event='fast_committed',query_id=1)],cfg)['handoff_audits']==1
    h['revision_rad']=.5
    with pytest.raises(ValueError,match='revision exceeds'):validate_handoff_events([r,h],cfg)

def test_replay_preserves_actual_previous_query_time(tmp_path):
    import numpy as np
    from dynamicwam.dualrate.continuity_data import ContinuityDataset
    sample=dict(fast_known_actions=torch.zeros(16,14),fast_known_mask=torch.arange(16)<2,
        source_known_actions=torch.zeros(16,14),source_known_mask=torch.arange(16)<2,
        source_action_intervals=torch.arange(32,dtype=torch.float64).reshape(16,2),
        action_intervals=torch.arange(32,dtype=torch.float64).reshape(16,2),
        teacher_available_time=torch.tensor(.2),measured_state_valid=torch.tensor(True),
        query_time=torch.tensor(.4,dtype=torch.float64),previous_query_time=torch.tensor(.18,dtype=torch.float64),
        previous_reference_same=torch.tensor(False),previous_handoff=torch.tensor(True))
    p=tmp_path/'r.pt';torch.save(sample,p)
    d=ContinuityDataset.__new__(ContinuityDataset);d.replay=[dict(path=str(p))];d.template=sample
    r=d.replay_item(0)
    assert r['previous_query_time']==sample['previous_query_time'] and r['previous_handoff']

def test_mixed_batch_does_not_treat_real_prior_query_as_paired_teacher_time():
    from test_residual_v10 import fixture,generator
    from dynamicwam.dualrate.continuity_v11 import configure,CONTRACT as V11
    s,b,w=fixture()
    configure(s,dict(contract=V11,reference_hold_steps=4,seam_weight=.1,revision_weight=.1,out_of_radius_weight=.2,revision_radius_fraction=1/3,
        handoff=dict(contract=CONTRACT,local_rows=4,handoff_step_rad=.06,same_step_rad=.02)))
    b['feedback_noise_seed']=torch.tensor([17]);b['action_grid_origin']=b['source_time'].clone()
    b={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in b.items()}
    b=defaults(b);b.update({'previous_'+k:v.clone() for k,v in list(b.items()) if not k.startswith('previous_')})
    b['pair_valid']=torch.tensor([True,False]);b['previous_query_time'][1]-=100.
    s.residual_v10['executor']='phase_c2'
    result=s.training_loss(b,generator=generator())
    assert torch.isfinite(result['loss'])

def test_expired_reference_defers_query_but_does_not_block_slow_refresh():
    from test_dualrate_hardtime_deployment import observation,DT
    from dynamicwam.dualrate.deployment import dispatch_due_queries
    from dynamicwam.dualrate.online import ObservationHistory
    from dynamicwam.dualrate.reference_lease import ReferenceUnavailable
    from dynamicwam.dualrate.timing import ExecutionQueue,TimedPlan
    q=ExecutionQueue(TimedPlan(0.,DT,tuple((float(i),) for i in range(16)),0,0.))
    now=2*DT+.12
    for t in (0.,DT,2*DT,now):q.action_at(t)
    q.commit_prefix(query_time=now,until=4*DT)
    h=ObservationHistory();h.append(observation(),exposure=2*DT,available=2*DT+.1)
    calls=[];events=[];deferred={}
    def fast(*a,**kw):raise ReferenceUnavailable(now,4*DT)
    r=SimpleNamespace(due=lambda now:(True,True),store=SimpleNamespace(snapshot=lambda now:None),start_fast=fast,
        start_slow=lambda *a,**kw:calls.append(kw))
    b=SimpleNamespace(hardtime=True,max_observation_age=.24)
    dispatch_due_queries(r,b,h,q,now=now,initial=0.,vae=None,emit=events.append,deferred=deferred,seen_references=set())
    assert calls and deferred['fast']==now
    assert any(e.get('reason')=='no_usable_slow_reference' for e in events)
    assert q.action_at(3*DT)==(3.,)  # Original queue continues, no repeated old command.

def test_measured_cup_expiry_recovers_only_after_new_publication():
    from dynamicwam.dualrate.reference_lease import ReferenceLease,ReferenceUnavailable
    dt=45*float(torch.tensor(.004,dtype=torch.float32))
    def ref(version,source,completed):
        t=source+torch.arange(16,dtype=torch.float64)*dt
        return SimpleNamespace(version=version,source_time=source,completed_time=completed,expires_at=source+2.,
            payload=dict(reference_actions=torch.zeros(1,16,14),reference_action_intervals=torch.stack((t,t+dt),-1)[None],world_valid=torch.ones(1,16,dtype=torch.bool)))
    old=ref(14,51*dt,9.928000471554697);new=ref(15,59*dt,11.432000543)
    lease=ReferenceLease(4,dt)
    lease.select(old,(14,None),now=10.848000515252352,first=62*dt)
    with pytest.raises(ReferenceUnavailable):lease.select(old,(14,None),now=11.276000535581261,first=64*dt)
    # Even a ready device result may not become visible before publication.
    with pytest.raises(ReferenceUnavailable):lease.select(new,(15,None),now=11.40,first=65*dt)
    got,_,_=lease.select(new,(15,None),now=11.4360005432,first=65*dt)
    assert got is new and old.expires_at==51*dt+2.

def test_clean_feedback_writer_does_not_make_a_failed_episode_training_data(tmp_path):
    import json
    from dynamicwam.dualrate.feedback_replay import digest,build_replay_index
    for i,ok in enumerate([True,False]):
        ep=tmp_path/'task'/f'episode_{i:04d}';fb=ep/'feedback';fb.mkdir(parents=True)
        rows=[]
        for n in range(4):
            p=fb/f'query_{n}.pt';torch.save(dict(x=torch.tensor(n)),p);rows.append(dict(path=str(p),sha256=digest(p)))
        (ep/'report.json').write_text(json.dumps(dict(continuous_valid=ok,realtime_valid=ok)))
        (fb/'manifest.json').write_text(json.dumps(dict(complete=True,identity=dict(split='train',seed=1,task='task'),rows=rows)))
    result=build_replay_index([tmp_path],tmp_path/'index.json',['task'])
    index=json.loads((tmp_path/'index.json').read_text())
    assert result['rows']==4 and len(index['excluded_failed_reports'])==1
