"""Functional feedback, no-oracle-label, solver-count and refresh-budget checks."""
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from verify_multirate_temporal import temporal_sample
from dynamicwam.models.compact_fast_policy import CompactFastPolicy
from dynamicwam.models.action_feedback import future_patch_target,dynamics_loss
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable
from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy

def fixture():
    old,current,bridge,text=temporal_sample()
    c=dataclasses.replace(old.config,feedback_enabled=True,feedback_channels=4)
    p=CompactFastPolicy(c,motion_history_count=2,motion_mean=[0.]*12,motion_scale=[1.]*12)
    return p,p.prepare_memory(current,bridge,text)

class Feedback(unittest.TestCase):
    def setUp(self): torch.manual_seed(20261003)

    def test_action_changes_forecast_and_forecast_changes_action(self):
        p,m=fixture();z=torch.randn(2,16,4)
        y1=p.feedback_dynamics(m.tokens,z)
        y2=p.feedback_dynamics(m.tokens,z+1)
        self.assertGreater((y1-y2).abs().max().item(),1e-5)
        t=torch.ones(2)*1000
        a=p.predict_velocity(z,t,p.feedback_memory(m,y1))
        b=p.predict_velocity(z,t,p.feedback_memory(m,y2))
        self.assertGreater((a-b).abs().max().item(),1e-6)

    def test_two_actor_calls_and_final_plan_forecast(self):
        p,m=fixture();z=torch.randn(2,16,4)
        with patch.object(p,'predict_velocity',wraps=p.predict_velocity) as actor:
            actions,_,_=p.sample_feedback(z,m)
        self.assertEqual(actor.call_count,2)
        torch.testing.assert_close(p.last_feedback_future,p.feedback_dynamics(m.tokens,actions))

    def test_dynamics_frozen_during_policy_and_gradient_reaches_proposal(self):
        from verify_multirate_training import _Backbone
        p,m=fixture();model=MultirateWAM(_Backbone(),None,compact_policy=p)
        model.feedback_phase='policy';configure_multirate_trainable(model,'warmup')
        self.assertFalse(any(x.requires_grad for x in p.feedback_dynamics.parameters()))
        a=torch.randn(2,16,4,requires_grad=True)
        p.feedback_dynamics(m.tokens.detach(),a).square().mean().backward()
        self.assertGreater(a.grad.abs().sum().item(),0)
        self.assertFalse(any(x.requires_grad for x in p.encoder.parameters()))

    def test_expired_prior_never_forces_repeated_heavy_refresh(self):
        p=MultiratePolicy.__new__(MultiratePolicy)
        p.contract=dict(execute_horizon=4,max_age_actions=4,max_age_seconds=.72,feedback=dict(world_refresh_actions=16))
        p.worker=SimpleNamespace();p.bridge=object();p.bridge_action_index=0;p.bridge_sim_time=0.;p._feedback_last_request=0
        p._submitted_source=None;p.over_age_queries=0;p.scheduling='independent_worker'
        p._install_ready_bridge=lambda *_:None
        dropped=[];requests=[]
        p._sample_fast=lambda packet,age,**kw: dropped.append(kw['null_bridge']) or torch.zeros(16,4)
        p._ensure_request=lambda index,packet:requests.append(index)
        for i in range(4,33,4):
            p._query_feedback(dict(action_index=i,sim_time=i*.5),SimpleNamespace(execute_horizon=4,query_id=i//4,every_fast_queries=4))
        self.assertEqual(requests,[16,32]);self.assertTrue(all(dropped))

    def test_future_target_keeps_temporal_order(self):
        x=torch.randn(2,4,2,8,10)
        y=future_patch_target(x,4)
        self.assertEqual(y.shape,(2,32,4))
        self.assertEqual(float(dynamics_loss(y,y)),0.)

    def test_serial_budget_refreshes_exactly_once_per_sixteen_actions(self):
        p=MultiratePolicy.__new__(MultiratePolicy)
        p.contract=dict(execute_horizon=4,max_age_actions=4,max_age_seconds=.72,feedback=dict(world_refresh_actions=16))
        p.worker=None;p.bridge=None;p._feedback_last_request=0;p.over_age_queries=0;p._submitted_source=None
        p.scheduling='controlled_ratio';built=[]
        def build(packet):
            built.append(packet['action_index'])
            return object(),None,packet['sim_time'],packet['action_index']
        p._build_bridge_completed=build
        p._sample_fast=lambda *a,**kw:torch.zeros(16,4)
        for i in range(0,33,4):
            p._query_feedback(dict(action_index=i,sim_time=i*.5),SimpleNamespace(execute_horizon=4,query_id=i//4,every_fast_queries=4))
        self.assertEqual(built,[0,16,32])
        self.assertEqual(p.over_age_queries,6)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
