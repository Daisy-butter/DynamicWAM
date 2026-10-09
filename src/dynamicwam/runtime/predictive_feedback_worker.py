"""Complete frozen WAM plans with action-conditioned short-future feedback.

The simulator and timed queue own execution. Forecasts are conditional on an
explicit candidate installation assumption, not measured future joint states.
"""
from __future__ import annotations
import math
import time
import uuid
import numpy as np
import torch
from dynamicwam.runtime.world_context_worker import WorldContextWorker
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256
from dynamicwam.runtime.ciwam.wam.temporal_feedback_policy import fixed_feedback_text
from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path
from dynamicwam.models.predictive_feedback import PredictiveFeedbackHead, ABLATIONS
from dynamicwam.models import predictive_feedback_contract as contract
from dynamicwam.action_normalization import RoboTwinQposNormalizer


class PredictiveFeedbackWorker(WorldContextWorker):
    # Collection is available only through the explicit bounded-memory subclass.
    # The ordinary deployment path never materializes legacy label bundles.
    buffered_collection = False

    def __init__(self,config):
        cfg=dict(config)
        if cfg.get('mode')!='predictive' or (cfg.get('collection_dir') and not self.buffered_collection):
            raise ValueError('predictive collection requires the explicit buffered_predictive_feedback_worker factory')
        checkpoint=cfg.pop('feedback_checkpoint')
        ablation=cfg.get('ablation','none')
        if ablation not in ABLATIONS: raise ValueError('unknown predictive ablation')
        # Reuse the exact native planner and observation preprocessing. Do not
        # route the new checkpoint through legacy world-context deserialization.
        super().__init__(dict(cfg,mode='graph_zero'))
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (payload.get('format')!=contract.FORMAT or payload.get('version')!=contract.VERSION or
                payload.get('architecture_version')!=PredictiveFeedbackHead.ARCHITECTURE_VERSION or
                payload.get('training_scope')!='joint' or payload.get('deployable') is not True or
                payload.get('controller_protocol_sha256')!=self.protocol_sha or
                payload.get('base_checkpoint',{}).get('sha256')!=self.base_sha):
            raise ValueError('joint predictive checkpoint with matching controller/WAM is required')
        self.checkpoint_sha=file_sha256(checkpoint)
        if config.get('feedback_checkpoint_sha256',self.checkpoint_sha)!=self.checkpoint_sha:
            raise ValueError('feedback weight identity differs')
        with torch.random.fork_rng(devices=[]): self.head=PredictiveFeedbackHead(**payload['spec'])
        contract.validate_runtime_contract(payload['runtime_contract'],self.schedule,self.head.residual_limit)
        self.head.load_state_dict(payload['model'],strict=True)
        self.head.to(self.base.runner.device).float().eval().requires_grad_(False)
        normal=payload['action_normalization']
        saved=RoboTwinQposNormalizer.from_stats(normal['stats'],epsilon=normal['epsilon'])
        for key in ('_mean','_std'):
            if not torch.equal(getattr(saved,key),getattr(self.base.runtime.action_normalizer,key)):
                raise ValueError('policy and planner normalizations differ')
        self.mean,self.scale=saved._mean.numpy().copy(),saved._std.numpy().copy()
        from dynamicwam.inference.predictive_feedback_graph import PredictiveFeedbackGraph
        self.fast=PredictiveFeedbackGraph(self.head,enabled=bool(cfg.get('feedback_cuda_graph',True)),ablation=ablation)
        self.mode='predictive';self.ablation=ablation;self.forecast=None
        self.config=dict(config)

    def reset(self):
        super().reset()
        self.forecast=None;self.plan_numpy=None

    def _collect_this_query(self,packet):
        return (self.buffered_collection and not packet.get('_warmup_only',False)
                and self.record_count<int(self.config.get('samples_per_episode',64)))

    def _retain_collection_source(self,cache,noise):
        raise RuntimeError('collection source retention requires the buffered worker')

    def _forecast_feedback(self,packet,age):
        started=time.perf_counter()
        if not packet.get('_warmup_only'):
            dispatch=packet.get('controller_state',{}).get('dispatch_controller',{})
            if dispatch.get('source_physics_time')!=packet.get('dispatch_sim_time') or not dispatch:
                raise ValueError('real deployment requires a causal dispatch controller and known command prefix')
        # Keep the legacy relabel inputs separate from proposal/forecast data.
        # No imagined or actual future action may enter the recorded teacher input.
        causal_inputs=self._inputs(packet,age)
        inputs=dict(causal_inputs)
        path,dt,path_metadata=proposal_action_path(controller=packet['controller_state'],
            plan=self.plan_numpy,age=age,observation_time=float(packet['sim_time']),mean=self.mean,scale=self.scale)
        def tensor(x): return torch.as_tensor(x,device=self.plan.device).unsqueeze(0)
        inputs.update(proposal_path=tensor(path),proposal_dt=tensor(dt))
        aux=self.fast.run_head(inputs)
        # This one transfer is also needed for the action queue; the visual
        # encoder and source tokens are reused for the corrected forecast.
        residual=aux['residual'][0].float().cpu().numpy().copy()
        length=min(len(residual),16-age)
        candidate=self.plan_numpy.copy()
        candidate[age:age+length]+=residual[:length]
        corrected_path,corrected_dt,corrected_metadata=proposal_action_path(
            controller=packet['controller_state'],plan=candidate,age=age,
            observation_time=float(packet['sim_time']),mean=self.mean,scale=self.scale)
        predicted=self.fast.run_forecast(aux,history_dt=inputs['history_dt'],history_valid=inputs['history_valid'],
            action_path=tensor(corrected_path),future_dt=tensor(corrected_dt))
        # Finish both stages before recording service time. No hidden unfinished
        # forecast is excluded from the latency budget.
        if predicted.is_cuda: torch.cuda.synchronize(predicted.device)
        self.forecast=dict(latents=predicted.detach(),plan_id=self.plan_id,
            observation_time=float(packet['sim_time']),future_dt=corrected_dt.copy(),
            corrected_proposal=corrected_path.copy(),conditional_install=corrected_metadata)
        details=dict(
            proposal=path_metadata,corrected_proposal=corrected_metadata,
            full_fast_wall_s=time.perf_counter()-started)
        details['recording_wall_s']=0.
        if self._collect_this_query(packet):
            record_started=time.perf_counter()
            self._record(packet,causal_inputs,age)
            details['recording_wall_s']=time.perf_counter()-record_started
        return candidate[age:],residual[:length],details

    @torch.inference_mode()
    def sample(self,packet):
        if packet.get('_reset_policy',False): self.reset()
        instruction=str(packet['instruction'])
        if instruction!=self.instruction:
            self.base.set_instruction(instruction);self.instruction=instruction;self.plan=None
        now=float(packet['sim_time'])
        if not math.isfinite(now) or now<0 or float(packet['control_period_s'])!=self.period:
            raise ValueError('invalid physical clock protocol')
        age=16 if self.plan is None else int(math.floor((now-self.source_time)/self.period+1e-6))
        if age<0: raise ValueError('observation predates source plan')
        acknowledged=packet.get('active_plan_id')
        mismatch=acknowledged is not None and acknowledged!=self.plan_id
        refresh=self.plan is None or age>=self.refresh_age or mismatch
        collection_source_wall_s=0.
        if refresh:
            source,noise,text=self._source(packet)
            plan,cache=self.sampler.sample(source,noise)
            self.plan=plan.float().clone();self.plan_numpy=self.plan[0].cpu().numpy().copy()
            self.source_state=source.state.float().clone();self.source_time=now
            self.source_action_index=int(packet['command_index']);self.source_sampling_seed=packet['_sampling_seed']
            self.plan_id=uuid.uuid4().hex;self.text,self.text_mask=fixed_feedback_text(text)
            self.forecast=None;age=0
            if self._collect_this_query(packet):
                record_started=time.perf_counter()
                self._retain_collection_source(cache,noise)
                # Attribute the extra GPU pooling/copies to collection, never
                # hide unfinished work in the next fast query's latency.
                if self.plan.is_cuda: torch.cuda.synchronize(self.plan.device)
                collection_source_wall_s=time.perf_counter()-record_started
            else:
                self.context=None;self.source_noise=None
        details={};residual=np.zeros((min(4,16-age),14),dtype=np.float32)
        tail=self.plan_numpy[age:].copy()
        if age>0:
            tail,residual,details=self._forecast_feedback(packet,age)
        elif packet.get('_warmup_only'):
            # Graph creation must never block live physics. A dummy causal
            # controller is only used here, while physical time is still zero.
            from dynamicwam.runtime.domino_rt.timed_control import LimitedHermiteController
            from dynamicwam.runtime.domino_rt.trajectory_recorder import controller_snapshot
            ctrl=LimitedHermiteController(np.asarray(packet['state'],dtype=np.float64))
            ctrl.begin(None)
            snapshot=controller_snapshot(ctrl)
            snapshot.update(source_physics_time=now,future_knots=[])
            self._forecast_feedback(dict(packet,controller_state=snapshot),4)
            self.forecast=None
        actions=(tail*self.scale+self.mean).astype(np.float32)
        if not np.isfinite(actions).all(): raise FloatingPointError('nonfinite predictive action')
        details.setdefault('recording_wall_s',0.)
        details['recording_wall_s']+=collection_source_wall_s
        details['collection_source_wall_s']=collection_source_wall_s
        details['collection_enabled']=self.buffered_collection
        return dict(actions=actions,metadata=dict(mode='refresh' if refresh else 'feedback',
            policy_arm=self.mode,ablation=self.ablation,plan_id=self.plan_id,
            source_sim_time=self.source_time,action_origin_sim_time=self.source_time+age*self.period,
            age_actions=age,feedback_rows=len(residual) if age>0 else 0,
            feedback_row_l2=np.linalg.norm(residual,axis=-1).tolist(),feedback_row_l2_units='normalized_action',
            branch_mismatch_refresh=mismatch,controller_protocol_sha256=self.protocol_sha,
            base_checkpoint_sha256=self.base_sha,checkpoint_sha256=self.checkpoint_sha,
            source_sampling_seed=self.source_sampling_seed,query_sampling_seed=packet.get('_sampling_seed'),
            context_semantics='current_grounded_short_future_conditional_on_timed_corrected_proposal',
            cached_future_used_as_new_observation=False,planner_graph=dict(self.sampler.stats),
            fast_graph=dict(self.fast.stats),**details))


def create_worker_policy(config): return PredictiveFeedbackWorker(config)
