"""Actual SmallWAM joint attention + native actor + tiny native Wan blocks, CPU."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys,torch
from torch.nn import functional as F
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from check_joint_motionkv_v3_training_cpu import v3_fixture,cpu_attention
from dynamicwam.models.small_wam import SmallWAMActionModel,SmallWAMActionConfig
from dynamicwam.vendor.wan.modules.model import WanAttentionBlock
from dynamicwam.models.rtc_prefix import PrefixCondition,apply_prefix,masked_suffix_mse
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable,prepare_multirate_batch,MultirateTrainer
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner

class TinyCompact(nn.Module):
    def __init__(self,text_dim=16):
        super().__init__()
        self.video_model=nn.Module();self.video_model.precision=torch.float32
        self.video_model.wan_model=nn.Module()
        self.video_model.wan_model.blocks=nn.ModuleList([WanAttentionBlock(12,24,2) for _ in range(2)])
        self.projection=nn.Linear(12,12);self.text=nn.Linear(text_dim,12)
        self.time=nn.Linear(1,12);self.mod=nn.Linear(12,72);self.head=nn.Linear(12,12)
    def prepare_multiscale_video_tokens(self,condition,future):
        b=condition.shape[0]
        summary=lambda x:F.adaptive_avg_pool1d(x.flatten(1).unsqueeze(1),12)
        tokens=torch.cat((summary(condition),summary(future).expand(-1,6,-1)),1)
        return self.projection(tokens),torch.full((b,),7),dict(condition_seq_len=1,future_seq_len=6,all_video_tokens_valid=True,future_shape=tuple(future.shape[1:])),torch.ones(10,3,dtype=torch.complex64)
    def prepare_text_context(self,embeddings):return self.text(torch.stack(embeddings))
    def prepare_time_embeddings(self,t,n):
        e=self.time(t[:,None,None].expand(-1,n,1)/1000)
        return e,self.mod(e).reshape(t.shape[0],n,6,12)
    def apply_multiscale_video_head(self,tokens,embedding,grid):
        result=self.head(tokens[:,1:]+embedding[:,1:]).mean(1)
        count=1
        for n in grid['future_shape']:count*=n
        return result.repeat(1,(count+11)//12)[:,:count].reshape(result.shape[0],*grid['future_shape'])
    def apply_multiscale_rope(self,x,grid,freqs):return x

def run():
    torch.set_num_threads(2);checks=[]
    patches=[patch('dynamicwam.models.small_wam.flash_attention',cpu_attention),
        patch('dynamicwam.vendor.wan.modules.model.flash_attention',cpu_attention),
        patch('dynamicwam.models.action_core_ops.flash_attention',cpu_attention),
        patch('dynamicwam.models.action_core_ops.dense_action_attention',cpu_attention)]
    for p in patches:p.start()
    try:
        model,batch=v3_fixture();old=model.backbone
        config=SmallWAMActionConfig(SimpleNamespace(dim=12,num_heads=2,num_layers=2,head_dim=6),14,14,16,24,48,2,False,4,(0.,)*12,(1.,)*12)
        model.backbone=SmallWAMActionModel(config,TinyCompact())
        model.backbone.action_expert.load_state_dict(old.action_expert.state_dict())
        model.backbone.absolute_motion_tokens.load_state_dict(old.absolute_motion_tokens.state_dict())
        core=model.backbone;source=batch['source'];noise=batch['source_noise']
        legacy=replace(source,committed_prefix=None,committed_prefix_lengths=None)
        zero=replace(source,committed_prefix=torch.full_like(noise.actions,float('nan')),committed_prefix_lengths=torch.zeros(2,dtype=torch.long))
        a=model.bridge(core,legacy,noise,detach_cache=False);b=model.bridge(core,zero,noise,detach_cache=False)
        assert torch.equal(a.source_action_plan,b.source_action_plan)
        assert all(torch.equal(x,y) for x,y in zip(a.video_k+a.video_v,b.video_k+b.video_v))
        checks.append('actual_native_joint_zero_prefix_is_bit_exact_to_legacy_no_prefix')
        prefix=PrefixCondition(torch.randn_like(noise.actions),torch.tensor([0,4]))
        source=replace(source,committed_prefix=prefix.actions,committed_prefix_lengths=prefix.lengths)
        with patch.object(core,'_build_action_tokens',wraps=core._build_action_tokens) as tokens,patch.object(core,'_build_action_time_embeddings',wraps=core._build_action_time_embeddings) as times:
            bridge=model.bridge(core,source,noise,detach_cache=False)
        assert len(tokens.call_args_list)==2 and len(times.call_args_list)==2
        for args in tokens.call_args_list:assert torch.equal(args.args[1][1,:4],prefix.actions[1,:4])
        for args in times.call_args_list:
            t=args.args[0];assert t[1,1:5].eq(0).all() and t[1,0]>0 and t[1,17:21].gt(0).all()
        assert torch.equal(bridge.source_action_plan[1,:4],prefix.actions[1,:4])
        changed=replace(source,committed_prefix=prefix.actions+10)
        other=model.bridge(core,changed,noise,detach_cache=False)
        assert not torch.equal(bridge.video_v[-1][1],other.video_v[-1][1])
        checks.append('both_joint_steps_use_clean_source_prefix_t0_native_motion_t_preserved_and_video_kv_changes')
        configure_multirate_trainable(model,'joint')
        aligned=dict(batch['source_aligned'],rtc_prefix=prefix,noisy_actions=apply_prefix(batch['source_aligned']['noisy_actions'],prefix))
        batch=dict(batch,source=source,source_aligned=aligned,source_rtc_prefix=prefix)
        loss=model(batch)['native_loss'];loss.backward()
        for module in (core.compact_wan,core.action_expert,core.absolute_motion_tokens):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
        altered=dict(batch,source_action_target=batch['source_action_target'].clone())
        altered['source_action_target'][1,:4]=float('nan')
        assert torch.equal(model(batch)['native_metrics']['loss/action_retention'],model(altered)['native_metrics']['loss/action_retention'])
        assert model(batch)['native_metrics']['loss/video_retention']>0
        checks.append('native_joint_gradients_reach_video_actor_motion_and_source_action_retention_masks_only_committed_prefix')
        fixture=Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/data/joint_motionkv_physical_grid_20261015/replay_cpu_smoke/synthetic_fixture.pt')
        payload=torch.load(fixture,weights_only=True);raw=payload['batch']
        text_dim=raw['text_embeddings'][0].shape[-1]
        replay_model,_=v3_fixture()
        replay_model.backbone=SmallWAMActionModel(config,TinyCompact(text_dim=text_dim))
        replay_model.backbone.action_expert.load_state_dict(old.action_expert.state_dict())
        replay_model.fast_conditioner=JointMotionKVConditioner(**dict(replay_model.fast_conditioner.constructor_spec,text_dim=text_dim))
        prepared=prepare_multirate_batch(raw,timesteps=torch.linspace(1000,0,20),sigmas=torch.linspace(1,0,20),
            device='cpu',dtype=torch.float32,phase='cross_time_joint',null_bridge_probability=0.,sim_scale=1.,action_scale=16.,
            native_multirate=True,joint_interface_version=3)
        trainer=MultirateTrainer.__new__(MultirateTrainer);trainer.config=dict(rtc_prefix_max=4,native_source_bridge='student_joint')
        trainer._attach_rtc_prefix(prepared)
        assert prepared['source'].committed_prefix_lengths.eq(0).all() and prepared['rtc_prefix'].lengths.eq(4).all()
        assert prepared['source'].action_target_offsets[0,0].item()==raw['joint_targets']['source_action_target_times'][0,0].item()-raw['source_sim_time'][0].item()
        configure_multirate_trainable(replay_model,'joint')
        replay_loss=replay_model(prepared)['native_loss'];assert torch.isfinite(replay_loss);replay_loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in replay_model.backbone.compact_wan.parameters() if p.grad is not None)>0
        checks.append('data_agent_raw_recorder_converter_collate_fixture_reaches_actual_native_joint_loss_with_distinct_source0_currentH4_prefixes')
        print(json.dumps(dict(passed=True,checks=checks,loss=float(loss),replay_fixture_loss=float(replay_loss),replay_fixture_is_synthetic=True,device='cpu',video='two tiny native WanAttentionBlocks; compact token projection and identity video RoPE stand-in',attention='CPU SDPA')))
    finally:
        for p in reversed(patches):p.stop()
if __name__=='__main__':run()
