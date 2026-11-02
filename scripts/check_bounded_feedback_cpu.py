"""Production-width CPU backward/optimizer/reload check; NOT GPU acceptance."""
import argparse,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only check requires CUDA visibility disabled')
    import torch
    from dynamicwam.dualrate.fast import FastConfig
    from dynamicwam.dualrate.bounded_feedback import BoundedFeedback
    from dynamicwam.dualrate.optimizer import FP32MasterAdamW
    from dynamicwam.dualrate.feedback_data import FeedbackDataset
    from dynamicwam.dualrate.native_interaction import validate_interaction
    from dynamicwam.dualrate.train import file_sha256,source_identity,state_sha256,validate_row
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output).resolve()
    if not str(out).startswith('/SSD_DISK/users/wuruihan/'):raise ValueError('SSD output required')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(20261102)
    model=BoundedFeedback(FastConfig(**cfg['architecture']),[.1]*14).eval()
    kw=dict(observation_tokens=torch.randn(1,20,384),observation_valid=torch.ones(1,20,dtype=torch.bool),
        world_tokens=torch.randn(1,3,384),world_valid=torch.ones(1,3,dtype=torch.bool),
        world_intervals=torch.tensor([[[9.,10.],[11.,18.],[19.,26.]]],dtype=torch.float64),world_types=torch.tensor([[0,1,1]]),
        query_time=torch.tensor([10.4],dtype=torch.float64),source_time=torch.tensor([10.],dtype=torch.float64),
        completed_time=torch.tensor([10.],dtype=torch.float64),native_source_index=torch.tensor([10]),
        native_query_index=torch.tensor([12]),native_available=torch.tensor([True]),
        draft_actions=torch.randn(1,16,14),draft_valid=torch.ones(1,16,dtype=torch.bool),
        interaction_history=torch.randn(1,8,64),interaction_valid=torch.ones(1,8,dtype=torch.bool))
    kw['draft_valid'][:,-2:]=False
    with torch.no_grad():assert torch.equal(model.predict(model.prepare_context(**kw))['actions'],kw['draft_actions'])
    opt=FP32MasterAdamW([dict(params=model.parameters(),lr=1e-4,name='new')])
    losses=[]
    for _ in range(3):
        opt.zero_grad();model.train();result=model.predict(model.prepare_context(**kw))
        loss=(result['actions'][:,:14,0]-kw['draft_actions'][:,:14,0]-.04).square().mean()+.01*result['future'].square().mean()
        loss.backward();opt.accumulate_gradients(1.);opt.clip_accumulated_grad_norm_(1.);opt.step();opt.zero_grad()
        assert torch.isfinite(loss);losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():before=model.predict(model.prepare_context(**kw))['actions']
    saved=dict(model=model.state_dict(),optimizer=opt.state_dict());digest=state_sha256(saved)
    checkpoint=out/'cpu_fast_fixture_only.pt';torch.save(saved,checkpoint)
    # Reload into the same owner after deliberate parameter corruption; all
    # state is verified, not merely the final output on one input.
    with torch.no_grad():model.action_out.bias.add_(3)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(loaded['model'],strict=True);opt.load_state_dict(loaded['optimizer'])
    assert state_sha256(dict(model=model.state_dict(),optimizer=opt.state_dict()))==digest
    with torch.no_grad():after=model.predict(model.prepare_context(**kw))['actions']
    assert torch.equal(before,after) and not torch.cuda.is_initialized()
    assert torch.equal(after[:,:,[6,13]],kw['draft_actions'][:,:,[6,13]])
    # The actual expert subset is read, but no frozen Full/VAE GPU forward is claimed.
    manifest=Path(cfg['manifest']);m=json.loads(manifest.read_text())
    m.update(native_v9_dataset=True,bounded_feedback_dataset=True,expert_manifest=str(manifest),expert_manifest_sha256=file_sha256(manifest),native_replay_samples=[])
    mp=out/'cpu_manifest.json';mp.write_text(json.dumps(m));ds=FeedbackDataset(mp)
    sampled=[]
    for i in (0,2,len(ds)//2,len(ds)-1):
        row=ds[i];validate_row(row)
        validate_interaction({k:v[None] for k,v in row.items() if isinstance(v,torch.Tensor)})
        sampled.append(dict(index=i,age=int(row['native_query_index']-row['native_source_index']),pair=bool(row['pair_valid'])))
    result=dict(passed=True,formal_training_ready=False,gpu_used=False,source_sha256=source_identity(),
        architecture=cfg['architecture'],fast_parameters=sum(p.numel() for p in model.parameters()),
        synthetic_cpu_updates=3,losses=losses,strict_optimizer_model_reload=True,reload_max_error=0.,
        actual_expert_rows=len(ds),real_samples=sampled,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint))
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
