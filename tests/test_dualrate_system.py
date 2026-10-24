"""Tiny CPU system acceptance; does not claim pretrained/GPU/DOMINO acceptance."""
import pytest
import torch
from types import SimpleNamespace

from dynamicwam.dualrate.fast import CurrentObservationEncoder, FastConfig, FastJointWAM
from dynamicwam.dualrate.slow import SlowReferenceAdapter, StructuredWorldReadout
from dynamicwam.dualrate.system import DualRateSystem, LossConfig
from test_dualrate_slow import NativeContractFixture


def make_system():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    slow = SlowReferenceAdapter(NativeContractFixture(), gradient_checkpointing=True)
    readout = StructuredWorldReadout(latent_channels=2, action_dim=2, world_dim=12, native_hidden_dim=2, test_fixture=True)
    observation = CurrentObservationEncoder(dim=12, views=1, motion_dim=3, state_dim=2, text_dim=4, test_fixture=True)
    fast = FastJointWAM(FastConfig(dim=12,layers=2,heads=3,action_dim=2,chunk_size=4,
                                  future_dim=10,future_tokens=16,observation_dim=12,world_dim=12,test_fixture=True))
    cfg = LossConfig(slow_video=0,slow_action=0,fast_action=1,future_visual=0,
                     future_geometry=0,contact=0,action_conditioned_probability=0,
                     cache_dropout=0,visual_dim=4)
    system = DualRateSystem(slow,readout,observation,fast,future_latent_shape=(2,2,4,4),loss_config=cfg)
    query = torch.tensor([2.1],dtype=torch.float64)
    action_times = torch.tensor([[2.1,2.2,2.3,2.4]],dtype=torch.float64)
    known = torch.randn(1,4,2)
    known_mask = torch.tensor([[True,False,False,False]])
    batch = dict(
        source_condition_latent=torch.randn(1,2,1,4,4),
        source_initial_state=torch.randn(1,2),source_motion=torch.randn(1,4,3),
        source_motion_valid=torch.ones(1,4,dtype=torch.bool),
        source_motion_acceleration_valid=torch.ones(1,4,dtype=torch.bool),source_text=torch.randn(1,3,4),
        source_action=torch.randn(1,4,2), fast_action=torch.randn(1,4,2),
        source_future_latent=torch.randn(1,2,2,4,4),
        source_future_intervals=torch.tensor([[[2.1,2.3],[2.4,2.6]]],dtype=torch.float64),
        source_action_times=action_times.clone(),source_known_actions=known.clone(),
        source_known_mask=known_mask.clone(),fast_known_actions=known.clone(),fast_known_mask=known_mask.clone(),
        rgb=torch.rand(1,1,3,32,32),rgb_times=torch.tensor([[2.09]],dtype=torch.float64),
        motion=torch.randn(1,4,3),motion_times=torch.tensor([[[1.4,1.55],[1.55,1.7],[1.7,1.85],[1.85,2.0]]],dtype=torch.float64),
        motion_valid=torch.ones(1,4,dtype=torch.bool),motion_acceleration_valid=torch.ones(1,4,dtype=torch.bool),
        state=torch.randn(1,2),text=torch.randn(1,3,4),text_valid=torch.ones(1,3,dtype=torch.bool),
        source_time=torch.tensor([2.],dtype=torch.float64),completed_time=torch.tensor([2.05],dtype=torch.float64),
        query_time=query,action_times=action_times,
        future_times=query[:,None]+torch.tensor([[.1,.2,.4,.8]*4],dtype=torch.float64),
        future=torch.randn(1,16,10),future_loss_valid=torch.ones(1,16,10,dtype=torch.bool),
        contact=torch.zeros(1,16,2),contact_valid=torch.ones(1,16,2,dtype=torch.bool),
    )
    batch["source_action"][known_mask] = known[known_mask]
    batch["fast_action"][known_mask] = known[known_mask]
    batch['source_action_intervals'] = torch.stack((action_times,action_times+.1),dim=-1)
    batch['action_intervals'] = batch['source_action_intervals'].clone()
    from test_dualrate_slow import metadata_fixture
    sm = metadata_fixture(batch['source_condition_latent'],torch.tensor([[[1.7,2.]]],dtype=torch.float64))
    fm = metadata_fixture(batch['source_future_latent'],batch['source_future_intervals'])
    for prefix,meta in [('source',sm),('future',fm)]:
        batch[prefix+'_view_overlap']=meta['view_overlap']
        batch[prefix+'_token_valid']=meta['valid']
        batch[prefix+'_input_type']=meta['input_type']
    batch['source_temporal_support']=sm['temporal_support']
    return system,batch


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


def test_sampling_keeps_unsupported_future_fields_at_fixed_noise():
    system,batch=make_system()
    system.future_field_support[:,0]=False
    world=system.world_readout(system.predict_reference(batch,generator=generator()))
    seen=[]
    def constant_velocity(a,z,sigma,context,**kwargs):
        seen.append(z.clone())
        return dict(action_velocity=torch.zeros_like(a),future_velocity=torch.ones_like(z),
                    contact_logits=torch.zeros(*z.shape[:2],2))
    system.fast.forward_cached=constant_velocity
    result=system.sample_fast(batch,world,steps=3,generator=generator())
    for value in seen[1:]:
        torch.testing.assert_close(value[...,0],seen[0][...,0],atol=0,rtol=0)
    torch.testing.assert_close(seen[-1][...,1],seen[0][...,1]-1)
    assert not result['future'][...,0].any()
    assert not result['future_field_valid'][...,0].any()


def copy_batch(batch):
    return {key:value.clone() for key,value in batch.items()}


def test_fast_action_only_loss_reaches_slow_video_action_motion():
    system,batch = make_system()
    result = system.training_loss(batch,generator(),action_conditioned=torch.tensor([False]))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for parameter in (system.slow.native.video_weight,system.slow.native.action_weight,
                      system.slow.native.motion_weight,
                      system.slow.native.compact_wan.video_model.wan_model.blocks[0].self_attn.k.weight,
                      system.fast.future_in.weight,system.world_readout.future_projection.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_predict_reference_ignores_all_labels_current_query_rgb_and_clean_suffixes():
    system,batch = make_system()
    changed = copy_batch(batch)
    for key in ("source_future_latent","source_action","fast_action","future","contact","rgb","state"):
        changed[key] = torch.full_like(changed[key],99.)
    changed["future_loss_valid"].zero_()
    changed["contact_valid"].zero_()
    changed["source_known_actions"][:,1:] = torch.nan
    changed["fast_known_actions"][:,1:] = -99
    first = system.predict_reference(batch,generator=generator())
    second = system.predict_reference(changed,generator=generator())
    for key in ("predicted_video_latent","reference_actions"):
        torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)


def test_joint_mode_only_prefix_conditional_mode_sees_clean_actions():
    system,batch = make_system()
    captures = []
    def capture(module,args,kwargs):
        captures.append({key:value.detach().clone() for key,value in kwargs.items() if isinstance(value,torch.Tensor)})
    handle = system.fast.register_forward_pre_hook(capture,with_kwargs=True)
    normal = system.training_loss(batch,generator(),action_conditioned=torch.tensor([False]))
    auxiliary = system.training_loss(batch,generator(),action_conditioned=torch.tensor([True]))
    handle.remove()
    assert torch.equal(captures[0]["action_known_mask"],batch["fast_known_mask"])
    assert torch.equal(captures[0]["known_actions"],batch["fast_known_actions"])
    assert captures[1]["action_known_mask"].all()
    assert torch.equal(captures[1]["known_actions"],batch["fast_action"])
    assert normal["fast_action"] > 0
    assert auxiliary["fast_action"] == 0
    assert "future_valid" not in captures[0] and "future_valid" not in captures[1]


def test_label_visibility_never_becomes_forward_condition():
    system,batch = make_system()
    reference = system.predict_reference(batch,generator=generator())
    world = system.world_readout(reference)
    changed = copy_batch(batch)
    changed["future_loss_valid"].zero_()
    changed["contact_valid"].zero_()
    changed["future"].fill_(torch.nan)
    changed["contact"].fill_(torch.nan)
    a,b = system.conditions(batch,world),system.conditions(changed,world)
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert "future_valid" not in a


def test_sample_fast_clamps_prefix_every_step_and_never_reads_labels():
    system,batch = make_system()
    world = system.world_readout(system.predict_reference(batch,generator=generator()))
    inference = {key:value for key,value in batch.items() if key not in
                 ("source_future_latent","source_action","fast_action","future","future_loss_valid","contact","contact_valid")}
    seen = []
    def capture(module,args,kwargs):
        seen.append(args[0].detach().clone())
    original = system.fast.forward_cached
    def tracked(*args, **kwargs):
        seen.append(args[0].detach().clone())
        return original(*args, **kwargs)
    system.fast.forward_cached = tracked
    output = system.sample_fast(inference,world,steps=4,generator=generator())
    system.fast.forward_cached = original
    assert len(seen) == 5  # four denoising steps and final contact readout
    for actions in seen+[output["actions"]]:
        torch.testing.assert_close(actions[:,:1],batch["fast_known_actions"][:,:1],atol=0,rtol=0)
    assert output["future"].shape == (1,16,10)
    assert torch.isfinite(output["contact_probability"]).all()


def test_valid_future_nan_and_nonbinary_contact_fail_closed():
    system,batch = make_system()
    batch["future"][0,1,0] = torch.nan
    with pytest.raises(ValueError,match="future labels must be finite"):
        system.training_loss(batch,generator())
    batch["future_loss_valid"][0,1,0] = False
    batch["contact"][0,0,0] = .5
    with pytest.raises(ValueError,match="binary"):
        system.training_loss(batch,generator())
    batch["contact_valid"][0,0,0] = False
    assert torch.isfinite(system.training_loss(batch,generator())["loss"])


@pytest.mark.parametrize("field", ["source_time","completed_time"])
def test_unavailable_reference_is_a_training_error(field):
    system,batch = make_system()
    batch[field] = torch.tensor([2.2],dtype=torch.float64)
    with pytest.raises(ValueError,match="source <= completed <= query"):
        system.training_loss(batch,generator())
    with pytest.raises(ValueError,match="source <= completed <= query"):
        system.predict_reference(batch,generator=generator())


def test_source_commitment_cannot_be_uncommitted_by_fast_branch():
    system,batch = make_system()
    batch["source_known_mask"][0,1] = True
    batch["source_action"][0,1] = batch["source_known_actions"][0,1]
    with pytest.raises(ValueError,match="drops a still-future source commitment"):
        system.training_loss(batch,generator())
    batch["fast_known_mask"][0,1] = True
    batch["fast_known_actions"][0,1] += 1
    with pytest.raises(ValueError,match="disagree"):
        system.training_loss(batch,generator())
    batch["fast_known_actions"][0,1] = batch["source_known_actions"][0,1]
    batch["fast_action"][0,1] = batch["fast_known_actions"][0,1]
    assert torch.isfinite(system.training_loss(batch,generator())["loss"])


@pytest.mark.parametrize("case", ["mask_dtype","contact_shape","time_order"])
def test_label_schema_and_requested_physical_times_are_checked(case):
    system,batch = make_system()
    if case == "mask_dtype":
        batch["future_loss_valid"] = batch["future_loss_valid"].float()
    elif case == "contact_shape":
        batch["contact"] = batch["contact"][:,:4]
    else:
        batch["future_times"] = batch["future_times"].flip(1)
    with pytest.raises(ValueError):
        system.training_loss(batch,generator())


def test_new_fast_commitments_are_allowed_and_source_not_rewritten():
    system,batch = make_system()
    batch["fast_known_mask"][0,1] = True
    batch["fast_action"][0,1] = batch["fast_known_actions"][0,1]
    before = batch["source_known_mask"].clone()
    assert torch.isfinite(system.training_loss(batch,generator())["loss"])
    assert torch.equal(before,batch["source_known_mask"])


@pytest.mark.parametrize("wrapped", [False,True])
def test_train_mode_keeps_codec_eval_without_freezing_trainable_experts(wrapped):
    system,_ = make_system()
    codec = torch.nn.Sequential(torch.nn.Dropout(.5),torch.nn.Linear(2,2))
    system.slow.native.compact_wan.video_model.vae = SimpleNamespace(model=codec) if wrapped else codec
    assert system.train() is system
    assert system.slow.native.training and system.slow.native.action_expert.training
    assert system.fast.training and system.observation.training
    assert not codec.training
    assert system.slow.native.video_weight.requires_grad
    assert system.slow.native.action_weight.requires_grad
    assert system.slow.native.motion_weight.requires_grad
    system.eval()
    assert not system.slow.native.training


def test_train_mode_fixture_without_vae_is_safe():
    system,_ = make_system()
    system.train()
    assert system.slow.native.training


@pytest.mark.parametrize("prefix", ["source","fast"])
def test_expert_future_cannot_supervise_a_different_committed_trajectory(prefix):
    system,batch = make_system()
    batch[prefix+"_action"][0,0,0] += 1
    with pytest.raises(ValueError,match="differs from the supervised trajectory"):
        system.training_loss(batch,generator())


def test_task_known_entity_presence_gates_requested_slots_not_gt_visibility():
    system,batch = make_system()
    batch["query_entity_valid"] = torch.tensor([[True,False,True,True]])
    world = system.world_readout(system.predict_reference(batch,generator=generator()))
    before = system.conditions(batch,world)
    expected = torch.tensor([[True]*4+[False]*4+[True]*8])
    assert torch.equal(before["future_valid"],expected)
    batch["future_loss_valid"].zero_()
    batch["contact_valid"].zero_()
    after = system.conditions(batch,world)
    for key in before:
        torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
    result = system.fast(torch.randn(1,4,2),torch.randn(1,16,10),torch.ones(1),**after)
    assert not result["future_velocity"][:,4:8].any()
    assert not result["contact_logits"][:,4:8].any()
    batch["query_entity_valid"] = batch["query_entity_valid"].float()
    with pytest.raises(ValueError,match="query_entity_valid"):
        system.conditions(batch,world)
