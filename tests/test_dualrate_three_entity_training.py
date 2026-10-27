"""Production-capacity 12Z fast path with an explicitly tiny CPU native fixture."""
import torch

from dynamicwam.dualrate.fast import CurrentObservationEncoder, FastConfig, FastJointWAM
from dynamicwam.dualrate.optimizer import FP32MasterAdamW
from dynamicwam.dualrate.slow import StructuredWorldReadout
from dynamicwam.dualrate.system import DualRateSystem, LossConfig
from test_dualrate_system import make_system, generator


def test_three_entity_full_fast_capacity_joint_backward_and_strict_reload():
    original, batch = make_system()
    torch.manual_seed(31)
    arch = FastConfig(future_tokens=12, future_entities=3, action_dim=2, chunk_size=4, test_fixture=True)
    fast = FastJointWAM(arch)
    system = DualRateSystem(original.slow,
        StructuredWorldReadout(latent_channels=2, action_dim=2, world_dim=384,
                              native_hidden_dim=2, test_fixture=True),
        CurrentObservationEncoder(dim=384, views=1, motion_dim=3, state_dim=2,
                                  text_dim=4, test_fixture=True),
        fast, future_latent_shape=(2, 2, 4, 4),
        loss_config=LossConfig(slow_video=.001, slow_action=1., fast_action=1.,
            future_visual=.25, future_geometry=1., contact=.1,
            action_conditioned_probability=0., cache_dropout=0., visual_dim=128))
    indices = torch.tensor(list(range(4))+list(range(8,16)))
    batch['future_times'] = batch['future_times'][:, indices]
    batch['future'] = torch.randn(1,12,134)
    batch['future_loss_valid'] = torch.ones(1,12,134,dtype=torch.bool)
    batch['contact'] = batch['contact'][:,indices]
    batch['contact_valid'] = batch['contact_valid'][:,indices]
    optimizer = FP32MasterAdamW(system.parameters(), lr=1e-4)
    result = system(batch, generator=generator())
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for module in (system.slow,system.world_readout,system.observation,system.fast):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in module.parameters() if p.requires_grad)
    assert system.fast.future_in.weight.grad.abs().sum()>0
    optimizer.accumulate_gradients()
    optimizer.clip_accumulated_grad_norm_(1.)
    optimizer.step()
    optimizer.zero_grad()
    # Strict load after an actual full-capacity A/Z update, including 12-role buffers.
    system.load_state_dict(system.state_dict(), strict=True)
    optimizer.load_state_dict(optimizer.state_dict())
    assert system.future_field_support.shape == (12,134)
    assert system.contact_field_support.shape == (12,2)
    assert len(system.fast.blocks) == 6
