"""CPU integration of versioned loading and the 1x deployment contract."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from check_joint_motion_kv_conditioner import joint_fixture
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.inference.multirate_model_loader import build_fast_conditioner, install_immutable_source_backbone
from dynamicwam.inference.fast_runtime import validate_runtime_contract, RTC_TIMESTAMPED_CONTROLLER

def run():
    torch.set_num_threads(2)
    model, *_ = joint_fixture()
    loaded = build_fast_conditioner(model.constructor_spec)
    loaded.load_state_dict(model.state_dict(), strict=True)
    assert type(loaded) is type(model)
    backbone = torch.nn.Linear(3, 4)
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    native = dict(version=2, source_bridge='student_joint', world_refresh_actions=16, fast_steps=10)
    install_immutable_source_backbone(backbone, dict(native_multirate=native), {})
    assert not backbone.training and all(not p.requires_grad for p in backbone.parameters())
    assert all(torch.equal(v, before[k]) for k, v in backbone.state_dict().items())
    schedule = BridgeSchedule.from_flow_match()
    norm = dict(stats=dict(mean=[0.] * 14, std=[1.] * 14), epsilon=1e-6)
    rtc = dict(version=1, mode='training_time_action_prefix', max_prefix_actions=4,
        clean_timestep=0., loss='per_sample_suffix_mean', includes_zero_prefix=True,
        queue_clock='absolute_action_effect_times', overflow='reject_and_refresh', execute_suffix_actions=4)
    contract = dict(version=1, native_multirate=native, max_age_actions=15,
        preprocess='three_view_160_imagenet_mask_v1', clock_mode='required',
        schedule=schedule.to_config(), execute_horizon=4, rtc=rtc)
    modules = dict(native_multirate=True, architecture='native', fast_conditioner=model.constructor_spec,
        fast_steps=10, fast_schedule=schedule.with_action_steps(10).to_config(),
        schedule=schedule.to_config(), runtime_contract=contract,
        action_normalization=norm, action_expert=dict(chunk_size=16))
    payload = dict(modules=modules, config=dict(runtime_contract=contract, action_normalization=norm))
    sc = schedule.to_config()
    runtime = SimpleNamespace(flow_match_shift=sc['shift'], flow_match_sigma_min=sc['sigma_min'],
        flow_match_extra_one_step=sc['extra_one_step'], num_inference_steps=sc['action_steps'],
        num_video_inference_steps=sc['video_steps'], chunk_size=16,
        action_normalizer=RoboTwinQposNormalizer.from_stats(norm['stats'], epsilon=norm['epsilon']))
    validate_runtime_contract(payload, runtime, rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
    for field, value in [('version', 1), ('source_bridge', 'unknown'), ('world_refresh_actions', 12)]:
        bad = deepcopy(payload)
        bad['config']['runtime_contract']['native_multirate'][field] = value
        try:
            validate_runtime_contract(bad, runtime, rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
        except RuntimeError:
            pass
        else:
            raise AssertionError(field)
    print(json.dumps(dict(passed=True, checks=6, scope='CPU loading/provenance/RTC contracts; no CUDA capture or closed loop')))

if __name__ == '__main__':
    run()
