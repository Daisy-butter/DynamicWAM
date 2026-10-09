"""Executable fast-only models and explicit deployment contracts."""
from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn

from dynamicwam.models.action_core_ops import ActionCoreOps, ActionAttentionSpec
from dynamicwam.models.compact_fast_policy import CompactFastPolicy, CompactFastConfig
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.action_normalization import RoboTwinQposNormalizer


class StandaloneActionCore(ActionCoreOps, nn.Module):
    def __init__(self, action_expert, motion, spec):
        super().__init__()
        self.action_expert = action_expert
        self.absolute_motion_tokens = motion
        cfg = action_expert.config
        self.config = SimpleNamespace(ae_dim=cfg.dim, ae_num_layers=cfg.num_layers, chunk_size=cfg.chunk_size)
        attn = spec['wan_attention']
        precision = getattr(torch, attn.get('precision', 'bfloat16'))
        windows = attn.get('window_sizes', [(-1, -1)] * cfg.num_layers)
        self.action_attention_spec = ActionAttentionSpec(precision, tuple(tuple(w) for w in windows))


def finalize_inference(model, *, device, dtype=torch.float32):
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise RuntimeError('inference model contains training-mode modules')
    return model


def build_fast_runtime(payload, *, device='cpu', dtype=torch.float32):
    from dynamicwam.inference.multirate_model_loader import assert_fast_only_payload, load_fast_only_modules
    state = assert_fast_only_payload(payload)
    spec = payload['modules']
    schedule = BridgeSchedule.from_flow_match(**spec['schedule'])
    if spec.get('architecture') == 'compact':
        if spec['compact_fast'].get('feedback_enabled') and not spec.get('runtime_contract',{}).get('feedback',{}).get('ready'):
            raise RuntimeError('feedback fast export is not ready for endpoint inference')
        motion = spec['motion']
        model = CompactFastPolicy(CompactFastConfig(**spec['compact_fast']),
            motion_history_count=int(motion['history_count']), motion_mean=motion['mean'],
            motion_scale=motion['scale'], schedule=schedule)
        if any(not key.startswith('fast_policy.') for key in state):
            raise RuntimeError('compact fast export contains non-student weights')
        model.load_state_dict({k.removeprefix('fast_policy.'):v for k,v in state.items()}, strict=True)
    else:
        modules = load_fast_only_modules(payload)
        core = StandaloneActionCore(modules.action_expert, modules.absolute_motion_tokens, spec)
        native = bool(spec.get('native_multirate', False))
        model = MultirateWAM(core, modules.fast_conditioner, schedule,
            native_multirate=native, fast_steps=int(spec.get('fast_steps', 10)))
        if native and spec.get('fast_schedule') != model.fast_schedule.to_config():
            raise RuntimeError('native fast schedule differs from serialized solver')
    return finalize_inference(model, device=device, dtype=dtype)


_WORLD_PARAMETER_MARKERS = ("compact_wan", "vae", "text_encoder", ".t5")


def isolated_fast_runtime(payload, *, device="cpu", dtype=torch.float32):
    """Load a second fast copy that does not own WAN, VAE, or T5.

    The slow bridge keeps the full world model. This copy is the online action
    sampler, so the two forwards do not share video buffers or parameters.
    """

    from dynamicwam.inference.multirate_model_loader import export_fast_only_payload

    modules = payload.get("modules")
    state = payload.get("model")
    if not isinstance(modules, dict) or "schedule" not in modules:
        raise RuntimeError("isolated fast runtime requires the checkpoint module schedule")
    if not isinstance(state, dict):
        raise RuntimeError("isolated fast runtime requires checkpoint weights")
    fast_payload = export_fast_only_payload(
        state,
        module_spec=modules,
        global_step=int(payload.get("global_step", 0)),
    )
    model = build_fast_runtime(fast_payload, device=device, dtype=dtype)
    if isinstance(model, CompactFastPolicy):
        # Keep the online sampler on MultirateWAM so compact and reference
        # deployments share prepare_fast_memory / sample_fast_actions.
        model = finalize_inference(
            MultirateWAM(nn.Identity(), None, model.schedule, compact_policy=model),
            device=device,
            dtype=dtype,
        )
    for name, _parameter in model.named_parameters():
        lowered = name.lower()
        if any(marker in lowered for marker in _WORLD_PARAMETER_MARKERS):
            raise RuntimeError(f"fast runtime still owns a world parameter: {name}")
    return model


def validate_runtime_contract(payload, runtime):
    config = payload.get('config', {})
    contract = config.get('runtime_contract')
    if not isinstance(contract, dict) or contract.get('version') != 1:
        raise RuntimeError('checkpoint lacks verified multirate runtime contract; re-export with explicit metadata')
    modules = payload.get('modules', {})
    if modules.get('runtime_contract') != contract:
        raise RuntimeError('module and checkpoint runtime contracts disagree')
    if modules.get('schedule') != contract.get('schedule'):
        raise RuntimeError('module and checkpoint schedules disagree')
    native = contract.get('native_multirate')
    if native:
        if not modules.get('native_multirate') or modules.get('architecture') != 'native':
            raise RuntimeError('native checkpoint architecture contract mismatch')
        if not modules.get('fast_conditioner', {}).get('world_aware') or native.get('version') != 1:
            raise RuntimeError('native checkpoint lacks observation-guided world router')
        steps = int(native.get('fast_steps', 0))
        if steps != modules.get('fast_steps') or steps not in (1, 2, 4, 10):
            raise RuntimeError('native fast step contract mismatch')
        expected = BridgeSchedule.from_flow_match(**contract['schedule']).with_action_steps(steps)
        if modules.get('fast_schedule') != expected.to_config():
            raise RuntimeError('serialized fast solver differs from trained native schedule')
        if native.get('source_bridge') != 'immutable_official' or not native.get('teacher', {}).get('sha256'):
            raise RuntimeError('native slow bridge must identify its immutable official weights')
        if native.get('world_refresh_actions') != 16 or contract.get('max_age_actions') != 12:
            raise RuntimeError('native trained world refresh support mismatch')
    if modules.get('action_normalization') != config.get('action_normalization'):
        raise RuntimeError('module and checkpoint normalizers disagree')
    if contract.get('preprocess') != 'three_view_160_imagenet_mask_v1':
        raise RuntimeError('fast image preprocessing contract mismatch')
    saved = BridgeSchedule.from_flow_match(**contract['schedule'])
    actual = BridgeSchedule.from_flow_match(shift=runtime.flow_match_shift,
        sigma_min=runtime.flow_match_sigma_min, extra_one_step=runtime.flow_match_extra_one_step,
        action_steps=runtime.num_inference_steps, video_steps=runtime.num_video_inference_steps)
    if saved.to_config() != actual.to_config():
        raise RuntimeError('deploy scheduler differs from multirate checkpoint')
    norm = config['action_normalization']
    candidate = RoboTwinQposNormalizer.from_stats(norm['stats'], epsilon=float(norm['epsilon']))
    reference = runtime.action_normalizer
    if candidate.epsilon != reference.epsilon or not torch.equal(candidate._mean, reference._mean) or not torch.equal(candidate._std, reference._std):
        raise RuntimeError('deploy action/state normalizer differs from multirate checkpoint')
    if contract.get('clock_mode') not in {'required', 'action_only'}:
        raise RuntimeError('invalid trained clock mode')
    if int(contract.get('max_age_actions', -1)) < 0:
        raise RuntimeError('missing trained age support')
    if not 1 <= int(contract.get('execute_horizon', 0)) <= int(runtime.chunk_size):
        raise RuntimeError('trained execute horizon is outside the action chunk')
    if int(modules['action_expert']['chunk_size']) != int(runtime.chunk_size):
        raise RuntimeError('deploy action chunk differs from multirate checkpoint')
    return saved, contract
