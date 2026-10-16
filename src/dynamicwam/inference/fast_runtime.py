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


RTC_TIMESTAMPED_CONTROLLER = 'domino_timestamped_prefix_v1'


def require_rtc_deployment_mode(payload, *, allow_zero_prefix_ablation=False,
                                rtc_controller=None):
    """Require an explicit deployment path for trained RTC prefixes.

    Legacy callers remain closed except labelled d=0 ablations. The timestamped
    DOMINO path validates controller packets and preserves prefixes in its own
    serving adapter; this selector alone does not certify queue timing.
    """
    from dynamicwam.inference.multirate_model_loader import rtc_prefix_contract
    if type(allow_zero_prefix_ablation) is not bool:
        raise TypeError("allow_zero_prefix_ablation must be an explicit bool")
    if rtc_controller not in (None, RTC_TIMESTAMPED_CONTROLLER):
        raise ValueError('unknown RTC controller contract')
    if rtc_controller is not None and allow_zero_prefix_ablation:
        raise ValueError('RTC deployment and zero-prefix ablation are exclusive')
    rtc = rtc_prefix_contract(payload)
    if rtc_controller is not None and rtc is None:
        raise RuntimeError('timestamped RTC deployment requires trained prefix metadata')
    if rtc is not None and not allow_zero_prefix_ablation and rtc_controller is None:
        raise RuntimeError(
            "RTC-trained checkpoint cannot run through the legacy controller: "
            "a timestamped prefix controller is not integrated. For an explicit "
            "d=0 ablation only, pass allow_zero_prefix_ablation=True."
        )
    return rtc


def build_fast_runtime(payload, *, device='cpu', dtype=torch.float32,
                       allow_zero_prefix_ablation=False, rtc_controller=None):
    from dynamicwam.inference.multirate_model_loader import assert_fast_only_payload, load_fast_only_modules
    rtc = require_rtc_deployment_mode(payload,
        allow_zero_prefix_ablation=allow_zero_prefix_ablation, rtc_controller=rtc_controller)
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
    model.rtc_prefix_max = 0 if rtc is None else int(rtc['max_prefix_actions'])
    model.rtc_deployment_mode = ('not_trained_for_rtc' if rtc is None else
        (rtc_controller or 'zero_prefix_ablation'))
    model.rtc_runtime_contract = rtc
    return finalize_inference(model, device=device, dtype=dtype)


_WORLD_PARAMETER_MARKERS = ("compact_wan", "vae", "text_encoder", ".t5")


def isolated_fast_runtime(payload, *, device="cpu", dtype=torch.float32,
                          allow_zero_prefix_ablation=False, rtc_controller=None):
    """Load a second fast copy that does not own WAN, VAE, or T5.

    The slow bridge keeps the full world model. This copy is the online action
    sampler, so the two forwards do not share video buffers or parameters.
    """

    from dynamicwam.inference.multirate_model_loader import export_fast_only_payload

    require_rtc_deployment_mode(payload,
        allow_zero_prefix_ablation=allow_zero_prefix_ablation, rtc_controller=rtc_controller)
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
    model = build_fast_runtime(fast_payload, device=device, dtype=dtype,
        allow_zero_prefix_ablation=allow_zero_prefix_ablation, rtc_controller=rtc_controller)
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


def validate_runtime_contract(payload, runtime, *, allow_zero_prefix_ablation=False,
                              rtc_controller=None):
    require_rtc_deployment_mode(payload,
        allow_zero_prefix_ablation=allow_zero_prefix_ablation, rtc_controller=rtc_controller)
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
        if not modules.get('fast_conditioner', {}).get('world_aware') or native.get('version') not in (1, 2, 3):
            raise RuntimeError('native checkpoint lacks observation-guided world router')
        steps = int(native.get('fast_steps', 0))
        if steps != modules.get('fast_steps') or steps not in (1, 2, 4, 10):
            raise RuntimeError('native fast step contract mismatch')
        expected = BridgeSchedule.from_flow_match(**contract['schedule']).with_action_steps(steps)
        if modules.get('fast_schedule') != expected.to_config():
            raise RuntimeError('serialized fast solver differs from trained native schedule')
        source = native.get('source_bridge')
        if source == 'student_joint':
            version = native.get('version')
            if version not in (2, 3) or modules['fast_conditioner'].get('interface_version') != version:
                raise RuntimeError('joint slow bridge and trained motion interface versions disagree')
            if version == 2 and contract.get('max_age_actions') != 15:
                raise RuntimeError('v2 slow bridge requires causal age support 0..15')
            if version == 3:
                if (contract.get('max_age_actions') != 31 or contract.get('max_age_seconds') != 4.96 or
                    native.get('slow_schedule') != 'asynchronous_latest_completed_v1'):
                    raise RuntimeError('v3 requires asynchronous publication with trained 31-action/4.96-second cache bounds')
                feedback = bool(modules['fast_conditioner'].get('proposal_feedback', False))
                expected_feedback = 'sigma_conditioned_physical_plan_v1' if feedback else 'disabled'
                if contract.get('proposal_feedback', 'disabled') != expected_feedback:
                    raise RuntimeError('proposal feedback architecture and trained runtime contract disagree')
                completed = bool(modules['fast_conditioner'].get('complete_source_plan', False))
                expected_completed = 'same_noise_native_K10_tail_stopgrad_v1' if completed else 'disabled'
                if contract.get('source_plan_completion', 'disabled') != expected_completed:
                    raise RuntimeError('completed source plan architecture and trained contract disagree')
                source_time = bool(modules['action_expert'].get('source_physical_time_conditioning', False))
                expected_source_time = 'native_action_tokens_target_offset_phase_v1' if source_time else 'disabled'
                if (contract.get('source_physical_time_conditioning', 'disabled') != expected_source_time
                        or bool(config.get('model', {}).get('action_expert', {}).get('source_physical_time_conditioning', False)) != source_time):
                    raise RuntimeError('source physical time architecture and trained contract disagree')
                physical = contract.get('physical_grid_contract', {})
                def valid_sha256(value):
                    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                if (not valid_sha256(contract.get('physical_grid'))
                        or not valid_sha256(physical.get('latent_manifest_sha256'))
                        or physical.get('schema_version') != 1
                        or physical.get('action_interval_seconds') != .16
                        or physical.get('action_horizon') != 16
                        or physical.get('source_plan_alignment') != 'physical_time_linear_arm_previous_hold_gripper'
                        or getattr(runtime, 'action_interval_seconds', None) != .16
                        or contract.get('previous_plan') != 'soft_context_only; hard_prefix_is_irrevocable'
                        or contract.get('source_retention') != 'fixed_grid_source_video_action_fm'
                        or contract.get('slow_source_prefix') != 'clean_committed_v1'):
                    raise RuntimeError('v3 physical grid, latent provenance or proposal semantics mismatch')
        elif source == 'immutable_official':
            if native.get('version') != 1 or not native.get('teacher', {}).get('sha256'):
                raise RuntimeError('native slow bridge must identify its immutable official weights')
            if contract.get('max_age_actions') != 12:
                raise RuntimeError('legacy native cache age support changed')
        else:
            raise RuntimeError('unknown native source bridge provenance')
        if native.get('world_refresh_actions') != 16:
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
