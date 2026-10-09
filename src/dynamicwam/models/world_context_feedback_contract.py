"""A separately identifiable continuous-clock, on-policy world-feedback format."""
from __future__ import annotations
import math
from dynamicwam.models.world_context_feedback import WORLD_CONTEXT_SPEC

FORMAT = 'dynamicwam_world_context_feedback_checkpoint'
VERSION = 1
AGES = tuple(range(16))


def runtime_contract(schedule, limit, history_mode='temporal', world_mode='context'):
    if history_mode not in ('static','temporal') or world_mode not in ('context','disabled'):
        raise ValueError('unsupported history/world mode')
    if not math.isfinite(float(limit)) or not 0 < float(limit) <= .2:
        raise ValueError('invalid bounded residual limit')
    if schedule.to_config()['action_steps'] != 10 or schedule.to_config()['video_steps'] != 2:
        raise ValueError('the complete official native sampler is required')
    return dict(source_plan='immutable_official_full_16',source_plan_length=16,output_horizon=4,
        execute_horizons=[2],refresh_actions=12,ages=list(AGES),
        deployed_nonfresh_query_ages=list(range(1,12)),control_period_seconds=.1,
        age0_correction=False,gripper_correction=False,tail='mask_beyond_16_no_repeated_labels',
        residual_limit=float(limit),residual_units='normalized_joint_drive_targets',
        state_semantics='joint_drive_targets',target='teacher_only_clipped_residual',
        preprocess='three_view_160_imagenet_mask_v1',history='actual_previous_snapshot',
        history_mode=history_mode,history_dropout=0.,world_mode=world_mode,
        world_context_spec=dict(WORLD_CONTEXT_SPEC),context_identity='immutable_plan_id_and_source_clock',
        context_validity='same_bundle_and_age_less_than_16',
        world_semantics='reference_second_joint_denoising_features_not_corrected_action_rollout',
        architecture='pretrained_feedback_zero_additive_world_readout',
        clock='required_real_simulator_seconds',controller_protocol='external_sha256_required',
        plan_age_seconds='telemetry_only',schedule=schedule.to_config(),
        precision=dict(video='bfloat16',action='float32',solver='float32',
            feedback_parameters='float32',feedback_autocast='bfloat16'))


def validate_runtime_contract(contract,schedule,limit,execute_horizon):
    expected=runtime_contract(schedule,limit,contract.get('history_mode'),contract.get('world_mode'))
    if contract != expected or execute_horizon not in expected['execute_horizons']:
        raise ValueError('world-feedback training/deployment contract mismatch')
    return contract['history_mode'],contract['world_mode']
