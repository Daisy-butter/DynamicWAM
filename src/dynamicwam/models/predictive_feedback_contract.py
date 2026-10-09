"""Versioned future-supervised feedback contract, separate from old K/V reuse."""
import math

FORMAT='dynamicwam_predictive_feedback_checkpoint'
VERSION=1


def runtime_contract(schedule,limit):
    cfg=schedule.to_config()
    if cfg['action_steps']!=10 or cfg['video_steps']!=2:
        raise ValueError('complete official WAM schedule required')
    if not math.isfinite(float(limit)) or not 0<float(limit)<=.2:
        raise ValueError('invalid bounded correction')
    return dict(architecture='fixed_spatial_short_future_feedback_v1',
        source_plan='immutable_official_full_16',source_plan_length=16,
        output_horizon=4,execute_horizons=[2],refresh_actions=12,
        control_period_seconds=.1,physics_dt_seconds=.004,
        age0_correction=False,gripper_correction=False,
        ages=list(range(16)),residual_limit=float(limit),
        residual_units='normalized_joint_drive_targets',state_semantics='joint_drive_targets',
        target='teacher_only_clipped_residual',preprocess='three_view_160_imagenet_mask_v1',
        history='actual_previous_snapshot_with_positive_physical_dt',
        future_representation='frozen_v1_spatial_3view_4x4_tokens',
        future_horizons='first_boundary_after_dispatch_plus_20ms_then_100ms_and_300ms',
        action_path_bins=4,
        action_path_units='normalized_drive_q_and_mean_drive_velocity',
        world_supervision='observed_future_rgb_and_actual_covered_commands',
        policy_condition='causal_controller_proposal_never_future_executed_commands',
        proposal_install='first_boundary_after_dispatch_plus_20ms_not_measured_arrival',
        known_command_prefix='complete_observation_to_dispatch_only',
        post_correction='reforecast_corrected_proposal_reusing_current_encoding',
        hierarchical_latent_model=False,clock='required_real_simulator_seconds',
        controller_protocol='external_sha256_required',schedule=cfg,
        precision=dict(video='bfloat16',action='float32',solver='float32',
            feedback_parameters='float32',feedback_autocast='bfloat16',visual_targets='float32'))


def validate_runtime_contract(value,schedule,limit,execute_horizon=2):
    expected=runtime_contract(schedule,limit)
    if value!=expected or execute_horizon not in expected['execute_horizons']:
        raise ValueError('predictive feedback train/deploy contract differs')
