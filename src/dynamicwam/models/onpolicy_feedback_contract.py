"""Identity for feedback trained on states reached by the deployed controller.

DOMINO's canonical joint_action is the drive target, not measured joint qpos.
This format is intentionally separate from offline expert-pair checkpoints.
"""
FORMAT = 'dynamicwam_onpolicy_feedback_checkpoint'
VERSION = 1
AGES = (2, 4, 6, 8, 10, 12, 14)


def runtime_contract(schedule, limit, history_mode='temporal'):
    if history_mode not in ('static', 'temporal'):
        raise ValueError('on-policy history_mode must be static or temporal')
    if not 0 < float(limit) <= 1:
        raise ValueError('finite positive bounded residual limit required')
    return dict(source_plan='immutable_official_full_16', execute_horizons=[2, 4],
        output_horizon=4, refresh_actions=16, ages=list(AGES),
        tail='mask_beyond_16_no_repeated_labels', age0_correction=False,
        gripper_correction=False, residual_units='normalized_joint_drive_targets',
        residual_limit=float(limit), state_semantics='joint_drive_targets',
        target='teacher_only_clipped_residual',
        preprocess='three_view_160_imagenet_mask_v1',
        history='actual_previous_command', history_mode=history_mode,
        clock='required_simulator_seconds', history_dropout=0.0,
        architecture='pretrained_static_head_additive_zero_temporal_v3',
        age_initialization='legacy_0_4_8_12_exact_linear_between_hold_after12',
        plan_age_seconds='telemetry_only', schedule=schedule.to_config(),
        precision=dict(video='bfloat16', action='float32', solver='float32',
            feedback_parameters='float32', feedback_autocast='bfloat16'))


def validate_runtime_contract(contract, schedule, limit, execute_horizon):
    mode = contract.get('history_mode')
    if contract != runtime_contract(schedule, limit, mode):
        raise ValueError('on-policy feedback train/deploy contract mismatch')
    if execute_horizon not in contract['execute_horizons']:
        raise ValueError('on-policy feedback supports only trained even-age H2/H4 execution')
    return mode
