"""Shared train/export/deploy identity for temporal bounded feedback."""
FORMAT = 'dynamicwam_temporal_plan_feedback_checkpoint'
VERSION = 3
AGES = tuple(range(16))


def runtime_contract(schedule, limit):
    return dict(source_plan='immutable_official_full_16', execute_horizons=[1, 2, 4],
        output_horizon=4, refresh_actions=16, ages=list(AGES),
        tail='mask_beyond_16_no_repeated_labels', age0_correction=False,
        gripper_correction=False, residual_units='normalized_qpos', residual_limit=limit,
        preprocess='three_view_160_imagenet_mask_v1',
        history='previous_command_head_160_causal_v1', action_downsample=3,
        history_training_raw_lags=[1, 2, 3, 6, 9, 12, 18, 24, 36],
        history_sampling='pair_identity_sha256_v1',
        history_candidate_rule='strictly_positive_real_dt',
        clock='required_simulator_seconds', history_dropout=0.15,
        architecture='pretrained_static_head_additive_zero_temporal_v3',
        age_initialization='legacy_0_4_8_12_exact_linear_between_hold_after12',
        plan_age_seconds='telemetry_only', schedule=schedule.to_config(),
        precision=dict(video='bfloat16', action='float32', solver='float32',
            feedback_parameters='float32', feedback_autocast='bfloat16'))
