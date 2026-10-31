"""Reject silent native-event -> continuous-time dataset reuse."""
import hashlib
import json
from pathlib import Path


def validate_policy_contract(config, contract):
    """Parameter shapes alone cannot distinguish full/residual or gating modes."""
    if config.get('continuous_v8') != contract.get('continuous_v8'):
        raise ValueError('continuous V8 train/deploy policy semantics differ')
    if config.get('continuous_v8') and any(config.get(k) for k in ('native_v5','native_v6','native_v7')):
        raise ValueError('continuous V8 cannot use native event-coordinate policy')


def validate_training_contract(config, manifest):
    expected = dict(version='continuous_rtc_v8',clock='physical_seconds',
        action_semantics='fixed_interval_endpoint_arm_absolute_gripper',
        action_interval_seconds=config['action_interval_seconds'],
        history_endpoint_seconds=.72,source='sampled_slow_prediction',
        labels='same_absolute_physical_interval',prefix='actual_committed_values_and_intervals')
    if (config.get('time_contract')!='exposure_anchor_wall_release_v4'
            or manifest.get('time_contract')!=config['time_contract']
            or manifest.get('continuous_rtc_contract')!=expected
            or any(config.get(k) for k in ('native_v5','native_v6','native_v7'))):
        raise ValueError('V8 requires audited continuous physical-time labels/prefixes, not native event data')
    if config['run_kind']=='formal_distributed_training':
        path=Path(manifest['continuous_acceptance_file']).resolve(strict=True)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['continuous_acceptance_sha256']:
            raise ValueError('continuous data acceptance checksum differs')
        receipt=json.loads(raw)
        checks=('same_command_state_parity','history_train_deploy_parity',
                'injected_delay_queue_parity','actual_gpu_parallel_timing',
                'physical_label_audit','all_task_smoke')
        if (receipt.get('contract')!=expected or receipt.get('source_sha256')!=config['source_sha256']
                or any(receipt.get(k) is not True for k in checks)):
            raise ValueError('V8 formal training is gated on simulator, data and GPU acceptance')
