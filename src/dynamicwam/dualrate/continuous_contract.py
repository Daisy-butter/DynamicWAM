"""Reject silent native-event -> continuous-time dataset reuse."""
import hashlib
import json
from pathlib import Path


def validate_initialization_fixture(config,payload):
    """A zero-step full-model diagnostic is never an accepted trained policy."""
    from .train import source_identity
    mode=(config.get('run_kind')=='domino_deployment_smoke' or
          (config.get('run_kind')=='dualrate_eval_suite' and config.get('stage')=='smoke'
           and config.get('engineering_initialization_only') is True))
    if (not mode or not config.get('continuous_v8')
            or payload.get('schema')!='dualrate-deployment-initialization-fixture-v1'
            or payload.get('run_kind')!='deployment_engineering_fixture_not_formal_init'
            or payload.get('phase')!='untrained_initialization'
            or payload.get('phase_step')!=0 or payload.get('global_step')!=0
            or payload.get('formal_training_ready') is not False
            or payload.get('policy_quality_accepted') is not False):
        raise ValueError('explicit untrained V8 diagnostic fixture required')
    contract=payload['contract']
    validate_policy_contract(config,contract)
    for key in ('architecture','codec_id','init_checkpoint_sha256','action_normalization_sha256',
                'motion_normalization_sha256','time_contract','action_interval_seconds'):
        if contract.get(key)!=config[key]:raise ValueError('diagnostic contract differs: '+key)
    if contract.get('source_sha256')!=source_identity():raise ValueError('diagnostic source changed')
    return contract


def validate_policy_contract(config, contract):
    """Parameter shapes alone cannot distinguish full/residual or gating modes."""
    if config.get('continuous_v8') != contract.get('continuous_v8'):
        raise ValueError('continuous V8 train/deploy policy semantics differ')
    if config.get('continuous_v8') and any(config.get(k) for k in ('native_v5','native_v6','native_v7')):
        raise ValueError('continuous V8 cannot use native event-coordinate policy')
    if config.get('continuous_v8'):
        runtime=config.get('deployment',{}).get('runtime',config.get('runtime'))
        if runtime is None or runtime!=contract.get('runtime_contract'):
            raise ValueError('continuous V8 train/deploy clock and query contracts differ')


def validate_training_contract(config, manifest):
    expected = dict(version='continuous_rtc_v8',clock='physical_seconds',
        action_semantics='fixed_interval_endpoint_arm_start_gripper',
        action_interval_seconds=config['action_interval_seconds'],
        history_endpoint_seconds=.72,source='sampled_slow_prediction',
        labels='same_absolute_physical_interval',prefix='actual_committed_values_and_intervals')
    if (config.get('time_contract')!='exposure_anchor_wall_release_v4'
            or manifest.get('time_contract')!=config['time_contract']
            or manifest.get('continuous_rtc_contract')!=expected
            or manifest.get('continuous_gripper_sampling')!='interval_start_previous_event'
            or manifest.get('continuous_flow_sampling')!='raw_frames_v1'
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
