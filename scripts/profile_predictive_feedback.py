#!/usr/bin/env python3
"""Loaded WAM/predictive-head compute fixture, not a continuous-control result.

Measures one CNN, policy forecast+decoder, then a second forecast sharing the
encoded images. Both CPU candidate-controller rollouts and the residual's
device-to-host transfer are included in the complete fast-query budget.
No training, evaluation rollout, or modification of the deployed worker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'external/flash_attn_runtime'))

from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.models.predictive_feedback import PredictiveFeedbackHead
from dynamicwam.inference.predictive_feedback_graph import PredictiveFeedbackGraph
from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path
from dynamicwam.runtime.domino_rt.timed_control import LimitedHermiteController, TimedPlanQueue
from dynamicwam.runtime.domino_rt.trajectory_recorder import controller_snapshot
from dynamicwam.runtime.world_context_worker import WorldContextWorker
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256


ENCODED_KEYS = ('current_tokens', 'previous_head_tokens', 'token_valid', 'previous_valid')


def delayed_controller_fixture(initial_state,physical_plan):
    """A source at .4 s and dispatch at .564 s, with 41 causal past ticks.

    This is a synthetic controller trace, not a captured scene. We first execute
    the unmodified plan from zero through the observation boundary, then retain
    only commands already executed by dispatch. No future label is supplied.
    """
    dt,source_tick,dispatch_tick=.004,100,141
    controller=LimitedHermiteController(np.asarray(initial_state).reshape(14),period=.1)
    queue=TimedPlanQueue(.1)
    queue.stage(dict(actions=np.asarray(physical_plan),observation_sim_time=0.,
                     request_id='profile_native_plan',metadata={}),now=0.)
    queue.publish(0)
    def snapshot(stamp):
        return controller_snapshot(controller,source_physics_time=stamp,
            future_knots=[dict(end_slot=slot,q=value[0]) for slot,value in sorted(queue.knots.items())])
    starts=[];ends=[];commands=[];velocities=[];source=None
    for tick in range(dispatch_tick):
        if tick%25==0:
            slot=tick//25;active=queue.pop(slot+1)
            controller.begin(None if active is None else active[0],queue.peek(slot+2))
        if tick==source_tick:source=snapshot(tick*dt)
        q,velocity=controller.step(dt)
        if tick>=source_tick:
            starts.append(tick*dt);ends.append((tick+1)*dt)
            commands.append(q.tolist());velocities.append(velocity.tolist())
    if source is None or len(commands)!=41:raise RuntimeError('invalid delayed-controller fixture')
    source['dispatch_controller']=snapshot(dispatch_tick*dt)
    source['executed_prefix']=dict(start_sim_time=starts,end_sim_time=ends,
        q_command=commands,velocity_command=velocities)
    return source


def runtime_identity():
    result = dict(executable=os.path.abspath(sys.executable), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        prefix=sys.prefix)
    if torch.cuda.is_available():
        result.update(gpu_name=torch.cuda.get_device_name(),
            gpu_capability=list(torch.cuda.get_device_capability()),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return result


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('latency samples must be finite and nonempty')
    return dict(samples_ms=a.tolist(), mean_ms=float(a.mean()),
        **{f'p{q}_ms':float(np.percentile(a, q)) for q in (50, 95, 99)})


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000


def measure(fn, samples):
    for _ in range(3):
        fn()
    return summary([timed(fn)[1] for _ in range(samples)])


def load_head(args, policy, default_path=None):
    path = args.checkpoint or args.init_plan_feedback or default_path
    if not path:
        raise ValueError('provide an identified feedback checkpoint in CLI or policy config')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('base_checkpoint', {}).get('sha256') != policy.base_sha:
        raise ValueError('predictive head and WAM base checkpoint SHA differ')
    normal = payload['action_normalization']
    saved = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
    deployed = policy.base.runtime.action_normalizer
    if saved.epsilon != deployed.epsilon or any(not torch.equal(getattr(saved, key), getattr(deployed, key))
                                             for key in ('_mean', '_std')):
        raise ValueError('predictive and native action normalization differ')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261009)
        head = PredictiveFeedbackHead(**payload['spec'])
    if args.checkpoint or (not args.init_plan_feedback and payload.get('format') == 'dynamicwam_predictive_feedback_checkpoint'):
        if payload.get('format') != 'dynamicwam_predictive_feedback_checkpoint' or payload.get('version') != 1:
            raise ValueError('expected predictive checkpoint version 1')
        if payload.get('architecture_version') != head.ARCHITECTURE_VERSION:
            raise ValueError('predictive architecture version mismatch')
        head.load_state_dict(payload['model'], strict=True)
        initialization = payload.get('initialization')
        trained = dict(predictor=True, policy=bool(payload.get('policy_labels_used', False)),
                       training_scope=payload.get('training_scope'), global_step=payload.get('global_step'))
    else:
        if payload.get('format') != 'dynamicwam_plan_feedback_checkpoint' or payload.get('version') != 1:
            raise ValueError('expected trained v1 plan-feedback initialization')
        if not isinstance(payload.get('global_step'), int) or payload['global_step'] < 1:
            raise ValueError('initialization is not a trained v1 checkpoint')
        initialization = head.initialize_from_plan_feedback(payload['model'])
        trained = dict(predictor=False, policy=False, training_scope='v1_migration_compute_fixture',
                       inherited_v1_global_step=payload['global_step'])
    if not all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()):
        raise ValueError('nonfinite checkpoint tensors')
    head.to(policy.base.runner.device).float().eval().requires_grad_(False)
    return head, dict(path=os.path.abspath(path), sha256=file_sha256(path),
        format=payload['format'], version=payload['version'], trained=trained,
        initialization=initialization, action_normalization=normal)


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('loaded CUDA compute profile requires a CUDA device')
    torch.set_num_threads(1)
    config = json.loads(args.policy_config.read_text())
    default_checkpoint = config.get('predictive_checkpoint') or config.get('feedback_checkpoint')
    config.pop('collection_dir', None)
    config.pop('feedback_checkpoint', None)
    config['mode'] = 'graph_zero'
    started = time.perf_counter()
    policy = WorldContextWorker(config)
    model_setup_ms = (time.perf_counter() - started) * 1000
    head, head_identity = load_head(args, policy, default_checkpoint)
    with np.load(args.packet, allow_pickle=False) as archive:
        raw = {k:np.array(archive[k], copy=True) for k in archive.files}
    if raw['frame'].shape != (360, 320, 3):
        raise ValueError('expected the original unletterboxed 360x320 packet')
    raw.update(instruction='pick up the object', control_period_s=.1, command_index=0,
        sim_time=0., _sampling_seed=20261009, history_dt=.2,
        previous_head_rgb=raw['frame'][:240].copy())
    policy.base.set_instruction(raw['instruction'])
    policy.instruction = raw['instruction']

    def refresh():
        policy.plan = None
        return policy.sample(raw)

    _, cold_plan = timed(refresh)
    graph_refresh = measure(refresh, args.samples)
    graph_zero = measure(refresh, args.samples)
    # The original production runner is the oracle, not another implementation
    # of the same refactored native sampler. Both consume identical noise.
    anchors = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for variation in range(3):
            for seed in (20261009, 20261010, 20261011):
                fixture = dict(raw, state=raw['state'] + variation * .0001,
                               frame=np.roll(raw['frame'], variation, axis=1).copy())
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                expected = policy.base.sample(fixture)
                source, noise, _ = policy._source(fixture, sampling_seed=seed)
                normalized, _ = policy.sampler.sample(source, noise)
                observed = policy.base.runtime.action_normalizer.denormalize(normalized).squeeze(0).float().cpu().numpy()
                error = float(np.max(np.abs(expected-observed)))
                exact = bool(np.array_equal(expected, observed))
                anchors.append(dict(variation=variation, seed=seed, max_abs_error=error, exact=exact))
    eager_refresh = measure(lambda:policy.base.sample(raw), args.samples)
    if not all(row['exact'] for row in anchors):
        raise RuntimeError(f'production HeadFlowRunner exact parity failed: {anchors}')
    refresh()
    packet = dict(raw, sim_time=.4, command_index=4)
    age, device = 4, policy.plan.device
    plan_np = policy.plan.squeeze(0).float().cpu().numpy().copy()
    normalizer = policy.base.runtime.action_normalizer
    mean, scale = normalizer._mean.numpy().copy(), normalizer._std.numpy().copy()
    physical_plan = plan_np * scale + mean
    snapshot = delayed_controller_fixture(raw['state'],physical_plan)
    packet.update(state=np.asarray(snapshot['q'],dtype=np.float32),controller_state=snapshot,
                  dispatch_sim_time=snapshot['dispatch_controller']['source_physics_time'])

    def proposal(candidate):
        return proposal_action_path(controller=snapshot, plan=candidate, age=age,
            observation_time=packet['sim_time'], mean=mean, scale=scale)

    def tensor(value):
        return torch.as_tensor(np.ascontiguousarray(value), device=device).unsqueeze(0)

    def query_inputs():
        inputs = policy._inputs(packet, age)
        path, duration, provenance = proposal(plan_np)
        return dict(inputs, proposal_path=tensor(path), proposal_dt=tensor(duration))

    graph = PredictiveFeedbackGraph(head, enabled=True, ablation=args.ablation)
    eager = PredictiveFeedbackGraph(head, enabled=False, ablation=args.ablation)

    def corrected_candidate(residual):
        candidate = plan_np.copy()
        candidate[age:age + 4] += residual.squeeze(0).float().cpu().numpy()
        return candidate

    def full_fast(engine):
        current = query_inputs()  # RGB preprocessing and the FIRST CPU rollout.
        aux = engine.run_head(current)
        candidate = corrected_candidate(aux['residual'])  # Real D2H dependency.
        path, duration, _ = proposal(candidate)  # SECOND CPU rollout after correction.
        future = engine.run_forecast(aux, history_dt=current['history_dt'],
            history_valid=current['history_valid'], action_path=tensor(path), future_dt=tensor(duration))
        actions = candidate[age:] * scale + mean
        return actions, future

    _, cold_fast = timed(lambda:full_fast(graph))
    inputs = query_inputs()
    first = eager.run_head(inputs)
    path, duration, proposal_provenance = proposal(corrected_candidate(first['residual']))
    revised_path, revised_dt = tensor(path), tensor(duration)

    def core(engine):
        aux = engine.run_head(inputs)
        return aux, engine.run_forecast(aux, history_dt=inputs['history_dt'],
            history_valid=inputs['history_valid'], action_path=revised_path, future_dt=revised_dt)

    # A fresh predictor copies current by construction. Exercise nonzero
    # conditional branches as a separate graph fixture, then restore EXACT
    # checkpoint weights before latency measurement. This is never training.
    parameters = [head.latent_predictor.output.weight, head.latent_predictor.output.bias,
                  head.future_readout.weight]
    saved_parameters = [p.clone() for p in parameters]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20261009)
        for parameter in parameters:
            parameter.normal_(0, .01)
    head_parity = []
    try:
        for key in ('baseline', 'rgb', 'previous_head', 'state', 'proposal_path',
                    'corrected_proposal_path', 'proposal_dt', 'history_valid', 'rgb_valid_mask'):
            changed = {k:None if v is None else v.clone() for k,v in inputs.items()}
            second_path = revised_path.clone()
            if key == 'corrected_proposal_path':
                second_path.add_(.03)
            elif key != 'baseline':
                if changed[key].dtype == torch.bool:
                    changed[key].logical_not_()
                else:
                    changed[key].add_(.03)
            expected, actual = eager.run_head(changed), graph.run_head(changed)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
            kwargs = dict(history_dt=changed['history_dt'], history_valid=changed['history_valid'],
                          action_path=second_path, future_dt=changed['proposal_dt'])
            expected_final = eager.run_forecast(expected, **kwargs)
            actual_final = graph.run_forecast(actual, **kwargs)
            torch.testing.assert_close(actual_final, expected_final, atol=2e-5, rtol=2e-5)
            head_parity.append(dict(changed=key,
                residual_max_error=float((actual['residual']-expected['residual']).abs().max()),
                final_forecast_max_error=float((actual_final-expected_final).abs().max())))
    finally:
        for parameter, saved in zip(parameters, saved_parameters):
            parameter.copy_(saved)
    if not all(torch.equal(parameter, saved) for parameter,saved in zip(parameters,saved_parameters)):
        raise RuntimeError('graph fixture failed to restore checkpoint parameters')
    restored_eager, restored_graph = core(eager), core(graph)
    for name in restored_eager[0]:
        torch.testing.assert_close(restored_graph[0][name], restored_eager[0][name], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(restored_graph[1], restored_eager[1], atol=2e-5, rtol=2e-5)
    head_eager = measure(lambda:core(eager), args.samples)
    head_graph = measure(lambda:core(graph), args.samples)
    fast_eager = measure(lambda:full_fast(eager), args.samples)
    fast = measure(lambda:full_fast(graph), args.samples)
    helper = measure(lambda:proposal(plan_np), args.samples)
    cycle = graph_refresh['mean_ms'] + 5 * fast['mean_ms']
    baseline = eager_refresh['mean_ms']
    gate = dict(passed=bool(cycle <= .85 * baseline and fast['p95_ms'] < 100),
        saving_fraction=1-cycle/baseline, amortized_ms_per_slot=cycle/12,
        eager_ms_per_slot=baseline/12, required_saving_fraction=.15,
        required_fast_p95_ms=100, refresh_slots=12, fast_queries_per_refresh=5,
        includes_camera_render=False, includes_candidate_controller=True,
        candidate_controller_calls_per_fast_query=2, includes_action_device_to_host=True,
        includes_rgb_preprocessing=True, includes_final_corrected_candidate_forecast=True)
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('dynamicwam.') and path and Path(path).is_file():
            loaded[name] = dict(path=os.path.abspath(path), sha256=file_sha256(path))
    return dict(status='complete', passed=gate['passed'], runtime=runtime_identity(),
        scope='loaded_worker_compute_fixture; excludes_camera_and_physics; no_quality_claim',
        proposal_fixture='synthetic native-plan controller trace: observation .4s, dispatch .564s, 41 already-executed ticks',
        proposal_provenance=proposal_provenance, controller_fixture=snapshot,
        observation_fixture=dict(source_physics_time=.4,dispatch_physics_time=.564,
            camera_and_dispatch_lag_s=.164,executed_prefix_ticks=41,age_actions=age,
            source_state_time=.4,dispatch_controller_elapsed_s=snapshot['dispatch_controller']['elapsed'],
            image='synthetic packet; not a scene rendering of the controller trace',
            includes_elapsed_prefix_helper_cost=True,fixture_construction_timed=False),
        deployment_budget_verified=False, worker_compute_budget_verified=gate['passed'],
        gate=gate, model_setup_ms=model_setup_ms, ablation=args.ablation,
        cold=dict(native_plan_ms=cold_plan, predictive_graph_ms=cold_fast),
        cold_start_ms=cold_plan,
        measurements=dict(native_graph_refresh=graph_refresh, graph_zero_refresh=graph_zero,
            official_eager_refresh=eager_refresh, fast_model_eager=head_eager,
            fast_model_graph=head_graph, fast_complete_eager=fast_eager,
            fast_complete_graph=fast, candidate_controller_single_call=helper),
        fast_samples_ms=fast['samples_ms'], graph_samples_ms=graph_refresh['samples_ms'],
        eager_samples_ms=eager_refresh['samples_ms'], graph_zero_samples_ms=graph_zero['samples_ms'],
        native_action_max_abs_error=max(row['max_abs_error'] for row in anchors),
        amortized_ms_per_slot=gate['amortized_ms_per_slot'], eager_ms_per_slot=gate['eager_ms_per_slot'],
        saving_fraction=gate['saving_fraction'], head_checkpoint_sha256=head_identity['sha256'],
        native_zero_anchors=anchors, native_full_plan_exact=True,
        native_parity_anchors=anchors,
        head_graph_parity=head_parity, head_graph=dict(graph.stats),
        native_graph=dict(policy.sampler.stats), head_checkpoint=head_identity,
        base_checkpoint_sha256=policy.base_sha, controller_protocol_sha256=policy.protocol_sha,
        packet_sha256=file_sha256(args.packet), policy_config_sha256=file_sha256(args.policy_config),
        source_hashes={p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted((ROOT/'src').rglob('*.py'))},
        profile_source_sha256=file_sha256(__file__), loaded_dynamicwam_modules=loaded,
        note='Run a trained export again; an initialized head only establishes code/compute feasibility.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--init-plan-feedback', type=Path)
    group.add_argument('--checkpoint', type=Path)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--ablation', choices=('none','nohistory','noaction','current_only'), default='none')
    args = parser.parse_args()
    if args.samples < 30 or args.output.exists():
        parser.error('at least 30 samples and a fresh output path are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        result = dict(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}',
                      runtime=runtime_identity(), profile_source_sha256=file_sha256(__file__))
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        raise
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
