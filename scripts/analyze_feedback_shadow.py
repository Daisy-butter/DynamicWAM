#!/usr/bin/env python3
"""CPU-only, read-only analysis of bounded native-plan shadow diagnostics.

No simulator, teacher inference, training, or GPU is constructed. Missing
episodes remain missing; teacher-only labels never receive invented experts.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
LEGACY_AGES = (4, 8, 12)
OLD_KEYS = ('rgb', 'rgb_valid_mask', 'state', 'source_state', 'plan', 'age_actions', 'text', 'text_mask')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*p
    lo = int(position)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(position-lo)


def distribution(values):
    values = list(values)
    if any(not math.isfinite(v) for v in values):
        raise ValueError('nonfinite value in audited distribution')
    return dict(count=len(values), mean=sum(values)/len(values) if values else None,
                **{f'p{int(p*100):02d}': quantile(values, p) for p in (0., .5, .9, .95, .99, 1.)})


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    path.relative_to(root)
    return path


def trusted_shadow_load(path, expected_sha):
    """Frozen producer may contain np.str_ instructions; permit only its dtype.

    Content must first match this run's committed index. Keep weights_only;
    never permit arbitrary pickle globals or relax model checkpoint loading.
    """
    import numpy as np
    import torch
    if sha256(path) != expected_sha:
        raise ValueError('sample SHA differs from committed index')
    scalar = np.core.multiarray.scalar
    allowed = [np.dtype, type(np.dtype('U1')), type(np.dtype('S1')),
               (scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar')]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def aggregate_status(root, seeds):
    """Validate completed worker reports; absence is partial, corruption invalid."""
    path = Path(root)/'aggregate.json'
    if not path.exists():
        return dict(status='partial', reason='aggregate not present', completed_seeds=[], success_count=None)
    result = json.loads(path.read_text())
    episodes = result.get('episodes', [])
    if result.get('requested_episodes') != len(seeds):
        raise ValueError('aggregate requested episode count differs from manifest')
    actual = [row.get('seed') for row in episodes]
    if (len(actual) != len(set(actual)) or any(seed not in seeds for seed in actual)
            or result.get('reported_episodes') != len(actual) or result.get('seed_sequence') != actual):
        raise ValueError('aggregate includes missing, replaced, duplicate or inconsistent seed identity')
    slots = result.get('slots', [])
    complete = (actual == seeds and result.get('seeds_reported') is True
        and result.get('episode_seed_manifest_matches') is True
        and slots and all(slot.get('status') == 'complete' and slot.get('return_code') == 0 for slot in slots))
    evidence = []
    for slot in slots:
        if slot.get('status') != 'complete':
            continue
        report_path = within(slot['report'], root)
        raw = json.loads(report_path.read_text())
        raw_seeds = [next((row[k] for k in ('seed', 'episode_seed', 'env_seed') if k in row), None) for row in raw]
        compact = slot.get('episodes', [])
        if (not isinstance(raw, list) or len(raw) != slot['requested_episodes']
                or raw_seeds != [row['seed'] for row in compact]
                or any(type(row.get('success')) is not bool for row in raw)
                or [row['success'] for row in raw] != [row.get('success') for row in compact]):
            raise ValueError('worker raw episode report disagrees with aggregate')
        evidence.append(dict(path=str(report_path), sha256=sha256(report_path), seeds=raw_seeds))
    if complete and (not evidence or any(type(row.get('success')) is not bool for row in episodes)):
        raise ValueError('completed aggregate lacks complete boolean outcome evidence')
    return dict(status='complete' if complete else 'partial', completed_seeds=actual,
        success_count=sum(row['success'] for row in episodes) if complete else None,
        path=str(path), sha256=sha256(path), reports=evidence,
        missing_seeds=[seed for seed in seeds if seed not in actual],
        note='success describes the zero-plan diagnostic behavior only; missing preparation is never policy failure')


def validate_record(row, index, *, task, seeds, base_sha, schedule):
    import torch
    ep = index['episode']
    if (type(ep) is not int or not 0 <= ep < len(seeds)
            or row.get('format') != 'dynamicwam_on_policy_shadow' or row.get('version') != 1
            or row.get('task') != task or row.get('seed') != seeds[ep]
            or index.get('seed') != seeds[ep] or row.get('episode_index') != ep
            or row.get('command_index') != index.get('command_index') or row.get('age') != index.get('age')
            or row.get('base_checkpoint_sha256') != base_sha or row.get('schedule') != schedule):
        raise ValueError('shadow record identity differs from index/manifest/checkpoint')
    age, command = row['age'], row['command_index']
    if type(age) is not int or age not in range(2, 16, 2) or command < age or command % 16 != age:
        raise ValueError('invalid zero-plan H2 command/age correspondence')
    for name in ('sim_time', 'source_sim_time', 'history_dt'):
        if not math.isfinite(float(row[name])):
            raise ValueError('nonfinite shadow clock')
    if row['sim_time'] < row['source_sim_time'] or row['history_dt'] < 0 or (row['history_valid'] and row['history_dt'] <= 0):
        raise ValueError('noncausal shadow clock/history')
    for name, shape in dict(plan=(1,16,14), teacher=(1,16,14), state=(1,14), source_state=(1,14)).items():
        if tuple(row[name].shape) != shape or not bool(torch.isfinite(row[name]).all()):
            raise ValueError(f'invalid/nonfinite shadow {name}')
    if (row['text'].ndim != 3 or row['text'].shape[0] != 1 or row['text'].shape[2] != 4096
            or row['text_mask'].shape != row['text'].shape[:2] or not bool(row['text_mask'].any())
            or not bool(torch.isfinite(row['text']).all())):
        raise ValueError('invalid shadow language')


def shadow_inputs(row):
    import numpy as np
    import torch
    from dynamicwam.observations.multiview import fast_views_from_composite, letterbox_view, imagenet_normalize
    rgb, mask = fast_views_from_composite(np.asarray(row['fast_frame']))
    if row['history_valid']:
        previous, pmask = letterbox_view(np.asarray(row['previous_head']))
        previous = imagenet_normalize(previous).transpose(2, 0, 1)
    else:
        previous, pmask = rgb[0], mask[0]
    return dict(rgb=torch.from_numpy(rgb), rgb_valid_mask=torch.from_numpy(mask),
        previous_head=torch.from_numpy(np.ascontiguousarray(previous)), previous_head_mask=torch.from_numpy(pmask),
        state=row['state'][0].float(), source_state=row['source_state'][0].float(), plan=row['plan'][0].float(),
        age_actions=torch.tensor(row['age']), history_dt=torch.tensor(row['history_dt'], dtype=torch.float32),
        history_valid=torch.tensor(row['history_valid']),
        plan_age_seconds=torch.tensor(row['sim_time']-row['source_sim_time'], dtype=torch.float32),
        text=row['text'][0], text_mask=row['text_mask'][0])


def load_shadow(run, v1):
    import torch
    import yaml
    from dynamicwam.action_normalization import RoboTwinQposNormalizer
    run = Path(run).resolve()
    launch_path, manifest_path = run/'shadow_launch.json', run/'shadow_seed_manifest.json'
    launch, manifest = json.loads(launch_path.read_text()), json.loads(manifest_path.read_text())
    if launch['seed_manifest_sha256'] != sha256(manifest_path):
        raise ValueError('launch seed manifest changed')
    source_manifest = run/'source_manifest.json'
    if launch['source_manifest_sha256'] != sha256(source_manifest):
        raise ValueError('launch source identity changed')
    source = Path(launch['source']).resolve()
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        if sha256(within(source/name, source)) != digest:
            raise ValueError(f'frozen shadow source changed: {name}')
    tasks = manifest['tasks']
    if set(tasks) != set(launch['tasks']):
        raise ValueError('launch task set differs from manifest')
    normal = v1['action_normalization']
    normalizer = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
    base_sha, schedule = v1['base_checkpoint']['sha256'], v1['runtime_contract']['schedule']
    rows, statuses, errors = [], {}, []
    for task, seeds in tasks.items():
        if launch['tasks'][task]['seeds'] != seeds or seeds != sorted(set(seeds)):
            raise ValueError('launch seeds differ or are unordered')
        directory = run/'shadow'/'samples'/task
        config_path = run/'shadow'/f'{task}.yaml'
        if sha256(config_path) != launch['tasks'][task]['config_sha256']:
            raise ValueError('shadow task configuration changed after launch')
        config = yaml.safe_load(config_path.read_text())
        if (config['task_name'] != task or config['fixed_episode_seeds'] != seeds
                or Path(config['dynamicwam_root']).resolve() != source
                or Path(config['shadow_output_root']).resolve() != directory.resolve()
                or config.get('temporal_feedback_disabled') is not True or config.get('execute_horizon') != 2):
            raise ValueError('shadow config is not the declared frozen zero-plan behavior')
        prefix_limit = int(config['shadow_samples_per_episode'])
        try:
            status = aggregate_status(run/'shadow'/task, seeds)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            status = dict(status='invalid', error=str(exc), completed_seeds=[])
            errors.append(dict(task=task, artifact='aggregate', error=str(exc)))
        statuses[task] = status
        index_path = directory/'index.jsonl'
        if not index_path.exists():
            status.update(samples=0, index_status='missing')
            continue
        index_bytes = index_path.read_bytes()
        raw_lines = index_bytes.splitlines()
        pending_index_tail = bool(index_bytes and not index_bytes.endswith(b'\n'))
        if pending_index_tail:
            raw_lines = raw_lines[:-1]
        seen, seen_names, episode_commands = set(), set(), collections.defaultdict(list)
        accepted = 0
        for ordinal, line in enumerate(raw_lines):
            try:
                item = json.loads(line)
                path = within(directory/item['path'], directory)
                if path.parent != directory.resolve() or item['path'] in seen_names:
                    raise ValueError('duplicate or nonlocal indexed sample')
                seen_names.add(item['path'])
                row = trusted_shadow_load(path, item['sha256'])
                validate_record(row, item, task=task, seeds=seeds, base_sha=base_sha, schedule=schedule)
                key = (item['episode'], item['command_index'])
                if key in seen:
                    raise ValueError('duplicate episode/command sample')
                seen.add(key)
                ep_ordinal = len(episode_commands[item['episode']])
                if ep_ordinal >= prefix_limit or path.name != f"episode{item['episode']:03d}_query{ep_ordinal:04d}.pt":
                    raise ValueError('sample file name/prefix ordinal exceeds collector contract')
                expected = normalizer.denormalize(row['plan'][:,row['age']:row['age']+2])[0].float()
                if not torch.equal(expected, torch.as_tensor(row['executed_actions'])):
                    raise ValueError('saved executed actions differ from unchanged source plan')
                episode_commands[item['episode']].append(item['command_index'])
                rows.append(dict(record_id=item['sha256'], episode=(task, row['seed']), task=task,
                    age=row['age'], inputs=shadow_inputs(row), teacher=row['teacher'][0,:4].float(),
                    provenance=dict(path=str(path), label_seed=row['label_seed'], command_index=row['command_index'])))
                accepted += 1
            except (ValueError, KeyError, OSError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                errors.append(dict(task=task, index_line=ordinal+1, error=str(exc)))
        # Collectors commit file then index. An unindexed tail may be in flight;
        # never load it without its content hash or call it policy failure.
        unindexed = sorted(path.name for path in directory.glob('*.pt') if path.name not in seen_names)
        for ep, commands in episode_commands.items():
            expected = [i for i in range(2, 10000, 2) if i % 16][:len(commands)]
            if commands != expected:
                errors.append(dict(task=task, episode=ep, error='indexed samples are not a deterministic contiguous H2 prefix'))
        status.update(samples=accepted, index_sha256=hashlib.sha256(index_bytes).hexdigest(),
            index_bytes_at_read=len(index_bytes), pending_index_tail=pending_index_tail,
            prefix_samples_per_episode=prefix_limit,index_status='read', unindexed=unindexed,
            sampled_seeds=[seeds[e] for e in sorted(episode_commands)], samples_by_episode={str(e):len(v) for e,v in episode_commands.items()})
        if unindexed or pending_index_tail or (status['status']=='complete' and set(episode_commands) != set(range(len(seeds)))):
            status['sample_coverage_status'] = 'partial'
        else:
            status['sample_coverage_status'] = 'available'
    complete = all(row['status']=='complete' and row.get('sample_coverage_status')=='available' for row in statuses.values())
    return rows, dict(status='invalid' if errors else ('complete' if complete else 'partial'), tasks=statuses, errors=errors,
        launch_sha256=sha256(launch_path), seed_manifest_sha256=sha256(manifest_path),
        source_manifest_sha256=sha256(source_manifest),
        limitation='at most first 64 nonfresh H2 queries per episode; missing/unindexed data is never a policy failure')


def load_offline(path, v1):
    import torch
    path = Path(path)
    sidecar = json.loads(path.with_suffix('.json').read_text())
    if sidecar['cache_sha256'] != sha256(path):
        raise ValueError('offline cache SHA mismatch')
    cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = cache['metadata']
    if (meta.get('format') != 'dynamicwam_temporal_feedback_probe_cache' or meta.get('version') not in (1,2,3)
            or any(sidecar.get(k) != v for k,v in meta.items())
            or meta['base_checkpoint']['sha256'] != v1['base_checkpoint']['sha256']
            or meta['action_normalization'] != v1['action_normalization']
            or meta['dataset_identity'] != v1['dataset_identity']):
        raise ValueError('offline cache/checkpoint identity mismatch')
    episode_sets = {}
    for split in ('train','validation'):
        episode_sets[split] = {(r['record']['split'],r['record']['task_name'],r['record']['episode_name']) for r in cache[split]}
        if any(r['record_id'] != json_hash(r['record']) for r in cache[split]):
            raise ValueError('offline record identity changed')
    if episode_sets['train'] & episode_sets['validation']:
        raise ValueError('offline cache training/validation episode overlap')
    rows = []
    for row in cache['validation']:
        data = dict(row['inputs'])
        data['rgb'], data['previous_head'] = data['rgb'].float(), data['previous_head'].float()
        text = cache['texts'][row['text_id']]
        data.update(text=text['text'], text_mask=text['text_mask'])
        rows.append(dict(record_id=row['record_id'], inputs=data, teacher=row['teacher'].float(),
            episode=(row['record']['split'],row['record']['task_name'],row['record']['episode_name']),
            task=row['record']['task_name'], age=int(data['age_actions']), provenance=dict(partition='validation')))
    valid_dt = [float(r['inputs']['history_dt']) for r in cache['train'] if bool(r['inputs']['history_valid'])]
    return rows, dict(path=str(path.resolve()), sha256=sha256(path), heldout_only=True,
        training_history_dt=distribution(valid_dt), metadata=meta)


def target_and_rule(inputs, teacher, limit):
    import torch
    from dynamicwam.models.temporal_plan_feedback import remaining_window
    base, tail = remaining_window(inputs['plan'], inputs['age_actions'])
    age = inputs['age_actions']
    arm = torch.ones(14, dtype=torch.bool); arm[[6,13]] = False
    active = tail[:,:,None] & (age[:,None,None]>0) & arm[None,None]
    delta = teacher-base
    preceding = inputs['plan'].gather(1, (age-1).clamp_min(0)[:,None,None].expand(-1,1,14))[:,0]
    error = inputs['state']-preceding
    rule = error[:,None].expand(-1,4,-1).clamp(-limit,limit)*active
    return delta.clamp(-limit,limit), active, delta, error, rule


def collate(rows, donor_rows=None):
    import torch
    keys = set(rows[0]['inputs'])-{'text','text_mask'}
    values = {k:torch.stack([r['inputs'][k] for r in rows]) for k in keys}
    if donor_rows is not None:
        for k in ('previous_head','previous_head_mask'):
            values[k] = torch.stack([r['inputs'][k] for r in donor_rows])
    length = max(r['inputs']['text'].shape[0] for r in rows)
    values['text'] = torch.zeros(len(rows),length,4096,dtype=rows[0]['inputs']['text'].dtype)
    values['text_mask'] = torch.zeros(len(rows),length,dtype=torch.bool)
    for i,r in enumerate(rows):
        n = r['inputs']['text'].shape[0]
        values['text'][i,:n] = r['inputs']['text'];values['text_mask'][i,:n] = r['inputs']['text_mask']
    return values


def history_donors(rows, seed):
    """Same-task cross-episode history; hold measured recipient dt fixed."""
    rng, result = random.Random(seed), {}
    for i,row in enumerate(rows):
        if not bool(row['inputs']['history_valid']):
            continue
        candidates = [j for j,other in enumerate(rows) if other['task']==row['task']
            and other['episode']!=row['episode'] and bool(other['inputs']['history_valid'])]
        if candidates:
            result[i] = rng.choice(candidates)
    return result


def load_heads(v1_path, v3_path=None):
    import torch
    from dynamicwam.models.plan_feedback import PlanFeedbackHead
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
    from dynamicwam.models.temporal_feedback_contract import FORMAT, VERSION, runtime_contract
    from dynamicwam.models.flow_schedule import BridgeSchedule
    v1 = torch.load(v1_path,map_location='cpu',weights_only=True)
    if v1.get('format')!='dynamicwam_plan_feedback_checkpoint' or v1.get('version')!=1 or v1.get('global_step',0)<=0:
        raise ValueError('expected a trained v1 feedback export')
    old = PlanFeedbackHead(**v1['spec']);old.load_state_dict(v1['model'],strict=True)
    initial = TemporalPlanFeedbackHead(**v1['spec']);mapping=initial.initialize_from_plan_feedback(v1['model'])
    heads = dict(v1_legacy=old, initial_v3=initial)
    identity = dict(v1=dict(path=str(Path(v1_path).resolve()),sha256=sha256(v1_path)),mapping=mapping)
    if v3_path:
        payload=torch.load(v3_path,map_location='cpu',weights_only=True)
        schedule=BridgeSchedule.from_flow_match(**v1['runtime_contract']['schedule'])
        expected=dict(format=FORMAT,version=VERSION,spec=v1['spec'],action_normalization=v1['action_normalization'],
            dataset_identity=v1['dataset_identity'],runtime_contract=runtime_contract(schedule,initial.residual_limit))
        if any(payload.get(k)!=v for k,v in expected.items()) or payload['base_checkpoint']['sha256']!=v1['base_checkpoint']['sha256']:
            raise ValueError('v3 checkpoint is incompatible with the diagnostic baseline')
        if payload.get('initialization',{}).get('sha256') != identity['v1']['sha256']:
            raise ValueError('v3 checkpoint was not initialized from the supplied v1')
        if payload.get('arm') not in (None,'temporal') or payload.get('global_step',0)<=0:
            raise ValueError('v3 checkpoint is not the trained temporal arm')
        model=TemporalPlanFeedbackHead(**payload['spec']);model.load_state_dict(payload['model'],strict=True)
        heads['new_v3']=model
        identity['v3']=dict(path=str(Path(v3_path).resolve()),sha256=sha256(v3_path),phase=payload.get('phase'),global_step=payload['global_step'])
    for head in heads.values():
        head.float().eval().requires_grad_(False)
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError('nonfinite head checkpoint')
    return v1,heads,identity


def score(rows, heads, *, batch_size, seed, limit):
    import torch
    donors = history_donors(rows,seed)
    result=[]
    arm = torch.ones(14,dtype=torch.bool);arm[[6,13]]=False
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            selected=rows[start:start+batch_size];inputs=collate(selected)
            teacher=torch.stack([r['teacher'] for r in selected])
            target,mask,delta,error,rule=target_and_rule(inputs,teacher,limit)
            predictions=dict(zero=torch.zeros_like(target),state_rule=rule)
            # CPU FP32 is shared by all arms; do not claim deployment BF16
            # kernel equivalence or use this diagnostic to report latency.
            for name,head in heads.items():
                if name=='v1_legacy':
                    indices=[i for i,r in enumerate(selected) if r['age'] in LEGACY_AGES]
                    value=torch.zeros_like(target)
                    if indices:value[indices]=head(**{k:inputs[k][indices] for k in OLD_KEYS})
                    predictions[name]=value
                else:
                    predictions[name]=head(**inputs)
                    chosen=[rows[donors.get(start+i,start+i)] for i in range(len(selected))]
                    predictions[name+'_shuffled']=head(**collate(selected,chosen))
            for i,row in enumerate(selected):
                active=mask[i];count=int(active.sum())
                if not count:continue
                metrics={}
                for name,value in predictions.items():
                    if name=='v1_legacy' and row['age'] not in LEGACY_AGES:continue
                    if name.endswith('_shuffled') and start+i not in donors:continue
                    prediction=value[i].float()
                    if not bool(torch.isfinite(prediction).all()):raise ValueError('nonfinite prediction')
                    metrics[name]=dict(sse=float((prediction-target[i])[active].square().sum()),
                        raw_sse=float((prediction-delta[i])[active].square().sum()),
                        residual_abs_sum=float(prediction[active].abs().sum()))
                result.append(dict(record_id=row['record_id'],episode=list(row['episode']),task=row['task'],age=row['age'],
                    count=count,history_dt=float(inputs['history_dt'][i]),history_valid=bool(inputs['history_valid'][i]),
                    plan_age_seconds=float(inputs['plan_age_seconds'][i]),state_error=error[i,arm].tolist(),
                    teacher_delta=delta[i][active].tolist(),teacher_clip_count=int((delta[i][active].abs()>limit).sum()),
                    metrics=metrics,shuffle_donor=None if start+i not in donors else rows[donors[start+i]]['record_id'],
                    provenance=row['provenance']))
    return result


def episode_ci(rows, left, right, *, seed, repetitions=2000):
    grouped=collections.defaultdict(lambda:[0.,0])
    episode_tasks={}
    for row in rows:
        if left not in row['metrics'] or right not in row['metrics']:continue
        g=grouped[tuple(row['episode'])]
        g[0]+=row['metrics'][right]['sse']-row['metrics'][left]['sse'];g[1]+=row['count']
        episode_tasks[tuple(row['episode'])]=row['task']
    groups=list(grouped.values());count=sum(g[1] for g in groups)
    point=sum(g[0] for g in groups)/count if count else None
    draws=[];rng=random.Random(seed)
    strata=collections.defaultdict(list)
    for episode,values in grouped.items():strata[episode_tasks[episode]].append(values)
    if len(groups)>=2:
        for _ in range(repetitions):
            sample=[g for task in sorted(strata) for g in rng.choices(strata[task],k=len(strata[task]))]
            draws.append(sum(g[0] for g in sample)/sum(g[1] for g in sample))
    return dict(difference=point,ci95=[quantile(draws,.025),quantile(draws,.975)] if draws else None,
        episodes=len(groups),coordinates=count,method='paired task-stratified episode-cluster percentile bootstrap; right minus left',
        episodes_per_task={task:len(values) for task,values in strata.items()},
        warning='only six planned shadow episodes; exploratory interval cannot support task-generalization claims',repetitions=repetitions)


def summarize(rows, *, seed, train_dt_max=None):
    total=sum(r['count'] for r in rows)
    names=sorted({name for r in rows for name in r['metrics']})
    metrics={}
    for name in names:
        common=[r for r in rows if name in r['metrics']];n=sum(r['count'] for r in common)
        z=sum(r['metrics']['zero']['sse'] for r in common)
        s=sum(r['metrics'][name]['sse'] for r in common)
        metrics[name]=dict(rows=len(common),episodes=len({tuple(r['episode']) for r in common}),coordinates=n,
            target_mse=s/n,raw_teacher_mse=sum(r['metrics'][name]['raw_sse'] for r in common)/n,
            mean_abs_residual=sum(r['metrics'][name]['residual_abs_sum'] for r in common)/n,
            zero_mse_on_identical_support=z/n,relative_gain_vs_zero=1-s/z if z else None,
            paired_vs_zero=episode_ci(common,'zero',name,seed=seed))
    error=[x for r in rows for x in r['state_error']]
    dt=[r['history_dt'] for r in rows if r['history_valid']]
    comparisons={}
    for left,right in [('initial_v3','new_v3'),('initial_v3_shuffled','initial_v3'),('new_v3_shuffled','new_v3'),('v1_legacy','new_v3')]:
        if left in names and right in names:comparisons[f'{right}_minus_{left}']=episode_ci(rows,left,right,seed=seed+1)
    return dict(rows=len(rows),episodes=len({tuple(r['episode']) for r in rows}),coordinates=total,metrics=metrics,
        comparisons=comparisons,state_error=distribution(error),absolute_state_error=distribution(abs(x) for x in error),
        state_error_fraction_above={str(v):sum(abs(x)>v for x in error)/len(error) if error else None for v in (.01,.05,.1)},
        absolute_teacher_delta=distribution(abs(x) for r in rows for x in r['teacher_delta']),
        teacher_clip_fraction=sum(r['teacher_clip_count'] for r in rows)/total if total else None,
        history_dt=distribution(dt),physical_plan_age=distribution(r['plan_age_seconds'] for r in rows),
        history_invalid=sum(not r['history_valid'] for r in rows),training_history_dt_max=train_dt_max,
        valid_history_fraction_above_training_max=sum(v>train_dt_max for v in dt)/len(dt) if dt and train_dt_max is not None else None)


def stratified(rows, *, seed, train_dt_max=None):
    groups={'all':rows,'legacy_age_support':[r for r in rows if r['age'] in LEGACY_AGES]}
    groups.update({f'age_{age}':[r for r in rows if r['age']==age] for age in sorted({r['age'] for r in rows})})
    groups.update({f'task_{task}':[r for r in rows if r['task']==task] for task in sorted({r['task'] for r in rows})})
    for low,high in zip((0.,.18,.5,1.,2.16,4.),(.18,.5,1.,2.16,4.,float('inf'))):
        groups[f'dt_[{low},{high})']=[r for r in rows if r['history_valid'] and low<=r['history_dt']<high]
    return {name:summarize(group,seed=seed,train_dt_max=train_dt_max) for name,group in groups.items()}


def common_cells(online, offline):
    cells={(r['task'],r['age']) for r in online}&{(r['task'],r['age']) for r in offline}
    names=sorted({n for r in online+offline for n in r['metrics']})
    result={}
    for name in names:
        values=[]
        for task,age in sorted(cells):
            sides=[[r for r in rows if (r['task'],r['age'])==(task,age) and name in r['metrics']] for rows in (online,offline)]
            if all(sides):
                values.append(dict(task=task,age=age,online_rows=len(sides[0]),offline_rows=len(sides[1]),
                    online_mse=sum(r['metrics'][name]['sse'] for r in sides[0])/sum(r['count'] for r in sides[0]),
                    offline_mse=sum(r['metrics'][name]['sse'] for r in sides[1])/sum(r['count'] for r in sides[1])))
        result[name]=dict(cells=values,equal_cell_online_mse=sum(v['online_mse'] for v in values)/len(values) if values else None,
            equal_cell_offline_mse=sum(v['offline_mse'] for v in values)/len(values) if values else None)
    return dict(arms=result,weighting='equal nonempty common task-by-age cells',
        interpretation='descriptive comparison of different states, seeds and phase mixtures; not paired or causal')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--v1-checkpoint',type=Path,required=True)
    parser.add_argument('--v3-checkpoint',type=Path)
    parser.add_argument('--offline-cache',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261006)
    args=parser.parse_args()
    if args.batch_size<1 or args.cpu_threads<1:parser.error('positive CPU batch/thread counts required')
    if args.output.exists():raise FileExistsError('use a fresh analysis output; never overwrite prior evidence')
    import torch
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_initialized():raise RuntimeError('this analysis must remain CPU-only')
    v1,heads,identities=load_heads(args.v1_checkpoint,args.v3_checkpoint)
    rows,integrity=load_shadow(args.run_root,v1)
    offline,offline_identity=(load_offline(args.offline_cache,v1) if args.offline_cache else ([],None))
    max_dt=offline_identity['training_history_dt']['p100'] if offline_identity else None
    # Evaluate only shared tasks/ages offline; preserve all readable shadow rows.
    support={(r['task'],r['age']) for r in rows}
    offline=[r for r in offline if (r['task'],r['age']) in support]
    shadow_scores=score(rows,heads,batch_size=args.batch_size,seed=args.seed,limit=v1['spec']['residual_limit'])
    offline_scores=score(offline,heads,batch_size=args.batch_size,seed=args.seed,limit=v1['spec']['residual_limit'])
    report=dict(format='dynamicwam_feedback_shadow_analysis',version=1,status=integrity['status'],integrity=integrity,
        model_identity=identities,offline_identity=offline_identity,analysis_source_sha256=sha256(__file__),
        target='clip(current_teacher[:4] - immutable_source_plan[age:age+4], +/- residual_limit); teacher only, no expert blend or L2',
        state_error='normalized q_current - immutable_plan[age-1]; shared official mean/std; grippers excluded',
        mask='age>0 AND age+slot<16 AND arm channels excluding 6,13',
        precision='CPU FP32 shared across heads; does not establish deployment BF16 kernel parity',
        shadow=stratified(shadow_scores,seed=args.seed,train_dt_max=max_dt),
        offline=stratified(offline_scores,seed=args.seed,train_dt_max=max_dt),
        common_task_age_comparison=common_cells(shadow_scores,offline_scores),
        raw_rows=dict(shadow=shadow_scores,offline=offline_scores),cuda_initialized=torch.cuda.is_initialized(),
        limitations=['development diagnostic, never an automatic formal-training gate',
            'one current-teacher label per state; no conditional teacher variance estimate',
            'shadow behavior uses training seeds and unchanged source plan; heads are not executed',
            'prefix sampling and at most six episodes limit phase coverage and bootstrap reliability',
            'same-task cross-episode image replacement holds recipient dt fixed; association test, not causal motion isolation',
            'old v1 scores only original ages4/8/12; initial_v3 other ages use interpolated embeddings',
            'missing preparation/report/data is incomplete evidence, never counted as policy failure'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],output=str(args.output),shadow_rows=len(shadow_scores),offline_rows=len(offline_scores))))
    return 2 if integrity['status']=='invalid' else 0


if __name__=='__main__':
    raise SystemExit(main())
