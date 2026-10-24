import json

import h5py
import numpy as np
import pytest

import test_dualrate_metadata_v3 as metadata_fixtures
from dynamicwam.dualrate.data import sha256_file
from dynamicwam.dualrate.producer import (
    DominoPhysicalDataset, _action_targets_supported, bind_metadata_manifests,
    completion_latency_seconds, derive_timing_manifests, eligible_timing_rows,
    file_stat_identity, manifest_completion_distribution, resample_actions,
    validate_completion_distribution,
)


def distribution(values=(.18, .42, .66, 1.02)):
    return dict(schema='dualrate-completion-discrete-v1',kind='synthetic',
                clock_domain='simulation_seconds',selection='row_hash_uniform_conditioned_available',
                latencies_seconds=list(values),seed=17,provenance='explicit_test_distribution_not_measured')


def test_completion_selection_is_deterministic_available_and_never_clipped():
    spec=validate_completion_distribution(distribution())
    actual=[completion_latency_seconds(spec,'episode',i,.72) for i in range(80)]
    assert set(actual)=={.18,.42,.66}
    assert actual==[completion_latency_seconds(spec,'episode',i,.72) for i in range(80)]
    assert completion_latency_seconds(spec,'episode',0,.36)==.18
    with pytest.raises(ValueError,match='no completed reference'):
        completion_latency_seconds(spec,'episode',0,.12)
    with pytest.raises(ValueError,match='no completed reference'):
        manifest_completion_distribution(dict(query_ages_seconds=[.36],completion_distribution=distribution([.5])))


def test_measured_distribution_requires_trace_hash_and_physical_clock(tmp_path):
    trace=tmp_path/'trace.json';trace.write_text('{"actual_latencies":[0.18,0.42]}')
    spec=distribution([.18,.42]);spec.update(kind='measured_quantiles',quantile_levels=[.5,.95],
        trace_path=str(trace),trace_sha256=sha256_file(trace),provenance='test actual timing trace')
    assert validate_completion_distribution(spec)==spec
    trace.write_text('{}')
    with pytest.raises(ValueError,match='trace changed'):
        validate_completion_distribution(spec)
    with pytest.raises(ValueError,match='physical model clock'):
        validate_completion_distribution({**distribution(),'clock_domain':'wall_seconds'})


def test_legacy_fixed_completion_is_explicit_and_missing_new_descriptor_fails():
    spec=manifest_completion_distribution({'query_ages_seconds':[.36,.72]})
    assert spec['kind']=='synthetic' and spec['latencies_seconds']==[.18]
    assert 'not_measured' in spec['provenance']
    with pytest.raises(ValueError,match='lost completion distribution'):
        manifest_completion_distribution({'timing_derivation':'v1'})


def test_support_only_math_matches_actual_action_resampling_with_holes_and_duplicates():
    times=np.array([0.,.06,.12,.12,.18,.30,.36,.42])
    targets=np.array([-.001,0.,.03,.12,.15,.18,.20,.24,.30,.42,.43])
    _,valid,_=resample_actions(times,np.zeros((len(times),14),np.float32),targets)
    assert np.array_equal(_action_targets_supported(times,targets),valid)


def test_timing_grid_enforces_last_endpoint_and_missing_action_or_rgb_support():
    times=np.arange(73,dtype=np.float64)*.06
    ages=[.36,.72,1.08,1.44]
    rows=eligible_timing_rows(times,.36,ages)
    assert (0,3) in rows and (1,3) not in rows
    assert set(age for _,age in rows)==set(range(4))
    # .54 is an action endpoint but not a native future RGB endpoint.
    hole=times[~np.isclose(times,.54)]
    assert not any(grid==0 for grid,_ in eligible_timing_rows(hole,.36,ages))
    video_hole=times[~np.isclose(times,.36)]
    assert not any(grid==0 for grid,_ in eligible_timing_rows(video_hole,.36,ages))


def timing_parent(tmp_path):
    legacy=tmp_path/'legacy';metadata_fixtures.parent_fixture(legacy)
    episodes=[];stats={}
    for index,split in enumerate(('train','validation')):
        raw=legacy/'raw/data'/f'episode{index}.hdf5'
        with h5py.File(raw,'w') as handle:
            handle.create_dataset('interception/sim_time_seconds',data=np.arange(101)*.06)
        uid=f'fixture/{split}';episodes.append(dict(episode_uid=uid,prompt_id='fixture_prompt',
            raw_path=str(raw),first_time=0.,last_time=6.,split=split))
        stats[uid]={'raw':file_stat_identity(raw)}
    (legacy/'episodes.json').write_text(json.dumps(episodes))
    stat_path=legacy/'source_file_stats.json';stat_path.write_text(json.dumps(dict(
        identity_kind='resolved_path_size_mtime_ns_not_content_hash',episodes=stats)))
    text_path=legacy/'text.json';text=json.loads(text_path.read_text())
    text['episodes']={episode['episode_uid']:'fixture_prompt' for episode in episodes}
    text_path.write_text(json.dumps(text))
    for split in ('train','validation'):
        path=legacy/f'{split}.json';manifest=json.loads(path.read_text())
        manifest.update(source_stride_seconds=.36,query_ages_seconds=[.36,.72],
                        source_file_stats_sha256=sha256_file(stat_path))
        path.write_text(json.dumps(manifest))
    bound=tmp_path/'full01';bind_metadata_manifests(legacy,bound)
    return bound


def test_new_timing_manifest_derivation_preserves_labels_stats_split_and_parent_bytes(tmp_path):
    parent=timing_parent(tmp_path)
    old={str(path):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    result=derive_timing_manifests(parent,tmp_path/'full02',query_ages_seconds=[.36,.72,1.08,1.44],
                                   completion_distribution=distribution())
    assert all(open(path,'rb').read()==payload for path,payload in old.items())
    assert result['audit']['raw_images_encoded']==0 and not result['audit']['simulator_started']
    for episode_index,split in enumerate(('train','validation')):
        path=tmp_path/'full02'/f'{split}.json'
        dataset=DominoPhysicalDataset(path)
        assert set(dataset.rows[:,0])=={episode_index}
        assert set(dataset.rows[:,2])=={0,1,2,3}
        assert dataset.completion_distribution==distribution()
        original=json.loads((parent/f'{split}.json').read_text())
        for key in ('statistics','action_statistics','episodes_file_sha256','text_index_sha256','prompts_file_sha256'):
            assert dataset.manifest[key]==original[key]
        assert dataset.manifest['parent_manifest_sha256']==sha256_file(parent/f'{split}.json')
    with pytest.raises(ValueError,match='already exists'):
        derive_timing_manifests(parent,tmp_path/'full02',query_ages_seconds=[.36],completion_distribution=distribution())


def test_loader_verifies_measured_trace_even_when_row_hashing_is_disabled(tmp_path):
    parent=timing_parent(tmp_path)
    trace=tmp_path/'latency.json';trace.write_text('[0.18,0.42]')
    spec=distribution([.18,.42]);spec.update(kind='measured_quantiles',quantile_levels=[.5,.95],
        trace_path=str(trace),trace_sha256=sha256_file(trace))
    derive_timing_manifests(parent,tmp_path/'full02',query_ages_seconds=[.36,.72],completion_distribution=spec)
    trace.write_text('[0.17,0.41]')
    with pytest.raises(ValueError,match='trace changed'):
        DominoPhysicalDataset(tmp_path/'full02/train.json',verify_hash=False)


def test_invalid_derivation_does_not_create_output(tmp_path):
    parent=timing_parent(tmp_path)
    with pytest.raises(ValueError,match='no completed reference'):
        derive_timing_manifests(parent,tmp_path/'impossible',query_ages_seconds=[.36],
                                completion_distribution=distribution([.5]))
    assert not (tmp_path/'impossible').exists()
