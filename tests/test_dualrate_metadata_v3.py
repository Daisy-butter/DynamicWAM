import json
import pickle
import numpy as np
import pytest

from dynamicwam.dualrate.data import sha256_file
from dynamicwam.dualrate.producer import (
    SCHEMA, CODEC_ID, DT, ACTIVE_ENTITY_ROLES, DominoPhysicalDataset, bind_metadata_manifests,
)


def parent_fixture(root):
    root.mkdir()
    (root/'raw/data').mkdir(parents=True)
    (root/'raw/_traj_data').mkdir()
    raw=root/'raw/data/episode0.hdf5';raw.write_bytes(b'fixture never read as HDF5')
    (root/'raw/_traj_data/episode0.pkl').write_bytes(pickle.dumps(
        {'dynamic_motion_info':{'target_actor_name':'basket_fixture'}}))
    episodes=[dict(episode_uid='fixture/episode',prompt_id='fixture_prompt',raw_path=str(raw))]
    (root/'episodes.json').write_text(json.dumps(episodes))
    (root/'prompts.json').write_text(json.dumps({'prompts':[]}))
    (root/'source_file_stats.json').write_text(json.dumps(dict(
        identity_kind='resolved_path_size_mtime_ns_not_content_hash',episodes={})))
    (root/'text.json').write_text(json.dumps(dict(complete=True,
        identity=dict(prompt_manifest_sha256=sha256_file(root/'prompts.json')),
        items={},episodes={'fixture/episode':'fixture_prompt'})))
    for split in ('train','validation'):
        rows=root/f'{split}.npy';np.save(rows,np.array([[0,0,0]],dtype=np.int64))
        value=dict(schema_version=SCHEMA,split=split,action_interval_seconds=DT,codec_id=CODEC_ID,
            native_checkpoint_sha256='a'*64,action_normalization_sha256='b'*64,
            motion_normalization_sha256='c'*64,statistics={'fit_split':'train'},
            rows_file=rows.name,rows_sha256=sha256_file(rows),episodes_file='episodes.json',
            source_file_stats='source_file_stats.json',source_file_stats_sha256=sha256_file(root/'source_file_stats.json'),
            text_index='text.json',action_statistics=dict(mean=[0.]*14,std=[1.]*14),
            future_field_support=[[True]*134]*4+[[False]*134]*4+[[True]*131+[False]*3]*8,
            contact_field_support=[[True]*2]*4+[[False]*2]*12)
        (root/f'{split}.json').write_text(json.dumps(value))


def test_metadata_binding_uses_unchanged_absolute_parent_payloads(tmp_path):
    parent=tmp_path/'v2';parent_fixture(parent)
    before={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    output=tmp_path/'v3'
    result=bind_metadata_manifests(parent,output)
    assert set(result)=={'train','validation'}
    assert {p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}==before
    dataset=DominoPhysicalDataset(output/'train.json',verify_hash=False)
    assert len(dataset)==1
    assert dataset.text_root==parent/'text.json'
    assert not (output/'prompts.json').exists()
    with pytest.raises(ValueError,match='already exists'):
        bind_metadata_manifests(parent,output)


@pytest.mark.parametrize('filename,field',[
    ('episodes.json','episodes_file'),('text.json','text_index'),('prompts.json','prompts_file')])
def test_metadata_tampering_rejected_even_without_row_hash_scan(tmp_path,filename,field):
    parent=tmp_path/'v2';parent_fixture(parent)
    output=tmp_path/'v3';bind_metadata_manifests(parent,output)
    path=parent/filename;path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match=field+' metadata hash changed'):
        DominoPhysicalDataset(output/'train.json',verify_hash=False)


def test_bound_manifest_cannot_drop_episode_hash(tmp_path):
    parent=tmp_path/'v2';parent_fixture(parent)
    output=tmp_path/'v3';bind_metadata_manifests(parent,output)
    path=output/'train.json';value=json.loads(path.read_text());del value['episodes_file_sha256']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='missing metadata hash episodes_file'):
        DominoPhysicalDataset(path,verify_hash=False)


def test_three_supervised_roles_keep_recorded_dynamic_target_identity(tmp_path):
    parent=tmp_path/'v2';parent_fixture(parent)
    old=(parent/'episodes.json').read_bytes()
    output=tmp_path/'v3'
    bind_metadata_manifests(parent,output,active_entity_roles=ACTIVE_ENTITY_ROLES,
                            capture_tracked_actor_identity=True)
    dataset=DominoPhysicalDataset(output/'train.json')
    assert dataset.active_entity_roles==ACTIVE_ENTITY_ROLES
    assert dataset.future_token_indices.tolist()==list(range(4))+list(range(8,16))
    assert len(dataset.manifest['future_field_support'])==12
    assert len(dataset.manifest['contact_field_support'])==12
    assert dataset.episodes[0]['tracked_actor_name']=='basket_fixture'
    assert dataset.episodes[0]['tracked_pose_field']=='interception/target_pose_wxyz'
    assert (parent/'episodes.json').read_bytes()==old
    assert (output/'episodes.json').read_bytes()!=old
    assert dataset.manifest['statistics']=={'fit_split':'train'}
