"""CPU fault injection for scheduling and immutable readiness receipts."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_feedback_launch as launch


def setup(tmp_path,monkeypatch):
    rows=tmp_path/'rows.npy';rows.write_bytes(b'fixed index')
    manifest=tmp_path/'data.json';manifest.write_text(json.dumps(dict(rows_file=str(rows),rows_sha256=launch.file_sha256(rows))))
    profile=tmp_path/'full.yaml';profile.write_text('frozen Full')
    core=tmp_path/'core.json';core.write_text(json.dumps(dict(manifest=str(manifest),native_profile=str(profile))))
    deploy=tmp_path/'deploy.json';deploy.write_text('{}')
    evidence=[]
    for i in (1,2,3):
        p=tmp_path/f'review{i}.json';p.write_text(json.dumps(dict(iteration=i,passed=True)))
        evidence.append(dict(path=str(p),sha256=launch.file_sha256(p)))
    monkeypatch.setattr(launch.runner,'frozen_source',lambda _:dict(source='frozen'))
    monkeypatch.setattr(launch.runner,'source_contract',lambda x:x)
    receipt=dict(passed=True,review_iterations=3,formal_training_ready=False,
        source_contract=dict(source='frozen'),core_sha256=launch.file_sha256(core),deployment_sha256=launch.file_sha256(deploy),
        data_manifest_sha256=launch.file_sha256(manifest),native_profile_sha256=launch.file_sha256(profile),evidence=evidence,
        reviews=[dict(iteration=i,passed=True) for i in (1,2,3)])
    return receipt,core,deploy,rows,profile


def test_changed_data_config_profile_or_failed_iteration_refuse_launch(tmp_path,monkeypatch):
    r,c,d,rows,profile=setup(tmp_path,monkeypatch)
    launch.verify(r,tmp_path,c,d)
    for p in (c,rows,profile,Path(r['evidence'][0]['path'])):
        original=p.read_bytes();p.write_bytes(original+b'changed')
        with pytest.raises((ValueError,json.JSONDecodeError)):launch.verify(r,tmp_path,c,d)
        p.write_bytes(original)
    r['reviews'][1]['passed']=False
    with pytest.raises(ValueError,match='iteration'):launch.verify(r,tmp_path,c,d)


def test_not_before_wait_is_cpu_only_and_does_not_claim_formal_ready(tmp_path,monkeypatch):
    r,c,d,_,_=setup(tmp_path,monkeypatch);rp=tmp_path/'receipt.json';rp.write_text(json.dumps(r))
    args=SimpleNamespace(core=str(c),deployment=str(d),cpu_acceptance=str(rp),not_before='2026-11-03T02:00:00+08:00')
    deadline=launch.datetime.fromisoformat(args.not_before).timestamp();now=[deadline-61];sleeps=[]
    def sleep(n):sleeps.append(n);now[0]+=n
    launch.wait_and_validate(args,tmp_path,tmp_path,dict(bounded_feedback={'contract':'x'}),dict(bounded_feedback={'contract':'x'}),clock=lambda:now[0],sleep=sleep)
    status=json.loads((tmp_path/'status.json').read_text())
    assert sleeps==[30.,30.,1.] and not status['formal_training_started'] and not status['gpu_context_created']
    assert status['state']=='gpu_acceptance_pending'


def test_wrong_timezone_or_train_deploy_contract_refuses(tmp_path,monkeypatch):
    r,c,d,_,_=setup(tmp_path,monkeypatch);rp=tmp_path/'receipt.json';rp.write_text(json.dumps(r))
    a=SimpleNamespace(core=str(c),deployment=str(d),cpu_acceptance=str(rp),not_before='2026-11-03T02:00:00')
    with pytest.raises(ValueError,match='offset'):launch.wait_and_validate(a,tmp_path,tmp_path,{}, {})
    a.not_before='2026-11-03T02:00:00+08:00'
    with pytest.raises(ValueError,match='contract'):launch.wait_and_validate(a,tmp_path,tmp_path,{'bounded_feedback':1},{'bounded_feedback':2})


def test_sampling_contract_changes_checkpoint_lineage(tmp_path):
    import torch
    from dynamicwam.dualrate.train_distributed import phase_contract,load_checkpoint,SCHEMA
    from test_dualrate_distributed_training import _config
    cfg=_config();identity=dict(source_sha256='s',manifest_sha256='d',init_checkpoint_sha256='i',device_type='cpu',torch_version='test')
    old=phase_contract(cfg,identity)
    cfg['bounded_feedback']=dict(action_steps=1,gripper_correction=False)
    current=phase_contract(cfg,identity)
    assert current!=old
    p=tmp_path/'old.pt';torch.save(dict(schema=SCHEMA,optimizer_boundary=True,contract=old),p)
    with pytest.raises(ValueError,match='contract differs'):
        load_checkpoint(p,None,None,None,None,cfg,identity)
    cfg['bounded_feedback']=dict(action_steps=2,gripper_correction=False)
    assert phase_contract(cfg,identity)!=current
