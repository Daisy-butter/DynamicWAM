#!/usr/bin/env python3
"""Bind a new policy profile to unchanged frozen DOMINO scenes without editing them.

CPU only. Historical full profile bytes must match the original prepare hash;
actual environment/task/camera/robot code and configurations must match its
recorded source manifest. Policy paths, checkpoints and model settings may vary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

KIND = 'frozen_domino_scene_compatibility_v1'
SCENE_KEYS = ('task_config', 'instruction_type', 'tasks', 'dynamic_coefficient')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)


def scene_profile(path):
    import yaml
    benchmark=yaml.safe_load(Path(path).read_text())['benchmark']
    require(all(key in benchmark for key in SCENE_KEYS),'scene profile is incomplete')
    return {key:benchmark[key] for key in SCENE_KEYS}


def domino_root(repo,config):
    import yaml
    value=yaml.safe_load(Path(config).read_text())['benchmark']['domino_root']
    return (Path(repo)/value).resolve()


def historical_domino_files(source,scene):
    """Normalize old manifest namespaces to relative paths in the new DOMINO."""
    hashes={}
    for name,value in source['source_sha256'].items():
        if name.startswith('selected_domino/'):
            relative=name[len('selected_domino/'):]
        elif name.startswith('external/DOMINO/envs/') or name=='external/DOMINO/script/eval_metrics.py':
            relative=name[len('external/DOMINO/'):]
        else:continue
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'unsafe historical scene path')
        require(relative not in hashes or hashes[relative]==value,'conflicting historical DOMINO hashes')
        hashes[relative]=value
    required={f'envs/{task}.py' for task in scene['tasks']}
    required.update({'envs/_base_task.py','envs/robot/robot.py','envs/camera/camera.py',
        'task_config/_camera_config.yml','task_config/_embodiment_config.yml',
        'task_config/_eval_step_limit.yml',f"task_config/{scene['task_config']}.yml"})
    require(required<=set(hashes),'historical source manifest lacks required scene/camera/robot provenance')
    return hashes


def verify_domino_files(root,hashes):
    for relative,value in hashes.items():
        path=Path(root)/relative
        require(path.is_file() and sha(path)==value,'DOMINO scene/runtime differs from prepared evidence: '+relative)


def build_receipt(*,repo,config,source_config,frozen_scenes,expected_frozen_sha256):
    repo,config,source_config,frozen_scenes=map(lambda p:Path(p).resolve(),(repo,config,source_config,frozen_scenes))
    suite=read(frozen_scenes/'suite_started.json');frozen=read(frozen_scenes/'frozen_manifest.json')
    require(sha(frozen_scenes/'frozen_manifest.json')==expected_frozen_sha256,'wrong frozen 700-scene manifest')
    require(suite.get('prepare_only') is True and frozen.get('complete') is True
            and len(frozen['episodes'])==700 and len(suite['tasks'])==35 and suite['episodes_per_task']==20,
            'compatibility requires the complete original 35x20 preparation')
    require(sha(source_config)==suite['config_sha256']==frozen['config_sha256'],
            'historical source profile bytes do not match prepared config identity')
    original,current=scene_profile(source_config),scene_profile(config)
    require(original==current,'scene-generation profile changed')
    require(set(original['tasks'])==set(suite['tasks']),'scene profile task list differs from prepared suite')
    hashes=historical_domino_files(read(frozen_scenes/'source_manifest.json'),original)
    verify_domino_files(domino_root(repo,config),hashes)
    return dict(schema_version=1,kind=KIND,passed=True,
        frozen_manifest_sha256=expected_frozen_sha256,
        prepared_suite_sha256=sha(frozen_scenes/'suite_started.json'),
        prepared_source_manifest_sha256=sha(frozen_scenes/'source_manifest.json'),
        source_config=str(source_config),source_config_sha256=sha(source_config),
        evaluation_config_sha256=sha(config),scene_profile=original,domino_files=hashes,
        policy_fields_may_differ=True,original_frozen_artifacts_modified=False)


def validate_receipt(receipt_path,*,repo,config,frozen_scenes):
    receipt=read(receipt_path)
    require(receipt.get('schema_version')==1 and receipt.get('kind')==KIND and receipt.get('passed') is True,
            'scene compatibility receipt is invalid')
    expected=build_receipt(repo=repo,config=config,source_config=receipt['source_config'],
        frozen_scenes=frozen_scenes,expected_frozen_sha256=receipt['frozen_manifest_sha256'])
    require(receipt==expected,'scene compatibility receipt no longer matches actual inputs')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','config','source-config','frozen-scenes','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-frozen-sha256',required=True)
    a=p.parse_args()
    receipt=build_receipt(repo=a.repo,config=a.config,source_config=a.source_config,
        frozen_scenes=a.frozen_scenes,expected_frozen_sha256=a.expected_frozen_sha256)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(passed=True,receipt=str(a.output.resolve()),sha256=sha(a.output),
                         verified_domino_files=len(receipt['domino_files']))))
    return 0


if __name__=='__main__':raise SystemExit(main())
