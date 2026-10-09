#!/usr/bin/env python3
"""Storage relocation is compatible only with unchanged, rehashed source bytes."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from make_world_context_rt_manifest import object_hash,write_json
import run_world_context_rt_pipeline as core
import run_predictive_feedback_pipeline as predictive
from run_world_context_rt_pipeline import external_source_identity,compare_external_source_identities,IdentityViolation

tempfile.tempdir=str(Path.cwd())


class ExternalSourceRelocationTests(unittest.TestCase):
    def identity(self,root='/old/DOMINO'):
        files={'envs/task.py':hashlib.sha256(b'original source').hexdigest()}
        return dict(root=root,files=files,sha256=object_hash(files))

    def test_same_identity_and_moved_root_preserve_both_manifests(self):
        old=self.identity();before=copy.deepcopy(old)
        same=compare_external_source_identities(old,copy.deepcopy(old))
        self.assertFalse(same['relocated'])
        new=dict(old,root='/SSD_DISK/users/wuruihan/DynamicWAM/external/DOMINO')
        result=compare_external_source_identities(old,new)
        self.assertTrue(result['relocated']);self.assertEqual(result['old_root'],old['root'])
        self.assertEqual(result['new_root'],new['root']);self.assertEqual(old,before)
        self.assertEqual(result['content_sha256'],old['sha256'])

    def test_changed_file_even_with_correct_recomputed_aggregate_is_rejected(self):
        old=self.identity();new=copy.deepcopy(old);new['root']='/new/DOMINO'
        new['files']['envs/task.py']=hashlib.sha256(b'changed source').hexdigest()
        new['sha256']=object_hash(new['files'])
        with self.assertRaises(IdentityViolation):compare_external_source_identities(old,new)

    def test_added_or_removed_file_is_rejected(self):
        old=self.identity()
        for files in ({},dict(old['files'],**{'script/new.py':'b'*64})):
            new=dict(old,root='/new',files=files,sha256=object_hash(files))
            with self.assertRaises(IdentityViolation):compare_external_source_identities(old,new)

    def test_aggregate_hash_is_verified_on_both_sides(self):
        old=self.identity();wrong=dict(old,sha256='f'*64)
        for left,right in ((old,wrong),(wrong,old),(wrong,wrong)):
            with self.assertRaises(IdentityViolation):compare_external_source_identities(left,right)

    def test_format_version_and_other_metadata_cannot_be_relaxed(self):
        old=dict(self.identity(),format='registered_external',version=1)
        self.assertTrue(compare_external_source_identities(old,dict(old,root='/new'))['relocated'])
        for new in (dict(old,format='other'),dict(old,version=2),self.identity(),dict(old,extra='changed')):
            with self.assertRaises(IdentityViolation):compare_external_source_identities(old,new)

    def test_malformed_file_identity_is_rejected(self):
        old=self.identity()
        for files in ({'../outside.py':'a'*64},{'/absolute.py':'a'*64},{'envs/task.py':'not-a-digest'}):
            bad=dict(old,files=files,sha256=object_hash(files))
            with self.assertRaises(IdentityViolation):compare_external_source_identities(bad,bad)

    def test_real_files_are_rehashed_before_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new'
            for root in (old,new):
                (root/'envs').mkdir(parents=True);(root/'envs/task.py').write_bytes(b'unchanged\n')
            # Windows restricted hosts can deny realpath despite readable files;
            # only path canonicalization is isolated, file hashes remain real.
            with patch.object(Path,'resolve',lambda path,*args,**kwargs:path.absolute()):
                original=external_source_identity(old);moved=external_source_identity(new)
                self.assertTrue(compare_external_source_identities(original,moved)['relocated'])
                (new/'envs/task.py').write_bytes(b'tampered\n')
                with self.assertRaises(IdentityViolation):
                    compare_external_source_identities(original,external_source_identity(new))

    def test_new_runtime_records_relocation_without_rewriting_registered_manifest(self):
        with tempfile.TemporaryDirectory() as d,patch.object(Path,'resolve',lambda path,*args,**kwargs:path.absolute()):
            root=Path(d);run=root/'run';run.mkdir();old=root/'old';new=root/'new'
            for path in (old,new):
                (path/'envs').mkdir(parents=True);(path/'envs/task.py').write_bytes(b'source\n')
            expected=external_source_identity(old);actual=external_source_identity(new)
            base=root/'base.pt';base.write_bytes(b'identified base')
            write_json(run/'run_config.json',dict(resources={'devices':[dict(index='0',uuid='gpu')]},
                simulation_concurrency=1,base_checkpoint=str(base),inputs={}))
            write_json(run/'source_manifest.json',dict(sha256=object_hash({}),files={}))
            write_json(run/'controller_protocol.json',{'version':2})
            write_json(run/'scene_manifest.json',{'episodes':[]})
            manifest=run/'external_source_manifest.json';write_json(manifest,expected);before=manifest.read_bytes()
            with patch.object(core,'external_source_identity',return_value=actual):core.Pipeline(run)
            receipt=json.loads((run/'external_source_relocation.json').read_text())
            self.assertEqual(receipt['old_root'],expected['root']);self.assertEqual(receipt['new_root'],actual['root'])
            self.assertEqual(before,manifest.read_bytes())

    def test_new_freeze_records_accepted_scene_path_relocation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(Path,'resolve',lambda path,*args,**kwargs:path.absolute()):
            root=Path(d);run=root/'new_run';run.mkdir();old=root/'historical_run';old.mkdir()
            original=self.identity();current=dict(original,root='/new/DOMINO')
            historical=old/'external_source_manifest.json';write_json(historical,original);before=historical.read_bytes()
            write_json(old/'paired_scene_manifest.json',{'scenes':{}})
            write_json(run/'external_source_manifest.json',current)
            write_json(run/'run_config.json',{'resources':{}})
            write_json(run/'scene_manifest.json',dict(episodes=[dict(task=f't{t}',seed=s,scene_id=f'{t}/{s}',role='formal')
                for t in range(35) for s in range(20)]))
            budget=core.resource_budget
            try:
                with patch.object(core,'freeze',return_value=run):
                    predictive.freeze(SimpleNamespace(accepted_run=old))
            finally:core.resource_budget=budget
            config=json.loads((run/'run_config.json').read_text());comparison=config['accepted_external_source_comparison']
            self.assertEqual(comparison['old_root'],'/old/DOMINO');self.assertEqual(comparison['new_root'],'/new/DOMINO')
            self.assertTrue(comparison['relocated']);self.assertEqual(before,historical.read_bytes())
            self.assertEqual(json.loads((run/'status.json').read_text())['status'],'prepared')


if __name__=='__main__':unittest.main(verbosity=2)
