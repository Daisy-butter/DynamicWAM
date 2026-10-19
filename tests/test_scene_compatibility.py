"""Moving policy weights/paths may not silently alter frozen scene semantics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scene_compatibility import build_receipt,validate_receipt,sha


class SceneCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.scenes=self.root/'scenes';self.scenes.mkdir()
        self.domino=self.root/'DOMINO';self.domino.mkdir()
        self.tasks=[f'task{i}' for i in range(35)]
        benchmark=dict(task_config='demo_clean_dynamic',instruction_type='unseen',tasks=self.tasks,
                       dynamic_coefficient=.1,domino_root='DOMINO')
        self.original=dict(benchmark=benchmark,inference=dict(checkpoint='old',action_interval_ms=160),
                           paths=dict(packed_dataset='/old/data'))
        self.new=copy.deepcopy(self.original);self.new['inference']['checkpoint']='stage3_20k'
        self.new['paths']['packed_dataset']='/new/data'
        self.source=self.root/'source.yaml';self.source.write_text(json.dumps(self.original))
        self.config=self.root/'eval.yaml';self.config.write_text(json.dumps(self.new))
        self.save('suite_started.json',dict(prepare_only=True,config_sha256=sha(self.source),
                  tasks=self.tasks,episodes_per_task=20))
        self.save('frozen_manifest.json',dict(complete=True,config_sha256=sha(self.source),
                  episodes=[dict(task=t,episode_index=i) for t in self.tasks for i in range(20)]))
        self.frozen_sha=sha(self.scenes/'frozen_manifest.json')
        names=[f'envs/{t}.py' for t in self.tasks]+['envs/_base_task.py','envs/robot/robot.py',
            'envs/camera/camera.py','task_config/_camera_config.yml','task_config/_embodiment_config.yml',
            'task_config/_eval_step_limit.yml','task_config/demo_clean_dynamic.yml']
        hashes={}
        for name in names:
            p=self.domino/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original '+name)
            hashes['selected_domino/'+name]=sha(p)
        self.save('source_manifest.json',dict(source_sha256=hashes))

    def save(self,name,data):(self.scenes/name).write_text(json.dumps(data))
    def build(self):
        return build_receipt(repo=self.root,config=self.config,source_config=self.source,
                             frozen_scenes=self.scenes,expected_frozen_sha256=self.frozen_sha)

    def test_policy_only_change_can_reuse_identical_environment(self):
        receipt=self.build();path=self.root/'receipt.json';path.write_text(json.dumps(receipt))
        self.assertEqual(validate_receipt(path,repo=self.root,config=self.config,frozen_scenes=self.scenes),receipt)
        self.assertFalse(receipt['original_frozen_artifacts_modified'])
        self.assertEqual(sha(self.scenes/'frozen_manifest.json'),self.frozen_sha)

    def test_each_scene_setting_change_is_rejected(self):
        for key,value in [('task_config','different'),('instruction_type','seen'),
                          ('dynamic_coefficient',.2),('tasks',self.tasks[:-1])]:
            new=copy.deepcopy(self.new);new['benchmark'][key]=value;self.config.write_text(json.dumps(new))
            with self.assertRaisesRegex(ValueError,'scene-generation'):self.build()

    def test_camera_robot_task_code_or_limit_change_is_rejected(self):
        for name in ['envs/task0.py','envs/_base_task.py','envs/robot/robot.py',
                     'task_config/_camera_config.yml','task_config/_eval_step_limit.yml']:
            p=self.domino/name;old=p.read_text();p.write_text(old+' changed')
            with self.assertRaisesRegex(ValueError,'differs'):self.build()
            p.write_text(old)

    def test_historical_profile_cannot_be_substituted(self):
        self.source.write_text(json.dumps(self.new))
        with self.assertRaisesRegex(ValueError,'historical source profile'):self.build()

    def test_manifest_digest_is_bound(self):
        frozen=json.loads((self.scenes/'frozen_manifest.json').read_text())
        frozen['episodes'][0]['episode_index']=99;self.save('frozen_manifest.json',frozen)
        with self.assertRaisesRegex(ValueError,'wrong frozen'):self.build()

    def test_forged_boolean_cannot_override_real_config_comparison(self):
        receipt=self.build();path=self.root/'receipt.json';path.write_text(json.dumps(receipt))
        changed=copy.deepcopy(self.new);changed['benchmark']['dynamic_coefficient']=.5
        self.config.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):validate_receipt(path,repo=self.root,config=self.config,frozen_scenes=self.scenes)

    def test_missing_original_provenance_fails_closed(self):
        source=json.loads((self.scenes/'source_manifest.json').read_text())
        source['source_sha256'].pop('selected_domino/envs/robot/robot.py')
        self.save('source_manifest.json',source)
        with self.assertRaisesRegex(ValueError,'lacks required'):self.build()


if __name__=='__main__':unittest.main()
