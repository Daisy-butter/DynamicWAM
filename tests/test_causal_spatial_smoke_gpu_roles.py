"""GPU-free checks for explicit renderer/policy roles and measured identities."""
from pathlib import Path
import copy
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("causal_smoke_roles",ROOT/"scripts/smoke_causal_spatial_domino.py")
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)


def fixture(count=2):
    ids=[f"GPU-00000000-0000-0000-0000-{i+1:012x}" for i in range(8)]
    guard=dict(verified=True,devices={f"/dev/nvidia{i}":uuid for i,uuid in enumerate(ids)},
        inventory={uuid:dict(pci=f"0000:{i+16:02x}:00.0",device=f"/dev/nvidia{i}") for i,uuid in enumerate(ids)})
    env=dict(CUDA_VISIBLE_DEVICES=",".join(ids[:count]),DYNAMICWAM_SIM_CUDA="0",
        DYNAMICWAM_INFER_CUDA=str(count-1),DYNAMICWAM_GPU_PCI="00000000:10:00.0")
    return env,guard


def actual(requested):
    renderer=dict(enabled=True,cuda_id=0,can_render=True,pci=requested["renderer"]["pci"])
    policy=requested["policy"]
    startup=dict(gpu_identity=dict(local_index=policy["local_index"],current_device=policy["local_index"],
        uuid=policy["uuid"],pci=policy["pci"],model_device=f"cuda:{policy['local_index']}",
        visible_uuids=list(requested["visible_uuids"])))
    return renderer,startup


class SmokeGpuRoleTests(unittest.TestCase):
    def test_two_gpu_split_requires_actual_child_and_renderer_identity(self):
        env,guard=fixture();requested=smoke.requested_gpu_roles(env,guard)
        self.assertEqual(requested["allocation"],"two_gpu_render_policy_split")
        self.assertEqual(requested["compute_gpu_count"],2)
        self.assertEqual(requested["renderer"]["local_index"],0)
        self.assertEqual(requested["policy"]["local_index"],1)
        renderer,startup=actual(requested)
        receipt=smoke.verify_gpu_roles(requested,renderer,startup)
        self.assertTrue(receipt["verified"])
        self.assertNotEqual(receipt["renderer"]["uuid"],receipt["policy"]["uuid"])
        self.assertEqual(receipt["policy"]["model_device"],"cuda:1")
        self.assertEqual(receipt["renderer"]["identity_source"],"actual_sapien_device_pci_to_guarded_proc_uuid")
        self.assertEqual(receipt["policy"]["identity_source"],"worker_startup_gpu_identity")
        self.assertNotIn("verified",requested)

    def test_single_gpu_compatibility_still_verifies_actual_context(self):
        env,guard=fixture(1);requested=smoke.requested_gpu_roles(env,guard)
        receipt=smoke.verify_gpu_roles(requested,*actual(requested))
        self.assertEqual(receipt["allocation"],"single_gpu_shared")
        self.assertEqual(receipt["compute_gpu_count"],1)
        self.assertEqual(receipt["renderer"]["uuid"],receipt["policy"]["uuid"])
        self.assertEqual(receipt["policy"]["local_index"],0)

    def test_rejects_implicit_ordinal_partial_duplicate_or_extra_visible_devices(self):
        env,guard=fixture();ids=env["CUDA_VISIBLE_DEVICES"].split(",")
        for visible in ("","0,1","GPU-1234",",".join([ids[0],ids[0]]),",".join(ids+[ids[0]])):
            with self.subTest(visible=visible),self.assertRaises(ValueError):
                smoke.requested_gpu_roles(dict(env,CUDA_VISIBLE_DEVICES=visible),guard)

    def test_role_indices_are_explicit_and_local_to_visible_uuid_order(self):
        env,guard=fixture()
        for changes in ({"DYNAMICWAM_SIM_CUDA":"1"},{"DYNAMICWAM_INFER_CUDA":"0"},
                        {"DYNAMICWAM_INFER_CUDA":"2"},{"DYNAMICWAM_INFER_CUDA":""}):
            with self.assertRaises(ValueError):smoke.requested_gpu_roles(dict(env,**changes),guard)
        env["CUDA_VISIBLE_DEVICES"]=",".join(reversed(env["CUDA_VISIBLE_DEVICES"].split(",")))
        with self.assertRaisesRegex(ValueError,"renderer PCI"):
            smoke.requested_gpu_roles(env,guard)

    def test_guard_coverage_and_renderer_pci_must_match(self):
        env,guard=fixture()
        invalid=copy.deepcopy(guard);invalid["verified"]=False
        with self.assertRaises(ValueError):smoke.requested_gpu_roles(env,invalid)
        invalid=copy.deepcopy(guard);del invalid["devices"]["/dev/nvidia1"]
        with self.assertRaises(ValueError):smoke.requested_gpu_roles(env,invalid)
        with self.assertRaises(ValueError):
            smoke.requested_gpu_roles(dict(env,DYNAMICWAM_GPU_PCI="0000:11:00.0"),guard)

    def test_requested_environment_alone_is_not_actual_gpu_evidence(self):
        requested=smoke.requested_gpu_roles(*fixture());renderer,startup=actual(requested)
        with self.assertRaisesRegex(RuntimeError,"identity is missing"):
            smoke.verify_gpu_roles(requested,renderer,{})
        changes=dict(local_index=0,current_device=0,uuid=requested["renderer"]["uuid"],
            pci=requested["renderer"]["pci"],model_device="cuda:0",
            visible_uuids=list(reversed(requested["visible_uuids"])))
        for key,value in changes.items():
            bad=copy.deepcopy(startup);bad["gpu_identity"][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                smoke.verify_gpu_roles(requested,renderer,bad)

    def test_actual_renderer_must_remain_local_zero_with_verified_pci(self):
        requested=smoke.requested_gpu_roles(*fixture());renderer,startup=actual(requested)
        for key,value in dict(enabled=False,cuda_id=1,can_render=False,pci="0000:11:00.0").items():
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                smoke.verify_gpu_roles(requested,dict(renderer,**{key:value}),startup)


if __name__=="__main__":unittest.main()
