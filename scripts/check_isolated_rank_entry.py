#!/usr/bin/env python3
"""CPU-only tests for pre-import rank/UUID isolation; never imports torch."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).with_name("isolated_rank_entry.py")
spec = importlib.util.spec_from_file_location("isolated_rank_entry", path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class IsolatedRankTest(unittest.TestCase):
    def fixture(self, rank=0, count=8):
        uuids = [f"GPU-{index:08x}-0000-0000-0000-000000000001" for index in range(count)]
        return dict(CUDA_VISIBLE_DEVICES=",".join(uuids), RANK=str(rank), WORLD_SIZE=str(count),
            LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(count), GROUP_RANK="0", GROUP_WORLD_SIZE="1",
            NCCL_P2P_DISABLE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="12345"), uuids

    def test_all_eight_rank_mappings_preserve_global_collective(self):
        assigned = []
        for rank in range(8):
            original, uuids = self.fixture(rank)
            result = entry.isolated_environment(original)
            self.assertEqual(result["RANK"], str(rank))
            self.assertEqual(result["WORLD_SIZE"], "8")
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], uuids[rank])
            self.assertEqual(result["LOCAL_RANK"], "0")
            self.assertEqual(result["LOCAL_WORLD_SIZE"], "1")
            self.assertEqual(result["GROUP_RANK"], str(rank))
            self.assertEqual(result["GROUP_WORLD_SIZE"], "8")
            self.assertEqual(result["WAM_PHYSICAL_LOCAL_RANK"], str(rank))
            self.assertEqual(result["WAM_PHYSICAL_GROUP_RANK"], "0")
            self.assertEqual(result["WAM_PHYSICAL_HOST_COUNT"], "1")
            self.assertEqual(result["MASTER_ADDR"], original["MASTER_ADDR"])
            self.assertEqual(result["MASTER_PORT"], original["MASTER_PORT"])
            self.assertEqual(original["LOCAL_RANK"], str(rank))
            assigned.append(result["CUDA_VISIBLE_DEVICES"])
        self.assertEqual(len(set(assigned)), 8)

    def test_invalid_topologies_fail_before_import(self):
        for change in (
                {"RANK": "8"}, {"RANK": "1"}, {"LOCAL_WORLD_SIZE": "4"},
                {"WORLD_SIZE": "7"}, {"GROUP_RANK": "1"}, {"GROUP_WORLD_SIZE": "2"},
                {"LOCAL_RANK": "-1"}, {"RANK": "bad"}, {"NCCL_P2P_DISABLE": "0"},
                {"CUDA_VISIBLE_DEVICES": "0,1"}, {"WAM_RANK_VISIBILITY": "isolated"}):
            original, _ = self.fixture()
            original.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.isolated_environment(original)
        original, uuids = self.fixture()
        original["CUDA_VISIBLE_DEVICES"] = ",".join([uuids[0]] * 8)
        with self.assertRaises(ValueError):
            entry.isolated_environment(original)
        original, _ = self.fixture()
        original.pop("LOCAL_RANK")
        with self.assertRaises(ValueError):
            entry.isolated_environment(original)

    def test_masked_torchrun_parent_assigns_all_eight_unique_children(self):
        assigned = []
        for rank in range(8):
            original, uuids = self.fixture(rank)
            original["WAM_ASSIGNED_GPU_UUIDS"] = original["CUDA_VISIBLE_DEVICES"]
            original["CUDA_VISIBLE_DEVICES"] = ""
            result = entry.isolated_environment(original)
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], uuids[rank])
            self.assertEqual(result["RANK"], str(rank))
            self.assertEqual(result["WORLD_SIZE"], "8")
            self.assertEqual(result["LOCAL_RANK"], "0")
            self.assertEqual(result["WAM_PHYSICAL_CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(result["WAM_PHYSICAL_ASSIGNED_GPU_UUIDS"], ",".join(uuids))
            self.assertEqual(result["WAM_LAUNCHER_CUDA_MASKED"], "1")
            self.assertEqual(original["CUDA_VISIBLE_DEVICES"], "")
            assigned.append(result["CUDA_VISIBLE_DEVICES"])
        self.assertEqual(len(set(assigned)), 8)
        original, uuids = self.fixture()
        original["WAM_ASSIGNED_GPU_UUIDS"] = ",".join(uuids)
        with self.assertRaisesRegex(ValueError, "empty CUDA visibility"):
            entry.isolated_environment(original)
        original["CUDA_VISIBLE_DEVICES"] = ""
        original.pop("WAM_ASSIGNED_GPU_UUIDS")
        with self.assertRaises(ValueError):
            entry.isolated_environment(original)

    def test_no_torch_import_and_target_sees_mapped_environment(self):
        tree = ast.parse(path.read_text())
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and "torch" in ast.unparse(node)
            for node in ast.walk(tree)))
        original, uuids = self.fixture(rank=7)
        original["WAM_ASSIGNED_GPU_UUIDS"] = original["CUDA_VISIBLE_DEVICES"]
        original["CUDA_VISIBLE_DEVICES"] = ""
        # Existing source file stands in as a target: runpy is mocked, no files
        # or GPU processes are created by this test.
        target = Path(__file__).resolve()
        observed = {}
        def run_target(*args, **kwargs):
            observed.update(env=dict(os.environ), argv=list(entry.sys.argv), target=args[0])
        with patch.dict(os.environ, original, clear=True), patch.object(entry.sys, "argv", [str(path), str(target), "--config", "config.yaml"]), patch.object(entry.runpy, "run_path", side_effect=run_target), patch.object(entry.sys, "path", list(entry.sys.path)):
            entry.main()
        self.assertEqual(observed["env"]["CUDA_VISIBLE_DEVICES"], uuids[7])
        self.assertEqual(observed["env"]["RANK"], "7")
        self.assertEqual(observed["env"]["LOCAL_RANK"], "0")
        self.assertEqual(observed["argv"], [str(target), "--config", "config.yaml"])


if __name__ == "__main__":
    unittest.main()
