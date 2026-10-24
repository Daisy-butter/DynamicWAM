import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dualrate_text.py"
spec = importlib.util.spec_from_file_location("dualrate_text_worker_test", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeEncoder:
    def tokenizer(self, texts, **kwargs):
        ids = torch.zeros(len(texts), 512, dtype=torch.long)
        masks = torch.zeros_like(ids)
        for i, text in enumerate(texts):
            count = len(text)
            ids[i, :count] = torch.arange(count) + 1
            masks[i, :count] = 1
        return ids, masks

    def __call__(self, texts, device):
        return [torch.full((len(text), 4096), float(len(text))) for text in texts]


class TextTests(unittest.TestCase):
    def test_prompt_id_is_exact_text_and_episode_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            text = "scene: actual episode instruction"
            row = dict(prompt_id=hashlib.sha256(text.encode()).hexdigest(), text=text, episode_uid="task/ep0", task="task")
            data = dict(schema_version=1, text_semantics=worker.TEXT_SEMANTICS,
                        prompts=[row, dict(row, episode_uid="task/ep1")])
            path.write_text(json.dumps(data), encoding="utf-8")
            prompts, episodes = worker.read_prompts(path)
            self.assertEqual(len(prompts), 1)
            self.assertEqual(len(episodes), 2)
            data["prompts"][1]["text"] += " changed"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                worker.read_prompts(path)

    def test_packing_keeps_true_mask_zero_padding_bf16_and_identity(self):
        result = worker.pack_contexts(FakeEncoder(), ["ab", "abc"], "cpu")
        for index, (item, token_sha) in enumerate(result):
            n = index + 2
            self.assertEqual(item["context"].shape, (512, 4096))
            self.assertEqual(item["context"].dtype, torch.bfloat16)
            self.assertEqual(item["mask"].dtype, torch.bool)
            self.assertEqual(item["mask"].sum().item(), n)
            self.assertEqual(item["context"][n:].abs().sum().item(), 0)
            self.assertEqual(len(token_sha), 64)
        self.assertNotEqual(result[0][1], result[1][1])

    def test_cpu_path_prohibits_cuda_initialization(self):
        with worker.cpu_without_cuda():
            worker.pack_contexts(FakeEncoder(), ["abc"], "cpu")
            with self.assertRaisesRegex(RuntimeError, "attempted CUDA"):
                torch.cuda._lazy_init()

    def test_tokenizer_hash_changes_with_any_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("a", encoding="utf-8")
            first = worker.tokenizer_identity(root)
            (root / "tokenizer.json").write_text("b", encoding="utf-8")
            self.assertNotEqual(first, worker.tokenizer_identity(root))

    def test_native_resolved_schema_uses_nested_prefix_but_flattened_wan_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"explicit fixture")
            (root / "google" / "umt5-xxl").mkdir(parents=True)
            profile = {"wan_root": str(root), "inference": {"scene_prefix": "native scene: "}}
            prompts = {"episode_prompt": "native scene: actual episode instruction"}
            with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU preflight only")):
                prefix, checkpoint, tokenizer = worker.native_text_inputs(profile, prompts)
            self.assertEqual(prefix, "native scene: ")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(tokenizer.is_dir())
            # Raw-YAML paths nesting is not the resolved deployment contract.
            with self.assertRaisesRegex(ValueError, "top-level absolute wan_root"):
                worker.native_text_inputs({"paths": {"wan_root": str(root)}, "inference": profile["inference"]}, prompts)
            # Regression: the failed first attempt looked at this wrong level.
            with self.assertRaisesRegex(ValueError, "inference mapping"):
                worker.native_text_inputs({"wan_root": str(root), "scene_prefix": prefix}, prompts)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                worker.native_text_inputs(profile, {"p": prefix + prefix + "task"})
            with self.assertRaisesRegex(ValueError, "does not start"):
                worker.native_text_inputs(profile, {"p": "task without the scene prefix"})

    def test_worker_arguments_bound_before_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "prompts.json"
            manifest.write_text("{}", encoding="utf-8")
            args_file = root / "args.json"
            expected = ["--prompts", str(manifest), "--output", str(root)]
            args_file.write_text(json.dumps(expected), encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(dict(worker=dict(path=str(SCRIPT), worker_sha256=worker.file_sha256(SCRIPT),
                worker_args_path=str(args_file), worker_args_sha256=worker.file_sha256(args_file)))), encoding="utf-8")
            argv = expected + ["--config", "cfg.json", "--device", "cuda:0", "--authorization", str(receipt)]
            config = dict(run_kind="data_preparation", manifest=str(manifest), init_checkpoint_sha256="f" * 64)
            with mock.patch.object(worker, "_gpu_authorization") as gate:
                worker.validate_worker_receipt(receipt, argv, config, root)
                self.assertEqual(gate.call_args.kwargs["expected_mode"], "data_preparation")
                with self.assertRaises(ValueError):
                    worker.validate_worker_receipt(receipt, argv + ["--limit", "0"], config, root)
                with self.assertRaises(ValueError):
                    worker.validate_worker_receipt(receipt, argv + ["--device", "cuda:1"], config, root)

    def test_native_t5_default_is_lazy(self):
        path = SCRIPT.parents[1] / "src/dynamicwam/vendor/wan/modules/t5.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "T5EncoderModel")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        names = [arg.arg for arg in init.args.args][-len(init.args.defaults):]
        default = dict(zip(names, init.args.defaults))["device"]
        self.assertIsInstance(default, ast.Constant)
        self.assertIsNone(default.value)


if __name__ == "__main__":
    unittest.main()
