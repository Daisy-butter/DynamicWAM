"""Build episode-bound native T5 caches on the verified artifact disk.

CPU is the default and processes at most four unique prompts unless --limit is
explicit. CUDA requires a fresh external data_preparation guard receipt. This
worker never launches a guard, probes NVML, downloads weights, or starts training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import torch

from dynamicwam.dualrate.train import (atomic_checkpoint, canonical_hash,
    file_sha256, source_identity, _gpu_authorization, _validate_storage)


TEXT_SCHEMA = "dualrate-native-t5-text-v1"
TEXT_SEMANTICS = "exact_encoder_input_with_scene_prefix"


def read_prompts(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("text_semantics") != TEXT_SEMANTICS:
        raise ValueError("prompt manifest must declare exact native encoder input including scene prefix")
    rows = value.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt manifest is empty")
    prompts, episodes = {}, {}
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key] for key in ("prompt_id", "text", "episode_uid", "task")):
            raise ValueError("prompt rows require prompt_id, exact text, episode_uid and task")
        prompt_id = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if row["prompt_id"] != prompt_id:
            raise ValueError("prompt_id must hash the exact UTF8 encoder text")
        if row["episode_uid"] in episodes and episodes[row["episode_uid"]] != prompt_id:
            raise ValueError("one episode has conflicting prompt identities")
        episodes[row["episode_uid"]] = prompt_id
        if prompt_id in prompts and prompts[prompt_id] != row["text"]:
            raise ValueError("prompt hash collision")
        prompts[prompt_id] = row["text"]
    return prompts, episodes


def tokenizer_identity(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("tokenizer must be an existing local directory")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("empty tokenizer directory")
    digest = hashlib.sha256()
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("tokenizer symlink escapes the bound directory")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


@contextmanager
def cpu_without_cuda():
    """Turn accidental CUDA initialization into a failure on the CPU path."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation entered with an existing CUDA context")
    original = torch.cuda._lazy_init
    def reject(*_args, **_kwargs):
        raise RuntimeError("CPU text preparation attempted CUDA initialization")
    torch.cuda._lazy_init = reject
    try:
        yield
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    finally:
        torch.cuda._lazy_init = original


def pack_contexts(encoder, texts, device):
    """Use native encoder outputs and its actual tokenizer mask, with zero pad."""
    ids, masks = encoder.tokenizer(texts, return_mask=True, add_special_tokens=True)
    if ids.shape != (len(texts), 512) or masks.shape != ids.shape:
        raise ValueError("native tokenizer must produce a fixed 512-token grid")
    masks = masks.detach().cpu().bool()
    lengths = masks.sum(-1)
    expected = torch.arange(512)[None] < lengths[:, None]
    if (lengths == 0).any() or not torch.equal(masks, expected):
        raise ValueError("native T5 interface requires nonempty right-padded token masks")
    with torch.no_grad():
        outputs = encoder(texts, device)
    if not isinstance(outputs, list) or len(outputs) != len(texts):
        raise TypeError("native encoder must return one unpadded context per text")
    result = []
    for index, context in enumerate(outputs):
        length = int(lengths[index])
        if context.shape != (length, 4096) or not torch.isfinite(context).all():
            raise ValueError("T5 context must be finite [actual token length,4096]")
        packed = torch.zeros(512, 4096, dtype=torch.bfloat16)
        packed[:length] = context.detach().to(device="cpu", dtype=torch.bfloat16)
        ids_bytes = ids[index].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        result.append(({"context": packed, "mask": masks[index]}, hashlib.sha256(ids_bytes).hexdigest()))
    return result


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_worker_receipt(path, argv, config, output):
    """Check script/argv binding before reusable guard ancestry/FD validation."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    worker = receipt.get("worker", {})
    script = Path(__file__).resolve()
    if Path(worker.get("path", "/nonexistent")).resolve() != script or worker.get("worker_sha256") != file_sha256(script):
        raise ValueError("GPU worker script hash/path mismatch")
    args_path = Path(worker.get("worker_args_path", "/nonexistent")).resolve(strict=True)
    if file_sha256(args_path) != worker.get("worker_args_sha256"):
        raise ValueError("GPU worker argv file hash mismatch")
    expected = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
        raise ValueError("worker argv file must be a JSON string array")
    reserved = ("--config", "--device", "--authorization")
    if any(x in reserved or x.startswith(tuple(k + "=" for k in reserved)) for x in expected):
        raise ValueError("worker argv must not override guard-controlled options")
    actual, counts = [], dict.fromkeys(reserved, 0)
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in reserved:
            counts[token] += 1
            if i + 1 >= len(argv):
                raise ValueError("missing guard-controlled option value")
            i += 2
        else:
            if token.startswith(tuple(k + "=" for k in reserved)):
                raise ValueError("guard options require separate arguments")
            actual.append(token)
            i += 1
    if actual != expected or any(n != 1 for n in counts.values()):
        raise ValueError("actual worker argv differs from the authorized list")
    if config.get("run_kind") != "data_preparation":
        raise ValueError("GPU text preparation requires data_preparation config")
    manifest = Path(config["manifest"]).resolve(strict=True)
    identity = {"source_sha256": source_identity(), "config_sha256": canonical_hash(config),
        "manifest_sha256": file_sha256(manifest), "init_checkpoint_sha256": config["init_checkpoint_sha256"],
        "device_type": "cuda", "torch_version": str(torch.__version__)}
    _gpu_authorization(Path(path), identity, output, expected_mode="data_preparation")


def prepare(args, argv):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = _validate_storage(config)
    if output != Path(args.output).resolve():
        raise ValueError("text output must exactly match the receipt-bound config output")
    if args.limit < 0 or args.batch_size < 1:
        raise ValueError("limit must be nonnegative and batch_size positive")
    target = torch.device(args.device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("only explicit CPU/CUDA supported")
    prompt_path = Path(args.prompts).resolve(strict=True)
    prompts, episodes = read_prompts(prompt_path)
    prompt_sha = file_sha256(prompt_path)
    if (Path(config["manifest"]).resolve(strict=True) != prompt_path and
            config.get("prompt_manifest_sha256") != prompt_sha):
        raise ValueError("config/receipt must bind the actual prompt manifest bytes")
    if target.type == "cuda":
        if target.index != 0:
            raise ValueError("guarded worker must use logical cuda:0 on its single UUID-pinned device")
        if not args.authorization:
            raise ValueError("CUDA data preparation requires an external guarded receipt")
        validate_worker_receipt(args.authorization, argv, config, output)
    elif args.authorization:
        raise ValueError("CPU path does not consume a GPU authorization")
    if not Path(config["init_checkpoint"]).is_absolute() or file_sha256(config["init_checkpoint"]) != config["init_checkpoint_sha256"]:
        raise ValueError("bound native checkpoint identity changed")

    def work():
        # Delay native imports until after the GPU guard / CPU no-CUDA boundary.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from dynamicwam.inference.model_loader import _build_t5_encoder, load_deploy_config
        profile = load_deploy_config(config["native_profile"])
        prefix = profile.get("scene_prefix")
        if not isinstance(prefix, str) or not prefix or any(not text.startswith(prefix) for text in prompts.values()):
            raise ValueError("episode prompts must already contain the exact native scene prefix once")
        if any(text[len(prefix):].startswith(prefix) for text in prompts.values()):
            raise ValueError("duplicated native scene prefix")
        root = Path(profile["wan_root"]).resolve(strict=True)
        encoder_path = root / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer_root = root / "google" / "umt5-xxl"
        encoder_sha = file_sha256(encoder_path)
        tokenizer_sha = tokenizer_identity(tokenizer_root)
        for key, actual in (("text_encoder_sha256", encoder_sha), ("tokenizer_sha256", tokenizer_sha)):
            if key in config and config[key] != actual:
                raise ValueError(key + " differs from the bound configuration")
        identity = dict(schema=TEXT_SCHEMA, prompt_manifest_sha256=file_sha256(prompt_path),
            native_profile_sha256=file_sha256(config["native_profile"]),
            source_encoder_sha256=encoder_sha, tokenizer_sha256=tokenizer_sha,
            source_sha256=source_identity(), text_semantics=TEXT_SEMANTICS,
            context_shape=[512, 4096], context_dtype="bfloat16", mask_dtype="bool")
        output.mkdir(parents=True, exist_ok=True)
        index_path = output / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("identity") != identity or index.get("episodes") != episodes:
                raise ValueError("existing text index identity/episode bindings differ")
        else:
            index = dict(identity=identity, episodes=episodes, items={}, complete=False)
        selected = sorted(prompts)
        if args.limit:
            selected = selected[:args.limit]
        pending = []
        for prompt_id in selected:
            entry = index["items"].get(prompt_id)
            path = output / (prompt_id + ".pt")
            if entry is not None:
                if entry.get("path") != path.name or entry.get("text_sha256") != prompt_id or entry.get("sha256") != file_sha256(path):
                    raise ValueError("existing prompt cache hash/path binding changed")
            elif path.exists():
                raise FileExistsError("orphan prompt cache preserved; review before selecting a fresh output")
            else:
                pending.append(prompt_id)
        encoder = _build_t5_encoder(profile, str(target)) if pending else None
        floor = int(config.get("min_free_bytes", 10 * (1 << 30)))
        for first in range(0, len(pending), args.batch_size):
            ids = pending[first:first + args.batch_size]
            contexts = pack_contexts(encoder, [prompts[k] for k in ids], str(target))
            for prompt_id, (payload, token_sha) in zip(ids, contexts):
                path = output / (prompt_id + ".pt")
                atomic_checkpoint(path, payload, min_free_bytes=floor)
                index["items"][prompt_id] = dict(path=path.name, sha256=file_sha256(path),
                    text_sha256=prompt_id, token_ids_sha256=token_sha,
                    source_encoder_sha256=encoder_sha, tokenizer_sha256=tokenizer_sha,
                    token_count=int(payload["mask"].sum()))
                _atomic_json(index_path, index)
            print(json.dumps(dict(event="text_batch_saved", completed=len(index["items"]), total=len(prompts))), flush=True)
        index["complete"] = set(index["items"]) == set(prompts)
        _atomic_json(index_path, index)
        return dict(index=str(index_path), cached=len(index["items"]), total=len(prompts), complete=index["complete"], device=str(target))

    if target.type == "cpu":
        with cpu_without_cuda():
            return work()
    return work()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorization")
    parser.add_argument("--limit", type=int, default=4, help="unique prompts; zero processes all")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args, argv), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
