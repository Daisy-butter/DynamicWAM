"""Eight-rank training with explicit phase lineage and exact update-boundary resume.

The external guard owns allocation, telemetry, initial acceptance and process
cleanup. This entry cannot start an unguarded CUDA context or retry a failed job.
"""
from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from . import train as base

SCHEMA = "dualrate-distributed-checkpoint-v1"
PHASES = ("warmup", "small_age_joint", "cross_time_joint")
OPERATIONAL_FIELDS = {"run_kind", "output", "steps", "sequence_id", "acceptance_receipt",
    "acceptance_receipt_sha256", "stage_init_checkpoint", "stage_init_checkpoint_sha256", "resume"}


def profile_sha256(config):
    # Kept identical to the pure-stdlib external guard, which must not import
    # this torch-dependent module before acquiring its device lifetime handles.
    return base.canonical_hash({key: value for key,value in config.items() if key not in OPERATIONAL_FIELDS})


class DistributedEpochCursor:
    """Disjoint shuffled rank batches; drop only the incomplete global tail.

    Every rank reconstructs the same permutation without touching model RNG.
    position counts global rows, so rank cursors cannot silently drift. Epoch
    reshuffling includes prior tail rows, rather than permanently removing them.
    """
    def __init__(self, length, batch_size, seed, rank, world_size, epoch=0, position=0):
        self.length, self.batch_size, self.seed = int(length), int(batch_size), int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.epoch, self.position = int(epoch), int(position)
        self.global_batch = self.batch_size * self.world_size
        if min(self.length, self.batch_size, self.world_size) <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed sampler dimensions")
        self.usable = self.length // self.global_batch * self.global_batch
        if self.usable == 0 or self.epoch < 0 or not 0 <= self.position <= self.usable or self.position % self.global_batch:
            raise ValueError("invalid distributed cursor or dataset smaller than a global microbatch")
        self._order_epoch, self._order = None, None

    def next_indices(self):
        if self.position == self.usable:
            self.epoch += 1
            self.position = 0
        if self._order_epoch != self.epoch:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
            self._order = torch.randperm(self.length, generator=generator)
            self._order_epoch = self.epoch
        start = self.position + self.rank * self.batch_size
        result = self._order[start:start + self.batch_size].tolist()
        self.position += self.global_batch
        return result

    def state_dict(self):
        return {key: getattr(self, key) for key in ("length", "batch_size", "seed", "rank", "world_size", "epoch", "position")}

    def load_state_dict(self, value):
        for key in ("length", "batch_size", "seed", "rank", "world_size"):
            if value.get(key) != getattr(self, key):
                raise ValueError(f"distributed sampler {key} changed")
        checked = type(self)(**value)
        self.epoch, self.position = checked.epoch, checked.position
        self._order_epoch, self._order = None, None


def selected_rows(dataset, query_age_seconds):
    """Select existing physical-age rows; never relabel timestamps or targets."""
    import numpy as np
    allowed = tuple(float(age) for age in query_age_seconds)
    if getattr(dataset,'exposure_anchored',False):
        # rows[:,2] is an event-template id, not an independent age id.
        available=tuple(float(row['query_time']) for row in dataset.event_templates['templates'])
    else:
        available = tuple(float(age) for age in dataset.manifest["query_ages_seconds"])
    if not allowed or any(not any(abs(age-candidate) < 1e-9 for candidate in available) for age in allowed):
        raise ValueError("requested stage ages are absent from the prepared physical-grid data")
    ids = [i for i, age in enumerate(available) if any(abs(age-value) < 1e-9 for value in allowed)]
    rows = np.flatnonzero(np.isin(dataset.rows[:, 2], ids)).astype(np.int64)
    if not len(rows):
        raise ValueError("stage age filter produced no samples")
    return rows


def validate_phase_config(config):
    phase = config.get("phase")
    if phase not in PHASES:
        raise ValueError("explicit supported phase required")
    if config.get("run_kind") not in {"distributed_training_smoke", "formal_distributed_training"}:
        raise ValueError("explicit distributed run kind required")
    if int(config.get("world_size", 0)) != 8:
        raise ValueError("formal and engineering distributed runs require eight ranks")
    batch, accumulation = int(config["batch_size"]), int(config["gradient_accumulation_steps"])
    if min(batch, accumulation, int(config["steps"])) < 1:
        raise ValueError("positive step, microbatch and accumulation counts required")
    if config.get("global_batch_size") != 8 * batch * accumulation:
        raise ValueError("global batch must equal eight ranks times microbatch times accumulation")
    if config.get("optimizer_kind") != "fp32_master_adamw" or config.get("gradient_accumulation_dtype") != "float32":
        raise ValueError("native-compatible FP32 masters and accumulation required")
    if config.get("ddp_find_unused_parameters") is not True:
        raise ValueError("conditional losses require explicit unused-parameter discovery")
    if not isinstance(config.get("sequence_id"), str) or not config["sequence_id"]:
        raise ValueError("sequence_id binds all phase lineage")
    if config.get("sampler") != "disjoint_global_shuffle_drop_tail":
        raise ValueError("unsupported sampler contract")
    roles = config.get("active_entity_roles")
    if roles != ["tracked_dynamic_target", "left_gripper", "right_gripper"]:
        raise ValueError("V3 training requires the three actually supervised entity roles")
    if config["architecture"].get("future_entities") != len(roles) or config["architecture"].get("future_tokens") != 4*len(roles):
        raise ValueError("future token budget must match active roles times four target times")
    if not (0 < float(config["min_lr_ratio"]) <= 1) or int(config["lr_warmup_steps"]) < 0:
        raise ValueError("invalid LR schedule")
    if not (float(config["fast_lr"]) > 0 and float(config["max_grad_norm"]) > 0):
        raise ValueError("positive fast LR and gradient norm required")
    if config.get("continuous_v8"):
        if float(config["slow_lr"]) != 0:
            raise ValueError("continuous V8 first candidate freezes the Full slow system")
    elif config.get("native_v7"):
        if float(config["slow_lr"]) != 0:
            raise ValueError("compact V7 requires a frozen Full teacher")
    elif config.get("native_v6"):
        if float(config["slow_lr"]) != 0 or not float(config["motor_lr"]) > 0:
            raise ValueError("V6 requires frozen Full teacher and positive motor LR")
    elif not float(config["slow_lr"]) > 0:
        raise ValueError("positive slow LR required")
    if phase == "warmup" and config.get("stage_init_checkpoint"):
        raise ValueError("first phase must use official full native initialization, never smoke weights")
    if phase != "warmup" and (not config.get("stage_init_checkpoint") or not config.get("stage_init_checkpoint_sha256")):
        raise ValueError("joint phases require the exact previous phase full checkpoint")


def phase_contract(config, identity):
    contract = dict(sequence_id=config["sequence_id"], world_size=int(config["world_size"]),
        source_sha256=identity["source_sha256"], manifest_sha256=identity["manifest_sha256"],
        init_checkpoint_sha256=config["init_checkpoint_sha256"],
        action_normalization_sha256=config["action_normalization_sha256"],
        motion_normalization_sha256=config["motion_normalization_sha256"],
        architecture=config["architecture"], codec_id=config["codec_id"],
        device_type=identity["device_type"], torch_version=identity["torch_version"],
        optimizer_kind=config["optimizer_kind"], gradient_accumulation_dtype="float32")
    # Legacy checkpoints retain their exact contract. V4 must bind the clock
    # interpretation as well as tensor architecture across all phase transitions.
    if config.get("time_contract") is not None:
        contract.update(time_contract=config["time_contract"],
                        action_interval_seconds=float(config["action_interval_seconds"]))
    if config.get("native_v5"):contract["native_v5"]=config["native_v5"]
    if config.get("native_v6"):contract["native_v6"]=config["native_v6"]
    if config.get("native_v7"):contract["native_v7"]=config["native_v7"]
    if config.get("continuous_v8"):
        contract["continuous_v8"]=config["continuous_v8"]
        contract["runtime_contract"]=config["runtime"]
    return contract


def set_phase(system, phase):
    if phase not in PHASES:
        raise ValueError("unknown phase")
    system.train()
    system.slow.requires_grad_(phase != "warmup" and not getattr(system,"native_v6",None) and not getattr(system,"continuous_v8",None))
    if getattr(system,"native_v5",None) and hasattr(system.fast,"motor"):
        system.fast.motor.requires_grad_(phase != "warmup")
    if phase == "warmup" or getattr(system,"native_v6",None) or getattr(system,"continuous_v8",None):
        system.slow.eval()


def set_learning_rate(optimizer, config, step):
    """Deterministic phase-local schedule; step counts completed updates."""
    warmup, total = int(config["lr_warmup_steps"]), int(config["steps"])
    if step < warmup:
        ratio = (step + 1) / max(1, warmup)
    else:
        progress = min(1., max(0., (step-warmup) / max(1, total-warmup)))
        floor = float(config["min_lr_ratio"])
        ratio = floor + (1-floor) * .5 * (1+math.cos(math.pi*progress))
    for group in optimizer.param_groups:
        group["lr"] = float(config["slow_lr" if group["name"] == "slow" else "motor_lr" if group["name"] == "motor" else "fast_lr"]) * ratio
    return ratio


def _all_objects(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _rank_state(cursor, generator, device):
    return dict(rank=dist.get_rank(), cursor=cursor.state_dict(),
                rng=base.capture_rng(generator, cuda_device=device if device.type == "cuda" else None))


def save_checkpoint(path, system, optimizer, cursor, generator, device, config, identity,
                    phase_step, global_step, *, min_free_bytes=0):
    """All ranks contribute RNG/cursor; only rank zero serializes shared tensors."""
    rank_states = _all_objects(_rank_state(cursor, generator, device))
    if [state["rank"] for state in rank_states] != list(range(dist.get_world_size())):
        raise RuntimeError("checkpoint rank ordering mismatch")
    positions = {(s["cursor"]["epoch"], s["cursor"]["position"]) for s in rank_states}
    if len(positions) != 1:
        raise RuntimeError("rank data cursors drifted")
    if dist.get_rank() == 0:
        base.atomic_checkpoint(path, dict(schema=SCHEMA, identity=identity,
            contract=phase_contract(config, identity), phase=config["phase"],
            run_kind=config["run_kind"], phase_step=phase_step, global_step=global_step,
            configured_steps=int(config["steps"]), phase_complete=phase_step == int(config["steps"]),
            formal_training_ready=config["run_kind"] == "formal_distributed_training",
            policy_quality_accepted=False,
            optimizer_boundary=True, model=system.state_dict(), optimizer=optimizer.state_dict(),
            rank_states=rank_states, model_training=system.training,
            parameter_requires_grad={name: p.requires_grad for name,p in system.named_parameters()},
            module_training={name: module.training for name,module in system.named_modules()},
            optimizer_precision=optimizer.precision_report(),
            lineage=dict(stage_init_checkpoint=config.get("stage_init_checkpoint"),
                         stage_init_checkpoint_sha256=config.get("stage_init_checkpoint_sha256")),
            timing_distribution=config.get("data_distribution", "offline_synthetic_expert_prefix_not_measured_on_policy"),
            action_conditioning_distribution="offline_expert_executed_and_committed_prefix_not_on_policy"), min_free_bytes=min_free_bytes)
    dist.barrier()


def load_checkpoint(path, system, optimizer, cursor, generator, config, identity, *, transition=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("optimizer_boundary") is not True:
        raise ValueError("distributed full-state checkpoint required")
    if payload.get("contract") != phase_contract(config, identity):
        raise ValueError("checkpoint source/data/native-init/architecture/sequence contract differs")
    if payload.get("run_kind") != config["run_kind"]:
        raise ValueError("smoke checkpoints cannot initialize or resume formal training")
    states = payload.get("rank_states", [])
    if len(states) != dist.get_world_size() or [value["rank"] for value in states] != list(range(dist.get_world_size())):
        raise ValueError("checkpoint lacks full per-rank state")
    if transition:
        phase_id = PHASES.index(config["phase"])
        if phase_id == 0 or payload.get("phase") != PHASES[phase_id-1] or payload.get("phase_complete") is not True:
            raise ValueError("phase transition requires completed immediate predecessor")
    elif payload.get("identity") != identity or payload.get("phase") != config["phase"]:
        raise ValueError("strict resume source/config/runtime/phase identity differs")
    if not transition:
        cursor.load_state_dict(states[dist.get_rank()]["cursor"])
        expected = {name: p.requires_grad for name,p in system.named_parameters()}
        if payload.get("parameter_requires_grad") != expected:
            raise ValueError("resume requires_grad contract differs")
    system.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if not transition:
        for name, module in system.named_modules():
            module.training = payload["module_training"][name]
        base.restore_rng(states[dist.get_rank()]["rng"], generator)
    # A stage transition deliberately resets phase sampler/RNG and LR schedule,
    # while preserving every trained tensor, FP32 master and Adam moment.
    return (0 if transition else int(payload["phase_step"])), int(payload["global_step"])


def verify_reload(path, system, optimizer, cursor, generator, device, config, identity, phase_step, global_step, output):
    def snapshot():
        return dict(model=base.state_sha256(system.state_dict()), optimizer=base.state_sha256(optimizer.state_dict()),
            rank=base.state_sha256(_rank_state(cursor, generator, device)),
            modes=base.state_sha256({name: m.training for name,m in system.named_modules()}),
            requires_grad=base.state_sha256({name: p.requires_grad for name,p in system.named_parameters()}))
    before = snapshot()
    restored_step, restored_global = load_checkpoint(path, system, optimizer, cursor, generator, config, identity)
    after = snapshot()
    result = dict(rank=dist.get_rank(), passed=before == after and restored_step == phase_step and restored_global == global_step,
                  before=before, after=after, phase_step=restored_step, global_step=restored_global)
    results = _all_objects(result)
    equal_model = len({item["after"]["model"] for item in results}) == 1
    equal_optimizer = len({item["after"]["optimizer"] for item in results}) == 1
    passed = all(item["passed"] for item in results) and equal_model and equal_optimizer
    report = dict(schema="dualrate-distributed-strict-reload-v1", passed=passed, ranks=results,
                  equal_model=equal_model, equal_optimizer=equal_optimizer, checkpoint=str(path),
                  extra_training_updates=0, identity=identity, world_size=dist.get_world_size())
    if dist.get_rank() == 0:
        report["checkpoint_sha256"] = base.file_sha256(path)
        base._atomic_json(output / "reload.json", report)
    if not passed:
        raise RuntimeError("distributed strict checkpoint reload or rank equality failed")
    dist.barrier()
    return report


def _validate_acceptance(config, identity):
    if config["run_kind"] != "formal_distributed_training":
        return
    path = Path(config["acceptance_receipt"]).resolve(strict=True)
    if base.file_sha256(path) != config["acceptance_receipt_sha256"]:
        raise ValueError("formal acceptance receipt hash mismatch")
    receipt = json.loads(path.read_text())
    if (receipt.get("passed") is not True or receipt.get("source_sha256") != identity["source_sha256"]
            or receipt.get("manifest_sha256") != identity["manifest_sha256"]
            or receipt.get("init_checkpoint_sha256") != identity["init_checkpoint_sha256"]
            or receipt.get("world_size") != 8):
        raise ValueError("formal training requires exact-source eight-rank and deployment acceptance")
    if receipt.get("profiles", {}).get(config["phase"]) != profile_sha256(config):
        raise ValueError("formal numerical/model/data profile differs from accepted phase smoke")


def validate_resume_authorization(resume, receipt, output):
    """Bind an explicit resume to immutable bytes before creating CUDA context."""
    declared, digest = receipt.get("resume_checkpoint"), receipt.get("resume_checkpoint_sha256")
    if resume is None:
        if declared is not None or digest is not None:
            raise ValueError("resume-bound authorization cannot launch fresh training")
        return
    argument = Path(resume)
    if not argument.is_absolute():
        raise ValueError("resume checkpoint must be absolute")
    resolved = argument.resolve(strict=True)
    if (declared != str(resolved) or resolved != (Path(output).resolve() / "last.pt").resolve(strict=True)
            or not isinstance(digest, str) or base.file_sha256(resolved) != digest):
        raise ValueError("resume checkpoint path/bytes differ from this guarded run authorization")


def run_distributed(config, *, device="cuda", resume=None, authorization=None, stop_after=None):
    validate_phase_config(config)
    rank, local_rank, world = (int(os.environ.get(key, "-1")) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 8 or rank != local_rank or not 0 <= rank < world or not str(device).startswith("cuda"):
        raise ValueError("single-node guarded launcher with eight local CUDA ranks required")
    target = torch.device(f"cuda:{local_rank}")
    output = base._validate_storage(config)
    from .dependencies import validate_dependency_overlay, activate_dependency_overlay, probe_dependency_imports
    dependencies = validate_dependency_overlay(config)
    activate_dependency_overlay(dependencies)
    manifest_path = Path(config["manifest"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if config.get('native_v5') or config.get('native_v6'):
        if manifest.get('time_contract')!=('native_event_single_v6' if config.get('native_v6') else 'native_sync_ordinal_v5') or config.get('time_contract')!=manifest['time_contract']:
            raise ValueError('native training/data contract mismatch')
        if config['runtime']['protect_steps']!=0 or config['runtime']['action_dt']!=manifest['action_interval_seconds']:
            raise ValueError('native synchronous training must use zero inference prefix')
    elif config.get('time_contract') is not None or manifest.get('time_contract')=='exposure_anchor_wall_release_v4':
        from .exposure_timing import TIME_CONTRACT, load_templates
        if config.get('time_contract')!=TIME_CONTRACT or manifest.get('time_contract')!=TIME_CONTRACT:
            raise ValueError('training and data physical-time contracts differ')
        events=load_templates(manifest['event_templates_file'],manifest['event_templates_file_sha256'])
        if (config.get('action_interval_seconds')!=manifest['action_interval_seconds'] or
                events['action_interval_seconds']!=manifest['action_interval_seconds']):
            raise ValueError('training, data and measured event execution grids differ')
        if config['run_kind']=='formal_distributed_training' and events['kind']!='measured_parallel':
            raise ValueError('formal V4 training requires measured parallel event templates')
    if config.get('continuous_v8'):
        from .continuous_contract import validate_training_contract
        validate_training_contract(config,manifest)
    base._validate_normalization(config, manifest)
    if base.file_sha256(config["init_checkpoint"]) != config["init_checkpoint_sha256"]:
        raise ValueError("official native initialization checksum differs")
    identity = dict(source_sha256=base.source_identity(), config_sha256=base.canonical_hash(config),
        manifest_sha256=base.file_sha256(manifest_path), init_checkpoint_sha256=config["init_checkpoint_sha256"],
        device_type="cuda", torch_version=str(torch.__version__))
    if authorization is None:
        raise ValueError("fresh external guarded rank authorization required")
    base._gpu_authorization(Path(authorization), identity, output, expected_mode=config["run_kind"],
                            expected_dependencies=dependencies, rank=rank, world_size=world)
    validate_resume_authorization(resume, json.loads(Path(authorization).read_text()), output)
    _validate_acceptance(config, identity)
    probe = probe_dependency_imports(config, require_flash_attention=True)
    # No CUDA context is touched before guard/identity/acceptance validation.
    torch.cuda.set_device(target)
    dist.init_process_group("nccl", timeout=timedelta(seconds=int(config.get("distributed_timeout_seconds", 1800))))
    try:
        return _train(config, target, output, rank, world, identity, dependencies, probe, resume, stop_after)
    finally:
        # The external supervisor retains device FDs until all ranks exit.
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, target, output, rank, world, identity, dependencies, probe, resume, stop_after):
    from .data import open_training_dataset
    from .factory import build_system
    from .optimizer import FP32MasterAdamW
    from .producer import materialize_native_latents
    from .dependencies import validate_dependency_overlay
    if rank == 0:
        if resume is None and output.exists() and any(output.iterdir()):
            raise FileExistsError("fresh training output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        base._atomic_json(output / "dependencies.json", probe)
    dist.barrier()
    floor, save_every = int(config["min_free_bytes"]), int(config["save_every"])
    if floor < 0 or save_every < 1 or shutil.disk_usage(output).free < floor:
        raise OSError("invalid checkpoint interval or insufficient verified artifact disk space")
    dataset = open_training_dataset(config["manifest"], verify_hash=True)
    if config["codec_id"] != dataset.manifest["codec_id"] or config["runtime"]["action_dt"] != dataset.manifest["action_interval_seconds"]:
        raise ValueError("training/runtime codec or physical grid mismatch")
    for key in ("future_field_support", "contact_field_support"):
        if config[key] != dataset.manifest[key]:
            raise ValueError("training/deployment capability mask mismatch")
    for key in ("active_entity_roles", "visual_target_semantics", "visual_validity_semantics"):
        if not config.get(key) or config[key] != dataset.manifest.get(key):
            raise ValueError(f"training/manifest {key} differs")
    eligible = selected_rows(dataset, config["query_age_seconds"])
    base.validate_row(dataset[int(eligible[0])])
    seed = int(config["seed"])
    # Identical initialization across ranks; RNG becomes rank-specific only
    # after DDP broadcasts parameters and buffers.
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    import numpy as np
    np.random.seed(seed % 2**32)
    system = build_system(config, target)
    slow = list(system.slow.parameters())
    slow_ids = {id(p) for p in slow}
    new = [p for p in system.parameters() if id(p) not in slow_ids and p.requires_grad]
    # Fixed optimizer groups across phases preserve fast moments and slow masters.
    groups=[dict(params=slow,lr=config["slow_lr"],name="slow")]
    if (config.get("native_v5") or config.get("native_v6")) and not config.get("native_v7"):
        motor=list(system.fast.motor.parameters());motor_ids={id(p) for p in motor}
        new=[p for p in new if id(p) not in motor_ids]
        groups.append(dict(params=motor,lr=config["motor_lr"],name="motor"))
    groups.append(dict(params=new,lr=config["fast_lr"],name="new"))
    optimizer = FP32MasterAdamW(groups, weight_decay=float(config["weight_decay"]),
        record_updates=bool(config.get("record_master_updates", True)))
    set_phase(system, config["phase"])
    ddp = DistributedDataParallel(system, device_ids=[target.index], output_device=target.index,
        find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=False,
        bucket_cap_mb=int(config.get("ddp_bucket_cap_mb", 25)))
    rank_seed = seed + 100003 * rank
    random.seed(rank_seed)
    torch.random.default_generator.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    generator = torch.Generator(device=target).manual_seed(rank_seed+1)
    cursor = DistributedEpochCursor(len(eligible), int(config["batch_size"]), seed, rank, world)
    start, global_start = 0, 0
    if resume:
        start, global_start = load_checkpoint(resume, system, optimizer, cursor, generator, config, identity)
    elif config.get("stage_init_checkpoint"):
        path = Path(config["stage_init_checkpoint"]).resolve(strict=True)
        if base.file_sha256(path) != config["stage_init_checkpoint_sha256"]:
            raise ValueError("phase predecessor checkpoint hash differs")
        start, global_start = load_checkpoint(path, system, optimizer, cursor, generator, config, identity, transition=True)
    steps = int(config["steps"])
    end = steps if stop_after is None else min(steps, int(stop_after))
    if end < start or end < 1:
        raise ValueError("stop boundary precedes resume")
    counts = base.parameter_counts(system)
    if rank == 0:
        base._atomic_json(output / "parameters.json", dict(**counts, optimizer_precision=optimizer.precision_report(),
            eligible_rows=len(eligible), dropped_epoch_tail=cursor.length-cursor.usable,
            global_batch_size=config["global_batch_size"], phase=config["phase"],
            accumulation="FP32 after each microbatch DDP reduction; communication each microbatch",
            initialization="official native full; new modules seeded; previous phases full model and optimizer retained"))
    accumulation = int(config["gradient_accumulation_steps"])
    checkpoint = output / "last.pt"
    log_path = output / f"rank_{rank}.jsonl"
    for step in range(start, end):
        started = time.monotonic()
        lr_ratio = set_learning_rate(optimizer, config, step)
        optimizer.zero_grad(set_to_none=True)
        loss_sums = {}
        for micro in range(accumulation):
            rows = [dataset[int(eligible[i])] for i in cursor.next_indices()]
            for row in rows:
                base.validate_row(row)
            batch = base._device_batch(default_collate(rows), target)
            if "source_condition_frames" in batch:
                batch = materialize_native_latents(batch, system.slow.native.compact_wan.video_model.vae)
            # Calling .module.training_loss would bypass DDP reducer discovery.
            result = ddp(batch, generator=generator)
            loss = result["loss"]
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite/non-scalar training loss")
            loss.backward()
            optimizer.accumulate_gradients(scale=1./accumulation)
            loss_values = torch.stack([value.detach().float() for value in result.values()]).cpu().tolist()
            for key, value in zip(result, loss_values):
                loss_sums[key] = loss_sums.get(key, 0.) + value/accumulation
        gradients = dict.fromkeys(("slow", "world_readout", "observation", "fast"), 0)
        owner = {id(p): name for name,module in system.named_children() for p in module.parameters()}
        pairs = optimizer.accumulated_gradients()
        norms = torch.stack([torch.linalg.vector_norm(gradient) for _,gradient in pairs])
        # One compact host transfer, never a GPU synchronization for each weight.
        for (parameter, _), norm_value in zip(pairs, norms.detach().cpu().tolist()):
            if not math.isfinite(norm_value):
                raise FloatingPointError("nonfinite gradient")
            if norm_value > 0:
                gradients[owner[id(parameter)]] += 1
        required = ("world_readout", "observation", "fast") if config["phase"] == "warmup" or config.get("native_v6") or config.get("continuous_v8") else tuple(gradients)
        if any(gradients[name] == 0 for name in required):
            raise RuntimeError(f"disconnected module in {config['phase']}: {gradients}")
        norm = optimizer.clip_accumulated_grad_norm_(float(config["max_grad_norm"]),
                                                    total_norm=torch.linalg.vector_norm(norms))
        optimizer.step()
        del pairs, norms  # Do not retain FP32 accumulation buffers during next forward.
        torch.cuda.synchronize(target)
        record = dict(rank=rank, step=step+1, global_step=global_start+step-start+1, phase=config["phase"],
            losses=loss_sums, grad_norm=float(norm), nonzero_finite_gradients=gradients,
            seconds=time.monotonic()-started, lr_ratio=lr_ratio, optimizer_update=optimizer.last_step_statistics,
            global_batch_size=config["global_batch_size"], peak_allocated_bytes=torch.cuda.max_memory_allocated(target),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(target), cursor=cursor.state_dict())
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
            handle.flush()
        # All-rank finite telemetry catches missing or diverged ranks promptly.
        scalars = torch.tensor([loss_sums["loss"], record["seconds"]], device=target, dtype=torch.float64)
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
        if rank == 0 and (step < 10 or (step+1) % int(config.get("log_every", 10)) == 0):
            summary = dict(step=step+1, global_step=record["global_step"], phase=config["phase"],
                loss=float(scalars[0]/world), mean_rank_seconds=float(scalars[1]/world), lr_ratio=lr_ratio)
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(summary)+"\n")
            print(json.dumps(summary), flush=True)
        if (step+1) % save_every == 0 or step+1 == end:
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint(checkpoint, system, optimizer, cursor, generator, target, config, identity,
                step+1, record["global_step"], min_free_bytes=floor)
    final = checkpoint if end > start else Path(resume)
    global_end = global_start + end-start
    if config.get("verify_checkpoint_reload", True):
        verify_reload(final, system, optimizer, cursor, generator, target, config, identity, end, global_end, output)
    if validate_dependency_overlay(config) != dependencies:
        raise RuntimeError("dependency overlay mutated while running")
    result = dict(completed_steps=end, configured_steps=steps, global_step=global_end, phase=config["phase"],
        run_kind=config["run_kind"], world_size=world, identity=identity, checkpoint=str(final),
        checkpoint_sha256=base.file_sha256(final) if rank == 0 else None,
        formal_training_ready=config["run_kind"] == "formal_distributed_training",
        policy_quality_accepted=False, result_kind="completed_phase_not_policy_quality_acceptance")
    if rank == 0:
        base._atomic_json(output / "result.json", result)
    dist.barrier()
    # Release references before distributed teardown; root guard outlives ranks.
    del ddp, optimizer, system, dataset
    return result
