"""Deployment policy for a causal bridge plus a 10-step fast action sample.

The synchronous ``sample_chunk`` path is not used. A bridge rebuild encodes the
current frame with the VAE and runs the two joint prefix steps. Later fast
queries keep that bridge and read the current cameras, motion, and state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from dynamicwam.inference.multirate_model_loader import (
    install_full_checkpoint,
    load_checkpoint_payload,
    build_fast_conditioner,
    install_immutable_source_backbone,
)
from dynamicwam.models.fast_conditioner import FastConditioner
from dynamicwam.models.compact_fast_policy import CompactFastPolicy, CompactFastConfig
from dynamicwam.inference.fast_runtime import (
    finalize_inference,
    isolated_fast_runtime,
    validate_runtime_contract,
)
from dynamicwam.runtime.ciwam.wam.fast_slow_worker import LatestWorldWorker
from dynamicwam.models.multirate_types import (
    BridgeAge,
    CurrentObservation,
    SourceNoise,
    SourceObservation,
    pad_text_embeddings,
)
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.runtime.ciwam.wam.multirate_control import ControlledMultirateController
from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy


class MultiratePolicy:
    def __init__(
        self,
        deploy_config_path: str,
        multirate_checkpoint: str,
        *,
        project_root: str,
        fast_conditioner: dict[str, Any] | None = None,
        scheduling: str = "controlled_ratio",
        reuse_action_noise: bool = False,
        bridge_wait_budget_ms: float = 150.0,
        native_cuda_graph: bool = True,
        allow_zero_prefix_ablation: bool = False,
    ) -> None:
        if scheduling not in {"controlled_ratio", "independent_worker"}:
            raise ValueError("unknown multirate scheduling mode")
        if bridge_wait_budget_ms < 0:
            raise ValueError("bridge wait budget must be non-negative")
        self.scheduling = scheduling
        self.reuse_action_noise = bool(reuse_action_noise)
        self.bridge_wait_budget_ms = float(bridge_wait_budget_ms)
        self.native_cuda_graph = bool(native_cuda_graph)
        if type(allow_zero_prefix_ablation) is not bool:
            raise TypeError("allow_zero_prefix_ablation must be an explicit bool")
        self.allow_zero_prefix_ablation = allow_zero_prefix_ablation
        self.fast_graph = None
        self.worker = None
        self._producer_stream = None
        self.contract = None
        self.blocking_refreshes = 0
        self.over_age_queries = 0
        self._synchronous_slow_builds = 0
        self.last_query_trace = {}
        self._previous_fast_noise = None
        self._previous_noise_index = None
        self._episode_index = 0
        self._episode_seed = 0
        self.base = DynamicWAMPolicy(
            deploy_config_path,
            project_root=project_root,
        )
        checkpoint = Path(multirate_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(project_root).expanduser() / checkpoint
        self.checkpoint_path = str(checkpoint.resolve())
        self.fast_conditioner_config = dict(fast_conditioner or _default_fast_config())
        self.wrapper: MultirateWAM | None = None
        self.fast_model = None
        self.bridge = None
        self.bridge_state: torch.Tensor | None = None
        self.bridge_sim_time: float | None = None
        self.source_id = 0
        self._submitted_source: int | None = None

    def setup(self) -> None:
        from dynamicwam.inference.model_loader import (
            build_runtime_from_config,
            load_deploy_config,
        )
        from dynamicwam.runtime.ciwam.wam.bootstrap import add_dynamicwam_to_path
        from dynamicwam.runtime.ciwam.wam.flow_runner import HeadFlowRunner

        add_dynamicwam_to_path(self.base.project_root)
        config = load_deploy_config(self.base.deploy_config_path)
        self.base.runner = HeadFlowRunner(
            runtime=build_runtime_from_config(config)
        )
        runner = self.base.runner
        payload = load_checkpoint_payload(self.checkpoint_path)
        schedule, self.contract = validate_runtime_contract(payload, self.runtime,
            allow_zero_prefix_ablation=self.allow_zero_prefix_ablation)
        validate_native_scheduling(self.contract, self.scheduling)
        spec = payload["modules"]
        compact = spec.get("compact_fast")
        fast_cfg = spec["fast_conditioner"]
        if self.fast_conditioner_config != _default_fast_config() and self.fast_conditioner_config != fast_cfg:
            raise RuntimeError("external fast conditioner overrides disagree with checkpoint")
        conditioner = build_fast_conditioner(fast_cfg).to(runner.device)
        if compact is not None:
            motion = spec["motion"]
            student = CompactFastPolicy(CompactFastConfig(**compact),
                motion_history_count=motion["history_count"], motion_mean=motion["mean"],
                motion_scale=motion["scale"], schedule=schedule)
            self.wrapper = MultirateWAM(runner.model, None, schedule, compact_policy=student).to(runner.device)
            del conditioner
        else:
            self.wrapper = MultirateWAM(runner.model, conditioner, schedule,
                native_multirate=bool(spec.get('native_multirate', False)),
                fast_steps=int(spec.get('fast_steps', 10))).to(runner.device)
        install_full_checkpoint(self.wrapper, payload, allow_v2_init=False)
        if spec.get('native_multirate', False):
            install_immutable_source_backbone(runner.model, self.contract,
                payload['config']['action_normalization'])
        saved = payload.get("modules", {}).get("fast_conditioner", {})
        if saved:
            if float(saved["sim_scale"]) != float(fast_cfg["sim_scale"]):
                raise RuntimeError("fast conditioner sim_scale differs from the checkpoint")
            if float(saved["action_scale"]) != float(fast_cfg["action_scale"]):
                raise RuntimeError(
                    "fast conditioner action_scale differs from the checkpoint"
                )
        # Keep the existing backbone precision policy. Only the new branch is
        # converted; original time/modulation operations have explicit dtypes.
        fast = self.wrapper.fast_policy if self.wrapper.fast_policy is not None else self.wrapper.fast_conditioner
        precision = runner.model.compact_wan.video_model.precision if torch.device(runner.device).type == "cuda" else torch.float32
        finalize_inference(fast, device=runner.device, dtype=precision)
        self.wrapper.eval().requires_grad_(False)
        # Slow bridge keeps runner.model (VAE + Video DiT). Action sampling uses
        # a second read-only copy with no world weights.
        self.fast_model = isolated_fast_runtime(payload, device=runner.device, dtype=precision,
            allow_zero_prefix_ablation=self.allow_zero_prefix_ablation)
        if self.contract.get('native_multirate'):
            from dynamicwam.inference.native_fast_graph import NativeFastGraph
            self.fast_graph = NativeFastGraph(self.fast_model,
                enabled=self.native_cuda_graph and torch.device(runner.device).type == 'cuda'
                    and len(self.fast_model.fast_schedule.action_sigmas) <= 2)
        if self.contract.get('feedback'):
            if not self.contract['feedback'].get('ready'):
                raise RuntimeError('dynamics-only checkpoint has no distilled actor; finish policy training')
        if self.scheduling == "independent_worker":
            self.worker = LatestWorldWorker(self._build_bridge_completed)
        self.reset()

    @property
    def runtime(self):
        return self.base.runtime

    def set_instruction(self, instruction: str) -> None:
        if self.base.runner is not None and self.base.runner.current_instruction != str(instruction or ""):
            self.reset()
        self.base.set_instruction(instruction)

    def reset(self) -> None:
        if self.worker is not None:
            self.worker.reset()
        self.base.reset()
        self.bridge = None
        self.bridge_state = None
        self.bridge_sim_time = None
        self.source_id = 0
        self._episode_index += 1
        # Mix the process seed with the episode counter so each reset draws a
        # fresh noise stream. torch.initial_seed() alone is constant per process.
        seed_material = f"{int(torch.initial_seed())}:episode:{self._episode_index}".encode()
        self._episode_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "little"
        ) % (2**63 - 1)
        self.bridge_action_index = 0
        self._submitted_source = None
        self.blocking_refreshes = 0
        self.over_age_queries = 0
        self._synchronous_slow_builds = 0
        self.last_query_trace = {}
        self._previous_fast_noise = None
        self._previous_noise_index = None
        self._feedback_last_request = 0

    @property
    def actual_slow_builds(self):
        return self._synchronous_slow_builds + (0 if self.worker is None else self.worker.completed_count())

    def close(self):
        if self.worker is not None and not self.worker.close():
            raise TimeoutError("world producer is still running; do not unload its model yet")

    def query(
        self,
        packet: dict[str, Any],
        controller: ControlledMultirateController,
    ) -> np.ndarray:
        max_age = int(self.contract["max_age_actions"])
        lead = int(getattr(controller, "bridge_lead_actions", 0) or 0)
        if int(controller.execute_horizon) != int(self.contract["execute_horizon"]):
            # The trained checkpoint supervised the full action chunk. A longer
            # deploy horizon is allowed only when the bridge is snapshotted
            # early enough to stay inside max_age_actions.
            if lead < 1 or lead > max_age:
                raise RuntimeError("execute horizon differs from trained contract")
        native = self.contract.get('native_multirate')
        if native and (lead or int(controller.every_fast_queries) * int(controller.execute_horizon)
                       != int(native['world_refresh_actions'])):
            raise RuntimeError('native policy requires its trained H=4, R=4 source cadence')
        packet = dict(packet)
        packet["action_index"] = int(controller.executed_total)
        packet["_episode_seed"] = self._episode_seed
        packet["_query_index"] = int(controller.query_id)
        packet["_text"] = self.base.runner._get_text_embeddings()
        if self.contract.get('feedback'):
            return self._query_feedback(packet,controller)
        max_age = int(self.contract["max_age_actions"])
        compact_cfg = getattr(getattr(getattr(self, 'fast_model', None), 'fast_policy', None), 'config', None)
        if compact_cfg is not None and compact_cfg.temporal_world_memory:
            if int(controller.execute_horizon) != int(self.contract['execute_horizon']):
                raise RuntimeError('temporal student requires its trained execution horizon')
            # The next executed prefix must fit inside the source forecast.
            # The remaining sampled tail may extend beyond it; never label that
            # tail as covered by the historical future prediction.
            max_age = min(max_age, int(compact_cfg.chunk_size) - int(controller.execute_horizon))
        supervised_horizon = self.contract.get('supervised_action_horizon')
        if supervised_horizon is not None and int(controller.execute_horizon) > int(supervised_horizon):
            raise RuntimeError('execution horizon exceeds supervised fast prefix')
        null_bridge = False
        mode, wait_ms = "reuse", 0.0
        if lead > 0:
            if self.worker is None:
                raise RuntimeError("same-cadence deployment requires the world worker")
            mode, wait_ms, age = self._resolve_same_cadence_bridge(packet, max_age)
        elif self.worker is None:
            if controller.rebuild_bridge() or self.bridge is None:
                mode = "inline_startup" if self.bridge is None else "inline_scheduled"
                self._install_bridge(self._build_bridge_completed(packet))
            age = packet["action_index"] - self.bridge_action_index
            if self.bridge is None or age > max_age:
                mode = "inline_overage"
                self.blocking_refreshes += 1
                self._install_bridge(self._build_bridge_completed(packet))
                age = packet["action_index"] - self.bridge_action_index
        else:
            self._install_ready_bridge(packet["action_index"], max_age)
            action, source = plan_worker_bridge(
                action_index=packet["action_index"],
                bridge_index=self.bridge_action_index,
                bridge_present=self.bridge is not None,
                max_age=max_age,
                horizon=int(controller.execute_horizon),
                submitted_source=self._submitted_source,
            )
            if action == "blocking":
                self.blocking_refreshes += 1
                if self._submitted_source is None:
                    self._ensure_request(source, packet)
                waiting = time.perf_counter()
                self._install_bridge(self.worker.wait_for(source).value)
                wait_ms += (time.perf_counter() - waiting) * 1000.0
                mode = "wait_startup" if action == "blocking" else "wait_overage"
                self._submitted_source = None
            elif action == "prefetch":
                self._ensure_request(source, packet)
            elif action == "stale":
                # A late producer must not expose ages absent from training.
                # Wait for a pending bridge only if its source is still usable;
                # otherwise replace pending work with this current observation.
                self.over_age_queries += 1
                self.blocking_refreshes += 1
                if packet["action_index"] - int(source) > max_age:
                    source = packet["action_index"]
                self._ensure_request(source, packet)
                waiting = time.perf_counter()
                self._install_bridge(self.worker.wait_for(source).value)
                wait_ms += (time.perf_counter() - waiting) * 1000.0
                mode = "wait_startup" if action == "blocking" else "wait_overage"
                self._submitted_source = None
            age = packet["action_index"] - self.bridge_action_index
        # Action count alone does not bound DOMINO TOPP execution duration.
        # Never consume seconds outside the checkpoint's declared support.
        ttl = self.contract.get("max_age_seconds")
        if ttl is not None and float(packet["sim_time"]) - float(self.bridge_sim_time) > float(ttl):
            waiting = time.perf_counter()
            if self.worker is None:
                value = self._build_bridge_completed(packet)
            else:
                source = packet["action_index"]
                self._ensure_request(source, packet)
                value = self.worker.wait_for(source).value
                self._submitted_source = None
            self._install_bridge(value)
            wait_ms += (time.perf_counter() - waiting) * 1000.0
            self.blocking_refreshes += 1
            mode, age = "refresh_seconds_expired", packet["action_index"] - self.bridge_action_index
        if age < 0:
            raise RuntimeError("world source is in the future")
        result = self._sample_fast(packet, age, null_bridge=null_bridge)
        self.last_query_trace = dict(
            query_id=int(controller.query_id), scheduling=self.scheduling,
            bridge_mode=mode, world_wait_ms=wait_ms,
            source_action_index=int(self.bridge_action_index), action_index=packet["action_index"],
            age_actions=int(age), age_sim_seconds=float(packet["sim_time"]) - float(self.bridge_sim_time),
            execute_horizon=int(controller.execute_horizon), refresh_ratio=int(controller.every_fast_queries),
            bridge_lead_actions=lead,
            max_age_seconds=self.contract.get("max_age_seconds"),
            reuse_action_noise=bool(self.reuse_action_noise),
            fast_denoising_steps=(len(self.fast_model.fast_schedule.action_sigmas)
                                  if native else len(self.wrapper.schedule.action_sigmas)),
            source_bridge=('immutable_official' if native else 'checkpoint_backbone'),
            cuda_graph=(dict(self.fast_graph.stats) if self.fast_graph is not None else None),
        )
        return result

    def _query_feedback(self, packet, controller):
        # Only the heavyweight WAM prior has a slow clock. Fresh action-dependent
        # futures are generated on every fast query, even when that prior expires.
        if controller.execute_horizon != self.contract['execute_horizon']:
            raise RuntimeError('feedback policy requires its trained execution horizon')
        wait_ms, mode = 0., 'reuse'
        period = int(self.contract['feedback']['world_refresh_actions'])
        due = packet['action_index'] - self._feedback_last_request >= period
        if self.bridge is None or (self.worker is None and due):
            mode = 'inline_startup' if self.bridge is None else 'inline_budget_refresh'
            started = time.perf_counter()
            self._install_bridge(self._build_bridge_completed(packet))
            if self.worker is not None: self._synchronous_slow_builds += 1
            self._feedback_last_request = packet['action_index']
            wait_ms = (time.perf_counter()-started)*1000
        elif self.worker is not None:
            self._install_ready_bridge(packet['action_index'],2**30)
        age = packet['action_index'] - self.bridge_action_index
        seconds = float(packet['sim_time']) - float(self.bridge_sim_time)
        null = age > self.contract['max_age_actions'] or seconds > self.contract['max_age_seconds']
        if null: self.over_age_queries += 1
        actions = self._sample_fast(packet,age,null_bridge=null)
        period = int(self.contract['feedback']['world_refresh_actions'])
        submitted = False
        if self.worker is not None and packet['action_index'] - self._feedback_last_request >= period and self._submitted_source is None:
            self._ensure_request(packet['action_index'],packet)
            self._feedback_last_request = packet['action_index']
            submitted = True
        self.last_query_trace = dict(query_id=int(controller.query_id),scheduling=self.scheduling,
            bridge_mode=mode,world_wait_ms=wait_ms,source_action_index=int(self.bridge_action_index),
            action_index=int(packet['action_index']),age_actions=int(age),age_sim_seconds=seconds,
            execute_horizon=int(controller.execute_horizon),refresh_ratio=int(controller.every_fast_queries),
            bridge_lead_actions=0,expired_prior_dropped=bool(null),world_request_submitted=submitted,
            actor_evaluations=2,forecast_updates=2,forecast_kind='fresh_action_conditioned_vae_patches')
        return actions

    def submit_bridge_snapshot(
        self,
        packet: dict[str, Any],
        controller: ControlledMultirateController,
    ) -> None:
        """Queue the next world bridge without sampling actions or waiting."""

        if not controller.needs_bridge_snapshot():
            return
        if self.worker is None:
            raise RuntimeError("same-cadence bridge snapshots require the world worker")
        packet = dict(packet)
        packet["action_index"] = int(controller.executed_total)
        packet["_episode_seed"] = self._episode_seed
        packet["_text"] = self.base.runner._get_text_embeddings()
        self._ensure_request(packet["action_index"], packet)
        controller.note_bridge_snapshot()

    def _resolve_same_cadence_bridge(
        self,
        packet: dict[str, Any],
        max_age: int,
    ) -> tuple[str, float, int]:
        """Use a prefetched bridge. Wait only inside the sync-query slack.

        Startup has no bridge and waits for the first build. A decision whose
        installed bridge is already inside ``max_age`` spends at most
        ``bridge_wait_budget_ms`` on a newer prefetch, then samples. An
        over-age bridge keeps waiting, because that age was not trained.
        """

        self._install_ready_bridge(packet["action_index"], max_age)
        mode, wait_ms = "reuse", 0.0
        age = None if self.bridge is None else packet["action_index"] - self.bridge_action_index
        budget_s = self.bridge_wait_budget_ms / 1000.0

        if self.bridge is None:
            source = packet["action_index"] if self._submitted_source is None else self._submitted_source
            self.blocking_refreshes += 1
            self._ensure_request(int(source), packet)
            waiting = time.perf_counter()
            self._install_bridge(self.worker.wait_for(source).value)
            wait_ms += (time.perf_counter() - waiting) * 1000.0
            self._submitted_source = None
            mode = "wait_startup"
        elif age > max_age:
            self.over_age_queries += 1
            self.blocking_refreshes += 1
            source = self._submitted_source
            if source is None or packet["action_index"] - int(source) > max_age:
                source = packet["action_index"]
            self._ensure_request(int(source), packet)
            completed, spent = self._wait_bridge(int(source), budget_s)
            wait_ms += spent
            if completed is None:
                waiting = time.perf_counter()
                completed = self.worker.wait_for(source)
                wait_ms += (time.perf_counter() - waiting) * 1000.0
                mode = "wait_overage"
            else:
                mode = "wait_budget"
            self._install_bridge(completed.value)
            self._submitted_source = None
        elif (
            self._submitted_source is not None
            and int(self._submitted_source) > int(self.bridge_action_index)
            and packet["action_index"] - int(self._submitted_source) <= max_age
        ):
            source = int(self._submitted_source)
            completed, spent = self._wait_bridge(source, budget_s)
            wait_ms += spent
            if completed is not None:
                self._install_bridge(completed.value)
                self._submitted_source = None
                mode = "wait_budget"
        age = packet["action_index"] - self.bridge_action_index
        if age < 0 or age > max_age:
            raise RuntimeError("fast sample would use a bridge outside trained age support")
        return mode, wait_ms, int(age)

    def _wait_bridge(self, source: int, timeout_s: float):
        waiting = time.perf_counter()
        try:
            completed = self.worker.wait_for(source, timeout=timeout_s)
        except TimeoutError:
            completed = None
        return completed, (time.perf_counter() - waiting) * 1000.0

    def _install_ready_bridge(self, action_index: int, max_age: int) -> None:
        latest = self.worker.latest()
        if latest is None:
            return
        if self._submitted_source is not None and int(latest.source_index) >= self._submitted_source:
            self._submitted_source = None
        age = int(action_index) - int(latest.source_index)
        if age < 0 or age > max_age:
            return
        if self.bridge is not None and int(latest.source_index) <= int(self.bridge_action_index):
            return
        self._install_bridge(latest.value)

    def _ensure_request(self, source_index: int, packet: dict[str, Any]) -> None:
        source_index = int(source_index)
        if self._submitted_source is not None and self._submitted_source >= source_index:
            return
        latest = self.worker.latest()
        if latest is not None and int(latest.source_index) >= source_index:
            return
        self.worker.request(source_index, capture_observation_packet(packet))
        self._submitted_source = source_index

    def _require_wrapper(self) -> MultirateWAM:
        if self.wrapper is None or self.base.runner is None:
            raise RuntimeError("multirate policy setup() must run before execution")
        return self.wrapper

    def _install_bridge(self, value):
        self.bridge, self.bridge_state, self.bridge_sim_time, self.bridge_action_index = value

    def _build_bridge_completed(self, packet):
        runner = self.base.runner
        device = torch.device(runner.device)
        stream = None
        if device.type == "cuda" and self.worker is not None:
            if self._producer_stream is None:
                self._producer_stream = torch.cuda.Stream(device=device)
            stream = self._producer_stream
            # Clones may have been produced on any caller stream, not default.
            for event in packet.get("_cuda_ready_events", ()):
                stream.wait_event(event)
        with torch.inference_mode(), (torch.cuda.stream(stream) if stream is not None else nullcontext()):
            value = self._build_bridge(packet)
            if stream is not None:
                done = torch.cuda.Event()
                done.record(stream)
        if stream is not None:
            done.synchronize()  # producer thread only; consumers never see partial K/V
        if self.worker is None:
            self._synchronous_slow_builds += 1
        return value

    def _build_bridge(self, packet: dict[str, Any]):
        wrapper = self._require_wrapper()
        runner = self.base.runner
        assert runner is not None
        frame = self.base._frame_tensor(packet["frame"])
        generator = packet_noise_generator(packet, runner.device, branch="world")
        future, condition = runner._initialize_video_latent(frame, packet["flow_frames"], generator=generator)
        state = runner._normalize_state(self.base._state_tensor(packet["state"]))
        motion = runner._motion_batch(
            packet["motion_features"],
            packet["motion_interval_valid_mask"],
            packet["motion_acceleration_valid_mask"],
        )
        text = packet["_text"]
        source_id = int(packet["action_index"])
        source = SourceObservation(
            condition_latent=condition,
            state=state,
            motion=motion["absolute_motion_features"],
            motion_interval_valid_mask=motion["absolute_motion_interval_valid_mask"],
            motion_acceleration_valid_mask=motion[
                "absolute_motion_acceleration_valid_mask"
            ],
            text_embeddings=text,
            source_id=source_id,
            sim_time=float(packet["sim_time"]),
        )
        noise = SourceNoise(future_latent=future, actions=runner._fresh_noisy_actions(generator=generator))
        with torch.inference_mode():
            bridge = wrapper.bridge(
                wrapper.backbone,
                source,
                noise,
                detach_cache=True,
            )
        return bridge, state.detach(), float(packet["sim_time"]), int(packet["action_index"])

    def _sample_fast(
        self,
        packet: dict[str, Any],
        age_actions: int,
        *,
        null_bridge: bool = False,
    ) -> np.ndarray:
        if self.fast_model is None:
            raise RuntimeError("fast sample requires the isolated fast runtime")
        wrapper = self.fast_model
        runner = self.base.runner
        assert runner is not None
        if self.bridge is None or self.bridge_state is None or self.bridge_sim_time is None:
            raise RuntimeError("fast sample requires a published bridge")
        views, masks = fast_views_from_composite(packet["fast_frame"])
        rgb = torch.from_numpy(views).unsqueeze(0).to(runner.device)
        state = runner._normalize_state(self.base._state_tensor(packet["state"]))
        motion = runner._motion_batch(
            packet["motion_features"],
            packet["motion_interval_valid_mask"],
            packet["motion_acceleration_valid_mask"],
        )
        text = runner._get_text_embeddings()
        text_tensor, text_mask = pad_text_embeddings(text)
        age = BridgeAge(
            delta_sim_seconds=(float(packet["sim_time"]) - float(self.bridge_sim_time)) if self.contract["clock_mode"] == "required" else 0.0,
            delta_actions=int(age_actions),
            bridge_valid=not null_bridge,
            state_delta=(state - self.bridge_state).reshape(-1),
        )
        current = CurrentObservation(
            rgb=rgb,
            state=state,
            motion=motion["absolute_motion_features"],
            motion_interval_valid_mask=motion["absolute_motion_interval_valid_mask"],
            motion_acceleration_valid_mask=motion[
                "absolute_motion_acceleration_valid_mask"
            ],
            text_embeddings=text,
            age=age,
            rgb_valid_mask=torch.from_numpy(masks).unsqueeze(0).to(runner.device),
        )
        bridge = self.bridge
        if null_bridge:
            bridge = dataclasses.replace(bridge, null_bridge=True)
        with torch.inference_mode():
            noise = runner._fresh_noisy_actions(generator=packet_noise_generator(packet, runner.device, branch="fast"))
            if self.reuse_action_noise:
                noise = align_replanning_noise(noise, self._previous_fast_noise,
                    packet['action_index'], self._previous_noise_index)
            if self.fast_graph is not None:
                normalized = self.fast_graph.sample(current, bridge, text_tensor, text_mask, noise)
                actions = self.runtime.action_normalizer.denormalize(normalized)
            else:
                memory = wrapper.prepare_fast_memory(current, bridge, text_tensor, text_mask)
                actions = wrapper.sample_fast_actions(current, bridge, memory, noise,
                    denormalize=self.runtime.action_normalizer.denormalize)
            if self.reuse_action_noise:
                self._previous_fast_noise = noise.detach().clone()
                self._previous_noise_index = int(packet['action_index'])
        array = actions.squeeze(0).float().cpu().numpy()
        if array.shape[0] != int(self.runtime.chunk_size):
            raise RuntimeError(
                "fast sample returned the wrong action chunk: "
                f"expected {self.runtime.chunk_size}, got {array.shape}"
            )
        return array


def align_replanning_noise(fresh, previous, action_index, previous_index):
    """Common random numbers for overlapping absolute action positions.

    Reuses sampling noise, never old predicted actions. Current observations can
    still revise every action. It reduces a source of replanning jitter, not a
    guarantee of smoothness or a replacement for RTC prefix constraints.
    """
    if previous is None or previous_index is None or previous.shape != fresh.shape:
        return fresh
    shift = int(action_index) - int(previous_index)
    if not 0 < shift < fresh.shape[1]:
        return fresh
    return torch.cat((previous[:, shift:].to(fresh), fresh[:, -shift:]), dim=1)


def packet_noise_generator(packet, device, *, branch):
    """Scheduling order must not change the random noise of a given query.

    Snapshot the episode seed with the packet so an old in-flight request cannot
    read a new episode's seed after reset. Branches never consume global RNG.
    """
    index = packet["action_index"] if branch == "world" else packet["_query_index"]
    key = f"{packet['_episode_seed']}:{branch}:{int(index)}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(seed)


def capture_observation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Copy arrays and tensors before another thread or the simulator reuses them."""

    captured: dict[str, Any] = {}
    devices = set()

    def clone(value):
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                devices.add(value.device)
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    for key, value in packet.items():
        captured[key] = clone(value)
    events = []
    for device in devices:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        events.append(event)
    captured["_cuda_ready_events"] = tuple(events)
    return captured


def validate_native_scheduling(contract: dict, scheduling: str) -> None:
    """Reject the legacy prefetch cadence for the trained native budget.

    The legacy worker prefetches at age 12 and would rebuild every 12 actions
    in steady state. Native training and latency accounting require one slow
    source per 16 executed actions. Concurrent source publication needs its own
    age/cadence contract before it can be enabled for native checkpoints.
    """
    if contract.get("native_multirate") and scheduling != "controlled_ratio":
        raise RuntimeError(
            "native multirate requires controlled_ratio scheduling: the legacy "
            "independent_worker does not preserve the trained 16-action source cadence"
        )


def plan_worker_bridge(
    *,
    action_index: int,
    bridge_index: int,
    bridge_present: bool,
    max_age: int,
    horizon: int,
    submitted_source: int | None,
) -> tuple[str, int | None]:
    """Decide whether this query blocks, prefetches, or reuses a world bridge.

    Reuse/prefetch within trained age support. A stale result asks the caller
    to wait for a usable refresh; it is not permission for unbounded stale use.
    """

    if horizon < 1:
        raise ValueError("execute horizon must be positive")
    if not bridge_present:
        source = action_index if submitted_source is None else submitted_source
        return "blocking", int(source)
    age = int(action_index) - int(bridge_index)
    if age < 0:
        raise RuntimeError("world source is in the future")
    if age > int(max_age):
        source = action_index if submitted_source is None else submitted_source
        return "stale", int(source)
    next_age = int(action_index) + int(horizon) - int(bridge_index)
    if next_age > int(max_age) and (submitted_source is None or submitted_source < action_index):
        return "prefetch", int(action_index)
    return "use", None


def _default_fast_config() -> dict[str, Any]:
    return {
        "dim": 768,
        "layers": 12,
        "heads": 8,
        "head_dim": 96,
        "text_dim": 4096,
        "state_dim": 14,
        "sim_scale": 1.0,
        "action_scale": 16.0,
    }
