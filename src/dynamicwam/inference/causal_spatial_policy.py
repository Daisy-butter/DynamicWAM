"""Independent causal-spatial deployment: background observation M, live b/Z/A.

The simulator owns the real clock and immutable H2 prefix. A spawned process
owns inference; one private CUDA stream builds observation-only memory while
the fast call uses the latest COMPLETED publication. No legacy policy/model is
loaded, and no CUDA Graph or speed claim is inferred from earlier architectures.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from dynamicwam.runtime.domino_continuous.worker import (
    ProcessInferenceWorker, _SharedMailbox, _REQUEST_CAPACITY, _RESULT_CAPACITY,
    _die_when_parent_dies, _configure_cpu_threads, _run_policy_requests,
)


from dynamicwam.inference.causal_spatial_config import EXPECTED_RUNTIME_CONTRACT, load_deployment


def capture_metadata(packet):
    """Require exposure-frozen commands and original controller availability."""
    commands = np.asarray(packet["_rtc_capture_drive_targets"], dtype=np.float64)
    captured = float(packet["_rtc_capture_wall_perf_s"])
    available = float(packet["_rtc_capture_available_wall_perf_s"])
    if commands.shape != (14,) or not np.isfinite(commands).all():
        raise ValueError("capture drive targets must be a finite fourteen-dimensional vector")
    if not np.isfinite([captured, available]).all() or captured < 0 or available < captured:
        raise ValueError("capture availability must follow the original controller exposure")
    return commands, available


def validate_packet(packet):
    capture_metadata(packet)
    bootstrap = bool(packet.get("_rtc_bootstrap_only", False))
    origin = packet["_rtc_observation_action_index"]
    length = packet["_rtc_prefix_length"]
    if type(origin) is not int or origin < 0 or origin % 2:
        raise ValueError("causal-spatial policy requires H2/R2 query origins")
    if type(length) is not int or length != (0 if bootstrap else 2):
        raise ValueError("only observation bootstrap d0 or trained committed H2 is allowed")
    prefix = np.asarray(packet["_rtc_prefix_actions"], dtype=np.float64)
    offsets = np.asarray(packet["_rtc_action_target_offsets"], dtype=np.float64)
    previous = np.asarray(packet["_rtc_previous_actions"], dtype=np.float32)
    valid = np.asarray(packet["_rtc_previous_valid"], dtype=bool)
    if prefix.shape != (length, 14) or offsets.shape != (16,) or previous.shape != (16, 14) or valid.shape != (16,):
        raise ValueError("invalid causal-spatial RTC packet geometry")
    if not np.isfinite(prefix).all() or not np.isfinite(offsets).all() or not np.isfinite(previous[valid]).all():
        raise ValueError("nonfinite RTC packet")
    if not np.isclose(float(packet["_rtc_action_interval_s"]), .16, atol=1e-10) or not np.allclose(offsets, np.arange(1, 17) * .16, atol=1e-8, rtol=0):
        raise ValueError("causal-spatial query left the immutable .16s endpoint grid")
    # Match training: only two replaceable overlap rows are soft conditioning.
    overlap = np.zeros(16, dtype=bool)
    if not bootstrap:
        overlap[2:4] = valid[2:4]
    previous = np.where(overlap[:, None], previous, 0)
    if bootstrap and origin != 0:
        raise ValueError("memory bootstrap is allowed only before action-grid origin zero")
    if not np.isfinite(float(packet["sim_time"])):
        raise ValueError("capture time must be finite")
    if packet.get("_rtc_contract") != "continuous_world_actual_latency_causal_spatial_h2r2_v1":
        raise ValueError("packet is not from the independently versioned H2 controller")
    return bootstrap, origin, prefix, offsets, previous, overlap


class CausalSpatialPolicy:
    def __init__(self, config_path, checkpoint, *, device="cuda:0", allow_diagnostic=False, cuda_graph=False):
        import torch
        from dynamicwam.inference.causal_spatial_loader import load_full, sha256
        from dynamicwam.action_normalization import RoboTwinQposNormalizer
        from dynamicwam.models.flow_schedule import BridgeSchedule
        from dynamicwam.runtime.causal_spatial_contract import CausalSpatialContract
        self.device = torch.device(device)
        self.model, payload = load_full(checkpoint, self.device, allow_diagnostic=allow_diagnostic)
        self.model.eval().requires_grad_(False)
        self.deployment, self.native_profile, self.config = load_deployment(config_path)
        self.contract = CausalSpatialContract()
        self.checkpoint_sha256 = sha256(checkpoint)
        self.checkpoint_runtime_contract = payload["runtime_contract"]
        if self.checkpoint_runtime_contract != EXPECTED_RUNTIME_CONTRACT:
            raise ValueError("checkpoint runtime contract differs from causal-spatial serving")
        exported = payload["data_identity"]["spatial_contract"]
        if (exported["window_size"] != 4 or exported["window_stride_seconds"] != .32
                or exported["prefix_length"] != 2
                or exported["command_history_semantics"] != "recorded_capture_drive_targets_not_dense_executed_control"
                or exported["future_time_semantics"] != "requested_grid_group_support_end_action_endpoints_8_16"):
            raise ValueError("checkpoint spatial data contract differs from deployment")
        normalization = payload["action_normalization"]
        self.normalizer = RoboTwinQposNormalizer.from_stats(normalization["stats"], epsilon=normalization["epsilon"])
        self.native_config = payload["native_config"]
        del payload
        self.sigmas = torch.cat((BridgeSchedule.from_flow_match().action_sigmas,
                                torch.zeros(1))).to(self.device)
        self._slow_stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="causal-observation-memory")
        self._future = None
        self._closed = False
        self._episode = 0
        self._text = None
        self._instruction = None
        self._warm = False
        self._graph_requested = bool(cuda_graph)
        self._graph = None
        self.vae = None
        self._source_origin = -1
        self.last_policy_timing = {}
        self.warmup_timing = dict(policy="causal_spatial", cuda_graph=False, fixed_simulation_rate=1.)
        self._new_episode()
        gc.collect()

    def _new_episode(self):
        from dynamicwam.runtime.causal_spatial_contract import FiniteObservationWindow, LatestObservationMemory
        self._episode += 1
        episode = "causal-spatial-" + str(self._episode)
        self.window = FiniteObservationWindow(episode, self.contract)
        self.memories = LatestObservationMemory(episode, self.contract)
        self._event = 0
        self._source_origin = -1

    def _sync_stream(self):
        import torch
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        event.synchronize()

    def prepare_instruction(self, instruction):
        import torch
        from dynamicwam.inference.model_loader import _build_t5_encoder
        from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE
        started = time.perf_counter()
        if self._instruction != str(instruction):
            # T5 is outside the timed episode. CPU ownership avoids adding an
            # 11GB language encoder to the live slow/fast/renderer GPU budget.
            encoder = _build_t5_encoder(self.config, "cpu")
            with torch.inference_mode():
                text = encoder([str(self.config["inference"]["scene_prefix"]) + str(instruction)], "cpu")
            self._text = [row.to(self.device, dtype=torch.bfloat16) for row in text]
            self._instruction = str(instruction)
            del encoder, text
            gc.collect()
        if self.vae is None:
            self.vae = Wan2_2_VAE(vae_pth=str(Path(self.config["wan_root"]) / "Wan2.2_VAE.pth"), device=self.device)
        self._sync_stream()
        if not self._warm:
            self._warmup()
            self._warm = True
        self.warmup_timing.update(instruction_prepare_ms=1000 * (time.perf_counter() - started),
            cuda_peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            cuda_peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)))
        self.last_policy_timing = dict(self.warmup_timing, prepared_instruction=True)
        return self.warmup_timing

    def _condition_latent(self, packet):
        import torch
        from dynamicwam.image import resize_with_padding
        from dynamicwam.runtime.ciwam.wam.flow_runner import flow_frames_to_pixels
        size = (int(self.config["common"]["video_height"]), int(self.config["common"]["video_width"]))
        if size != (384, 320):
            raise ValueError("condition VAE preprocessing differs from packed training")
        frame = resize_with_padding(np.asarray(packet["frame"]), size)
        first = torch.from_numpy(frame.astype(np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        flow = flow_frames_to_pixels(np.asarray(packet["flow_frames"]), count=4,
            height=size[0], width=size[1], device=self.device, dtype=torch.float32)
        pixels = torch.cat((flow, (first * 2 - 1).unsqueeze(2)), dim=2)
        condition = self.vae.encode(pixels)
        if tuple(condition.shape) != (1, 48, 2, 24, 20):
            raise ValueError("condition VAE shape differs from training")
        return condition

    def _build_memory(self, packet, token, origin):
        import torch
        started = time.perf_counter()
        done = None
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self._slow_stream), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    condition = self._condition_latent(packet)
                    memory = self.model.encode_memory(condition, self._text,
                        source_time=float(packet["sim_time"]), source_id=token.event_id)
                finally:
                    done = torch.cuda.Event(); done.record(self._slow_stream)
        finally:
            if done is not None:
                done.synchronize()
        finished = time.perf_counter()
        installed = self.memories.publish(token, memory, completed_wall_s=finished, completion_ready=True)
        return dict(origin=origin, source_event=token.event_id, source_time=float(packet["sim_time"]),
                    started=started, finished=finished, milliseconds=1000 * (finished - started), published=installed)

    def _observe(self, packet):
        commands, available = capture_metadata(packet)
        self._event += 1
        return self.window.append_composite(np.asarray(packet["frame"]), event_id=self._event,
            observation_physics_s=float(packet["sim_time"]), available_wall_s=available,
            capture_drive_targets=commands)

    def _slow_status(self):
        if self._future is not None and self._future.done():
            receipt = self._future.result()  # propagate an actual slow failure
            self._future = None
            self._last_slow = receipt
        return getattr(self, "_last_slow", None)

    def _submit_memory(self, packet, observation, origin):
        self._slow_status()
        if self._future is not None:
            return False
        token = self.memories.begin(observation)
        # IPC already owns packet bytes; the only retained producer inputs are
        # observations. No current action proposal or future latent is copied.
        source = {key: np.array(packet[key], copy=True) for key in ("frame", "flow_frames")}
        source["sim_time"] = float(packet["sim_time"])
        self._future = self._executor.submit(self._build_memory, source, token, origin)
        self._source_origin = origin
        return True

    def _model_batch(self, packet, observation, memory, prefix, offsets, previous, previous_valid):
        import torch
        from dynamicwam.models.rtc_prefix import PrefixCondition
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=self.device)[None]
        commands = self.normalizer.normalize(tensor(observation.recorded_capture_commands))
        command_valid = tensor(observation.command_valid, torch.bool)
        commands = torch.where(command_valid[..., None], commands, 0.)
        # The renderer may finish after control has advanced: use the state
        # frozen at the source exposure, as in the capture-indexed dataset.
        initial = self.normalizer.normalize(tensor(packet["_rtc_capture_drive_targets"]))
        normalized_prefix = torch.zeros((1, 16, 14), dtype=torch.float32, device=self.device)
        normalized_prefix[:, :len(prefix)] = self.normalizer.normalize(tensor(prefix))
        previous_tensor = self.normalizer.normalize(tensor(previous))
        previous_mask = tensor(previous_valid, torch.bool)
        previous_tensor = torch.where(previous_mask[..., None], previous_tensor, 0.)
        return dict(initial_state=initial,
            absolute_motion_features=tensor(packet["motion_features"]),
            absolute_motion_interval_valid_mask=tensor(packet["motion_interval_valid_mask"], torch.bool),
            absolute_motion_acceleration_valid_mask=tensor(packet["motion_acceleration_valid_mask"], torch.bool),
            spatial_rgb=tensor(observation.spatial_rgb), spatial_flow=tensor(observation.flow),
            spatial_valid=tensor(observation.valid, torch.bool), spatial_flow_valid=tensor(observation.flow_valid, torch.bool),
            spatial_dt=tensor(observation.dt, torch.float64), recorded_capture_commands=commands, command_valid=command_valid,
            previous_plan=previous_tensor, previous_plan_valid=previous_mask,
            observation_age=torch.tensor([float(packet["sim_time"]) - memory.source.observation_physics_s], dtype=torch.float64, device=self.device),
            action_target_offsets=tensor(offsets, torch.float64), future_target_offsets=tensor(offsets[[7, 15]], torch.float64),
            rtc_prefix=PrefixCondition(normalized_prefix, torch.tensor([len(prefix)], device=self.device, dtype=torch.long)))

    def _warmup(self):
        import torch
        count, height, width = 4, *[int(v) for v in self.config["head_flow"]["compute_size"]]
        packet = dict(frame=np.zeros((360, 320, 3), np.uint8), state=np.zeros(14, np.float32),
            flow_frames=np.zeros((count, height, width, 3), np.uint8),
            motion_features=np.zeros((4, 12), np.float32), motion_interval_valid_mask=np.zeros(4, bool),
            motion_acceleration_valid_mask=np.zeros(4, bool), sim_time=0.,
            _rtc_capture_drive_targets=np.zeros(14, np.float64),
            _rtc_capture_wall_perf_s=time.perf_counter(),
            _rtc_capture_available_wall_perf_s=time.perf_counter())
        observation = self._observe(packet)
        self._submit_memory(packet, observation, 0)
        self._future.result(timeout=180.); self._slow_status()
        memory = self.memories.snapshot(observation_physics_s=0., request_wall_s=time.perf_counter())
        batch = self._model_batch(packet, observation, memory, np.zeros((2, 14), np.float32),
            np.arange(1, 17) * .16, np.zeros((16, 14), np.float32), np.zeros(16, bool))
        noise = (torch.randn((1, 16, 14), device=self.device),
                 torch.randn((1, 48, 2, 12, 10), device=self.device))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):
                baseline = self.model.sample(batch, action_noise=noise[0], future_noise=noise[1],
                                             sigmas=self.sigmas, memory=memory.payload)
            if self._graph_requested:
                from dynamicwam.inference.causal_spatial_graph import JointGraph
                # Producer has completed and no live physical clock has begun.
                self._graph = JointGraph(self.model, batch, memory.payload, noise, self.sigmas)
                replay = self._graph.replay(batch, memory.payload, noise)
                for name in ("actions", "future_latent", "belief_state"):
                    torch.testing.assert_close(replay[name], baseline[name], atol=.002, rtol=.002)
                self.warmup_timing.update(cuda_graph=True, graph_captures=self._graph.captures,
                    graph_eager_parity=True, shared_graph_module="dynamicwam.inference.causal_spatial_graph")
        self._sync_stream()
        self._new_episode()

    def sample(self, packet):
        import torch
        started = time.perf_counter()
        if self._text is None or self.vae is None:
            raise RuntimeError("prepare_instruction must finish before starting the episode clock")
        if packet.get("_reset_policy"):
            self.drain_episode(); self._new_episode()
        bootstrap, origin, prefix, offsets, previous, previous_valid = validate_packet(packet)
        observation = self._observe(packet)
        self._slow_status()
        if bootstrap:
            self._submit_memory(packet, observation, origin)
            self._future.result(timeout=120.); self._slow_status()
            self.last_policy_timing = dict(bootstrap_only=True, bootstrap_actions_executed=False,
                memory=self._last_slow, cuda_graph=self._graph is not None,
                complete_request_ms=1000 * (time.perf_counter() - started))
            return np.repeat(np.asarray(packet["_rtc_capture_drive_targets"], np.float64)[None], 16, axis=0)
        memory = self.memories.snapshot(observation_physics_s=float(packet["sim_time"]), request_wall_s=time.perf_counter())
        launched = False
        if origin - self._source_origin >= 16:
            launched = self._submit_memory(packet, observation, origin)
        batch = self._model_batch(packet, observation, memory, prefix, offsets, previous, previous_valid)
        generator = torch.Generator(device=self.device).manual_seed(int(packet["_sampling_seed"]))
        action_noise = torch.randn((1, 16, 14), device=self.device, generator=generator)
        future_noise = torch.randn((1, 48, 2, 12, 10), device=self.device, generator=generator)
        fast_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self._graph is None:
                outputs = self.model.sample(batch, action_noise=action_noise, future_noise=future_noise,
                    sigmas=self.sigmas, memory=memory.payload)
            else:
                outputs = self._graph.replay(batch, memory.payload, (action_noise, future_noise))
        self._sync_stream()
        # Match training coordinates exactly: denormalize in FP32, never raw BF16.
        actions = self.normalizer.denormalize(outputs["actions"].float()).squeeze(0).cpu().numpy().astype(np.float64)
        if actions.shape != (16, 14) or not np.isfinite(actions).all():
            raise RuntimeError("nonfinite or malformed causal-spatial action plan")
        actions[:len(prefix)] = prefix  # preserve controller-owned original targets exactly
        finished = time.perf_counter()
        self.last_policy_timing = dict(policy="causal_spatial", cuda_graph=self._graph is not None, H=2, R=2, K=10,
            graph_captures=0 if self._graph is None else self._graph.captures,
            graph_replays=0 if self._graph is None else self._graph.replays,
            source_event_id=memory.source.event_id, source_sim_time=memory.source.observation_physics_s,
            source_publication=memory.publication, source_completed_wall_s=memory.completed_wall_s,
            source_consumed_wall_s=fast_started, memory_age_seconds=float(packet["sim_time"]) - memory.source.observation_physics_s,
            slow_launched=launched, slow_in_flight=self._future is not None,
            last_slow=self._slow_status(), spatial_window_times=observation.spatial_times.tolist(),
            command_history_semantics="recorded_capture_drive_targets_not_dense_executed_control",
            soft_plan_valid_rows=int(previous_valid.sum()), future_target_offsets=offsets[[7, 15]].tolist(),
            preprocessing_ms=1000 * (fast_started - started), fast_model_ms=1000 * (finished - fast_started),
            complete_request_ms=1000 * (finished - started))
        return actions

    def drain_episode(self, timeout=120.):
        started = time.perf_counter()
        if self._future is not None:
            self._future.result(timeout=float(timeout)); self._slow_status()
        return dict(slow_quiescent=True, waited_ms=1000 * (time.perf_counter() - started),
                    last_source=self._last_slow if hasattr(self, "_last_slow") else None)

    def close(self):
        if self._closed: return
        try:
            self.drain_episode()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._graph is not None:
                self._sync_stream()
                self._graph.graph.reset()
                self._graph = None
            self._closed = True


def _serve_causal_policy(root, config, checkpoint, infer_index, allow_diagnostic, cuda_graph, requests, results):
    policy = None
    try:
        _die_when_parent_dies()
        import torch
        cpu_runtime = _configure_cpu_threads()
        torch.cuda.set_device(infer_index)
        try:
            policy = CausalSpatialPolicy(config, checkpoint, device=f"cuda:{infer_index}",
                                         allow_diagnostic=allow_diagnostic, cuda_graph=cuda_graph)
        except BaseException as exc:
            results.put(f"{type(exc).__name__}: {exc}"); traceback.print_exc(); return
        results.put(dict(ready=True, metadata=dict(policy="causal_spatial", cpu_runtime=cpu_runtime,
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,python=__import__('sys').version.split()[0]),
            checkpoint_sha256=policy.checkpoint_sha256, cuda_graph=False, cuda_graph_requested=bool(cuda_graph))))
        def infer(packet):
            if "_prepare_instruction" in packet:
                policy.prepare_instruction(packet["_prepare_instruction"])
                return np.empty((0, 14), dtype=np.float32)
            return policy.sample(packet)
        # The existing transport's mode only gates the generic drain control
        # message; it does not instantiate or invoke the old MotionKV policy.
        _run_policy_requests(policy, requests, results, "motionkv_rtc", infer)
    finally:
        try:
            if policy is not None: policy.close()
        finally:
            requests.close(); results.close()


class CausalSpatialProcessWorker(ProcessInferenceWorker):
    """Reuse the tested owned-process/shared-mailbox lifecycle, new policy only."""
    def __init__(self, *, project_root, config_path, checkpoint, infer_index=0, allow_diagnostic=False, cuda_graph=False):
        context = multiprocessing.get_context("spawn")
        self._requests, child_requests = _SharedMailbox.pair(context, _REQUEST_CAPACITY)
        child_results, self._results = _SharedMailbox.pair(context, _RESULT_CAPACITY)
        self._ready = self._results
        self._closed = False
        self._pending_ipc = {}
        self._active_request_id = None
        self._busy = False
        self._next_id = 0
        self.calls = 0
        self.intervals = []
        self.startup_metadata = {}
        self._process = context.Process(target=_serve_causal_policy,
            args=(str(project_root), str(config_path), str(checkpoint), int(infer_index), bool(allow_diagnostic), bool(cuda_graph), child_requests, child_results),
            name="causal-spatial-inference", daemon=True)
        try:
            try:
                self._process.start()
            finally:
                child_requests.close(); child_results.close()
            status = self._wait_until_ready(900.)
            if status: raise RuntimeError(status)
        except BaseException:
            self.close(); raise

    def prepare_instruction(self, instruction, timeout_s=120.):
        result = super().prepare_instruction(instruction, timeout_s)
        measured = result["policy_timing"]
        for key in ("cuda_graph", "graph_captures", "graph_eager_parity", "shared_graph_module"):
            if key in measured: self.startup_metadata[key] = measured[key]
        if self.startup_metadata.get("cuda_graph_requested") and not (
                measured.get("cuda_graph") is True and measured.get("graph_captures") == 1
                and measured.get("graph_eager_parity") is True):
            raise RuntimeError("requested production Graph was not captured and parity checked")
        return result
