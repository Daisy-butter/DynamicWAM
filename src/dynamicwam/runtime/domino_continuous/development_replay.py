"""Opt-in raw, same-trajectory development replay. No expert observation splicing.

Only successful, fully validated episodes may become self-imitation targets.
This recorder does not create latents or authorize training. Missing future
frames/commands must be rejected by the offline converter, never fabricated.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "dynamicwam_development_replay_v1"
ARRAY_FIELDS = ("frame", "fast_frame", "state", "flow_frames", "motion_features",
    "motion_interval_valid_mask", "motion_acceleration_valid_mask",
    "_rtc_prefix_actions", "_rtc_previous_actions", "_rtc_previous_valid", "_rtc_action_target_offsets")


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):value.update(block)
    return value.hexdigest()


def _hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def validate_development_scope(allowlist_path, formal_manifest_path):
    """Freeze a small dev allowlist against all 700 protected test scenes."""
    allowed=json.loads(Path(allowlist_path).read_text())
    formal=json.loads(Path(formal_manifest_path).read_text())
    rows=allowed.get("scenes",[]); test=formal.get("episodes",formal.get("scenes",[]))
    rows=[dict(row,manifest_sha256=row.get("manifest_sha256",row.get("scene_sha256"))) for row in rows]
    test=[dict(row,manifest_sha256=row.get("manifest_sha256",row.get("scene_sha256"))) for row in test]
    if allowed.get("schema_version")!=1 or allowed.get("purpose")!="development_replay" or not 1<=len(rows)<=16:
        raise ValueError("development replay requires an explicit 1..16 scene allowlist")
    if not (formal.get("complete") is True or (formal.get("kind")=="formal700_scene_exclusion" and formal.get("schema_version")==1)) or len(test)!=700:
        raise ValueError("formal exclusion requires the complete frozen 700-scene manifest")
    def identity(row):
        if not isinstance(row.get("task"),str) or type(row.get("seed")) is not int:
            raise ValueError("scene identity missing task/seed")
        if not all(_hash(row.get(key)) for key in ("manifest_sha256","motion_sha256")):
            raise ValueError("scene identity missing frozen hashes")
        return row["task"],row["seed"]
    test_ids={identity(row) for row in test}
    test_hashes={row[key] for row in test for key in ("manifest_sha256","motion_sha256")}
    dev_ids=set()
    for row in rows:
        key=identity(row)
        if key in dev_ids or key in test_ids or any(row[field] in test_hashes for field in ("manifest_sha256","motion_sha256")):
            raise ValueError("development scene overlaps protected formal scenes or duplicates")
        dev_ids.add(key)
        if not isinstance(row.get("development_scene_id"),str) or not row["development_scene_id"]:
            raise ValueError("development_scene_id must be explicit")
    if len(test_ids)!=700:raise ValueError("formal scene identities are duplicated")
    return dict(schema_version=1,purpose="development_replay",scenes=rows,
        allowlist_sha256=sha256(allowlist_path),formal_manifest_sha256=sha256(formal_manifest_path),
        formal_exclusion_verified=True,formal_scene_count=700)


def select_development_scene(scope, *, task, seed, manifest_sha256, motion_sha256):
    for row in scope["scenes"]:
        if row["task"]==task and row["seed"]==seed:
            if row["manifest_sha256"]!=manifest_sha256 or row["motion_sha256"]!=motion_sha256:
                raise ValueError("development frozen scene hashes changed")
            return dict(row)
    raise ValueError("scene is absent from explicit development allowlist")


def validate_requested_development_scenes(scope, frozen_root, tasks, episodes):
    if frozen_root is None or not 1<=int(episodes)*len(tasks)<=16:
        raise ValueError("development replay must select at most 16 explicitly frozen episodes")
    root=Path(frozen_root);manifest=json.loads((root/"frozen_manifest.json").read_text())
    if manifest.get("complete") is not True:raise ValueError("development scenes are not frozen")
    entries={(row["task"],row["episode_index"]):row for row in manifest["episodes"]}
    for task in tasks:
        for episode in range(int(episodes)):
            row=entries[(task,episode)];directory=root/task/f"episode_{episode:04d}"
            if sha256(directory/"manifest.json")!=row["manifest_sha256"] or sha256(directory/"expert_motion.pkl")!=row["motion_sha256"]:
                raise ValueError("frozen development scene files changed")
            select_development_scene(scope,task=task,seed=row["seed"],
                manifest_sha256=row["manifest_sha256"],motion_sha256=row["motion_sha256"])


class DevelopmentReplayRecorder:
    def __init__(self, directory, *, scope, scene, trajectory_id, episode_id,
                 policy_checkpoint_sha256, policy_source_sha256, max_bytes=1024**3):
        self.directory=Path(directory)
        if not self.directory.is_absolute():raise ValueError("replay output must be an absolute preflight-approved SSD path")
        if not scope.get("formal_exclusion_verified") or not all(_hash(v) for v in (policy_checkpoint_sha256,policy_source_sha256)):
            raise ValueError("replay requires frozen development and policy provenance")
        self.directory.mkdir(parents=True,exist_ok=False)
        self.identity=dict(schema=SCHEMA,trajectory_id=str(trajectory_id),episode_id=str(episode_id),
            development_scene_id=scene["development_scene_id"],scene=scene,
            development_scope=scope,policy_checkpoint_sha256=policy_checkpoint_sha256,
            policy_source_sha256=policy_source_sha256,observation_origin="policy_capture",
            state_semantics="raw_14_drive_targets",command_semantics="actual_continuous_controller_commands")
        self.queries=[];self.query_arrays=[];self.frames=[];self.frame_times=[]
        self.executed=[];self.targets=[];self.bytes=0;self.max_bytes=int(max_bytes);self.closed=False
        self.last_command=None;self.initial_physics_s=None

    def _copy(self,value):
        result=np.array(value,copy=True)
        if result.dtype.hasobject:raise ValueError("object arrays are forbidden in replay")
        self.bytes+=result.nbytes
        if self.bytes>self.max_bytes:raise RuntimeError("development replay memory cap exceeded; no silent sample dropping")
        return result

    def begin(self, physics_s, drive_targets):
        self.initial_physics_s=float(physics_s)
        self.last_command=self._copy(drive_targets).astype(np.float64)
        if self.last_command.shape!=(14,) or not np.isfinite(self.last_command).all():
            raise ValueError("initial hold requires actual raw drive targets")

    def record_frame(self, physics_s, frame, *, for_inference):
        image=np.asarray(frame)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
            raise ValueError("replay camera must be real RGB uint8 composite")
        self.frames.append(self._copy(image));self.frame_times.append((float(physics_s),bool(for_inference)))

    def record_query(self, query_index, packet, *, dispatch_wall_s, dispatch_physics_s, previous_publication_physics_s, motion_history):
        if query_index!=len(self.queries):raise ValueError("replay query order changed")
        arrays={key:self._copy(packet[key]) for key in ARRAY_FIELDS}
        endpoints=motion_history.get("endpoint_capture_times_s",[])
        count=len(arrays["motion_features"])
        if len(endpoints)!=count+2:
            raise ValueError("actual native-flow endpoint timestamps are required for replay")
        times=np.asarray([np.nan if t is None else float(t) for t in endpoints],np.float64)
        arrays["motion_history_start_times"]=self._copy(times[1:-1])
        arrays["motion_history_end_times"]=self._copy(times[2:])
        offsets=arrays["_rtc_action_target_offsets"]
        capture=float(packet["sim_time"]);length=int(packet["_rtc_prefix_length"])
        self.query_arrays.append(arrays)
        self.queries.append(dict(query_index=query_index,path=f"queries/{query_index:06d}.npz",
            instruction=str(packet["instruction"]),capture_time=capture,motion_history_diagnostics=motion_history,
            completed_action_count=int(packet["_rtc_observation_action_index"]),
            query_wall_s=float(dispatch_wall_s),query_physics_s=float(dispatch_physics_s),
            prefix_length=length,prefix_endpoint_times=(capture+offsets[:length]).tolist(),
            previous_endpoint_times=(capture+offsets).tolist(),
            previous_publication_physics_s=previous_publication_physics_s,
            bootstrap_only=bool(packet.get("_rtc_bootstrap_only",False)),
            queue_generation=int(packet.get("_rtc_queue_generation",0)),result_received=False))

    def record_result(self, record, actions):
        index=int(record["query_index"]);row=self.queries[index]
        self.query_arrays[index]["returned_actions"]=self._copy(actions)
        info=record.get("policy_diagnostics",{})
        source=info.get("asynchronous_world",{})
        row.update(result_received=True,ready_wall_s=record["receive_wall_s"],
            ready_physics_s=record["receive_physics_s"],installed=bool(record.get("installed")),
            policy_diagnostics=info,source_capture_time=source.get("source_sim_time"),
            source_completed_action_count=source.get("source_action_index"),
            source_publication_physics_upper_bound=record["receive_physics_s"],
            source_publication_time_semantics="upper_bound_at_result_receipt; exact_monotonic_in_policy_diagnostics")

    def record_step(self, start, end, command, velocity, measured_state):
        held=command is None
        if not held:self.last_command=self._copy(command).astype(np.float64)
        if self.last_command is None:raise RuntimeError("hold command ownership missing")
        self.executed.append((float(start),float(end),self.last_command.copy(),
            np.zeros(14) if velocity is None else self._copy(velocity),self._copy(measured_state),held))

    def record_target(self, action_index, endpoint_time, target):
        self.targets.append((int(action_index),float(endpoint_time),self._copy(target)))

    def finalize(self, *, success, complete, error=None):
        if self.closed:raise RuntimeError("replay is already finalized")
        files={}
        def save_npz(name,**arrays):
            path=self.directory/name;path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,**arrays);files[name]=sha256(path)
        for row,arrays in zip(self.queries,self.query_arrays):save_npz(row["path"],**arrays)
        shape=self.frames[0].shape if self.frames else (0,0,3)
        save_npz("frames.npz",rgb=np.stack(self.frames) if self.frames else np.empty((0,*shape),np.uint8),
            capture_times=np.asarray([r[0] for r in self.frame_times],np.float64),
            for_inference=np.asarray([r[1] for r in self.frame_times],bool))
        def column(rows,index,shape=(),dtype=np.float64):
            return np.asarray([row[index] for row in rows],dtype=dtype).reshape((-1,*shape))
        save_npz("executed.npz",start_times=column(self.executed,0),end_times=column(self.executed,1),
            commands=column(self.executed,2,(14,)),velocities=column(self.executed,3,(14,)),
            measured_states=column(self.executed,4,(14,)),held=column(self.executed,5,dtype=bool))
        save_npz("targets.npz",action_indices=column(self.targets,0,dtype=np.int64),
            endpoint_times=column(self.targets,1),commands=column(self.targets,2,(14,)))
        queries=self.directory/"queries.json";queries.write_text(json.dumps(self.queries,indent=2,allow_nan=False)+"\n")
        files["queries.json"]=sha256(queries)
        manifest=dict(self.identity,files=files,initial_physics_s=self.initial_physics_s,
            success=bool(success),complete=bool(complete),error=error,
            action_target_source="successful_self_imitation" if success and complete else "unavailable_failed_trajectory",
            trainer_ready=False,missing_future_samples="reject; never pad or borrow expert observations",
            frame_count=len(self.frames),query_count=len(self.queries),executed_ticks=len(self.executed),
            completed_targets=len(self.targets),buffer_bytes=self.bytes,
            recording_overhead="included_in_actual_wall_clock; full_camera_history_in_development_only")
        (self.directory/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
        self.closed=True
        return manifest


def load_development_replay(directory, *, require_success=False):
    directory=Path(directory);manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("schema")!=SCHEMA or manifest.get("trainer_ready") is not False:
        raise ValueError("not a raw development replay")
    if not manifest["development_scope"].get("formal_exclusion_verified"):
        raise ValueError("missing formal-test exclusion evidence")
    if require_success and (manifest.get("success") is not True or manifest.get("complete") is not True
            or manifest.get("action_target_source")!="successful_self_imitation"):
        raise ValueError("failed/incomplete trajectory cannot provide behavior-cloning labels")
    for name,digest in manifest["files"].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path)!=digest:
            raise ValueError("replay artifact hash/path mismatch")
    queries=json.loads((directory/"queries.json").read_text())
    identities={(q["capture_time"],q["completed_action_count"]):q for q in queries}
    for q in queries:
        if q.get("result_received"):
            source=(q.get("source_capture_time"),q.get("source_completed_action_count"))
            if source not in identities or source[0]>q["capture_time"]:
                raise ValueError("actual source observation missing or noncausal; cannot substitute current")
            q["source_query_index"]=identities[source]["query_index"]
        publication=q.get("previous_publication_physics_s")
        if publication is not None and publication>q["capture_time"]+1e-9:
            raise ValueError("previous plan was published after observation")
    return manifest,queries
