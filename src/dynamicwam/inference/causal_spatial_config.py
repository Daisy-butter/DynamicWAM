"""Strict deployment wrapper; native assets never select the model checkpoint."""
from pathlib import Path
import json


EXPECTED_RUNTIME_CONTRACT = {
    "version": "causal_spatial_v4_h2r2", "dt": .16, "H": 2, "R": 2,
    "chunk_size": 16, "K": 10, "simulation_rate": 1.,
    "window_size": 4, "window_stride_seconds": .32,
    "max_memory_age_seconds": 4.96, "memory_refresh_actions": 16,
    "memory": "observation_only_native_video", "belief": "finite_causal_spatial_window",
    "future": "fresh_joint_action_conditioned_FM",
    "previous_plan": "time_aligned_policy_proposal_valid_overlap_only",
    "prefix_source_training": "expert_demonstration",
    "soft_plan_training": "frozen_native_teacher_on_previous_fast_query_observation",
    "on_policy_ready": False,
    "command_history": "recorded_capture_drive_targets_not_dense_executed_control",
    "normalization": "FP32_before_BF16",
}


def read_deployment(path):
    path = Path(path).resolve()
    deploy = json.loads(path.read_text(encoding="utf-8"))
    if set(deploy) != {"format", "native_profile", "model_loader", "runtime_contract"}:
        raise ValueError("expected a dedicated causal-spatial deployment wrapper, not a training recipe")
    if (deploy["format"] != "causal_spatial_deployment_v1"
            or deploy["model_loader"] != "causal_spatial_strict_checkpoint"
            or deploy["runtime_contract"] != EXPECTED_RUNTIME_CONTRACT):
        raise ValueError("deployment model loader/runtime contract differs")
    native = (path.parent / deploy["native_profile"]).resolve()
    if not native.is_file(): raise FileNotFoundError(native)
    return deploy, native


def load_deployment(path):
    from dynamicwam.config import load_profile
    deploy, native = read_deployment(path)
    profile = load_profile(native)
    config = profile.inference_config()
    if (config["common"] != {"num_video_frames": 8, "video_height": 384, "video_width": 320}
            or config["observation"] != {"head_size": [240, 320], "wrist_size": [120, 160]}
            or config["head_flow"]["count"] != 4
            or config["head_flow"]["compute_size"] != [64, 64]
            or config["inference"]["action_interval_ms"] != 160):
        raise ValueError("native preprocessing differs from causal-spatial training")
    # Only preprocessing and assets survive. The checkpoint argument and its
    # own serialized normalization/config are the sole model authorities.
    for key in ("checkpoint_path", "checkpoint_artifact_id", "checkpoint_manifest"):
        config.pop(key, None)
    return deploy, profile, config
