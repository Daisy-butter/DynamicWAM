"""Strict real full-checkpoint allocation on CPU, without simulation or CUDA."""
import argparse
import json
from pathlib import Path
import torch

from dynamicwam.dualrate.factory import build_system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(Path(args.config).read_text())
    model = build_system(config, "cpu")
    model.train()
    assert all(p.device.type == "cpu" for p in model.parameters())
    native = model.slow.native
    assert native.config.wan_frozen is False
    assert all(p.requires_grad for p in native.compact_wan.video_model.wan_model.parameters())
    assert all(p.requires_grad for p in native.action_expert.parameters())
    assert all(p.requires_grad for p in native.absolute_motion_tokens.parameters())
    value = {
        "strict_full_native_load": True, "cpu_only": True,
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "fast_parameters": sum(p.numel() for p in model.fast.parameters()),
        "observation_parameters": sum(p.numel() for p in model.observation.parameters()),
        "readout_parameters": sum(p.numel() for p in model.world_readout.parameters()),
        "initialization_identity": model.initialization_identity,
        "source_future_latent_shape": list(model.future_latent_shape),
        "full_model_forward": False, "gpu_acceptance": False,
        "formal_training_ready": False,
    }
    Path(args.output).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
