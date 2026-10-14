#!/usr/bin/env python3
"""Give each torchrun rank one CUDA UUID before importing its training program.

Global RANK/WORLD_SIZE retain the real distributed collective membership. Local
rank coordinates describe one logical device slot per rank on one physical host.
This is an application-level visibility workaround, not a driver repair.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys

sys.dont_write_bytecode = True


def isolated_environment(original):
    """Pure transformation; raises before target import on an invalid topology."""
    env = dict(original)
    if env.get("WAM_RANK_VISIBILITY") == "isolated":
        raise ValueError("rank visibility was already isolated; refusing to remap")
    assigned = env.get("WAM_ASSIGNED_GPU_UUIDS")
    if assigned is not None and env.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("assigned UUID mode requires the torchrun parent to have empty CUDA visibility")
    uuid_list = assigned if assigned is not None else env.get("CUDA_VISIBLE_DEVICES", "")
    uuids = uuid_list.split(",")
    if (not uuids or len(set(uuids)) != len(uuids) or
            any(not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", item) for item in uuids)):
        raise ValueError("isolated ranks require an ordered list of unique full GPU UUIDs")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    try:
        rank, world, local_rank, local_world = (int(env[key]) for key in required)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("torchrun must supply all global and local rank coordinates") from exc
    if world != len(uuids) or local_world != world or rank != local_rank or not 0 <= rank < world:
        raise ValueError("only one physical host with one original local rank per selected UUID is supported")
    if int(env.get("GROUP_RANK", "0")) != 0 or int(env.get("GROUP_WORLD_SIZE", "1")) != 1:
        raise ValueError("torchrun source topology must contain exactly one physical worker group")
    if env.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("isolated visibility requires explicitly disabled NCCL P2P")
    for key in (*required, "GROUP_RANK", "GROUP_WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
        env["WAM_PHYSICAL_" + key] = env.get(key, "0" if key == "GROUP_RANK" else "1")
    env["WAM_PHYSICAL_ASSIGNED_GPU_UUIDS"] = uuid_list
    env["WAM_LAUNCHER_CUDA_MASKED"] = "1" if assigned is not None else "0"
    env.update(CUDA_VISIBLE_DEVICES=uuids[local_rank], LOCAL_RANK="0", LOCAL_WORLD_SIZE="1",
        GROUP_RANK=str(rank), GROUP_WORLD_SIZE=str(world), WAM_RANK_VISIBILITY="isolated",
        WAM_SELECTED_GPU_UUID=uuids[local_rank], WAM_PHYSICAL_HOST_COUNT="1")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Python entrypoint to execute after rank isolation")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if not target.is_file() or target == Path(__file__).resolve():
        raise ValueError("target must be a different existing Python script")
    # Importing torch even before this point would defeat the visibility guard.
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before the rank visibility wrapper")
    transformed = isolated_environment(os.environ)
    os.environ.update(transformed)
    print(json.dumps(dict(stage="rank_visibility_isolated", host=socket.gethostname(),
        global_rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
        physical_local_rank=int(os.environ["WAM_PHYSICAL_LOCAL_RANK"]),
        local_rank=0, local_world_size=1, gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
        launcher_cuda_masked=os.environ["WAM_LAUNCHER_CUDA_MASKED"] == "1",
        nccl_p2p_disable=os.environ["NCCL_P2P_DISABLE"])), flush=True)
    arguments = args.target_args[1:] if args.target_args[:1] == ["--"] else args.target_args
    sys.argv = [str(target), *arguments]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
