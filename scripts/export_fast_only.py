#!/usr/bin/env python3
"""Write the fast-only view of a multirate checkpoint.

The output has the action expert, motion tokens, and fast conditioner. It does
not contain WAN, VAE, or T5 weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from dynamicwam.inference.multirate_model_loader import (
    MULTIRATE_CHECKPOINT_FORMAT,
    export_fast_only_payload,
    fast_only_matches,
    load_checkpoint_payload,
    load_fast_only_modules,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fast-only multirate weights")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = load_checkpoint_payload(args.checkpoint)
    if payload.get("format") != MULTIRATE_CHECKPOINT_FORMAT:
        raise SystemExit(
            f"expected {MULTIRATE_CHECKPOINT_FORMAT}, got {payload.get('format')!r}"
        )
    if "modules" not in payload:
        raise SystemExit("multirate checkpoint is missing its fast module spec")
    fast_payload = export_fast_only_payload(
        payload["model"],
        module_spec=payload["modules"],
        global_step=int(payload.get("global_step", 0)),
    )
    if not fast_only_matches(payload["model"], fast_payload):
        raise SystemExit("fast-only tensors do not match the full checkpoint")
    load_fast_only_modules(fast_payload)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(fast_payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"exported fast-only weights to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
