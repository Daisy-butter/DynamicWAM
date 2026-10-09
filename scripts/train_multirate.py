#!/usr/bin/env python3
"""Train the cross-time world/action model. Does not dispatch stage1鈥搒tage3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.training.train.common import build_accelerator, build_arg_parser, setup_logging
from dynamicwam.training.train.multirate_trainer import MultirateTrainer


def main() -> int:
    parser = build_arg_parser("DynamicWAM multirate world/action training")
    args = parser.parse_args()
    profile = load_multirate_profile(args.config)
    # Drop incomplete process rounds in the sampler instead of letting Accelerate
    # cycle the first batches so every rank has the same step count.
    accelerator = build_accelerator(
        profile.training, profile.deepspeed, even_batches=False
    )
    setup_logging(str(profile.training["log_level"]), rank=accelerator.process_index)
    from accelerate.utils import set_seed
    set_seed(int(profile.training.get("initialization_seed", 20260930)))
    trainer = MultirateTrainer(profile.training, accelerator)
    set_seed(int(profile.training.get("initialization_seed", 20260930)) + accelerator.process_index)
    trainer.run()
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
