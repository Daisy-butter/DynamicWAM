#!/usr/bin/env python3
"""Train the cross-time world/action model. Does not dispatch stage1鈥搒tage3."""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.training.train.common import build_accelerator, build_arg_parser, setup_logging
from dynamicwam.training.train.multirate_trainer import MultirateTrainer


def finish_training(accelerator, trainer, *, preserve_exception: bool) -> None:
    """Release application objects before the single process-group shutdown.

    No collective is entered here: a failed rank must preserve its original
    error without waiting forever for peers in a successful-path barrier.
    """
    logger = logging.getLogger(__name__)
    errors = []
    operations = []
    if trainer is not None:
        operations.append(("release_trainer", trainer.release_training_resources))
    operations.extend((
        ("free_accelerator", accelerator.free_memory),
        ("collect_application_objects", gc.collect),
        # Accelerate 1.14 end_training already destroys ALL registered groups.
        # Do not add a second destroy_process_group call.
        ("end_training", accelerator.end_training),
    ))
    for stage, operation in operations:
        logger.info("Training teardown: %s begin", stage)
        try:
            operation()
        except BaseException as exc:
            errors.append(exc)
            logger.exception("Training teardown failed at %s", stage)
        else:
            logger.info("Training teardown: %s complete", stage)
    if errors and not preserve_exception:
        raise errors[0]


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
    trainer = None
    try:
        trainer = MultirateTrainer(profile.training, accelerator)
        set_seed(int(profile.training.get("initialization_seed", 20260930)) + accelerator.process_index)
        trainer.run()
        # Only successful ranks enter this barrier. Failed ranks unwind locally
        # and torchrun handles peer termination; cleanup never adds a barrier.
        import torch
        torch.cuda.synchronize(accelerator.device)
        accelerator.wait_for_everyone()
    finally:
        finish_training(accelerator, trainer, preserve_exception=sys.exc_info()[0] is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
