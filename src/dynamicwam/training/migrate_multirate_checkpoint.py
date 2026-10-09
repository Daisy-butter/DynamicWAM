"""Checkpoint migration used by multirate training and deployment.

A v2 state dict is prefixed onto ``backbone`` and the live fast conditioner
fills every new key. ``load_state_dict(..., strict=True)`` then runs. Call
``install_full_checkpoint`` on a constructed ``MultirateWAM``; this module does
not build WAN by itself.
"""

from dynamicwam.inference.multirate_model_loader import (
    FAST_ONLY_FORMAT,
    MULTIRATE_CHECKPOINT_FORMAT,
    install_full_checkpoint,
    load_checkpoint_payload,
)

__all__ = [
    "FAST_ONLY_FORMAT",
    "MULTIRATE_CHECKPOINT_FORMAT",
    "install_full_checkpoint",
    "load_checkpoint_payload",
]
