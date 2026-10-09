"""Check token layout, zero residual, and the two-stage output-projection gradient."""

from __future__ import annotations

import sys

import torch

from dynamicwam.models.fast_conditioner import FastConditioner
from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation


def _current() -> tuple[CurrentObservation, torch.Tensor]:
    rgb = torch.randn(2, 3, 3, 32, 32)
    text = torch.randn(2, 5, 32)
    age = BridgeAge(
        delta_sim_seconds=0.2,
        delta_actions=4,
        bridge_valid=True,
        state_delta=torch.zeros(14),
    )
    current = CurrentObservation(
        rgb=rgb,
        state=torch.zeros(2, 14),
        motion=torch.zeros(2, 4, 12),
        motion_interval_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        motion_acceleration_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        text_embeddings=[],
        age=age,
    )
    return current, text


def main() -> None:
    conditioner = FastConditioner(
        dim=32,
        layers=2,
        heads=4,
        head_dim=8,
        text_dim=32,
        state_dim=14,
    )
    current, text = _current()
    memory = conditioner.prepare_memory(current, text)
    tokens = torch.randn(2, 25, 32)
    residual = conditioner.apply_layer(0, tokens, memory)
    if residual.shape != tokens.shape:
        raise SystemExit(f"residual changed the token layout: {tuple(residual.shape)}")
    if not torch.allclose(residual, tokens):
        raise SystemExit("zero output projection must leave action tokens unchanged")

    loss = residual.square().mean()
    loss.backward()
    output_grad = conditioner.blocks[0].out_proj.weight.grad
    encoder_grad = conditioner.encoder.stem[0].weight.grad
    if output_grad is None or float(output_grad.abs().sum()) == 0.0:
        raise SystemExit("the output projection must receive a gradient on the first step")
    if encoder_grad is not None and float(encoder_grad.abs().sum()) != 0.0:
        raise SystemExit("encoder gradient must stay zero while the output projection is zero")

    with torch.no_grad():
        conditioner.blocks[0].out_proj.weight.add_(0.01)
    conditioner.zero_grad(set_to_none=True)
    opened = conditioner.apply_layer(0, tokens.detach(), conditioner.prepare_memory(current, text))
    opened.square().mean().backward()
    encoder_grad = conditioner.encoder.stem[0].weight.grad
    if encoder_grad is None or float(encoder_grad.abs().sum()) == 0.0:
        raise SystemExit("encoder gradient must appear after the output projection moves")
    print("verified fast conditioner")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_fast_conditioner failed: {exc}", file=sys.stderr)
        raise
