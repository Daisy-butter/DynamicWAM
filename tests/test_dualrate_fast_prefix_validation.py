"""CPU regression for one-time validation and per-step hard-prefix clamping."""
import pytest
import torch

import dynamicwam.dualrate.system as system_module
from dynamicwam.dualrate.slow import _prefix
from test_dualrate_system import make_system, generator


def legacy_sample_fast(system, batch, world, *, steps, generator):
    """Frozen pre-optimization sampler: validation occurs on every update."""
    c = system.fast.config
    b, like = batch["state"].shape[0], batch["state"]
    a = system._noise((b, c.chunk_size, c.action_dim), like, generator)
    z = system._noise((b, c.future_tokens, c.future_dim), like, generator)
    z = torch.where(system.future_field_support[None], z, torch.zeros_like(z))
    known, mask = batch["fast_known_actions"], batch["fast_known_mask"]
    a = _prefix(a, known, mask)
    conditions = system.conditions(batch, world)
    context = system.fast.prepare_context(**conditions)
    valid = conditions.get("future_valid", torch.ones(z.shape[:2], dtype=torch.bool, device=z.device))
    fields = valid[..., None] & system.future_field_support[None]
    for step in range(steps):
        sigma = a.new_full((b,), 1 - step / steps)
        result = system.fast.forward_cached(a, z, sigma, context,
            known_actions=known, action_known_mask=mask)
        a = _prefix(a - result["action_velocity"] / steps, known, mask)
        z = torch.where(fields, z - result["future_velocity"] / steps, z)
    final = system.fast.forward_cached(a, z, a.new_zeros(b), context,
        known_actions=known, action_known_mask=mask)
    contacts = valid[..., None] & system.contact_field_support[None]
    return dict(actions=a, future=torch.where(fields, z, torch.zeros_like(z)),
        contact_probability=torch.where(contacts, final["contact_logits"].sigmoid(),
                                        torch.zeros_like(final["contact_logits"])),
        future_valid=valid, future_field_valid=fields, contact_valid=contacts)


@pytest.mark.parametrize("steps", [1, 10])
@pytest.mark.parametrize("lengths", [(0,), (4,), (0, 2, 4)])
@pytest.mark.parametrize("suffix_value", [0., float("nan"), float("inf")])
def test_sampler_exact_legacy_parity_and_rng(steps, lengths, suffix_value):
    system, batch = make_system()
    batch = {key: value.repeat(len(lengths), *([1] * (value.ndim - 1)))
             for key, value in batch.items()}
    mask = torch.arange(4)[None] < torch.tensor(lengths)[:, None]
    batch["fast_known_mask"] = mask
    batch["fast_known_actions"][~mask] = suffix_value
    known_before = batch["fast_known_actions"].clone()
    mask_before = mask.clone()
    with torch.no_grad():
        world = system.world_readout(system.predict_reference(batch, generator=generator()))
        old_rng, new_rng = generator(), generator()
        expected = legacy_sample_fast(system, batch, world, steps=steps, generator=old_rng)
        actual = system.sample_fast(batch, world, steps=steps, generator=new_rng)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert torch.equal(old_rng.get_state(), new_rng.get_state())
    torch.testing.assert_close(batch["fast_known_actions"], known_before,
                               atol=0, rtol=0, equal_nan=True)
    assert torch.equal(mask, mask_before)


@pytest.mark.parametrize("case, message", [
    ("action_shape", "known actions/mask must match"),
    ("mask_shape", "known actions/mask must match"),
    ("mask_dtype", "known_mask must be boolean"),
    ("gap", "contiguous prefix"),
    ("nan", "known action prefix must be finite"),
    ("inf", "known action prefix must be finite"),
])
def test_invalid_prefix_fails_before_conditions_or_context(monkeypatch, case, message):
    system, batch = make_system()
    if case == "action_shape":
        batch["fast_known_actions"] = batch["fast_known_actions"][:, :-1]
    elif case == "mask_shape":
        batch["fast_known_mask"] = batch["fast_known_mask"][:, :-1]
    elif case == "mask_dtype":
        batch["fast_known_mask"] = batch["fast_known_mask"].long()
    elif case == "gap":
        batch["fast_known_mask"][0] = torch.tensor([True, False, True, False])
    else:
        batch["fast_known_actions"][0, 0, 0] = float(case)
    def unexpected(*args, **kwargs):
        pytest.fail("invalid prefix reached condition/context preparation")
    monkeypatch.setattr(system, "conditions", unexpected)
    monkeypatch.setattr(system.fast, "prepare_context", unexpected)
    with pytest.raises(ValueError, match=message):
        system.sample_fast(batch, {}, steps=10, generator=generator())


def test_prefix_validated_once_and_clamped_despite_nonzero_prefix_velocity(monkeypatch):
    system, batch = make_system()
    with torch.no_grad():
        world = system.world_readout(system.predict_reference(batch, generator=generator()))
    for key in ("source_future_latent", "source_action", "fast_action", "future",
                "future_loss_valid", "contact", "contact_valid"):
        batch.pop(key)
    known, mask = batch["fast_known_actions"], batch["fast_known_mask"]
    known_before, mask_before = known.clone(), mask.clone()
    calls, inputs = [], []
    def tracked_prefix(*args):
        calls.append(1)
        return _prefix(*args)
    original = system.fast.forward_cached
    def nonzero_prefix_velocity(a, *args, **kwargs):
        inputs.append(a.detach().clone())
        result = original(a, *args, **kwargs)
        result["action_velocity"] = torch.where(mask[..., None],
            torch.full_like(a, 7.), result["action_velocity"])
        return result
    monkeypatch.setattr(system_module, "_prefix", tracked_prefix)
    monkeypatch.setattr(system.fast, "forward_cached", nonzero_prefix_velocity)
    with torch.no_grad():
        output = system.sample_fast(batch, world, steps=10, generator=generator())
    assert len(calls) == 1
    assert len(inputs) == 11  # Ten updates plus the final contact readout.
    for actions in inputs + [output["actions"]]:
        torch.testing.assert_close(actions[mask], known[mask], atol=0, rtol=0)
    assert torch.equal(known, known_before)
    assert torch.equal(mask, mask_before)
