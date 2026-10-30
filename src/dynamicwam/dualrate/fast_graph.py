"""Inference-only CUDA replay of the unchanged joint A/Z Euler sampler.

Noise, physical alignment, observation encoding and cache validation stay outside
the graph. Each query copies EVERY tensor used by the captured core. Capture and
real-device eager/replay comparisons must finish before the runtime is activated.
This plain Python helper adds no parameters or state_dict entries.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import torch


def eager_joint_sample(fast, a, z, known, mask, context, fields, contacts, *, steps):
    """Same K Euler updates and final contact forward as DualRateSystem."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('positive integer sampling steps required')
    batch = a.shape[0]
    grid=getattr(fast,'sampling_sigmas',None)
    if grid is not None and len(grid)!=steps:raise ValueError('untrained compact sampling schedule')
    for step in range(steps):
        sigma = a.new_full((batch,), 1 - step / steps) if grid is None else grid[step].expand(batch)
        delta = -1/steps if grid is None else ((grid[step+1] if step+1<steps else 0.)-grid[step])
        result = fast.forward_cached(a, z, sigma, context,
                                     known_actions=known, action_known_mask=mask)
        next_a=a-result['action_velocity']/steps if grid is None else a+result['action_velocity']*delta
        next_z=z-result['future_velocity']/steps if grid is None else z+result['future_velocity']*delta
        a = torch.where(mask[..., None], known, next_a)
        z = torch.where(fields, next_z, z)
    final = fast.forward_cached(a, z, a.new_zeros(batch), context,
                                known_actions=known, action_known_mask=mask)
    return dict(actions=a, future=torch.where(fields, z, torch.zeros_like(z)),
        contact_probability=torch.where(contacts, final['contact_logits'].sigmoid(),
                                        torch.zeros_like(final['contact_logits'])),
        future_valid=context.future_valid, future_field_valid=fields, contact_valid=contacts)


_CONTEXT_FIELDS = ('current_allowed', 'slow_allowed', 'time_features',
                   'action_deviation', 'future_valid')


def _flatten(a, z, known, mask, context, fields, contacts):
    values = dict(a=a, z=z, known=known, mask=mask, fields=fields, contacts=contacts)
    values.update({name: getattr(context, name) for name in _CONTEXT_FIELDS})
    for layer, group in enumerate(context.kv):
        for item, value in enumerate(group):
            values[f'kv_{layer}_{item}'] = value
    return values


def _context(template, values):
    # Input-version checks belong to the actual query context. The private
    # graph storage is intentionally updated before each replay, never reused
    # as a stale public cache. Parameter identity/version checks remain intact.
    return replace(template, inputs={}, versions={},
        kv=tuple(tuple(values[f'kv_{i}_{j}'] for j in range(len(group)))
                 for i, group in enumerate(template.kv)),
        **{name: values[name] for name in _CONTEXT_FIELDS})


def _layout(values, steps):
    return steps, tuple((name, tuple(value.shape), str(value.dtype), str(value.device))
                        for name, value in values.items())


class FastGraphSampler:
    def __init__(self, fast, *, warmup=2, atol=2e-5, rtol=2e-5):
        if fast.training:
            raise ValueError('CUDA sampler requires eval mode')
        if warmup < 1 or atol < 0 or rtol < 0:
            raise ValueError('invalid capture warmup/parity configuration')
        self.fast = fast
        self.warmup, self.atol, self.rtol = int(warmup), float(atol), float(rtol)
        self._slots = {}
        self._lock = threading.Lock()
        self.sealed = False
        grid = getattr(fast, 'sampling_sigmas', None)
        self._schedule_identity = None if grid is None else (id(grid), grid._version)

    def _validate(self, context):
        if self.fast.training or torch.is_grad_enabled():
            raise ValueError('CUDA sampler is inference-only; use eval and no_grad')
        grid = getattr(self.fast, 'sampling_sigmas', None)
        identity = None if grid is None else (id(grid), grid._version)
        if identity != self._schedule_identity:
            raise ValueError('sampling schedule changed after graph setup')
        # This includes actual model versions and all original query input
        # mutation guards. Never bypass it with the private capture context.
        self.fast._check_cache(context)

    @staticmethod
    def _copy(slot, values):
        if values.keys() != slot['inputs'].keys():
            raise ValueError('graph input field set changed')
        for name, value in values.items():
            slot['inputs'][name].copy_(value)

    def _eager(self, values, context, steps):
        return eager_joint_sample(self.fast, values['a'], values['z'], values['known'],
            values['mask'], context, values['fields'], values['contacts'], steps=steps)

    def _compare(self, slot, values, template, label):
        expected = self._eager(values, _context(template, values), slot['steps'])
        self._copy(slot, values)
        slot['graph'].replay()
        maxima = {}
        for name, actual in slot['outputs'].items():
            wanted = expected[name]
            if actual.dtype == torch.bool:
                if not torch.equal(actual, wanted):
                    raise RuntimeError(f'CUDA Graph parity failed ({label}/{name}, boolean)')
                maxima[name] = 0.
            else:
                if not torch.isfinite(actual).all() or not torch.isfinite(wanted).all():
                    raise RuntimeError(f'CUDA Graph parity nonfinite ({label}/{name})')
                torch.testing.assert_close(actual, wanted, atol=self.atol, rtol=self.rtol)
                maxima[name] = float((actual-wanted).abs().max().item()) if actual.numel() else 0.
        mask = values['mask']
        if not torch.equal(slot['outputs']['actions'][mask], values['known'][mask]):
            raise RuntimeError('CUDA Graph changed the exact known prefix')
        return dict(case=label, passed=True, max_absolute_error=max(maxima.values()), fields=maxima)

    def _parity_cases(self, slot, original, template):
        receipts = [self._compare(slot, original, template, 'original')]
        # These are buffer-mutation parity tests, not synthetic labels or an
        # assertion about task quality. They never mutate the caller's context.
        for count in range(min(3, original['mask'].shape[1])+1):
            values = dict(original)
            mask = torch.zeros_like(original['mask']);mask[:, :count] = True
            known = original['known'] + .125
            values.update(mask=mask, known=known,
                          a=torch.where(mask[..., None], known, original['a']))
            receipts.append(self._compare(slot, values, template, f'prefix_{count}'))
        for field in ('slow_allowed', 'current_allowed'):
            values = dict(original)
            missing = torch.zeros_like(values[field]);missing[..., 0] = True
            values[field] = missing
            receipts.append(self._compare(slot, values, template, field+'_all_missing'))
        values = dict(original)
        values['time_features'] = original['time_features'] + .03125
        values['action_deviation'] = original['action_deviation'] + .125
        for key in original:
            if key.startswith('kv_'):
                values[key] = original[key] * .875 + .015625
        receipts.append(self._compare(slot, values, template, 'time_deviation_and_all_kv_changed'))
        values = dict(original)
        values['fields'] = original['fields'].clone();values['fields'][..., 0] = False
        values['contacts'] = torch.zeros_like(original['contacts'])
        receipts.append(self._compare(slot, values, template, 'field_and_contact_masks_changed'))
        values = dict(original)
        values['future_valid'] = torch.zeros_like(original['future_valid'])
        values['fields'] = torch.zeros_like(original['fields'])
        values['contacts'] = torch.zeros_like(original['contacts'])
        receipts.append(self._compare(slot, values, template, 'future_entity_masks_missing'))
        self._copy(slot, original)
        return receipts

    def _capture(self, key, values, context, steps):
        if self.sealed:
            raise RuntimeError('new CUDA Graph shape after activation/seal is forbidden')
        device = values['a'].device
        if device.type != 'cuda' or any(value.device != device for value in values.values()):
            raise ValueError('CUDA Graph core inputs must reside on one fast CUDA device')
        static = {name: value.detach().clone() for name, value in values.items()}
        private = _context(context, static)
        caller = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._eager(static, private, steps)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self._eager(static, private, steps)
        caller.wait_stream(stream)
        slot = dict(graph=graph, inputs=static, outputs=outputs, context=private,
                    steps=steps, parameter_versions=context.parameter_versions)
        slot['parity'] = self._parity_cases(slot, values, context)
        # Install only after every real-device comparison passed.
        self._slots[key] = slot
        return slot

    @torch.no_grad()
    def sample(self, a, z, known, mask, context, fields, contacts, *, steps):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('one CUDA Graph replay/capture may be in flight')
        try:
            self._validate(context)
            values = _flatten(a, z, known, mask, context, fields, contacts)
            key = _layout(values, steps)
            slot = self._slots.get(key)
            if slot is None:
                slot = self._capture(key, values, context, steps)
            if slot['parameter_versions'] != context.parameter_versions:
                raise ValueError('fast parameters changed since CUDA Graph capture')
            self._copy(slot, values)
            slot['graph'].replay()
            # A later replay cannot overwrite a previous caller's outputs.
            return {name: value.clone() for name, value in slot['outputs'].items()}
        finally:
            self._lock.release()

    def seal(self, expected_shapes=2):
        if self._lock.locked():
            raise RuntimeError('cannot seal during graph work')
        if len(self._slots) != expected_shapes or not self._slots:
            raise RuntimeError(f'expected {expected_shapes} prepared graph shapes, got {len(self._slots)}')
        if self.fast.training:
            raise ValueError('cannot seal a training-mode sampler')
        for slot in self._slots.values():
            self.fast._check_cache(slot['context'])
        self.sealed = True
        return self.receipt()

    def receipt(self):
        shapes = [dict(steps=slot['steps'], device=str(slot['inputs']['a'].device),
                       world_key_count=slot['inputs']['slow_allowed'].shape[-1],
                       parity_cases=slot['parity']) for slot in self._slots.values()]
        passed = self.sealed and bool(shapes) and all(
            case['passed'] for shape in shapes for case in shape['parity_cases'])
        return dict(enabled=True, sealed=self.sealed, shape_count=len(shapes), passed=passed,
                    parity_passed=passed, steps=sorted({x['steps'] for x in shapes}),
                    atol=self.atol, rtol=self.rtol, shapes=shapes,
                    scope='actual_cuda_eager_vs_graph_core; no deployment_or_task_quality_claim')
