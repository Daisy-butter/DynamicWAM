"""Static inference-only CUDA graph for the temporal feedback head.

Every tensor is copied on every call. Frozen input contents, episode identity,
or tensor version counters are never used to infer cache validity.
"""
from __future__ import annotations

import time
import torch


class FeedbackGraph:
    def __init__(self, head, enabled=True, warmup=3):
        if warmup < 1:
            raise ValueError('graph warmup must be positive')
        self.head, self.enabled, self.warmup = head, bool(enabled), int(warmup)
        self.graph = self.signature = None
        self.stats = dict(captures=0, replays=0, capture_ms=None, eager_calls=0,
                          graph_enabled=False)

    def invalidate(self):
        self.graph = self.signature = None
        for key in ('static_inputs', 'output'):
            if hasattr(self, key):
                delattr(self, key)

    def _forward(self, inputs):
        cuda = inputs['plan'].is_cuda
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
            return self.head(**inputs, validate_inputs=False)

    def _capture(self, inputs):
        self.static_inputs = {key: value.detach().clone() if torch.is_tensor(value) else value
                              for key, value in inputs.items()}
        device = inputs['plan'].device
        start = time.perf_counter()
        with torch.cuda.device(device):
            active = torch.cuda.current_stream(device)
            side = torch.cuda.Stream(device=device)
            side.wait_stream(active)
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._forward(self.static_inputs)
            active.wait_stream(side)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.output = self._forward(self.static_inputs)
        self.graph = graph
        self.stats['captures'] += 1
        self.stats['capture_ms'] = (time.perf_counter() - start) * 1000

    @torch.inference_mode()
    def __call__(self, **inputs):
        if self.head.training or any(parameter.requires_grad for parameter in self.head.parameters()):
            raise RuntimeError('feedback graph requires an eval head with frozen parameters')
        if 'validate_inputs' in inputs:
            raise ValueError('the graph wrapper owns validation')
        self.head.validate_inputs(**inputs)
        inputs = dict(inputs)
        inputs['age_actions'] = torch.as_tensor(inputs['age_actions'], device=inputs['plan'].device)
        if any(not (torch.is_tensor(value) or value is None) for value in inputs.values()):
            raise TypeError('static graph inputs must be tensors or None')
        use_graph = self.enabled and inputs['plan'].is_cuda
        self.stats['graph_enabled'] = use_graph
        if not use_graph:
            self.stats['eager_calls'] += 1
            return self._forward(inputs)
        signature = tuple((key, None if value is None else
                           (tuple(value.shape), str(value.dtype), str(value.device)))
                          for key, value in sorted(inputs.items()))
        try:
            if signature != self.signature:
                self.invalidate()
                self._capture(inputs)
                self.signature = signature
            else:
                for key, value in inputs.items():
                    if value is not None:
                        self.static_inputs[key].copy_(value)
            with torch.cuda.device(inputs['plan'].device):
                self.graph.replay()
            self.stats['replays'] += 1
            # A later replay must not mutate an action already returned to a caller.
            return self.output.clone()
        except Exception as error:
            self.invalidate()
            raise RuntimeError('temporal feedback graph failed; no silent fallback') from error
