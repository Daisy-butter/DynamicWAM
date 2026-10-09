"""Two inference graphs separated by the CPU candidate-controller rollout.

The first graph encodes RGB once, predicts a candidate future, and decodes a
bounded residual. The second reuses those fixed observation tokens to forecast
the corrected candidate. There is no second CNN and no cross-query belief reuse.
"""
from __future__ import annotations

import time

import torch
from dynamicwam.models.predictive_feedback import ABLATIONS


ENCODED_KEYS = ('current_tokens', 'previous_head_tokens', 'token_valid', 'previous_valid')


class _StageGraph:
    def __init__(self, head, *, predictor_only, enabled, warmup, ablation):
        self.head = head
        self.predictor_only, self.enabled = bool(predictor_only), bool(enabled)
        self.warmup = int(warmup)
        if ablation not in ABLATIONS:
            raise ValueError('unknown predictive graph ablation')
        self.ablation = ablation
        if self.warmup < 1:
            raise ValueError('positive graph warmup required')
        self.graph = self.signature = None
        self.stats = dict(captures=0, replays=0, eager_calls=0, capture_ms=None,
                          graph_enabled=False)

    def invalidate(self):
        self.graph = self.signature = None
        for name in ('static_inputs', 'output'):
            if hasattr(self, name):
                delattr(self, name)

    def _forward(self, inputs):
        reference = inputs['current_tokens'] if self.predictor_only else inputs['plan']
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=reference.is_cuda):
            if self.predictor_only:
                return self.head.predict_latents(**inputs, validate_inputs=False, ablation=self.ablation)
            return self.head(**inputs, validate_inputs=False, return_aux=True, ablation=self.ablation)

    def _capture(self, inputs, device):
        self.static_inputs = {k:None if v is None else v.detach().clone() for k,v in inputs.items()}
        started = time.perf_counter()
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
        self.stats['capture_ms'] = (time.perf_counter()-started)*1000

    @torch.inference_mode()
    def __call__(self, inputs):
        if self.head.training or any(p.requires_grad for p in self.head.parameters()):
            raise RuntimeError('predictive graph requires a frozen eval head')
        if any(not (torch.is_tensor(v) or v is None) for v in inputs.values()):
            raise TypeError('predictive graph inputs must be tensors or None')
        if self.predictor_only:
            self.head.latent_predictor.validate_inputs(**inputs)
            reference = inputs['current_tokens']
        else:
            self.head.validate_inputs(**inputs)
            reference = inputs['plan']
        use_graph = self.enabled and reference.is_cuda
        self.stats['graph_enabled'] = use_graph
        if not use_graph:
            self.stats['eager_calls'] += 1
            return self._forward(inputs)
        signature = tuple((k, None if v is None else (tuple(v.shape), str(v.dtype), str(v.device)))
                          for k,v in sorted(inputs.items()))
        try:
            if signature != self.signature:
                self.invalidate()
                self._capture(inputs, reference.device)
                self.signature = signature
            else:
                for k,v in inputs.items():
                    if v is not None:
                        self.static_inputs[k].copy_(v)
            with torch.cuda.device(reference.device):
                self.graph.replay()
            self.stats['replays'] += 1
            if isinstance(self.output, dict):
                return {k:v.clone() for k,v in self.output.items()}
            return self.output.clone()
        except Exception as exc:
            self.invalidate()
            raise RuntimeError('predictive feedback CUDA graph failed; no silent eager fallback') from exc


class PredictiveFeedbackGraph:
    def __init__(self, head, enabled=True, warmup=3, ablation='none'):
        self.head = head
        self.ablation = ablation
        self._head = _StageGraph(head, predictor_only=False, enabled=enabled, warmup=warmup, ablation=ablation)
        self._forecast = _StageGraph(head, predictor_only=True, enabled=enabled, warmup=warmup, ablation=ablation)

    @property
    def stats(self):
        return dict(ablation=self.ablation, head=dict(self._head.stats), forecast=dict(self._forecast.stats))

    def invalidate(self):
        self._head.invalidate()
        self._forecast.invalidate()

    def run_head(self, inputs):
        return self._head(dict(inputs))

    def run_forecast(self, encoded, *, history_dt, history_valid, action_path, future_dt):
        inputs = {k:encoded[k] for k in ENCODED_KEYS}
        inputs.update(history_dt=history_dt, history_valid=history_valid,
                      action_path=action_path, future_dt=future_dt)
        return self._forecast(inputs)
