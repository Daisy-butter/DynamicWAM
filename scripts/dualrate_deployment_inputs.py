"""Deployment-only immutable per-device episode-language caches.

Imports CUDA-capable dependencies only when the guarded worker constructs the
builder. All copies and byte-parity checks finish before episode activation.
"""
import hashlib


def build_cached_online_batch_builder(*args, **kwargs):
    import torch
    from dynamicwam.dualrate.online import OnlineBatchBuilder

    class DeviceCachedOnlineBatchBuilder(OnlineBatchBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._language_sources = dict(self.language)
            self._language_source_versions = {
                name: self.language[name]._version for name in ('context', 'mask')}
            self._language_cache = {}
            self._language_cache_versions = {}
            cpu = {name: self.language[name].detach().cpu().contiguous().clone()
                   for name in ('context', 'mask')}
            digests = {name: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                       for name, value in cpu.items()}
            devices = {}
            for route, destination in (('fast', self.device), ('slow', self.slow_device)):
                destination = torch.device(destination)
                if destination.type == 'cuda' and destination.index is None:
                    destination = torch.device('cuda', torch.cuda.current_device())
                key = str(destination)
                if key not in self._language_cache:
                    local = {name: value.to(destination).unsqueeze(0).contiguous()
                             for name, value in cpu.items()}
                    for name, value in local.items():
                        returned = value.detach().cpu().squeeze(0).contiguous()
                        if value.device != destination or not torch.equal(
                                returned.view(torch.uint8), cpu[name].view(torch.uint8)):
                            raise ValueError('episode language device cache changed exact tensor bytes')
                    self._language_cache[key] = local
                    self._language_cache_versions[key] = {name: value._version for name, value in local.items()}
                devices[route] = dict(device=key, exact_byte_parity=True,
                    context_shape=list(self._language_cache[key]['context'].shape),
                    mask_shape=list(self._language_cache[key]['mask'].shape))
            self.deployment_language_receipt = dict(
                contract='episode_language_precopied_per_device_exact_v1', passed=True,
                copies_completed_before_activation=True, devices=devices,
                context_dtype=str(cpu['context'].dtype), mask_dtype=str(cpu['mask'].dtype),
                context_sha256=digests['context'], mask_sha256=digests['mask'])

        def _batch(self, values, *, device=None):
            destination = torch.device(self.device if device is None else device)
            if destination.type == 'cuda' and destination.index is None:
                destination = torch.device('cuda', torch.cuda.current_device())
            key = str(destination)
            if key not in self._language_cache:
                raise ValueError('uncached online input device')
            names = {'text': 'context', 'source_text': 'context', 'text_valid': 'mask'}
            # Retain original conversion semantics for every nonlanguage input.
            result = super()._batch({name: value for name, value in values.items() if name not in names},
                                    device=destination)
            for name, field in names.items():
                if name not in values:
                    continue
                source, cached = self._language_sources[field], self._language_cache[key][field]
                if (values[name] is not source or source._version != self._language_source_versions[field]
                        or cached._version != self._language_cache_versions[key][field]):
                    raise ValueError('immutable episode language cache/input changed')
                result[name] = cached
            return result

    return DeviceCachedOnlineBatchBuilder(*args, **kwargs)
