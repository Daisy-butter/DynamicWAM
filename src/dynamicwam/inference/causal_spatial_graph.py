"""Shared capture/replay implementation for acceptance and production serving.

Only graph-owned tensor storage is mutated. In particular each observation
memory publication, its source time/id, and text remain caller-owned snapshots.
Graph capture must finish before any background memory producer is dispatched.
"""
from dataclasses import fields, is_dataclass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def map_tensors(value, fn):
    import torch
    if torch.is_tensor(value): return fn(value)
    if is_dataclass(value):
        return type(value)(**{f.name: map_tensors(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict): return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(map_tensors(v, fn) for v in value)
    if isinstance(value, list): return [map_tensors(v, fn) for v in value]
    return value


def copy_tensors(destination, source):
    import torch
    if torch.is_tensor(destination):
        require(torch.is_tensor(source) and destination.shape == source.shape
                and destination.dtype == source.dtype and destination.device == source.device,
                "CUDA Graph input signature changed")
        destination.copy_(source)
        return
    if is_dataclass(destination):
        require(type(destination) is type(source), "dataclass input signature changed")
        for f in fields(destination): copy_tensors(getattr(destination, f.name), getattr(source, f.name))
        return
    if isinstance(destination, dict):
        require(isinstance(source, dict) and destination.keys() == source.keys(), "packet key signature changed")
        for key in destination: copy_tensors(destination[key], source[key])
        return
    if isinstance(destination, (list, tuple)):
        require(type(destination) is type(source) and len(destination) == len(source), "sequence signature changed")
        for a, b in zip(destination, source): copy_tensors(a, b)
        return
    require(destination == source, "static Graph metadata changed")


class JointGraph:
    def __init__(self, model, packet, memory, noise, sigmas):
        import torch
        require(not model.training, "CUDA Graph serving requires an eval model")
        self.model = model
        self.packet = map_tensors(packet, lambda t: t.clone())
        self.memory = map_tensors(memory, lambda t: t.clone())
        self.noise = map_tensors(noise, lambda t: t.clone())
        self.sigmas = sigmas.clone()
        self.captures = 0
        self.replays = 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3): self.output = self._sample()
        torch.cuda.current_stream().wait_stream(stream)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.cuda.graph(self.graph): self.output = self._sample()
        self.captures += 1

    def _sample(self):
        return self.model.sample(self.packet, action_noise=self.noise[0], future_noise=self.noise[1],
                                 sigmas=self.sigmas, memory=self.memory)

    def replay(self, packet, memory, noise):
        import torch
        from dynamicwam.models.rtc_prefix import validate_prefix_condition
        validate_prefix_condition(packet["rtc_prefix"], max_prefix=2)
        with torch.inference_mode():
            copy_tensors(self.packet, packet)
            copy_tensors(self.memory, memory)
            copy_tensors(self.noise, noise)
            self.graph.replay()
        self.replays += 1
        return self.output
