"""Native AdamW with FP32 masters for low-precision forward parameters.

The model keeps its original BF16/FP16 tensors. Masters start from those loaded
values cast to FP32; this cannot recover precision discarded by the native
checkpoint loader. FP32 optimizer accumulation preserves subsequent tiny steps.
"""
from __future__ import annotations

from collections import defaultdict
import copy

import torch
from torch import nn


class FP32MasterAdamW(torch.optim.Optimizer):
    """Eager optimizer with one device per process, including DDP ranks.

    FP32 model parameters go directly to torch.optim.AdamW. Low-precision model
    parameters use FP32 proxy Parameters, so native AdamW also allocates FP32
    moments. Checkpoint loading targets these FP32 proxies, avoiding PyTorch's
    automatic moment/master cast to a BF16 model Parameter.
    """
    schema = "dualrate-fp32-master-adamw-v1"

    def __init__(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=.01, amsgrad=False, maximize=False,
                 *, record_updates=False, sample_values_per_master=32):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, maximize=maximize, foreach=False)
        self._initialized = False
        super().__init__(params, defaults)
        self.record_updates = bool(record_updates)
        self.sample_values_per_master = int(sample_values_per_master)
        if self.sample_values_per_master < 1:
            raise ValueError("sample_values_per_master must be positive")
        self._masters, self._proxies, self._layout = [], [], []
        groups = []
        for group in self.param_groups:
            masters, proxies, layout = [], [], []
            for parameter in group["params"]:
                if parameter.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                    raise TypeError("master AdamW supports only FP32/BF16/FP16 model parameters")
                if parameter.layout != torch.strided:
                    raise TypeError("master AdamW requires dense parameters")
                low = parameter.dtype != torch.float32
                master = nn.Parameter(parameter.detach().float().clone(), requires_grad=True) if low else None
                masters.append(master)
                proxies.append(parameter if master is None else master)
                layout.append(dict(shape=list(parameter.shape), dtype=str(parameter.dtype), master=low))
            self._masters.append(masters)
            self._proxies.append(proxies)
            self._layout.append(layout)
            groups.append(dict(group, params=proxies))
        devices = {p.device for group in self.param_groups for p in group["params"]}
        if len(devices) != 1:
            raise ValueError("engineering master optimizer requires a single parameter device")
        self._adamw = torch.optim.AdamW(groups, **defaults)
        self.last_step_statistics = None
        self._accumulated = None
        self._accumulation_steps = 0
        self._initialized = True
        self._refresh_public_state()

    def add_param_group(self, param_group):
        if getattr(self, "_initialized", False):
            raise RuntimeError("parameter groups are fixed; construct a new explicitly bound optimizer")
        super().add_param_group(param_group)

    def _sync_groups(self):
        for group, internal in zip(self.param_groups, self._adamw.param_groups):
            if group.get("foreach", False) is not False:
                raise ValueError("foreach=False is required for the bounded-memory engineering optimizer")
            for key, value in group.items():
                if key != "params":
                    internal[key] = value

    def _refresh_public_state(self):
        # Inspection through optimizer.state[model_parameter] sees FP32 master
        # and moment tensors; serialization remains explicit and nonduplicated.
        public = defaultdict(dict)
        for group, proxies, masters in zip(self.param_groups, self._proxies, self._masters):
            for original, proxy, master in zip(group["params"], proxies, masters):
                state = dict(self._adamw.state.get(proxy, {}))
                if master is not None:
                    state["master"] = master.detach()
                if state:
                    public[original] = state
        self.state = public

    def _samples(self, *, forward=False):
        values = []
        for group, masters in zip(self.param_groups, self._masters):
            for original, master in zip(group["params"], masters):
                if master is not None and master.numel():
                    value = original if forward else master
                    values.append(value.detach().reshape(-1)[:self.sample_values_per_master].float())
        return torch.cat(values) if values else None

    def _trainable_samples(self):
        values=[p.detach().reshape(-1)[:self.sample_values_per_master].float()
                for group in self.param_groups for p in group['params'] if p.requires_grad and p.numel()]
        return torch.cat(values) if values else None

    @torch.no_grad()
    def accumulate_gradients(self, scale=1.):
        """Accumulate each already-DDP-reduced microbatch in FP32.

        DDP synchronizes every microbatch. Clearing model .grad after the copy
        prevents the next backward from adding into a rounded BF16 buffer.
        Checkpoints are allowed only after step(), at an update boundary.
        """
        if self._accumulated is None:
            self._accumulated = [[None for _ in group["params"]] for group in self.param_groups]
        for group, buffers in zip(self.param_groups, self._accumulated):
            for index, parameter in enumerate(group["params"]):
                grad = parameter.grad
                if grad is not None:
                    if grad.is_sparse:
                        raise RuntimeError("dense microbatch gradients required")
                    value = grad.detach().float()
                    if buffers[index] is None:
                        buffers[index] = value.clone().mul_(scale)
                    else:
                        buffers[index].add_(value, alpha=scale)
                parameter.grad = None
        self._accumulation_steps += 1

    def accumulated_gradients(self):
        if self._accumulated is None:
            raise RuntimeError("no accumulated microbatch gradients")
        return [(parameter, grad) for group, buffers in zip(self.param_groups, self._accumulated)
                for parameter, grad in zip(group["params"], buffers) if grad is not None]

    @torch.no_grad()
    def clip_accumulated_grad_norm_(self, max_norm, *, total_norm=None):
        gradients = [grad for _, grad in self.accumulated_gradients()]
        if not gradients:
            raise RuntimeError("training update has no gradients")
        # Keep FP32 buffers: assigning an FP32 grad to a BF16 model is illegal.
        norm = (torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
                if total_norm is None else total_norm)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite accumulated gradient norm")
        scale = (float(max_norm) / (norm + 1e-6)).clamp(max=1.)
        for grad in gradients:
            grad.mul_(scale)
        return norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_groups()
        before = self._samples() if self.record_updates else None
        before_forward = self._samples(forward=True) if self.record_updates else None
        before_trainable = self._trainable_samples() if self.record_updates else None
        active_master_tensors = 0
        for group_index, (group, masters, layout) in enumerate(zip(self.param_groups, self._masters, self._layout)):
            for index, (original, master, expected) in enumerate(zip(group["params"], masters, layout)):
                if list(original.shape) != expected["shape"] or str(original.dtype) != expected["dtype"]:
                    raise ValueError("model parameter shape/dtype changed after optimizer construction")
                gradient = original.grad if self._accumulated is None else self._accumulated[group_index][index]
                if master is None:
                    if self._accumulated is not None:
                        original.grad = gradient
                    continue
                if original.device != master.device:
                    raise ValueError("model/master devices differ")
                if original.grad is not None and original.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                master.grad = None if gradient is None else gradient.detach().float()
                active_master_tensors += int(master.grad is not None)
        try:
            self._adamw.step()
            for group, masters in zip(self.param_groups, self._masters):
                for original, master in zip(group["params"], masters):
                    if master is not None:
                        original.copy_(master)
        finally:
            # FP32 gradient copies are needed only during optimizer.step, never
            # retained through the next native forward/backward activation peak.
            for masters in self._masters:
                for master in masters:
                    if master is not None:
                        master.grad = None
            self._accumulated = None
            self._accumulation_steps = 0
        if before is not None:
            after = self._samples()
            after_forward = self._samples(forward=True)
            delta = after - before
            stats = torch.stack((delta.abs().max(), (delta != 0).sum().float(),
                                 (after_forward != before_forward).sum().float())).cpu().tolist()
            self.last_step_statistics = dict(scope="first_values_of_each_master_not_full_parameter_scan",
                sample_values_per_master=self.sample_values_per_master, sampled_elements=before.numel(),
                sampled_max_abs_master_update=stats[0], sampled_changed_master_values=int(stats[1]),
                sampled_changed_forward_values=int(stats[2]), active_master_tensors=active_master_tensors)
        elif self.record_updates:
            self.last_step_statistics = dict(scope="no_low_precision_parameters", sampled_elements=0,
                sampled_max_abs_master_update=0., sampled_changed_master_values=0,
                sampled_changed_forward_values=0, active_master_tensors=0)
        if before_trainable is not None:
            delta=self._trainable_samples()-before_trainable
            stats=torch.stack((delta.abs().max(),(delta!=0).sum().float())).cpu().tolist()
            self.last_step_statistics.update(sampled_max_abs_trainable_update=stats[0],
                sampled_changed_trainable_values=int(stats[1]),sampled_trainable_elements=before_trainable.numel())
        self._refresh_public_state()
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)
        self._accumulated = None
        self._accumulation_steps = 0

    def precision_report(self):
        low = sum(master.numel() for masters in self._masters for master in masters if master is not None)
        total = sum(p.numel() for group in self.param_groups for p in group["params"])
        return dict(kind="fp32_master_adamw", implementation="torch.optim.AdamW_on_FP32_parameters",
            master_initialization="loaded_model_values_cast_fp32_not_recovered_precast_bits",
            low_precision_parameter_elements=low, direct_fp32_parameter_elements=total-low,
            master_dtype="torch.float32", moment_dtype="torch.float32", master_bytes=4*low,
            adam_moment_bytes_after_all_parameters_receive_grad=8*total,
            temporary_master_gradient_bytes_upper_bound=4*low,
            foreach=False, forward_parameter_dtypes_unchanged=True)

    def state_dict(self):
        if self._accumulated is not None:
            raise RuntimeError("checkpoint requires a completed optimizer update, not a partial accumulation")
        self._sync_groups()
        return dict(schema=self.schema, precision=self.precision_report(), layout=copy.deepcopy(self._layout),
            master_params=[[None if master is None else master.detach() for master in masters] for masters in self._masters],
            adamw=self._adamw.state_dict(), last_step_statistics=copy.deepcopy(self.last_step_statistics))

    @torch.no_grad()
    def load_state_dict(self, value):
        if value.get("schema") != self.schema or value.get("layout") != self._layout or value.get("precision") != self.precision_report():
            raise ValueError("optimizer precision/layout identity mismatch; plain BF16 AdamW states cannot be resumed")
        saved = value.get("master_params")
        if not isinstance(saved, list) or len(saved) != len(self._masters):
            raise ValueError("optimizer master group count mismatch")
        # Validate every master and its projected model value before mutation.
        for group, masters, payload in zip(self.param_groups, self._masters, saved):
            if not isinstance(payload, list) or len(payload) != len(masters):
                raise ValueError("optimizer master parameter count mismatch")
            for original, master, stored in zip(group["params"], masters, payload):
                if master is None:
                    if stored is not None:
                        raise ValueError("FP32 direct parameters must not acquire a master")
                elif (not isinstance(stored, torch.Tensor) or stored.dtype != torch.float32 or
                      stored.shape != master.shape or not torch.isfinite(stored).all()):
                    raise ValueError("checkpoint master must be finite FP32 with exact shape")
                elif not torch.equal(stored.to(device=original.device, dtype=original.dtype), original):
                    raise ValueError("checkpoint master projection does not equal the restored forward model")
        internal = value.get("adamw")
        if not isinstance(internal, dict) or len(internal.get("param_groups", [])) != len(self.param_groups):
            raise ValueError("invalid native AdamW state/group count")
        for state in internal.get("state", {}).values():
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state and (not isinstance(state[name], torch.Tensor) or state[name].dtype != torch.float32):
                    raise ValueError("AdamW moments must remain FP32 on resume")
        # Underlying parameters are all FP32: native loader cannot downcast any
        # moment to the BF16 forward-model dtype.
        self._adamw.load_state_dict(internal)
        for masters, payload in zip(self._masters, saved):
            for master, stored in zip(masters, payload):
                if master is not None:
                    master.copy_(stored)
                    master.grad = None
        for public, restored in zip(self.param_groups, self._adamw.param_groups):
            for key in list(public):
                if key != "params" and key not in restored:
                    del public[key]
            public.update({key: val for key, val in restored.items() if key != "params"})
        self.last_step_statistics = copy.deepcopy(value.get("last_step_statistics"))
        self._refresh_public_state()
