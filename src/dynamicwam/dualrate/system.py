"""Joint training and sampling with explicit predictive/label boundaries."""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .slow import SourceCondition, _prefix


@dataclass(frozen=True)
class LossConfig:
    slow_video: float = 1.0
    slow_action: float = 1.0
    fast_action: float = 1.0
    future_visual: float = 0.25
    future_geometry: float = 1.0
    contact: float = 0.1
    action_conditioned_probability: float = 0.25
    cache_dropout: float = 0.1
    visual_dim: int = 128

    def __post_init__(self):
        if any(v < 0 for v in vars(self).values()):
            raise ValueError("loss settings must be nonnegative")
        if not 0 <= self.action_conditioned_probability <= 1 or not 0 <= self.cache_dropout <= 1:
            raise ValueError("probabilities must be in [0,1]")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = torch.broadcast_to(mask, values.shape)
    # Mask before arithmetic reduction, so unlabelled NaNs cannot poison loss.
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def fm_input(target: Tensor, sigma: Tensor, noise: Tensor) -> Tensor:
    scale = sigma.reshape(-1, *([1] * (target.ndim - 1)))
    return target * (1 - scale) + noise * scale


class DualRateSystem(nn.Module):
    def __init__(self, slow, world_readout, observation, fast, *, future_latent_shape,
                 loss_config=LossConfig(), motion_mean=None, motion_scale=None, codec_id=None,
                 future_field_support=None,contact_field_support=None):
        super().__init__()
        self.slow, self.world_readout = slow, world_readout
        self.observation, self.fast = observation, fast
        self.loss_config = loss_config
        self.test_fixture = bool(getattr(fast.config, "test_fixture", False))
        self.codec_id = codec_id or ("test_fixture_codec_not_real_data" if self.test_fixture else "")
        if not self.codec_id or (not self.test_fixture and self.codec_id.startswith("test_fixture")):
            raise ValueError("production system requires the audited dataset codec identity")
        fshape=(fast.config.future_tokens,fast.config.future_dim)
        cshape=(fast.config.future_tokens,2)
        self.register_buffer("future_field_support",torch.ones(fshape,dtype=torch.bool) if future_field_support is None
                             else torch.as_tensor(future_field_support,dtype=torch.bool))
        self.register_buffer("contact_field_support",torch.ones(cshape,dtype=torch.bool) if contact_field_support is None
                             else torch.as_tensor(contact_field_support,dtype=torch.bool))
        if self.future_field_support.shape!=fshape or self.contact_field_support.shape!=cshape:
            raise ValueError("static trained field capabilities differ from Z/contact layout")
        self.future_latent_shape = tuple(int(x) for x in future_latent_shape)
        if len(self.future_latent_shape) != 4 or min(self.future_latent_shape) < 1:
            raise ValueError("future_latent_shape is explicit [C,T,H,W] from the native codec")
        motion_dim = observation.motion.in_features
        self.register_buffer("motion_mean", torch.tensor(motion_mean if motion_mean is not None else [0.] * motion_dim, dtype=torch.float32))
        self.register_buffer("motion_scale", torch.tensor(motion_scale if motion_scale is not None else [1.] * motion_dim, dtype=torch.float32))
        if self.motion_mean.shape != (motion_dim,) or self.motion_scale.shape != (motion_dim,) or (self.motion_scale <= 0).any():
            raise ValueError("invalid native motion statistics")

    def train(self, mode: bool = True):
        super().train(mode)
        # Native codec wrappers are not uniformly nn.Module; frozen VAE dropout
        # or running statistics must not be enabled by recursive model.train().
        compact = getattr(getattr(self.slow, "native", None), "compact_wan", None)
        video = getattr(compact, "video_model", None)
        vae = getattr(video, "vae", None)
        if isinstance(vae, nn.Module):
            vae.eval()
        vae_model = getattr(vae, "model", None)
        if isinstance(vae_model, nn.Module):
            vae_model.eval()
        return self

    @staticmethod
    def source(batch):
        return SourceCondition(
            condition_latent=batch["source_condition_latent"], text_embeddings=batch["source_text"],
            initial_state=batch["source_initial_state"], absolute_motion_features=batch["source_motion"],
            absolute_motion_interval_valid_mask=batch["source_motion_valid"],
            absolute_motion_acceleration_valid_mask=batch["source_motion_acceleration_valid"])

    @staticmethod
    def _noise(shape, like, generator):
        return torch.randn(shape, device=like.device, dtype=like.dtype, generator=generator)

    def predict_reference(self, batch, *, generator=None):
        """Does not read source_future_latent, clean action suffixes or future labels."""
        self._validate_clocks(batch)
        source = self.source(batch)
        b = source.initial_state.shape[0]
        native = self.slow.native.config
        video_noise = self._noise((b, *self.future_latent_shape), source.condition_latent, generator)
        action_noise = self._noise((b, native.chunk_size, native.action_dim), source.initial_state, generator)
        return self.slow.sample_reference(source, video_noise, action_noise,
            batch["source_known_actions"], batch["source_known_mask"],
            batch["source_action_times"], batch["source_future_intervals"],
            batch["source_time"], batch["completed_time"],
            action_intervals=batch["source_action_intervals"],
            source_metadata=self._spatial_metadata(batch, "source"),
            future_metadata=self._spatial_metadata(batch, "future"))

    def _spatial_metadata(self, batch, prefix):
        return dict(codec_id=self.codec_id,
            temporal_support=batch["source_temporal_support" if prefix == "source" else "source_future_intervals"],
            view_overlap=batch[prefix+"_view_overlap"],valid=batch[prefix+"_token_valid"],
            input_type=batch[prefix+"_input_type"])

    def empty_world(self, batch):
        """Explicit missing-reference condition, trained by reference dropout."""
        state,time = batch["state"],batch["query_time"]
        b=state.shape[0];n=self.fast.config.world_dim
        return dict(world_tokens=state.new_zeros(b,1,n),
            world_valid=torch.zeros(b,1,dtype=torch.bool,device=state.device),
            world_intervals=time[:,None,None].expand(b,1,2).clone(),
            world_types=torch.zeros(b,1,dtype=torch.long,device=state.device),
            source_time=time.clone(),completed_time=time.clone(),
            reference_actions=state.new_empty(b,0,self.fast.config.action_dim),
            reference_action_intervals=time.new_empty(b,0,2))

    @staticmethod
    def mask_reference(world,batch,available):
        """Batched empty-world semantics without relabelling slow supervision.

        Masked tokens use the same null-attention sentinel as empty_world. Its
        query-relative clock convention must also match: age=latency=0, even
        when the independently supervised slow source is an expired old image.
        """
        query=batch['query_time']
        if available.dtype!=torch.bool or available.shape!=query.shape:
            raise ValueError('reference_available must be boolean [B] runtime state')
        result=dict(world)
        result['world_valid']=world['world_valid'] & available[:,None]
        result['source_time']=torch.where(available,world['source_time'],query)
        result['completed_time']=torch.where(available,world['completed_time'],query)
        return result

    def _action_deviation(self,batch,world):
        """Compare only executed history and current immutable prefix, not GT suffix."""
        tokens=world["world_tokens"]
        b,n=tokens.shape[:2];a=self.fast.config.action_dim
        delta=tokens.new_zeros(b,n,a)
        valid=torch.zeros(b,n,dtype=torch.bool,device=tokens.device)
        if world["reference_actions"].shape[1] == 0:
            return delta,valid
        needed={"executed_actions","executed_action_intervals","executed_action_valid","action_intervals"}
        if not needed<=batch.keys():
            if self.test_fixture:
                return delta,valid
            raise ValueError("physical executed trace and current action intervals required")
        acts=batch["executed_actions"]
        times=batch["executed_action_intervals"]
        known=batch["executed_action_valid"]
        if (times.dtype!=torch.float64 or times.shape!=(*acts.shape[:2],2) or
                known.dtype!=torch.bool or known.shape!=acts.shape[:2]):
            raise ValueError("execution trace contains unavailable future actions")
        unavailable=(known & (times[...,1]>batch["query_time"][:,None]+1e-8)).any()
        acts=torch.cat((acts,batch["fast_known_actions"]),dim=1)
        times=torch.cat((times,batch["action_intervals"]),dim=1)
        known=torch.cat((known,batch["fast_known_mask"]),dim=1)
        ref=world["reference_actions"]
        slots=world["world_types"]==2
        layout_invalid=slots.sum(-1)!=ref.shape[1]
        # Fixed-shape matches avoid nonzero/any/allclose host synchronizations
        # for every reference row. First match retains executed-before-prefix
        # precedence and the original differentiable action/reference paths.
        match=known[:,None,:] & ((times[:,None]-world["reference_action_intervals"][:,:,None]).abs().amax(-1)<1e-7)
        matched=match.any(-1)
        if acts.shape[1]:
            first=match.to(torch.long).argmax(-1)
            values=acts.gather(1,first[...,None].expand(-1,-1,a))
            agrees=torch.isclose(acts[:,None],values[:,:,None],atol=1e-5,rtol=0).all(-1)
            conflict=(match & ~agrees).any(-1).any(-1)
        else:
            values=acts.new_zeros(b,ref.shape[1],a)
            conflict=torch.zeros(b,dtype=torch.bool,device=acts.device)
        # One device-to-host check, preserving validation order by batch row.
        checks=torch.cat((unavailable.reshape(1),layout_invalid,conflict,matched.any().reshape(1))).tolist()
        if checks[0]:
            raise ValueError("execution trace contains unavailable future actions")
        for i in range(b):
            if checks[1+i]:
                raise ValueError("soft action tokens and reference plan differ")
            if checks[1+b+i]:
                raise ValueError("execution and commitment disagree at a physical interval")
        if not checks[-1]:
            return delta,valid
        rank=(slots.long().cumsum(-1)-1).clamp_min(0)
        per_ref=torch.where(matched[...,None],values.to(ref.dtype)-ref,0).to(tokens.dtype)
        delta=torch.where(slots[...,None],per_ref.gather(1,rank[...,None].expand(-1,-1,a)),0)
        valid=slots & matched.gather(1,rank)
        return delta,valid

    def conditions(self, batch, world):
        self._validate_clocks(dict(source_time=world["source_time"],
                                  completed_time=world["completed_time"],
                                  query_time=batch["query_time"]))
        # Native slow module accepts raw motion and normalizes internally. Fast
        # normalization is FP32 BEFORE model dtype conversion, using identical stats.
        motion = ((batch["motion"].float() - self.motion_mean) / self.motion_scale).to(self.observation.motion.weight.dtype)
        obs, valid = self.observation(batch["rgb"], motion, batch["state"], batch["text"],
            query_time=batch["query_time"], rgb_times=batch["rgb_times"],
            motion_times=batch["motion_times"], motion_valid=batch["motion_valid"],
            motion_acceleration_valid=batch["motion_acceleration_valid"],
            text_valid=batch["text_valid"],
            **{key:batch[key] for key in ("rgb_valid","rgb_pixel_valid","state_time",
                "spatial_flow","spatial_flow_valid","spatial_flow_times") if key in batch})
        action_delta,action_delta_valid=self._action_deviation(batch,world)
        result = dict(observation_tokens=obs, observation_valid=valid,
            world_tokens=world["world_tokens"], world_valid=world["world_valid"],
            world_intervals=world["world_intervals"],
            world_types=world["world_types"],world_action_delta=action_delta,
            world_action_delta_valid=action_delta_valid,
            # A row is the drive target reached at its interval END. Its start
            # remains the execution/commitment identity, not the prediction time.
            target_times=torch.cat((batch["action_intervals"][..., 1], batch["future_times"]), dim=1),
            query_time=batch["query_time"], source_time=world["source_time"],
            completed_time=world["completed_time"])
        # Task-known entity existence and STATIC trained output capabilities may
        # gate requested slots. A role with no trained continuous/contact fields
        # must not inject meaningless fixed noise into joint A/Z attention.
        # Per-example future visibility/GT label masks NEVER condition this path.
        capability = self.future_field_support.any(-1) | self.contact_field_support.any(-1)
        if "query_entity_valid" in batch:
            presence = batch["query_entity_valid"]
            c = self.fast.config
            if presence.shape != (batch["query_time"].shape[0], c.future_entities) or presence.dtype != torch.bool:
                raise ValueError("query_entity_valid must be boolean [B,entities] task-known presence")
            result["future_valid"] = presence.repeat_interleave(c.future_tokens // c.future_entities,dim=1) & capability[None]
        elif not bool(capability.all()):
            result["future_valid"] = capability[None].expand(batch["query_time"].shape[0], -1)
        return result

    @staticmethod
    def _validate_clocks(batch):
        source, complete, query = (batch[key] for key in ("source_time", "completed_time", "query_time"))
        if (source.ndim != 1 or source.shape != complete.shape or source.shape != query.shape or
                any(x.dtype != torch.float64 or not torch.isfinite(x).all() for x in (source,complete,query)) or
                (source > complete).any() or (complete > query).any()):
            raise ValueError("expected finite float64 [B] source <= completed <= query clocks")
        if 'source_request_time' in batch:
            request,available=batch['source_request_time'],batch['source_available_time']
            for value in (request,available):
                if value.shape!=source.shape or value.dtype!=torch.float64 or not torch.isfinite(value).all():
                    raise ValueError('exposure/request/availability require finite FP64 clocks')
            if ((available<source-1e-6).any() or (request<available-1e-6).any() or (request>complete+1e-6).any()):
                raise ValueError('expected source exposure <= availability <= request <= publication')
            if 'query_available_time' in batch and (batch['query_available_time']>query+1e-6).any():
                raise ValueError('query condition is not yet available')

    @staticmethod
    def _validate_prefix_pair(batch):
        for prefix in ("source", "fast"):
            _prefix(batch[prefix + "_known_actions"], batch[prefix + "_known_actions"], batch[prefix + "_known_mask"])
        st, ft = batch["source_action_times"], batch["action_times"]
        common = (st[:, :, None] - ft[:, None, :]).abs() <= 1e-7
        si,fi=batch["source_action_intervals"],batch["action_intervals"]
        common=common & ((si[:,:,None,1]-fi[:,None,:,1]).abs()<=1e-7)
        active=batch["source_known_mask"] & (si[...,1]>batch["query_time"][:,None]+1e-8)
        represented=(common & batch["fast_known_mask"][:,None,:]).any(-1)
        if (active & ~represented).any():
            raise ValueError("fast prefix drops a still-future source commitment")
        common = common & batch["source_known_mask"][:, :, None] & batch["fast_known_mask"][:, None, :]
        difference = (batch["source_known_actions"][:, :, None] - batch["fast_known_actions"][:, None]).abs()
        if (difference[common] > 1e-5).any():
            raise ValueError("source and fast committed actions disagree at the same physical time")

    def training_loss(self, batch, generator=None, *, action_conditioned=None):
        cfg = self.loss_config
        self._validate_clocks(batch)
        self._validate_prefix_pair(batch)
        source_video, source_action = batch["source_future_latent"], batch["source_action"]
        action, future = batch["fast_action"], batch["future"]
        b, device = action.shape[0], action.device
        if source_video.shape != (b, *self.future_latent_shape):
            raise ValueError("supervision latent shape differs from configured native codec layout")
        for name, value in (("source video", source_video), ("source actions", source_action), ("fast actions", action)):
            if not torch.isfinite(value).all():
                raise ValueError(name + " must be finite")
        for prefix, target in (("source",source_action),("fast",action)):
            known, mask = batch[prefix+"_known_actions"],batch[prefix+"_known_mask"]
            if target.shape != known.shape or not torch.allclose(target[mask],known[mask],atol=1e-5,rtol=0):
                raise ValueError(prefix+" committed prefix differs from the supervised trajectory")
        if not 0 < cfg.visual_dim < future.shape[-1]:
            raise ValueError("visual/geometry split must describe both future fields")
        future_mask = batch["future_loss_valid"]
        if future.shape != (b, self.fast.config.future_tokens, self.fast.config.future_dim):
            raise ValueError("future target shape differs from configured slots/fields")
        if future_mask.shape != future.shape or future_mask.dtype != torch.bool:
            raise ValueError("future_loss_valid must be a boolean field mask matching future")
        if (future_mask & ~self.future_field_support[None]).any():
            raise ValueError("label declares a field outside the training capability manifest")
        contact, contact_mask = batch["contact"], batch["contact_valid"]
        if (contact.shape != (*future.shape[:2], 2) or contact_mask.shape != contact.shape or
                contact_mask.dtype != torch.bool):
            raise ValueError("contact labels need matching boolean [B,Z,2] masks")
        if (contact_mask & ~self.contact_field_support[None]).any():
            raise ValueError("contact label outside declared trained capabilities")
        if not torch.isfinite(future[future_mask]).all():
            raise ValueError("valid future labels must be finite")
        if not (((contact[contact_mask] == 0) | (contact[contact_mask] == 1)).all()):
            raise ValueError("valid contact labels must be binary")
        times = batch["future_times"]
        if times.shape != future.shape[:2] or times.dtype != torch.float64 or not torch.isfinite(times).all():
            raise ValueError("future_times must be finite float64 [B,Z]")
        offsets = times.new_tensor((.1,.2,.4,.8)).repeat(self.fast.config.future_entities)
        if (offsets.numel() != times.shape[1] or
                not torch.allclose(times, batch["query_time"][:, None] + offsets, atol=1e-7, rtol=0)):
            raise ValueError("future labels must follow entity-major .1/.2/.4/.8s physical targets")
        future = torch.where(future_mask, future, torch.zeros_like(future))
        sigmas = torch.rand((3, b), device=device, generator=generator)
        nv, na, nf, nz = [self._noise(x.shape, x, generator) for x in (source_video, source_action, action, future)]

        # Supervised source pass has its own noisy GT and exposes no reusable cache.
        if any(parameter.requires_grad for parameter in self.slow.parameters()):
            supervised = self.slow.supervised_forward(self.source(batch),
                fm_input(source_video, sigmas[0], nv), fm_input(source_action, sigmas[1], na),
                sigmas[0], sigmas[1], batch["source_known_actions"], batch["source_known_mask"])
            losses = {
                "slow_video": (supervised["video_pred"].float() - (nv - source_video).float()).square().mean(),
                "slow_action": masked_mean((supervised["action_pred"].float() - (na - source_action).float()).square(), ~batch["source_known_mask"][..., None]),
            }
        else:
            # Short interface warmup freezes the complete native slow model.
            # Its supervised pass cannot train anything and need not run. The
            # separate source-only prediction below still produces genuine M;
            # readout/current/fast modules retain their ordinary live gradients.
            losses = {"slow_video": action.new_zeros((), dtype=torch.float32),
                      "slow_action": action.new_zeros((), dtype=torch.float32)}
        # A separate source-only prediction creates reference, with gradients live.
        world = self.world_readout(self.predict_reference(batch, generator=generator))
        available=batch.get('reference_available',torch.ones(b,dtype=torch.bool,device=device))
        if available.dtype!=torch.bool or available.shape!=(b,):
            raise ValueError('reference_available must be boolean [B] runtime state')
        keep = (torch.rand((b,), device=device, generator=generator) >= cfg.cache_dropout) & available
        world=self.mask_reference(world,batch,keep)
        conditions = self.conditions(batch, world)
        if action_conditioned is None:
            action_conditioned = torch.rand((b,), device=device, generator=generator) < cfg.action_conditioned_probability
        if action_conditioned.shape != (b,) or action_conditioned.dtype != torch.bool:
            raise ValueError("action_conditioned must be an explicit boolean [B] mode mask")
        # Only the declared auxiliary conditional-world mode sees clean suffixes.
        known_mask = batch["fast_known_mask"] | action_conditioned[:, None]
        known_actions = torch.where(action_conditioned[:, None, None], action, batch["fast_known_actions"])
        # Missing future labels must not become artificial clean zero states.
        noisy_future=torch.where(future_mask,fm_input(future,sigmas[2],nz),nz)
        # Universally unsupported fields are constant padding, not random
        # pseudo-world variables. This STATIC capability is known at inference;
        # sample-specific missing GT above remains noise and never a condition.
        noisy_future=torch.where(self.future_field_support[None],noisy_future,torch.zeros_like(noisy_future))
        result = self.fast(fm_input(action, sigmas[2], nf), noisy_future, sigmas[2],
            action_known_mask=known_mask, known_actions=known_actions, **conditions)
        losses["fast_action"] = masked_mean((result["action_velocity"].float() - (nf - action).float()).square(), ~known_mask[..., None])
        ze = (result["future_velocity"].float() - (nz - future).float()).square()
        d = cfg.visual_dim
        losses["future_visual"] = masked_mean(ze[..., :d], future_mask[..., :d])
        losses["future_geometry"] = masked_mean(ze[..., d:], future_mask[..., d:])
        contact = torch.where(batch["contact_valid"], batch["contact"], torch.zeros_like(batch["contact"]))
        ce = F.binary_cross_entropy_with_logits(result["contact_logits"].float(), contact.float(), reduction="none")
        losses["contact"] = masked_mean(ce, batch["contact_valid"])
        total = sum(getattr(cfg, name) * value for name, value in losses.items())
        return {"loss": total, **losses}

    def forward(self,batch,generator=None):
        return self.training_loss(batch,generator=generator)

    def enable_fast_graph(self):
        """Explicit deployment setup; no parameters, training changes or capture yet."""
        if self.training:
            raise ValueError('fast CUDA Graph deployment requires eval mode')
        from .fast_graph import FastGraphSampler
        if getattr(self, '_fast_graph_sampler', None) is not None:
            raise ValueError('fast CUDA Graph sampler already installed')
        engine = FastGraphSampler(self.fast)
        self._fast_graph_sampler = engine
        return engine

    @torch.no_grad()
    def sample_fast(self, batch, world, *, steps=10, generator=None):
        """Joint A/Z Euler flow with known prefix clamped at EVERY step.

        The caller supplies only timestamped observations, requested time grids,
        known executable actions and latest-complete reference. No label access.
        TimedPlan publication and deadline checks belong to timing.py.
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        c = self.fast.config
        b = batch["state"].shape[0]
        like = batch["state"]
        a = self._noise((b, c.chunk_size, c.action_dim), like, generator)
        z = self._noise((b, c.future_tokens, c.future_dim), like, generator)
        z = torch.where(self.future_field_support[None],z,torch.zeros_like(z))
        known, mask = batch["fast_known_actions"], batch["fast_known_mask"]
        a = _prefix(a, known, mask)
        conditions = self.conditions(batch, world)
        context=self.fast.prepare_context(**conditions)
        valid=conditions.get("future_valid",torch.ones(z.shape[:2],dtype=torch.bool,device=z.device))
        fields=valid[...,None] & self.future_field_support[None]
        contacts=valid[...,None] & self.contact_field_support[None]
        graph = getattr(self, '_fast_graph_sampler', None)
        if graph is not None:
            return graph.sample(a,z,known,mask,context,fields,contacts,steps=steps)
        # Static unsupported fields stay zero in both training and inference;
        # their unsupervised velocities cannot alter joint action generation.
        for step in range(steps):
            sigma = a.new_full((b,), 1 - step / steps)
            result = self.fast.forward_cached(a, z, sigma, context,
                known_actions=known, action_known_mask=mask)
            # The immutable query prefix was validated once before this loop.
            a = torch.where(mask[..., None], known, a - result["action_velocity"] / steps)
            z = torch.where(fields,z - result["future_velocity"] / steps,z)
        final = self.fast.forward_cached(a, z, a.new_zeros(b), context,
            known_actions=known, action_known_mask=mask)
        return dict(actions=a, future=torch.where(fields,z,torch.zeros_like(z)),
            contact_probability=torch.where(contacts,final["contact_logits"].sigmoid(),
                                            torch.zeros_like(final["contact_logits"])),
            future_valid=valid,future_field_valid=fields,contact_valid=contacts)
