"""Learned-policy collection, with all files deferred until physics has stopped.

Only this factory retains the legacy WAM cache/noise needed by offline relabel.
It shares the bounded 512 MiB raw/bundle writer with the zero-plan collector;
the extra copies are reported separately and are not deployment speed evidence.
"""
from __future__ import annotations

from pathlib import Path

from dynamicwam.models.world_context_feedback import pack_world_context
from dynamicwam.runtime.predictive_feedback_worker import PredictiveFeedbackWorker
from dynamicwam.runtime.buffered_world_context_worker import BufferedWorldContextWorker


class BufferedPredictiveFeedbackWorker(PredictiveFeedbackWorker):
    buffered_collection = True
    causal_input_keys = frozenset(('rgb','rgb_valid_mask','state','source_state','plan',
        'text','text_mask','previous_head','previous_head_mask','age_actions',
        'history_dt','history_valid','plan_age_seconds'))

    def __init__(self,config):
        if config.get('mode')!='predictive' or not config.get('collection_dir'):
            raise ValueError('buffered predictive collection requires predictive mode and a fresh destination')
        if Path(config['collection_dir']).exists():
            raise FileExistsError('collection destination must not exist')
        if type(config.get('samples_per_episode',64)) is not int or config.get('samples_per_episode',64)<1:
            raise ValueError('samples_per_episode must be a positive integer')
        self.pending_records=[];self.pending_bundles={};self.buffer_bytes=0
        self.finalized=False
        super().__init__(config)

    def _retain_collection_source(self,cache,noise):
        # pack_world_context returns owned tensors. Reuse the sampler's original
        # noise exactly; no generator call and no extra backbone invocation.
        self.context=pack_world_context(cache)
        self.source_noise=dict(actions=noise.actions.detach().clone(),
                               future_latent=noise.future_latent.detach().clone())

    def _record(self,packet,inputs,age):
        if set(inputs)-self.causal_input_keys:
            raise ValueError('raw relabel inputs must contain causal legacy observations only')
        if self.context is None or self.source_noise is None:
            raise RuntimeError('collection source bundle was not retained at refresh')
        return BufferedWorldContextWorker._record(self,packet,inputs,age)

    def finalize(self):
        """Worker RPC invokes this only after the physical episode has stopped."""
        return BufferedWorldContextWorker.finalize(self)


def create_worker_policy(config):
    return BufferedPredictiveFeedbackWorker(config)
