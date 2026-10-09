"""CPU regressions for the 2026-09-29 multirate fixes. No model/data downloads."""
from __future__ import annotations

import copy
from collections import Counter, OrderedDict
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.fast_runtime import StandaloneActionCore, validate_runtime_contract
from dynamicwam.models.action_core_ops import pack_world_action_kv
from dynamicwam.models.action_expert import ActionExpert, ActionExpertConfig
from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.models.fast_conditioner import FastConditioner, FastResidualBlock
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.models.multirate_types import BridgeTensorBundle
from dynamicwam.runtime.ciwam.wam.fast_slow_worker import LatestWorldWorker
from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, packet_noise_generator
from dynamicwam.training.data.multirate_sampling import BalancedPairSampler, ProcessGroupAlignedSampler
from dynamicwam.training.data.paired_world_action_dataset import PairedWorldActionDataset
from dynamicwam.training.train.multirate_trainer import MultirateTrainer
from verify_multirate_v2 import sample


def core():
    expert = ActionExpert(ActionExpertConfig(16, 32, 2, 4, 4, 2), dict(dim=16, num_heads=2, head_dim=8))
    # A zero-initialized decoder can make output equality tests vacuous.
    torch.nn.init.normal_(expert.decoder.action_head[0].weight, std=.1)
    motion = AbsoluteMotionTokenModule(dim=16, history_count=2, feature_mean=(0.,)*12, feature_scale=(1.,)*12)
    return StandaloneActionCore(expert, motion, dict(wan_attention=dict(precision='bfloat16', window_sizes=[[-1,-1]]*2)))


def cpu_attention(q, k, v, k_lens, window_size):
    assert window_size == (-1, -1)
    mask = torch.arange(k.shape[1], device=k.device)[None] < k_lens[:,None]
    return F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                                         attn_mask=mask[:,None,None]).transpose(1,2)


class MultirateFixes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)

    def test_real_mp4_decoder_and_handle_cache(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory)/'episode.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (32,32))
            self.assertTrue(writer.isOpened(), 'MP4 codec unavailable; real reader test must run')
            for value in (20, 100, 220):
                writer.write(np.full((32,32,3), value, np.uint8))
            writer.release()
            dataset = PairedWorldActionDataset.__new__(PairedWorldActionDataset)
            dataset.source_root = Path(directory)
            dataset.frame_reader = None
            dataset._captures = OrderedDict()
            dataset._clocks = OrderedDict()
            dataset.max_open_clocks = 1
            row = SimpleNamespace(split='clean', task_name='test', episode_name='ep0', fast_frame=2)
            with patch('dynamicwam.training.data.paired_world_action_dataset.episode_media_paths', return_value=(video, video.with_suffix('.json'))):
                frame = dataset._read_frame(row)
                self.assertGreater(frame.mean(), 200)
                row.fast_frame = 0
                self.assertLess(dataset._read_frame(row).mean(), 40)
            self.assertEqual(len(dataset._captures), 1)
            self.assertFalse(dataset.__getstate__()['_captures'])
            dataset.close()

    def test_accelerate_shards_global_sampler_once_and_resumes_epoch(self):
        from accelerate import skip_first_batches
        from accelerate.data_loader import prepare_data_loader
        records = [SimpleNamespace(task_name='test', episode_index=e, offset_label='steady_state', offset=0) for e in range(8)]
        sampler = BalancedPairSampler(records, samples_per_episode=4, seed=19)
        sampler.set_epoch(3)
        expected = list(sampler)
        streams = []
        for rank in range(2):
            loader = DataLoader(list(range(8)), batch_size=2, sampler=BalancedPairSampler(records, samples_per_episode=4, seed=19))
            prepared = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2, process_index=rank,
                                           split_batches=False, put_on_device=False, rng_types=[], even_batches=False)
            prepared.set_epoch(3)
            batches = [batch.tolist() for batch in prepared]
            streams.append(batches)
            skipped = skip_first_batches(prepared, 2)
            skipped.set_epoch(3)
            self.assertEqual([batch.tolist() for batch in skipped], batches[2:])
        reconstructed = [i for rows in zip(*streams) for row in rows for i in row]
        self.assertEqual(reconstructed, expected)
        self.assertEqual(Counter(reconstructed), {i:4 for i in range(8)})

    def test_process_group_alignment_drops_incomplete_round_without_recycling(self):
        from accelerate.data_loader import prepare_data_loader
        # 5 batches x 2 ranks would leave one leftover batch; Accelerate used to
        # recycle the first batches when even_batches=True. Alignment drops them.
        records = [SimpleNamespace(task_name='test', episode_index=e, offset_label='steady_state', offset=0) for e in range(5)]
        base = BalancedPairSampler(records, samples_per_episode=2, seed=7)
        aligned = ProcessGroupAlignedSampler(base, batch_size=2, num_processes=2)
        aligned.set_epoch(0)
        self.assertEqual(len(aligned), 8)
        self.assertEqual(len(list(aligned)), 8)
        streams = []
        for rank in range(2):
            loader = DataLoader(list(range(10)), batch_size=2, sampler=ProcessGroupAlignedSampler(
                BalancedPairSampler(records, samples_per_episode=2, seed=7), batch_size=2, num_processes=2))
            prepared = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2, process_index=rank,
                                           split_batches=False, put_on_device=False, rng_types=[], even_batches=False)
            prepared.set_epoch(0)
            streams.append([batch.tolist() for batch in prepared])
        self.assertEqual(len(streams[0]), len(streams[1]))
        reconstructed = [i for rows in zip(*streams) for row in rows for i in row]
        self.assertEqual(len(reconstructed), 8)
        self.assertEqual(reconstructed, list(ProcessGroupAlignedSampler(
            BalancedPairSampler(records, samples_per_episode=2, seed=7), batch_size=2, num_processes=2)))

    def test_bridge_publish_does_not_host_sync_on_seq_lens(self):
        seq = torch.tensor([2, 4])
        with patch.object(torch.Tensor, 'all', side_effect=AssertionError('host sync on publish')):
            with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host bool on publish')):
                bridge = BridgeTensorBundle(
                    video_k=(torch.randn(2, 4, 2, 8),),
                    video_v=(torch.randn(2, 4, 2, 8),),
                    seq_lens=seq,
                    grid_sizes={},
                    source_id=0,
                    sim_time=0.0,
                )
        self.assertFalse(bridge.all_video_tokens_valid)
        self.assertFalse(bridge.as_video_cache()['all_video_tokens_valid'])

    def test_episode_reset_changes_noise_seed(self):
        policy = MultiratePolicy.__new__(MultiratePolicy)
        policy.worker = None
        policy.base = SimpleNamespace(reset=lambda: None)
        policy._episode_index = 0
        policy.reset()
        first = policy._episode_seed
        policy.reset()
        second = policy._episode_seed
        self.assertNotEqual(first, second)
        packet = {'_episode_seed': first, 'action_index': 4, '_query_index': 1}
        other = {'_episode_seed': second, 'action_index': 4, '_query_index': 1}
        a = torch.randn(8, generator=packet_noise_generator(packet, 'cpu', branch='fast'))
        b = torch.randn(8, generator=packet_noise_generator(other, 'cpu', branch='fast'))
        self.assertFalse(torch.equal(a, b))

    def test_padded_world_never_drops_tail_action_tokens_or_gradients(self):
        video = torch.tensor([1.,2.,99.,99.]).view(1,4,1,1).requires_grad_()
        actions = torch.tensor([3.,4.,5.]).view(1,3,1,1).requires_grad_()
        k, v, lengths = pack_world_action_kv(video, video, actions, actions, torch.tensor([2]))
        self.assertEqual(lengths.tolist(), [5])
        self.assertEqual(k[0,:5,0,0].tolist(), [1.,2.,3.,4.,5.])
        v[:,:5].sum().backward()
        self.assertEqual(actions.grad.flatten().tolist(), [1.,1.,1.])
        self.assertEqual(video.grad.flatten().tolist(), [1.,1.,0.,0.])

    def test_action_attention_batch_null_parity_and_no_tensor_bool(self):
        model = core()
        tokens = torch.randn(2,3,16)
        modulation = tuple(torch.ones(2,1,1,16) if i==2 else torch.zeros(2,1,1,16) for i in range(6))
        cache = dict(seq_lens=torch.tensor([2,4]), video_k=[torch.randn(2,4,2,8)], video_v=[torch.randn(2,4,2,8)])
        with patch('dynamicwam.models.action_core_ops.flash_attention', cpu_attention):
            with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host bool in inner loop')):
                mixed = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=torch.tensor([True,False]))
                full = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=False)
                empty = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=True)
            torch.testing.assert_close(mixed[0], empty[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(mixed[1], full[1], rtol=1e-5, atol=1e-5)
            perturbed = dict(cache, video_v=[cache['video_v'][0]+10])
            changed = model._action_attention_with_video_cache(tokens, modulation, 0, perturbed)
            self.assertFalse(torch.allclose(full, changed))

    def test_residual_mask_never_reduces_to_host_bool(self):
        block = FastResidualBlock(16,2,8)
        torch.nn.init.normal_(block.out_proj.weight, std=.1)
        tokens, k, v = torch.randn(2,3,16), torch.randn(2,4,16), torch.randn(2,4,16)
        mask = torch.tensor([[1,1,0,0],[1,1,1,0]], dtype=torch.bool)
        with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host bool in residual')):
            result = block(tokens,k,v,mask)
        changed_v = v.clone(); changed_v[~mask] = 1e4
        torch.testing.assert_close(result, block(tokens,k,changed_v,mask))

    def test_bfloat_reference_real_decoder_ten_steps(self):
        model = MultirateWAM(core(), FastConditioner(dim=16,layers=2,heads=2,head_dim=8,text_dim=8,state_dim=4)).bfloat16().eval()
        current, bridge, text = sample()
        with patch('dynamicwam.models.action_core_ops.flash_attention',cpu_attention), torch.inference_mode():
            noise = torch.randn(2,2,4)
            result = model.sample_fast_actions(current,bridge,model.prepare_fast_memory(current,bridge,text),noise)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.allclose(result, noise))

    def test_worker_coalesces_active_request_and_ignores_old_episode_count(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def build(packet):
            calls.append(packet)
            if packet == 'old':
                entered.set()
                if not release.wait(3): raise TimeoutError('test gate')
            return packet
        worker = LatestWorldWorker(build)
        try:
            self.assertTrue(worker.request(7,'old'))
            self.assertTrue(entered.wait(3))
            self.assertFalse(worker.request(7,'duplicate'))
            worker.reset()
            self.assertTrue(worker.request(0,'new'))
            release.set()
            self.assertEqual(worker.wait_for(0,3).value, 'new')
            self.assertEqual(worker.completed_count(), 1)
            self.assertEqual(calls, ['old','new'])
        finally:
            release.set()
            self.assertTrue(worker.close())

    def test_stale_query_waits_for_bridge_within_trained_support(self):
        for submitted, expected_source in ((12,12),(0,16),(None,16)):
            with self.subTest(submitted=submitted):
                requests, waits = [], []
                worker = SimpleNamespace(latest=lambda:None,
                    request=lambda source,packet:requests.append(source),
                    wait_for=lambda source: (waits.append(source) or SimpleNamespace(value=('new',torch.zeros(4),1.,source))))
                policy = MultiratePolicy.__new__(MultiratePolicy)
                policy.worker=worker; policy.scheduling='independent_worker'; policy.contract={'execute_horizon':4,'max_age_actions':12}
                policy.base=SimpleNamespace(runner=SimpleNamespace(_get_text_embeddings=lambda:[]))
                policy.bridge='old'; policy.bridge_state=torch.zeros(4); policy.bridge_action_index=0
                policy._submitted_source=submitted; policy._episode_seed=19
                policy.over_age_queries=policy.blocking_refreshes=0
                policy._sample_fast=lambda packet,age,**kwargs:np.array([age,kwargs['null_bridge']])
                result=policy.query({'sim_time':2.}, SimpleNamespace(execute_horizon=4,every_fast_queries=4,executed_total=16,query_id=4))
                self.assertEqual(waits,[expected_source])
                self.assertLessEqual(result[0],12)
                self.assertEqual(result[1],False)
                self.assertEqual(policy.blocking_refreshes,1)
                self.assertEqual(policy.last_query_trace['bridge_mode'], 'wait_overage')
                self.assertEqual(policy.last_query_trace['age_actions'], int(result[0]))

    def test_scheduled_inline_refresh_is_reported_despite_zero_legacy_waits(self):
        policy = MultiratePolicy.__new__(MultiratePolicy)
        policy.worker = None; policy.scheduling = 'controlled_ratio'
        policy.contract = {'execute_horizon': 4, 'max_age_actions': 12}
        policy.base = SimpleNamespace(runner=SimpleNamespace(_get_text_embeddings=lambda: []))
        policy.bridge = 'old'; policy.bridge_action_index = 0; policy._episode_seed = 19
        policy.blocking_refreshes = 0
        policy._build_bridge_completed = lambda packet: ('new', torch.zeros(4), 2., 16)
        policy._sample_fast = lambda packet, age, **kwargs: np.zeros((16, 14))
        policy.query({'sim_time': 2.}, SimpleNamespace(execute_horizon=4, every_fast_queries=4,
                     executed_total=16, query_id=4, rebuild_bridge=lambda: True))
        self.assertEqual(policy.blocking_refreshes, 0)
        self.assertEqual(policy.last_query_trace['bridge_mode'], 'inline_scheduled')
        self.assertEqual(policy.last_query_trace['age_sim_seconds'], 0.)

    def _same_cadence_policy(self, worker):
        policy = MultiratePolicy.__new__(MultiratePolicy)
        policy.worker = worker
        policy.scheduling = 'independent_worker'
        policy.bridge_wait_budget_ms = 150.0
        policy.contract = {'execute_horizon': 4, 'max_age_actions': 12, 'clock_mode': 'required'}
        policy.base = SimpleNamespace(runner=SimpleNamespace(_get_text_embeddings=lambda: []))
        policy.bridge = 'old'
        policy.bridge_state = torch.zeros(4)
        policy.bridge_sim_time = 1.0
        policy.bridge_action_index = 0
        policy._submitted_source = None
        policy._episode_seed = 19
        policy.over_age_queries = policy.blocking_refreshes = 0
        policy._sample_fast = lambda packet, age, **kwargs: np.array([age, kwargs['null_bridge']])
        return policy

    def test_same_cadence_accepts_full_chunk_when_the_lead_stays_in_support(self):
        ready = SimpleNamespace(source_index=12, value=('bridge', torch.zeros(4), 1.0, 12))
        waits = []
        worker = SimpleNamespace(
            latest=lambda: ready,
            request=lambda source, packet: None,
            wait_for=lambda source, timeout=120: waits.append((source, timeout)),
        )
        policy = self._same_cadence_policy(worker)
        policy._submitted_source = 12
        result = policy.query(
            {'sim_time': 2.},
            SimpleNamespace(execute_horizon=16, every_fast_queries=1, executed_total=16,
                            query_id=1, bridge_lead_actions=4),
        )
        self.assertEqual(int(result[0]), 4)
        self.assertEqual(waits, [])
        self.assertEqual(policy.last_query_trace['bridge_mode'], 'reuse')
        self.assertEqual(policy.blocking_refreshes, 0)

    def test_deploy_horizon_without_an_in_support_lead_is_rejected(self):
        policy = self._same_cadence_policy(worker=None)
        with self.assertRaises(RuntimeError):
            policy.query(
                {'sim_time': 2.},
                SimpleNamespace(execute_horizon=16, every_fast_queries=1, executed_total=16,
                                query_id=1, bridge_lead_actions=0),
            )

    def test_same_cadence_keeps_waiting_when_the_installed_bridge_is_over_age(self):
        calls = []

        def wait_for(source, timeout=120):
            calls.append(timeout)
            if timeout < 10:
                raise TimeoutError('budget')
            return SimpleNamespace(value=('new', torch.zeros(4), 1.5, source))

        worker = SimpleNamespace(latest=lambda: None, request=lambda source, packet: None, wait_for=wait_for)
        policy = self._same_cadence_policy(worker)
        policy._submitted_source = 12
        result = policy.query(
            {'sim_time': 2.},
            SimpleNamespace(execute_horizon=16, every_fast_queries=1, executed_total=16,
                            query_id=1, bridge_lead_actions=4),
        )
        self.assertEqual(calls[0], 0.15)
        self.assertGreater(calls[1], 10)
        self.assertEqual(int(result[0]), 4)
        self.assertEqual(policy.last_query_trace['bridge_mode'], 'wait_overage')
        self.assertEqual(policy.over_age_queries, 1)

    def test_same_cadence_falls_back_inside_the_wait_budget(self):
        calls = []

        def wait_for(source, timeout=120):
            calls.append(timeout)
            raise TimeoutError('budget')

        worker = SimpleNamespace(latest=lambda: None, request=lambda source, packet: None, wait_for=wait_for)
        policy = self._same_cadence_policy(worker)
        policy.bridge_action_index = 4
        policy._submitted_source = 12
        result = policy.query(
            {'sim_time': 2.},
            SimpleNamespace(execute_horizon=16, every_fast_queries=1, executed_total=16,
                            query_id=1, bridge_lead_actions=4),
        )
        self.assertEqual(calls, [0.15])
        self.assertEqual(int(result[0]), 12)
        self.assertEqual(policy.last_query_trace['bridge_mode'], 'reuse')
        self.assertEqual(policy.over_age_queries, 0)

    def test_same_cadence_eval_snapshots_four_actions_before_the_decision(self):
        from dynamicwam.runtime.ciwam.adapters.domino import deploy_policy_multirate_flow as adapter
        from dynamicwam.runtime.ciwam.wam.multirate_control import ControlledMultirateController

        control = ControlledMultirateController(
            execute_horizon=16, every_fast_queries=1, chunk_size=16, bridge_lead_actions=4,
        )
        snapshots, queries = [], []
        actions = np.zeros((16, 14), dtype=np.float32)

        def query(packet, controller):
            queries.append(controller.executed_total)
            policy.last_query_trace = {'bridge_mode': 'reuse', 'world_wait_ms': 0.0}
            return actions

        def submit(packet, controller):
            snapshots.append(controller.executed_total)
            controller.note_bridge_snapshot()

        policy = SimpleNamespace(
            runtime=SimpleNamespace(observation_config={}),
            query=query,
            submit_bridge_snapshot=submit,
            last_query_trace={},
        )
        motion = SimpleNamespace(
            flow_rgb=[], motion_features=[], interval_valid_mask=[], acceleration_valid_mask=[],
        )
        model = SimpleNamespace(
            bind_env=lambda env: None,
            policy=policy,
            flow=SimpleNamespace(push=lambda *a, **kw: None, observation=lambda: motion),
            controller=control,
            stepper=SimpleNamespace(execute=lambda action: None, action_interval_s=0.1),
            pace_action_interval=False,
            query_ms=[],
            query_includes_bridge=[],
            query_traces=[],
        )
        env = SimpleNamespace(_scene_step_clock=SimpleNamespace(snapshot=lambda: SimpleNamespace(time_seconds=0.)))
        blank = np.zeros((1, 1, 3), dtype=np.uint8)
        with patch.object(adapter, 'composite_robotwin_frame', return_value=blank), \
             patch.object(adapter, 'fast_composite_from_observation', return_value=blank), \
             patch.object(adapter, 'head_camera_frame', return_value=None), \
             patch.object(adapter, 'extract_state', return_value=None):
            for _ in range(32):
                adapter.eval(env, model, {})
        self.assertEqual(queries, [0, 16])
        self.assertEqual(snapshots, [12, 28])
        self.assertTrue(all(trace['bridge_mode'] != 'inline_scheduled' for trace in model.query_traces))

    def test_sync_adapter_executes_selected_prefix_and_then_replans(self):
        from collections import deque
        from dynamicwam.runtime.ciwam.adapters.domino import deploy_policy_sync_flow as adapter
        motion = SimpleNamespace(flow_rgb=[], motion_features=[], interval_valid_mask=[], acceleration_valid_mask=[])
        env = SimpleNamespace(_scene_step_clock=SimpleNamespace(snapshot=lambda: SimpleNamespace(time_seconds=0.)))
        for horizon in (4, 16):
            queries, executed = [], []
            actions = np.arange(16 * 14).reshape(16,14)
            model = SimpleNamespace(bind_env=lambda env: None, pending=deque(), execute_horizon=horizon,
                policy=SimpleNamespace(runtime=SimpleNamespace(chunk_size=16, observation_config={}),
                    sample=lambda packet: (queries.append(packet) or actions)),
                flow=SimpleNamespace(push=lambda *a, **kw: None, observation=lambda: motion),
                stepper=SimpleNamespace(execute=lambda action: executed.append(action)), query_ms=[])
            with patch.object(adapter, 'composite_robotwin_frame', return_value=None), \
                 patch.object(adapter, 'head_camera_frame', return_value=None), \
                 patch.object(adapter, 'extract_state', return_value=None):
                for _ in range(horizon + 1): adapter.eval(env, model, {})
            self.assertEqual(len(queries), 2)
            np.testing.assert_array_equal(executed[:horizon], actions[:horizon])
            np.testing.assert_array_equal(executed[-1], actions[0])

    def test_aligned_validation_keeps_all_rows_and_reports_executed_prefix_by_age(self):
        from accelerate import Accelerator
        # Reproduce Accelerate's dataset/sampler remainder mismatch using its
        # real loader: 13 dataset rows, exactly 8 sampled rows, batch size 4.
        acc = Accelerator(cpu=True)
        loader = acc.prepare(DataLoader(list(range(13)), batch_size=4, sampler=list(range(8))))
        raw_count = trimmed_count = 0
        for rows in loader:
            raw_count += len(acc.gather(rows))
            trimmed_count += len(acc.gather_for_metrics(rows))
        self.assertEqual(raw_count, 8)
        self.assertLess(trimmed_count, raw_count)

        trainer = MultirateTrainer.__new__(MultirateTrainer)
        logs = []
        trainer.config = {'model': {'fast_conditioner': {'sim_scale': 1., 'action_scale': 16.}},
                          'dataset': {'partition_seed': 19, 'execute_horizon': 1}}
        trainer.device = 'cpu'; trainer.phase = 'warmup'; trainer.global_step = 3
        trainer._timesteps = trainer._sigmas = torch.ones(1)
        trainer.val_loader = [{'delta_actions': torch.tensor([0.,4.])} for _ in range(2)]
        trainer.val_sampler = SimpleNamespace(set_epoch=lambda epoch: None)
        trainer.objective = SimpleNamespace(fast_weight=1., action_weight=1., video_weight=0.)
        trainer.accelerator = SimpleNamespace(is_main_process=True, gather=lambda x: x,
            gather_for_metrics=lambda x: self.fail('truncating gather must not be used'),
            log=lambda metrics, step: logs.append(metrics))
        class Predictions(torch.nn.Module):
            def forward(self, *a, **kw):
                return {'fast_pred': torch.tensor([[[1.],[3.]], [[2.],[4.]]]),
                        'action_pred': torch.zeros(2,2,1)}
        trainer.model = Predictions().train()
        prepared = {'fast_target': torch.zeros(2,2,1), 'action_target': torch.zeros(2,2,1)}
        with patch('dynamicwam.training.train.multirate_trainer.prepare_multirate_batch', return_value=prepared):
            trainer._run_validation()
        self.assertTrue(trainer.model.training)
        self.assertEqual(logs[0]['val/samples'], 4)
        self.assertEqual(logs[0]['val/fast'], 7.5)
        self.assertEqual(logs[0]['val/fast_executed_prefix'], 2.5)
        self.assertEqual(logs[0]['val/age_4/fast_executed_prefix'], 4.)
        self.assertEqual(logs[0]['val/age_4/samples'], 2)

    def test_dense_cache_skips_gather_without_a_tensor_reduction(self):
        model = core()
        tokens = torch.randn(2,3,16)
        modulation = tuple(torch.ones(2,1,1,16) if i==2 else torch.zeros(2,1,1,16) for i in range(6))
        cache = dict(seq_lens=torch.tensor([4,4]), video_k=[torch.randn(2,4,2,8)],
                     video_v=[torch.randn(2,4,2,8)])
        with patch('dynamicwam.models.action_core_ops.flash_attention', cpu_attention):
            expected = model._action_attention_with_video_cache(tokens, modulation, 0, cache)
            with patch('dynamicwam.models.action_core_ops.pack_world_action_kv', side_effect=AssertionError('identity gather')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host bool')):
                result = model._action_attention_with_video_cache(tokens, modulation, 0,
                    dict(cache, all_video_tokens_valid=True))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_full_backbone_ablation_rejects_fast_only_and_unknown_weights(self):
        from dynamicwam.inference.multirate_model_loader import (
            extract_multirate_backbone, MULTIRATE_CHECKPOINT_FORMAT, MULTIRATE_CHECKPOINT_VERSION)
        payload = dict(format=MULTIRATE_CHECKPOINT_FORMAT, version=MULTIRATE_CHECKPOINT_VERSION,
                       model={'backbone.action.weight': torch.ones(1), 'fast_conditioner.extra': torch.ones(1)})
        self.assertEqual(list(extract_multirate_backbone(payload)), ['action.weight'])
        with self.assertRaises(RuntimeError): extract_multirate_backbone(dict(payload, format='fast_only'))
        with self.assertRaises(RuntimeError): extract_multirate_backbone(dict(payload, model={'unknown': torch.ones(1)}))

    def test_sampling_seed_can_change_without_changing_episode_partition(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        raw = yaml.safe_load((root/'configs/world_action_multirate_v1.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.yaml'
            raw['pairing']['sampler_seed'] = 20260930
            path.write_text(yaml.safe_dump(raw))
            profile = load_multirate_profile(path)
            config = profile.training
            self.assertEqual(config['dataset']['sampler_seed'], 20260930)
            self.assertEqual(config['dataset']['partition_seed'], raw['pairing']['partition_seed'])

    def test_noise_is_independent_of_thread_scheduling_and_global_rng(self):
        packet={'_episode_seed':71,'action_index':12,'_query_index':3}
        state=torch.get_rng_state().clone()
        first=torch.randn(16,generator=packet_noise_generator(packet,'cpu',branch='fast'))
        world=torch.randn(16,generator=packet_noise_generator(packet,'cpu',branch='world'))
        second=torch.randn(16,generator=packet_noise_generator(packet,'cpu',branch='fast'))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertFalse(torch.equal(first,world))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_profile_rejects_incompatible_world_layout(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        for name in ('v1','v2'):
            load_multirate_profile(root/f'configs/world_action_multirate_{name}.yaml')
        raw=yaml.safe_load((root/'configs/world_action_multirate_v2.yaml').read_text())
        raw['model']['compact_fast']['world_heads']+=1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.yaml';path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError,'world head layout'):
                load_multirate_profile(path)

    def test_contract_rejects_conflicting_duplicate_schedule(self):
        schedule=BridgeSchedule.from_flow_match().to_config()
        contract=dict(version=1,schedule=schedule)
        payload={'config':{'runtime_contract':contract},'modules':{'runtime_contract':copy.deepcopy(contract),'schedule':dict(schedule,shift=1.)}}
        with self.assertRaisesRegex(RuntimeError,'schedules disagree'):
            validate_runtime_contract(payload,SimpleNamespace())

    def test_resume_rejects_sampling_change_before_loading_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer=MultirateTrainer.__new__(MultirateTrainer)
            trainer.config={'resume_from':directory,'flow_matching':dict(shift=5.,sigma_min=0.,extra_one_step=True,training_timesteps=1000)}
            trainer.phase='warmup'
            trainer.dataset=SimpleNamespace(pairing_identity='same')
            loaded=[]
            trainer.accelerator=SimpleNamespace(num_processes=2,load_state=lambda path:loaded.append(path))
            trainer._resume_contract=lambda:{'sampling':{'samples_per_episode':10}}
            progress=dict(pairing_identity='same',phase='warmup',world_size=2,
                schedule=BridgeSchedule.from_flow_match().to_config(),resume_contract={'sampling':{'samples_per_episode':20}})
            (Path(directory)/'multirate_progress.json').write_text(json.dumps(progress))
            with self.assertRaisesRegex(RuntimeError,'sampling/optimization'):
                trainer._resume_if_needed()
            self.assertEqual(loaded,[])

    def test_custom_scheduler_ticks_only_at_optimizer_boundary(self):
        from verify_multirate_training import _Accel, _Backbone, _config, _batch
        from dynamicwam.training.train.common import build_training_flow_scheduler
        from dynamicwam.training.train.multirate_objective import MultirateLossObjective
        from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable
        from dynamicwam.training.utils.scheduler import create_scheduler
        trainer=MultirateTrainer.__new__(MultirateTrainer)
        trainer.config=_config(); trainer.device=torch.device('cpu'); trainer.phase='warmup'
        trainer.model=MultirateWAM(_Backbone(), FastConditioner(dim=32,layers=1,heads=4,head_dim=8,text_dim=8,state_dim=4))
        configure_multirate_trainable(trainer.model,'warmup')
        trainer.optimizer=torch.optim.AdamW([p for p in trainer.model.parameters() if p.requires_grad],lr=.001)
        trainer.scheduler=create_scheduler(trainer.optimizer,total_steps=4,warmup_steps=0,min_lr_ratio=.1)
        trainer.objective=MultirateLossObjective()
        flow=build_training_flow_scheduler(trainer.config)
        trainer._timesteps=flow.timesteps; trainer._sigmas=flow.sigmas
        trainer.accelerator=_Accel(); trainer.accelerator.sync_gradients=False
        self.assertIsNone(trainer.train_step(_batch()))
        self.assertEqual(trainer.scheduler.step_count,0)
        trainer.accelerator.sync_gradients=True
        trainer.train_step(_batch())
        self.assertEqual(trainer.scheduler.step_count,1)
        trainer.accelerator.optimizer_step_was_skipped=True
        trainer.train_step(_batch())
        self.assertEqual(trainer.scheduler.step_count,1)

    def test_registered_custom_scheduler_survives_accelerate_save_load(self):
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState, PartialState
        from dynamicwam.training.utils.scheduler import create_scheduler
        # An earlier test may already have bound PartialState to CUDA. A second
        # Accelerator(cpu=True) is then rejected; reuse that device instead.
        state = AcceleratorState._shared_state
        if state != PartialState._shared_state and getattr(state.get("device"), "type", "cpu") != "cpu":
            accelerator = Accelerator()
        else:
            accelerator = Accelerator(cpu=True)
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        model,optimizer=accelerator.prepare(model,optimizer)
        scheduler=create_scheduler(optimizer,total_steps=10,warmup_steps=0,min_lr_ratio=.1)
        accelerator.register_for_checkpointing(scheduler)
        scheduler.step();scheduler.step()
        saved_lr=scheduler.get_last_lr()
        with tempfile.TemporaryDirectory() as directory:
            accelerator.save_state(directory)
            scheduler.step()
            accelerator.load_state(directory)
        self.assertEqual(scheduler.step_count,2)
        self.assertEqual(scheduler.get_last_lr(),saved_lr)

    def test_schedule_mixture_weights_reused_ages(self):
        from dynamicwam.training.data.paired_index import deployment_age_weights
        weights = deployment_age_weights(
            execute_horizon=4, every_fast_queries=4, global_downsample_rate=3
        )
        self.assertEqual(weights, {0: 1.0, 12: 2.0, 24: 2.0, 36: 2.0})
        records = []
        for offset in (0, 12, 24, 36):
            records.append(SimpleNamespace(
                task_name="task", episode_index=0, offset_label="steady_state", offset=offset
            ))
        sampler = BalancedPairSampler(records, samples_per_episode=4000, seed=1, age_weights=weights)
        counts = {0: 0, 12: 0, 24: 0, 36: 0}
        for index in sampler:
            counts[records[index].offset] += 1
        self.assertLess(counts[0], counts[12])
        self.assertLess(counts[0], counts[24])
        self.assertLess(counts[0], counts[36])

    def test_isolated_fast_runtime_drops_world_weights(self):
        from dynamicwam.inference.fast_runtime import isolated_fast_runtime
        from dynamicwam.inference.multirate_model_loader import build_fast_only_modules
        spec = {
            "schedule": BridgeSchedule.from_flow_match().to_config(),
            "architecture": "reference",
            "action_expert": dict(dim=16, ffn_dim=32, num_layers=2, state_dim=4, action_dim=4, chunk_size=2),
            "wan_attention": dict(dim=16, num_heads=2, head_dim=8, precision="float32", window_sizes=[[-1, -1], [-1, -1]]),
            "motion": dict(history_count=2, mean=[0.0] * 12, scale=[1.0] * 12),
            "fast_conditioner": dict(dim=16, layers=2, heads=2, head_dim=8, text_dim=8, state_dim=4, sim_scale=1.0, action_scale=16.0),
        }
        modules = build_fast_only_modules(spec)
        state = {}
        for key, value in modules.state_dict().items():
            if key.startswith(("action_expert.", "absolute_motion_tokens.")):
                state["backbone." + key] = value
            else:
                state[key] = value
        state["backbone.compact_wan.video.weight"] = torch.zeros(1)
        state["vae.encoder.weight"] = torch.zeros(1)
        model = isolated_fast_runtime({"model": state, "modules": spec, "global_step": 3}, device="cpu", dtype=torch.float32)
        names = [name for name, _parameter in model.named_parameters()]
        self.assertTrue(any(name.startswith("backbone.action_expert.") for name in names))
        self.assertFalse(any("compact_wan" in name or name.startswith("vae") or ".t5" in name for name in names))
        self.assertIsNot(model, modules)

    def test_joint_sampler_seed_does_not_replay_warmup(self):
        import yaml
        from dynamicwam.config.multirate import JOINT_SAMPLER_SEED_SALT
        root = Path(__file__).resolve().parents[1]
        raw = yaml.safe_load((root / "configs" / "world_action_multirate_v1.yaml").read_text())
        raw["pairing"].pop("sampler_seed", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            raw["phase"] = "fast_warmup"
            raw["loss"]["video_weight"] = 0.0
            path.write_text(yaml.safe_dump(raw))
            warmup = load_multirate_profile(path).training
            raw["phase"] = "cross_time_joint"
            raw["loss"]["video_weight"] = 0.001
            path.write_text(yaml.safe_dump(raw))
            joint = load_multirate_profile(path).training
        self.assertEqual(warmup["dataset"]["partition_seed"], joint["dataset"]["partition_seed"])
        self.assertEqual(warmup["dataset"]["sampler_seed"], warmup["dataset"]["partition_seed"])
        self.assertEqual(
            joint["dataset"]["sampler_seed"],
            warmup["dataset"]["partition_seed"] + JOINT_SAMPLER_SEED_SALT,
        )

    def test_fast_loss_ignores_actions_deploy_discards(self):
        from dynamicwam.training.train.multirate_objective import MultirateLossObjective
        prediction = torch.tensor([[[1.0], [0.0]], [[3.0], [0.0]]])
        target = torch.zeros(2, 2, 1)
        _loss, metrics = MultirateLossObjective()(
            {"fast_pred": prediction, "action_pred": torch.zeros(2, 2, 1)},
            target,
            torch.zeros(2, 2, 1),
            executed_horizon=1,
        )
        self.assertAlmostEqual(metrics["loss/fast"], 5.0)
        self.assertAlmostEqual(metrics["loss/fast_chunk"], 2.5)

    def test_dense_mixed_null_skips_gather(self):
        model = core()
        tokens = torch.randn(2, 3, 16)
        modulation = tuple(torch.ones(2, 1, 1, 16) if i == 2 else torch.zeros(2, 1, 1, 16) for i in range(6))
        cache = dict(
            seq_lens=torch.tensor([4, 4]),
            video_k=[torch.randn(2, 4, 2, 8)],
            video_v=[torch.randn(2, 4, 2, 8)],
            all_video_tokens_valid=True,
        )
        mask = torch.tensor([False, True])
        with patch("dynamicwam.models.action_core_ops.flash_attention", cpu_attention):
            full = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=False)
            dropped = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=True)
            with patch("dynamicwam.models.action_core_ops.pack_world_action_kv", side_effect=AssertionError("identity gather")):
                mixed = model._action_attention_with_video_cache(tokens, modulation, 0, cache, null_bridge=mask)
        expected = torch.where(mask.view(-1, 1, 1), dropped, full)
        torch.testing.assert_close(mixed, expected, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
