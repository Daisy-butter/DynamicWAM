"""CPU parity; production constructor also verifies both real CUDA destinations."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_deployment_inputs import build_cached_online_batch_builder
from dynamicwam.dualrate.online import OnlineBatchBuilder


def builders(dtype=torch.float32):
    language = dict(context=torch.linspace(-3., 3., 512*4096).reshape(512,4096).to(dtype),
                    mask=torch.arange(512) % 3 != 0)
    kwargs = dict(action_mean=[0.]*14, action_scale=[1.]*14, language=language,
                  device='cpu', slow_device='cpu')
    return OnlineBatchBuilder({}, **kwargs), build_cached_online_batch_builder({}, **kwargs), language


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_language_aliases_are_bitwise_equal_and_reused_without_changing_other_inputs(dtype):
    original, cached, language = builders(dtype)
    values = dict(text=language['context'], text_valid=language['mask'],
                  source_text=language['context'], state=torch.arange(14.).float(),
                  query_time=torch.tensor(.36, dtype=torch.float64))
    expected, actual = original._batch(values), cached._batch(values)
    assert set(actual) == set(expected)
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].shape == expected[name].shape
        assert torch.equal(actual[name].contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))
    again = cached._batch(values, device=cached.slow_device)
    assert actual['text'].data_ptr() == actual['source_text'].data_ptr() == again['source_text'].data_ptr()
    assert cached.deployment_language_receipt['passed']
    assert all(row['exact_byte_parity'] for row in cached.deployment_language_receipt['devices'].values())
    assert len(cached.deployment_language_receipt['context_sha256']) == 64


def test_changed_episode_language_cannot_silently_reuse_an_old_cache():
    original, cached, language = builders()
    with pytest.raises(ValueError, match='immutable'):
        cached._batch(dict(text=language['context'].clone()))
    language['context'][0,0] = 8.
    with pytest.raises(ValueError, match='immutable'):
        cached._batch(dict(text=language['context']))


def test_returned_language_alias_cannot_be_mutated_silently():
    original, cached, language = builders()
    result = cached._batch(dict(text=language['context']))
    result['text'][0,0,0] = 8.
    with pytest.raises(ValueError, match='immutable'):
        cached._batch(dict(text=language['context']))
