import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dualrate_continuous_acceptance import validate_reference_leases


def history():
    activation = dict(event='activation', wall_time=100., physics_time=0.)
    start = dict(event='fast_started', query_id=1, reference_version=-1,
        source_time=None, reference_age=None, query_time=2.16,
        observation_time=1.98, request_wall_time=102.16,
        protected_until=2.52, submit_margin=.01)
    commit = dict(start, event='fast_committed', ready_wall_time=102.25,
        commit_wall_time=102.26, commit_time=2.26)
    return [dict(event='slow_completed', accepted=True, version=1,
                 source_time=0., completed_time=0.), activation, start, commit]


def validate(events):
    return validate_reference_leases(events, {'runtime': {'reference_max_age': 2.}},
                                    next(e for e in events if e['event'] == 'activation'))


def test_expired_reference_uses_trained_empty_world_and_preserves_deadlines():
    result = validate(history())
    assert result['active_empty_reference_queries'] == 1
    assert result['active_reference_queries'] == 0
    assert result['active_queries'] == result['active_commits'] == 1


@pytest.mark.parametrize('mutation', ['live', 'fake_source', 'fake_age', 'stale_tensor',
    'late_wall', 'late_physics', 'early_release', 'no_bootstrap', 'future_publication', 'duplicate'])
def test_empty_world_cannot_hide_invalid_reference_or_lease(mutation):
    events = history()
    if mutation == 'live': events[0].update(source_time=.18, completed_time=.3)
    elif mutation == 'fake_source': events[2]['source_time'] = 0.
    elif mutation == 'fake_age': events[2]['reference_age'] = 2.16
    elif mutation == 'stale_tensor': events[2].update(reference_version=1, source_time=0., reference_age=2.16)
    elif mutation == 'late_wall': events[3]['commit_wall_time'] = 102.52
    elif mutation == 'late_physics': events[3]['commit_time'] = 2.52
    elif mutation == 'early_release': events[3]['commit_time'] = 2.2
    elif mutation == 'no_bootstrap': events.pop(0)
    elif mutation == 'future_publication': events[0]['completed_time'] = 2.2
    else: events.insert(3, copy.deepcopy(events[2]))
    with pytest.raises(ValueError): validate(events)


def test_reference_refresh_is_consumed_after_empty_query():
    events = history()
    events.append(dict(event='slow_completed', accepted=True, version=2,
                       source_time=2.16, completed_time=2.4))
    start = dict(events[2], query_id=2, query_time=2.52, observation_time=2.34,
                 request_wall_time=102.52, protected_until=2.88,
                 reference_version=2, source_time=2.16, reference_age=.36)
    events.append(start)
    assert validate(events)['active_reference_queries'] == 1
    events[-1].update(reference_version=-1, source_time=None, reference_age=None)
    with pytest.raises(ValueError, match='bypassed'): validate(events)


def test_exact_ttl_boundary_and_cancelled_terminal_query():
    events = history()[:-1]
    events[2].update(query_time=2., request_wall_time=102.)
    assert validate(events)['active_empty_reference_queries'] == 1
    events[2].update(query_time=1.9999999, request_wall_time=101.9999999)
    with pytest.raises(ValueError, match='bypassed'): validate(events)
