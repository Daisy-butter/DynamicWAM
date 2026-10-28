import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_continuous_acceptance import validate_continuous_report, validate_reference_leases


def valid():
    evidence = dict(passed=True, checks={'immutable': True}, activation_wall_time=100.,
        activation_physics_time=0., final_wall_time=101.01, final_physics_time=1.)
    return dict(raw_clock_acceptance=evidence, physics_step_seconds=.004, physics_steps=250,
        active_wall_seconds=1.01, physical_duration=1., physics_steps_during_camera=20,
        physics_steps_during_fast_inflight=20, events=[
            dict(event='activation', wall_time=100., physics_time=0.),
            dict(event='query', clock_pair=dict(wall_time=100.5, physical_time=.48,
                expected_physical_time=.5, backlog_seconds=.02)),
            dict(event='terminal_clock', **evidence)])


def test_true_raw_clock_passes():
    assert validate_continuous_report(valid(), dict(max_backlog=.25, backlog_scheduling_tolerance=.005))['passed']


@pytest.mark.parametrize('mutation', ['credit', 'rebase', 'skipped'])
def test_self_reported_flags_cannot_hide_clock_failure(mutation):
    report = valid()
    if mutation == 'credit': report['active_wall_seconds'] = .8
    elif mutation == 'rebase': report['events'][1]['clock_pair']['expected_physical_time'] = .48
    elif mutation == 'skipped': report['physics_steps'] = 249
    with pytest.raises(ValueError):
        validate_continuous_report(report, dict(max_backlog=.25, backlog_scheduling_tolerance=.005))


def test_native_terminal_before_active_queries_needs_no_per_episode_overlap():
    report = valid()
    report.update(physics_steps_during_camera=0, physics_steps_during_fast_inflight=0)
    assert validate_continuous_report(report, dict(max_backlog=.25, backlog_scheduling_tolerance=.005))['passed']


def lease_events():
    activation = dict(event='activation', wall_time=100., physics_time=0.)
    start = dict(event='fast_started', query_id=1, reference_version=1, query_time=.36,
        source_time=0., reference_age=.36, observation_time=.18, request_wall_time=100.36,
        protected_until=.72, submit_margin=.01)
    commit = dict(start, event='fast_committed', ready_wall_time=100.46,
                  commit_wall_time=100.48, commit_time=.48)
    return [dict(event='slow_completed', accepted=True, version=1, source_time=0., completed_time=0.),
            activation, start, commit]


@pytest.mark.parametrize('mutation', ['missing_m', 'expired', 'future_m', 'lease', 'late_wall', 'early_release'])
def test_active_query_causality_and_commit_contract(mutation):
    events = lease_events()
    if mutation == 'missing_m': events[2]['reference_version'] = -1
    elif mutation == 'expired': events[2]['query_time'] = 2.
    elif mutation == 'future_m': events[0]['completed_time'] = .4
    elif mutation == 'lease': events[3]['protected_until'] = .9
    elif mutation == 'late_wall': events[3]['commit_wall_time'] = 100.72
    else: events[3]['commit_time'] = .44
    with pytest.raises(ValueError):
        validate_reference_leases(events, {'runtime': {'reference_max_age': 2.}}, events[1])


def test_bootstrap_reference_and_terminal_cancelled_query_are_valid():
    events = lease_events()
    assert validate_reference_leases(events, {'runtime': {'reference_max_age': 2.}}, events[1])['active_commits'] == 1
    assert validate_reference_leases(events[:-1], {'runtime': {'reference_max_age': 2.}}, events[1])['active_reference_queries'] == 1
