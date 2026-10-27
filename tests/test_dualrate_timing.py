"""Deterministic physical-clock tests; no GPU or simulator claims."""
import pytest
import torch

from dynamicwam.dualrate.timing import (
    BufferUnderrun, CompletedReferenceStore, DeadlineMiss, ExecutionQueue,
    StalePlan, TimedPlan, TimingError,
)


def publish(store, **kw):
    args = dict(source_time=0., completed_time=.2, version=1, generation_stage="joint-2of10",
                future_times=(.1,.2,.4,.8), future_valid=(True,True,True,True),
                payload={"features": torch.ones(4,8)}, now=.2)
    args.update(kw)
    return store.publish(**args)


def plan(values=(0,1,2,3,4,5), *, start=0., dt=.1, version=1, query=0.):
    return TimedPlan(start, dt, tuple((x,) for x in values), version, query)


def test_reference_empty_bootstrap_is_explicit():
    assert CompletedReferenceStore(test_fixture=True).snapshot(0.) is None


def test_reference_only_complete_available_at_query():
    store = CompletedReferenceStore(test_fixture=True)
    with pytest.raises(TimingError, match="in-flight"):
        publish(store, complete=False)
    with pytest.raises(TimingError, match="in-flight"):
        publish(store, now=.1)
    assert publish(store)
    with pytest.raises(TimingError, match="not-yet-complete"):
        store.snapshot(.199)
    assert store.snapshot(.2).completed_time == .2


def test_reference_age_is_source_age_and_support_uses_absolute_time():
    store = CompletedReferenceStore(test_fixture=True)
    publish(store)
    snapshot = store.snapshot(.4)
    assert snapshot.source_time == 0.
    assert snapshot.supports((.2,.4,.6,.8,.9),query_time=.4) == (False,True,True,True,False)


def test_reference_missing_support_not_bridged():
    store = CompletedReferenceStore(test_fixture=True)
    publish(store, future_valid=(True,True,False,True))
    snapshot = store.snapshot(.2)
    assert snapshot.supports((.2,.3,.4,.6,.8),query_time=.2) == (True,False,False,False,True)


def test_reference_stale_version_and_older_source_are_rejected():
    store = CompletedReferenceStore(test_fixture=True)
    publish(store, source_time=.1)
    assert not publish(store, version=0)
    assert not publish(store, source_time=0.,version=2)
    assert publish(store, source_time=.1,version=2)
    assert store.snapshot(.2).version == 2


def test_reference_original_and_reader_tensors_cannot_mutate_store():
    store = CompletedReferenceStore(test_fixture=True)
    tensor = torch.ones(4,8)
    publish(store,payload={"features": tensor})
    tensor.fill_(99)
    first = store.snapshot(.2)
    assert first.payload["features"].eq(1).all()
    first.payload["features"].fill_(42)
    assert store.snapshot(.2).payload["features"].eq(1).all()
    with pytest.raises(TypeError):
        first.payload["other"] = tensor


def test_runtime_reference_cannot_silently_detach_training_graph():
    with pytest.raises(ValueError, match="autograd"):
        publish(CompletedReferenceStore(test_fixture=True),payload={"live":torch.ones(1,requires_grad=True)})


def test_timed_plan_deeply_copies_mutable_actions():
    rows = [[1.],[2.]]
    p = TimedPlan(0.,.1,rows,1,0.)
    rows[0][0] = 99
    assert p.actions[0] == (1.,)


def test_mid_row_replace_keeps_current_row_and_new_suffix():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    prefix = q.commit_prefix(query_time=0.,until=.2)
    assert prefix.actions == ((0.,),(1.,))
    assert q.replace(plan((0,1,20,30),version=2),now=.15) == 2
    assert q.action_at(.15) == (1.,)
    assert q.action_at(.2) == (20.,)


def test_mid_step_query_may_include_already_started_committed_row():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    prefix = q.commit_prefix(query_time=.15, until=.2)
    assert prefix.start_time == pytest.approx(.1)
    p = plan((1,20,30), start=.1, query=.15, version=2)
    q.replace(p, now=.18)
    assert q.action_at(.18) == (1.,)
    assert q.action_at(.2) == (20.,)


def test_changed_committed_prefix_rejected_without_mutation():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    q.commit_prefix(query_time=0.,until=.3)
    with pytest.raises(TimingError,match="committed"):
        q.replace(plan((0,99,2,30),version=2),now=.15)
    assert q.action_at(.15) == (1.,)
    assert q.action_at(.3) == (3.,)


def test_expired_prediction_rows_discarded_not_replayed():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    assert q.replace(plan((99,1,20,30),version=2),now=.15) == 2
    assert q.action_at(.15) == (1.,)
    assert q.action_at(.2) == (20.,)


def test_late_plan_with_no_executable_suffix_fails():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    with pytest.raises(DeadlineMiss):
        q.replace(plan((0,1),version=2),now=.25)
    assert q.action_at(.3) == (3.,)


def test_stale_version_cannot_overwrite_even_uncommitted_actions():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    with pytest.raises(StalePlan):
        q.replace(plan((9,9,9),version=1),now=0.)
    assert q.action_at(0.) == (0.,)


def test_deadline_boundary_can_replace_before_buffer_runs_out():
    q = ExecutionQueue(plan((0,1)), allow_unleased_test_fixture=True)
    q.action_at(.1)
    assert q.replace(plan((2,3),start=.2,query=.1,version=2),now=.2) == 0
    assert q.action_at(.2) == (2.,)


def test_underrun_is_never_silent_hold():
    q = ExecutionQueue(plan((0,1)), allow_unleased_test_fixture=True)
    with pytest.raises(BufferUnderrun):
        q.action_at(.2)
    with pytest.raises(BufferUnderrun):
        q.replace(plan((3,4),start=.3,query=.1,version=2),now=.3)


def test_zero_length_commit_and_floating_boundary():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    prefix = q.commit_prefix(query_time=0.,until=0.)
    assert prefix.actions == ()
    q.replace(plan((9,8,7),version=2),now=0.)
    assert q.action_at(.1+.1) == (7.,)


def test_query_cannot_commit_missing_buffer():
    q = ExecutionQueue(plan((0,1)), allow_unleased_test_fixture=True)
    with pytest.raises(BufferUnderrun):
        q.commit_prefix(query_time=0.,until=.3)


def test_plan_alignment_and_backward_clock_are_errors():
    q = ExecutionQueue(plan(), allow_unleased_test_fixture=True)
    with pytest.raises(ValueError,match="aligned"):
        q.replace(plan(start=.05,version=2),now=.1)
    q.action_at(.2)
    with pytest.raises(TimingError,match="backwards"):
        q.action_at(.1)


def test_reference_metadata_rejects_invalid_future_grid():
    with pytest.raises(ValueError,match="increasing"):
        publish(CompletedReferenceStore(test_fixture=True),future_times=(.1,.1),future_valid=(True,True))
    with pytest.raises(ValueError,match="finite"):
        publish(CompletedReferenceStore(test_fixture=True),source_time=float("nan"))
