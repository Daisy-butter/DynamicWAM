"""CPU-only measurements of raw proposals and actual execution; never alter actions.

A gripper threshold is only a diagnostic convention, not a contact/success label.
Unknown queue tails and unmatched physical endpoints contribute no measurements.
"""
from __future__ import annotations
import numpy as np

GRIPPERS = (6, 13)
ARMS = tuple(i for i in range(14) if i not in GRIPPERS)


def _stats(values):
    x=np.asarray(values,dtype=np.float64)
    return dict(count=int(x.size),rms=float(np.sqrt(np.mean(x*x))) if x.size else None,
                max_abs=float(np.max(np.abs(x))) if x.size else None)


def diagnose_query(row, arrays, *, gripper_threshold=.5, endpoint_tolerance=1e-6):
    """Compare proposals with published queue at exactly matching future endpoints."""
    raw=np.asarray(arrays['returned_actions'],dtype=np.float64)
    offsets=np.asarray(arrays['_rtc_action_target_offsets'],dtype=np.float64)
    previous=np.asarray(arrays['_rtc_previous_actions'],dtype=np.float64)
    valid=np.asarray(arrays['_rtc_previous_valid'],dtype=bool)
    previous_times=np.asarray(row['previous_endpoint_times'],dtype=np.float64)
    n=len(raw);d=int(row['prefix_length'])
    times=float(row['capture_time'])+offsets
    if (raw.shape!=(n,14) or previous.shape!=(len(previous_times),14) or valid.shape!=previous_times.shape
        or offsets.shape!=(n,) or not 0<=d<=n or not np.isfinite(raw).all()
        or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
        or not np.isfinite(previous_times).all() or np.any(np.diff(previous_times)<=0)
        or not np.isfinite(previous[valid]).all() or not np.isfinite(gripper_threshold)):
        raise ValueError('invalid raw plan diagnostic arrays/timestamps')
    prefix=np.asarray(arrays['_rtc_prefix_actions'],dtype=np.float64)[:d]
    if prefix.shape!=(d,14) or not np.isfinite(prefix).all():raise ValueError('invalid committed prefix')
    pairs=[]
    for i,t in enumerate(times):
        if i<d or t<=float(row['ready_physics_s'])+endpoint_tolerance:continue
        candidates=np.flatnonzero(valid & (np.abs(previous_times-t)<=endpoint_tolerance))
        if len(candidates)>1:raise ValueError('ambiguous previous physical endpoint')
        if len(candidates):pairs.append((i,int(candidates[0])))
    new=np.asarray([raw[i] for i,j in pairs],dtype=np.float64).reshape(-1,14)
    old=np.asarray([previous[j] for i,j in pairs],dtype=np.float64).reshape(-1,14)
    delta=new-old
    disagreement=np.count_nonzero((new[:,GRIPPERS]>=gripper_threshold)!=(old[:,GRIPPERS]>=gripper_threshold),axis=0)
    transitions_new=np.zeros(2,dtype=int);transitions_old=np.zeros(2,dtype=int)
    for k in range(1,len(pairs)):
        i,j=pairs[k];pi,pj=pairs[k-1]
        # No transition may span an unknown or unmatched endpoint gap.
        if i==pi+1 and j==pj+1:
            transitions_new+=((raw[i,GRIPPERS]>=gripper_threshold)!=(raw[pi,GRIPPERS]>=gripper_threshold))
            transitions_old+=((previous[j,GRIPPERS]>=gripper_threshold)!=(previous[pj,GRIPPERS]>=gripper_threshold))
    return dict(query_index=int(row['query_index']),installed=bool(row.get('installed')),
        semantics='raw_model_proposal_vs_previous_published_queue; before_position_guard; not_execution_or_BC_labels',
        prefix_rows=d,prefix_delta=_stats(raw[:d]-prefix),future_overlap_rows=len(pairs),
        compared_endpoint_times=[float(times[i]) for i,j in pairs],
        arm_delta=_stats(delta[:,ARMS]),gripper_delta=_stats(delta[:,GRIPPERS]),
        gripper_state_disagreements=disagreement.tolist(),
        proposal_gripper_transitions=transitions_new.tolist(),previous_gripper_transitions=transitions_old.tolist())


def summarize_plan_switches(queries, arrays, targets, *, gripper_threshold=.5):
    if len(queries)!=len(arrays):raise ValueError('query diagnostic arrays are not one-to-one')
    rows=[diagnose_query(row,data,gripper_threshold=gripper_threshold)
          for row,data in zip(queries,arrays) if row.get('result_received')]
    transitions=np.zeros(2,dtype=int);adjacent=0
    for prev,current in zip(targets,targets[1:]):
        if int(current[0])!=int(prev[0])+1:continue
        a=np.asarray(prev[2],dtype=np.float64);b=np.asarray(current[2],dtype=np.float64)
        if a.shape!=(14,) or b.shape!=(14,) or not np.isfinite([*a,*b]).all():raise ValueError('invalid actual completed targets')
        transitions+=((a[list(GRIPPERS)]>=gripper_threshold)!=(b[list(GRIPPERS)]>=gripper_threshold));adjacent+=1
    return dict(schema='development_plan_switch_diagnostics_v1',measurement_only=True,
        action_semantics='raw_14_drive_targets',gripper_indices=list(GRIPPERS),
        gripper_threshold=float(gripper_threshold),threshold_semantics='diagnostic_state_only; not_contact_or_grasp_success',
        queries=rows,query_count=len(rows),compared_future_rows=sum(x['future_overlap_rows'] for x in rows),
        actual_completed_target_count=len(targets),actual_adjacent_target_pairs=adjacent,
        actual_completed_gripper_transitions=transitions.tolist(),
        target_semantics='actual_completed_controller_targets; distinct_from_returned_proposals')
