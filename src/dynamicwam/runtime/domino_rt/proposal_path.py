"""Causal controller rollouts and one shared training/deployment path encoding.

A proposal is a conditional forecast, not a claim about arrival time or measured
joint motion. The currently committed Hermite segment is kept, then the proposed
tail is installed after the observed dispatch time and a computation reserve.
Only already executed commands may appear in the known prefix.
"""
from __future__ import annotations
import math
import numpy as np
from .timed_control import LimitedHermiteController

PATH_BINS = 4
HORIZONS = (.2, .4)
TIME_TOLERANCE = 1e-8
COMPUTE_RESERVE_SECONDS = .02


def forecast_horizons(source_time, dispatch_time):
    """Two goals after candidate installation, relative to the RGB source time."""
    source,dispatch=float(source_time),float(dispatch_time)
    if not np.isfinite([source,dispatch]).all() or source<0 or dispatch<source-1e-8 or dispatch-source>2.:
        raise ValueError('invalid or unsupported causal observation age')
    install=math.ceil((dispatch+COMPUTE_RESERVE_SECONDS-1e-9)/.1)*.1
    return np.asarray((install+.1-source,install+.3-source),dtype=np.float64)


def aggregate_command_path(start, end, q, velocity, source_time, target_time,
                           mean, scale, *, bins=PATH_BINS, tolerance=TIME_TOLERANCE):
    """Four time bins: normalized final drive q and duration-weighted mean v."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    q, velocity = np.asarray(q, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (start.ndim != 1 or end.shape != start.shape or q.shape != (len(start), 14) or
            velocity.shape != q.shape or mean.shape != (14,) or scale.shape != (14,) or
            bins != PATH_BINS or tolerance < 0 or not np.isfinite(tolerance)):
        raise ValueError('invalid command path layout')
    if any(not np.isfinite(x).all() for x in (start, end, q, velocity, mean, scale)) or np.any(scale <= 0):
        raise ValueError('nonfinite command path or invalid normalization')
    if (np.any(end <= start) or np.any(np.diff(start) <= 0) or
            np.any(start[1:] < end[:-1] - tolerance)):
        raise ValueError('command intervals must be ordered and nonoverlapping')
    source, target = float(source_time), float(target_time)
    if not np.isfinite([source, target]).all() or source < 0 or target <= source:
        raise ValueError('future interval must have strictly positive real duration')
    output = np.zeros((bins, 28), dtype=np.float32)
    edges = np.linspace(source, target, bins + 1)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = np.flatnonzero((end > lo + tolerance) & (start < hi - tolerance))
        if not len(selected): return np.zeros_like(output), False
        s, e = start[selected], end[selected]
        if s[0] > lo + tolerance or e[-1] < hi - tolerance or np.any(s[1:] > e[:-1] + tolerance):
            return np.zeros_like(output), False
        duration = np.minimum(e, hi) - np.maximum(s, lo)
        if abs(duration.sum() - (hi - lo)) > tolerance: return np.zeros_like(output), False
        output[index, :14] = (q[selected[-1]] - mean) / scale
        output[index, 14:] = (velocity[selected] * duration[:, None]).sum(0) / (hi - lo) / scale
    if not np.isfinite(output).all(): raise ValueError('normalized command path overflow')
    return output, True


def proposal_action_path(*, controller, plan, age, observation_time, mean, scale):
    """Encode a normalized immutable 16-row plan under a source controller state.

    The source snapshot is captured with RGB. A dispatch snapshot and only the
    already executed prefix bridge camera delay; no future executed commands,
    teacher outputs or target images are accepted.
    Both training's proposal and deployment call this same pure NumPy function.
    """
    source = float(observation_time)
    if (type(age) is not int or not 0 <= age < 16 or not math.isfinite(source) or source < 0 or
            controller.get('initialized') is not True or controller.get('period') != .1 or
            abs(float(controller['source_physics_time']) - source) > 1e-7):
        raise ValueError('proposal requires the matching source-clock controller snapshot')
    if abs(source / .1 - round(source / .1)) > 1e-6:
        raise ValueError('source observation is off the physical control grid')
    plan, mean, scale = (np.asarray(x, dtype=np.float64) for x in (plan, mean, scale))
    if plan.shape != (16,14) or mean.shape != (14,) or scale.shape != (14,):
        raise ValueError('proposal/normalization layout differs')
    if not all(np.isfinite(x).all() for x in (plan,mean,scale)) or np.any(scale <= 0):
        raise ValueError('invalid proposal values')
    if abs(float(controller['elapsed']))>1e-8:
        raise ValueError('source-grid snapshot must follow begin_interval, before its first tick')
    current=controller.get('dispatch_controller',controller)
    dispatch=float(current['source_physics_time'])
    horizons=forecast_horizons(source,dispatch)
    if current.get('initialized') is not True or current.get('period')!=.1:
        raise ValueError('invalid dispatch controller')
    simulator = LimitedHermiteController(np.asarray(current['q']), period=.1)
    for name in ('q','v','start_q','start_v','goal','end_v','vmax','amax'):
        value = np.asarray(current[name],dtype=np.float64)
        if value.shape != (14,) or not np.isfinite(value).all(): raise ValueError('invalid controller vector')
        setattr(simulator,name,value.copy())
    if np.any(simulator.vmax <= 0) or np.any(simulator.amax <= 0): raise ValueError('invalid limits')
    simulator.elapsed=float(current['elapsed'])
    if not 0<=simulator.elapsed<.1+1e-8 or abs(dispatch/.004-round(dispatch/.004))>1e-6:
        raise ValueError('dispatch is off the real physics clock')
    phase=dispatch-math.floor((dispatch+1e-8)/.1)*.1
    if abs(simulator.elapsed-phase)>1e-7: raise ValueError('dispatch segment and clock disagree')
    simulator.ticks=int(current['ticks']);simulator.limited_ticks=int(current['limited_ticks'])
    origin_slot=round(source/.1)-age
    install_slot=round((source+horizons[0]-.1)/.1)
    # The active segment is immutable. The next interval uses candidate knots;
    # the candidate tail replaces all old uncommitted knots, just like publish.
    candidate=plan*scale+mean
    knots={origin_slot+i+1:row for i,row in enumerate(candidate) if origin_slot+i+1>install_slot}
    old_knots={int(k['end_slot']):np.asarray(k['q'],dtype=np.float64) for k in current.get('future_knots',[])}
    if any(q.shape!=(14,) or not np.isfinite(q).all() for q in old_knots.values()):
        raise ValueError('invalid causal queued command')
    dt=.004;count=round((source+max(horizons)-dispatch)/dt)
    starts=dispatch+np.arange(count)*dt;ends=starts+dt
    commands=np.empty((count,14),np.float64);velocities=np.empty_like(commands)
    # Hermite coefficients do not depend on intermediate q/v. Compute each
    # committed segment's desired positions together, retaining the exact
    # recurrence of the limited controller for command velocity/acceleration.
    # No approximate unconstrained interpolation or different command contract.
    elapsed=simulator.elapsed
    begin=0
    while begin<count:
        if begin:
            slot=round(starts[begin]/.1)
            queue=knots if slot>=install_slot else old_knots
            simulator.begin(queue.get(slot+1),queue.get(slot+2));elapsed=0.
        segment_count=min(count-begin,max(1,round((.1-elapsed)/dt)))
        ts=[]
        for _ in range(segment_count):
            elapsed=min(.1,elapsed+dt);ts.append(elapsed/.1)
        # Use Python scalar powers, like LimitedHermiteController.step; NumPy
        # vectorized power differs at near-zero cancellation on some builds.
        h00,h10,h01,h11=np.asarray([(2*t**3-3*t**2+1,t**3-2*t**2+t,
            -2*t**3+3*t**2,t**3-t**2) for t in ts]).T[:,:,None]
        desired=(h00*simulator.start_q+h10*.1*simulator.start_v
            +h01*simulator.goal+h11*.1*simulator.end_v)
        vmax=simulator.vmax;dv=simulator.amax*dt
        q,v=simulator.q,simulator.v
        for j,target in enumerate(desired):
            velocity=np.minimum(np.maximum((target-q)/dt,-vmax),vmax)
            velocity=np.minimum(np.maximum(velocity,v-dv),v+dv)
            q=q+velocity*dt;v=velocity
            commands[begin+j]=q;velocities[begin+j]=v
        simulator.q=q;simulator.v=v;simulator.elapsed=elapsed
        begin+=segment_count
    past_count=round((dispatch-source)/dt)
    if past_count:
        past=controller.get('executed_prefix')
        if not isinstance(past,dict): raise ValueError('camera delay requires a known executed command prefix')
        ps,pe,pq,pv=(np.asarray(past[k],dtype=np.float64) for k in
            ('start_sim_time','end_sim_time','q_command','velocity_command'))
        if (ps.shape!=(past_count,) or pe.shape!=(past_count,) or pq.shape!=(past_count,14) or pv.shape!=pq.shape or
            not np.allclose(ps,source+np.arange(past_count)*dt,atol=1e-8,rtol=0) or
            not np.allclose(pe,source+(np.arange(past_count)+1)*dt,atol=1e-8,rtol=0) or
            not np.isfinite(pq).all() or not np.isfinite(pv).all()):
            raise ValueError('known prefix is incomplete, retimestamped or includes a future command')
        starts=np.concatenate((ps,starts));ends=np.concatenate((pe,ends))
        commands=np.concatenate((pq,commands));velocities=np.concatenate((pv,velocities))
    paths=[]
    for horizon in horizons:
        path,valid=aggregate_command_path(starts,ends,commands,velocities,source,source+horizon,mean,scale)
        if not valid: raise RuntimeError('causal proposal integration left uncovered time')
        paths.append(path)
    return np.stack(paths),horizons.astype(np.float32),dict(
        format='dynamicwam_causal_proposal_path',version=1,
        source_physics_time=source,dispatch_sim_time=dispatch,assumed_install_sim_time=install_slot*.1,
        install_assumption='first_boundary_after_dispatch_plus_20ms_not_measured_arrival',
        compute_reserve_seconds=COMPUTE_RESERVE_SECONDS,known_executed_prefix_ticks=past_count,
        committed_segment_preserved=True,actual_future_commands_used=False,
        units='normalized_drive_q_and_mean_drive_velocity',physics_dt=dt,
        horizons=horizons.tolist(),bins=PATH_BINS)
