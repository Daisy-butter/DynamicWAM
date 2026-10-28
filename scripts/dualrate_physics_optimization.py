"""Reuse identical passive-force applications inside one actuator setter block.

The audited DOMINO setters only change drive targets/velocities. They do not
integrate physics or change qpos/qvel. A scope must end before any such state
change, and every actuator tick must start a fresh scope. Native setter order,
gripper clipping/rate limits and joint command arithmetic remain untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

_MISSING=object()
_MARKER='_full_passive_force_scope'


def _restore(robot,name,previous):
    if previous is _MISSING:
        if name in vars(robot):
            delattr(robot,name)
    else:
        setattr(robot,name,previous)


@contextmanager
def passive_force_scope(robot, *, enabled=False):
    """Temporarily deduplicate ``_entity_qf`` by articulation object identity.

    Disabled is the exact native path. Enabled yields counters and restores the
    original descriptor/instance binding even if a native setter raises. No cache
    survives this scope, so nothing is reused across physics steps or episodes.
    """
    if type(enabled) is not bool:
        raise ValueError('passive_force_dedup must be a boolean')
    if not enabled:
        yield None
        return
    owner=threading.get_ident()
    attributes=vars(robot)
    if getattr(robot,_MARKER,None) is not None:
        raise RuntimeError('passive force scope is already active')
    previous_method=attributes.get('_entity_qf',_MISSING)
    previous_marker=attributes.get(_MARKER,_MISSING)
    original=robot._entity_qf
    if not callable(original):
        raise TypeError('native robot passive-force method must be callable')
    cached={}
    counters=dict(native_evaluations=0,reused_calls=0)

    def apply(entity):
        if threading.get_ident()!=owner:
            raise RuntimeError('passive-force application requires the actuator owner thread')
        key=id(entity)
        if key in cached:
            kept,value=cached[key]
            if kept is not entity:
                raise RuntimeError('articulation identity changed inside actuator scope')
            counters['reused_calls']+=1
            return value
        value=original(entity)
        # Retain the object itself as well as its identity until the scope ends.
        cached[key]=(entity,value)
        counters['native_evaluations']+=1
        return value

    try:
        robot._entity_qf=apply
        setattr(robot,_MARKER,apply)
    except BaseException:
        _restore(robot,'_entity_qf',previous_method)
        _restore(robot,_MARKER,previous_marker)
        raise
    try:
        yield counters
    finally:
        changed=robot._entity_qf is not apply or getattr(robot,_MARKER,None) is not apply
        _restore(robot,'_entity_qf',previous_method)
        _restore(robot,_MARKER,previous_marker)
        cached.clear()
        if threading.get_ident()!=owner or changed:
            raise RuntimeError('passive-force scope ownership or method binding changed')
