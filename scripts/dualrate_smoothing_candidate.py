"""CPU reference for optional plan projection. NOT wired to evaluation.

Filter absolute arm commands only in uncommitted plan rows. The final committed
row is the authoritative anchor; both gripper commands are copied exactly.
This is an experimental interface, not a safety or closed-loop performance claim.
"""
import math


def smooth_actions(actions, known_mask, alpha=0.85, max_offset=None):
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('alpha must be finite and in (0, 1]')
    if max_offset is not None and (not math.isfinite(max_offset) or max_offset < 0):
        raise ValueError('max_offset must be a finite nonnegative number')
    result = [list(row) for row in actions]
    known = list(known_mask)
    if not result or len(known) != len(result):
        raise ValueError('action and mask lengths must agree and be nonempty')
    if any(len(row) != 14 or any(not math.isfinite(float(x)) for x in row) for row in result):
        raise ValueError('each action must have 14 finite coordinates')
    if any(type(x) is not bool for x in known):
        raise ValueError('mask must contain booleans')
    n = sum(known)
    if known != [True] * n + [False] * (len(known) - n) or n == 0:
        raise ValueError('a nonempty contiguous committed prefix is required')
    arm_indices = (*range(6), *range(7, 13))
    for i in range(n, len(result)):
        for j in arm_indices:
            correction = (1 - alpha) * (result[i - 1][j] - result[i][j])
            if max_offset is not None:
                correction = max(-max_offset, min(max_offset, correction))
            result[i][j] += correction
    return result
