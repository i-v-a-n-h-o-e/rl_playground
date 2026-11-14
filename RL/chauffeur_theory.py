"""Analytical tangential barrier in Merz §2.3 for gamma=0.5, beta=0.3.

These curves describe the continuous game. They are independent of the
simulator, the rollout horizon, and the domain of the feedback policies.
"""

from math import acos, cos, pi, sin

GAMMA, BETA = 0.5, 0.3
USABLE_PART_ANGLE = acos(GAMMA)
BARRIER_END_TIME = 2 * (pi - USABLE_PART_ANGLE)


def retrograde_position(theta: float, tau: float) -> tuple[float, float]:
    """Relative position along a right turn from terminal angle theta (Merz §2.1)."""
    q = BETA - GAMMA * tau
    return 1 - cos(tau) + q * sin(theta + tau), sin(tau) + q * cos(theta + tau)


def tangential_barrier(tau: float, *, side: int = 1) -> tuple[float, float]:
    """Point on the open barrier, from tangency at tau=0 to B at BARRIER_END_TIME."""
    if not 0 <= tau <= BARRIER_END_TIME:
        raise ValueError("tau must lie between 0 and BARRIER_END_TIME")
    if side not in (-1, 1):
        raise ValueError("side must be +1 (right) or -1 (left)")
    x, y = retrograde_position(USABLE_PART_ANGLE, tau)
    return side * x, y
