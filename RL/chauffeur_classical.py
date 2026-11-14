"""Partial classical feedback from §5 of the Notion strategy specification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from typing import Literal

from RL.chauffeur_theory import BETA, GAMMA, USABLE_PART_ANGLE, retrograde_position
from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv

DEFAULT_CONFIG = GameConfig(gamma=GAMMA, beta=BETA)
AXIS_TOLERANCE = 1e-9
RESIDUAL_TOLERANCE = 1e-8
BOUNDARY_TOLERANCE = 1e-10

State = tuple[float, float]
Policy = Callable[[State, GameConfig], float | None]
Branch = Literal["capture", "UL", "turn_to_UL", "primary"]


@dataclass(frozen=True)
class Feedback:
    branch: Branch
    phi: float | None
    psi: float | None
    time_to_capture: float


def _check_parameters(config: GameConfig) -> None:
    if (config.gamma, config.beta) != (GAMMA, BETA):
        raise ValueError(f"Classical feedback requires gamma={GAMMA} and beta={BETA}")


def _wrap(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def classical_feedback(state: State, config: GameConfig = DEFAULT_CONFIG) -> Feedback | None:
    """Evaluate the supported branches and their continuous time to capture.

    ``None`` means the state is outside the verified domain. The capture
    branch has zero time and no controls. Use the simulator's float64 state.
    """
    _check_parameters(config)
    x, y = state
    if not (isfinite(x) and isfinite(y)):
        raise ValueError("Classical feedback requires finite coordinates")

    u = abs(x)
    if hypot(u, y) <= BETA:
        return Feedback("capture", None, None, 0.0)
    if u <= AXIS_TOLERANCE and y > BETA:
        return Feedback("UL", 0.0, 0.0, (y - BETA) / (1 - GAMMA))
    side = 1.0 if x > 0 else -1.0

    # Merz §2.2. Recover the time a to the axis and the ordinate y1 there.
    h_squared = u * u + y * y - 2 * u
    if h_squared >= 0:
        h = sqrt(h_squared)
        a = atan2(y - h * (1 - u), 1 - u + h * y)
        y1 = h + GAMMA * a
        tolerance = BOUNDARY_TOLERANCE
        if 0 < a <= 0.55 + tolerance and 2.2 - tolerance <= y1 <= 2.4 + tolerance:
            residual = hypot(1 - cos(a) + h * sin(a) - u, sin(a) + h * cos(a) - y)
            if residual <= RESIDUAL_TOLERANCE:
                time_to_capture = a + (y1 - BETA) / (1 - GAMMA)
                return Feedback("turn_to_UL", side, _wrap(side * a), time_to_capture)

    if y < 0.26:
        return None

    def radial_residual(time: float) -> float:
        # Same sign and zeros as F in §5, since beta - gamma*time is positive.
        return hypot(u - 1 + cos(time), y - sin(time)) - (BETA - GAMMA * time)

    lo, hi = 0.0, 0.1
    if radial_residual(hi) > BOUNDARY_TOLERANCE:
        return None
    for _ in range(50):
        middle = (lo + hi) / 2
        if radial_residual(middle) > 0:
            lo = middle
        else:
            hi = middle
    time = (lo + hi) / 2
    angle = atan2(u - 1 + cos(time), y - sin(time))
    theta = angle - time
    if 0.2 - BOUNDARY_TOLERANCE <= theta <= 0.5 + BOUNDARY_TOLERANCE and theta < USABLE_PART_ANGLE:
        px, py = retrograde_position(theta, time)
        if hypot(px - u, py - y) <= RESIDUAL_TOLERANCE:
            # In this interval theta>0 and theta+time<pi, so S>0 throughout.
            return Feedback("primary", side, _wrap(side * angle), time)
    return None


def classical_pursuer(state: State, config: GameConfig = DEFAULT_CONFIG) -> float | None:
    """Return the chauffeur's curvature, or None outside play on a verified branch."""
    feedback = classical_feedback(state, config)
    return feedback.phi if feedback is not None else None


def classical_evader(state: State, config: GameConfig = DEFAULT_CONFIG) -> float | None:
    """Return the pedestrian's relative heading, or None outside play on a verified branch."""
    feedback = classical_feedback(state, config)
    return feedback.psi if feedback is not None else None


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    state: State
    # Controls on arrival at this point, absent at reset.
    phi: float | None = None
    psi: float | None = None
    axis_crossing: bool = False


@dataclass(frozen=True)
class Rollout:
    outcome: Literal["capture", "time_limit", "unsupported"]
    trajectory: tuple[TrajectoryPoint, ...]


def rollout_classical(
    start: State,
    config: GameConfig = DEFAULT_CONFIG,
    *,
    pursuer: Policy | None = None,
    evader: Policy | None = None,
) -> Rollout:
    """Run the classical pair or replace one side with a state-feedback policy.

    Check coverage at every decision, and recompute both controls after reaching
    the positive axis. Both policies see the same state in double precision.
    """
    _check_parameters(config)
    env = HomicidalChauffeurEnv(config)
    env.reset(options={"state": start})
    x, y = env.state
    state = float(x), float(y)
    trajectory = [TrajectoryPoint(0.0, state)]
    while True:
        feedback = classical_feedback(state, config)
        if feedback is None:
            return Rollout("unsupported", tuple(trajectory))
        phi = feedback.phi if pursuer is None else pursuer(state, config)
        psi = feedback.psi if evader is None else evader(state, config)
        if phi is None or psi is None:
            return Rollout("unsupported", tuple(trajectory))
        _, _, terminated, truncated, info = env.step_controls(
            phi, psi, stop_at_positive_axis=feedback.branch == "turn_to_UL"
        )
        x, y = info["state"]
        state = float(x), float(y)
        trajectory.append(
            TrajectoryPoint(
                info["elapsed_time"], state, info["phi"], info["psi"], info["axis_crossing"]
            )
        )
        if terminated:
            return Rollout("capture", tuple(trajectory))
        if truncated:
            return Rollout("time_limit", tuple(trajectory))
