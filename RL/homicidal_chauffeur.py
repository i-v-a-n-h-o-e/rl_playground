"""Minimal normalized homicidal-chauffeur game, without training.

Angles are clockwise from the pursuer's forward axis (+y). Both controls are
held constant in the moving coordinate system throughout each RK4 step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

Observation = NDArray[np.float32]
EvaderPolicy = Callable[[Observation], float]
Transition = tuple[Observation, float, bool, bool, dict[str, Any]]


@dataclass(frozen=True)
class GameConfig:
    gamma: float = 0.5
    beta: float = 0.3
    dt: float = 0.05
    horizon: float = 15.0
    initial_low: float = -4.0
    initial_high: float = 4.0

    def __post_init__(self) -> None:
        values = (self.gamma, self.beta, self.dt, self.horizon, self.initial_low, self.initial_high)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Configuration values must be finite")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if min(self.beta, self.dt, self.horizon) <= 0:
            raise ValueError("beta, dt and horizon must be positive")
        if self.initial_low >= self.initial_high:
            raise ValueError("initial_low must be less than initial_high")
        if np.sqrt(2) * max(abs(self.initial_low), abs(self.initial_high)) <= self.beta:
            raise ValueError("Initial square must contain states outside the capture circle")


def dynamics(state: ArrayLike, phi: float, psi: float, gamma: float) -> NDArray[np.float64]:
    """Relative dynamics: x'=-phi*y+gamma*sin(psi), y'=-1+phi*x+gamma*cos(psi)."""
    x, y = np.asarray(state, dtype=np.float64)
    return np.array([-phi * y + gamma * np.sin(psi), -1.0 + phi * x + gamma * np.cos(psi)])


def _rk4_path(
    state: NDArray[np.float64], phi: float, psi: float, gamma: float, dt: float
) -> NDArray[np.float64]:
    """Coefficients of the RK4 state at fractional time u in [0, 1].

    For this affine ODE, RK4 is s + t*f(s) + t^2*A*f(s)/2 + ... + t^4*A^3*f(s)/24.
    Evaluating the polynomial at u=1 is exactly one classical RK4 step.
    """
    coefficients = np.empty((5, 2), dtype=np.float64)
    coefficients[0] = state
    term = dt * dynamics(state, phi, psi, gamma)
    for order in range(1, 5):
        coefficients[order] = term
        term = dt * phi * np.array([-term[1], term[0]]) / (order + 1)
    return coefficients


def _capture_fraction(coefficients: NDArray[np.float64], beta: float) -> float | None:
    """First intersection with the capture circle along the numerical RK4 path.

    Checking the whole path also catches entry and exit within a single step.
    Normalizing time to [0, 1] avoids roots in powers of a small dt.
    """
    polynomial = np.polynomial.polynomial
    radius_squared = polynomial.polyadd(
        polynomial.polymul(coefficients[:, 0], coefficients[:, 0]),
        polynomial.polymul(coefficients[:, 1], coefficients[:, 1]),
    )
    radius_squared[0] -= beta**2
    roots = polynomial.polyroots(radius_squared)
    candidates = [
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1e-8 and 0 < root.real <= 1
    ]
    # Tangencies are double roots: roundoff can move them off the real axis.
    # Stationary points of squared distance remain well-conditioned there.
    tolerance = 64 * np.finfo(np.float64).eps * max(beta**2, np.sum(abs(radius_squared)))
    for root in polynomial.polyroots(polynomial.polyder(radius_squared)):
        if (
            abs(root.imag) <= 1e-8
            and 0 < root.real <= 1
            and abs(polynomial.polyval(root.real, radius_squared)) <= tolerance
        ):
            candidates.append(float(root.real))
    if candidates:
        return min(candidates)
    # A root at the endpoint can round above 1. Check the endpoint itself
    # with a roundoff allowance in distance, independent of the step length.
    endpoint = polynomial.polyval(1.0, coefficients)
    if np.linalg.norm(endpoint) - beta <= 64 * np.finfo(np.float64).eps * beta:
        return 1.0
    return None


class HomicidalChauffeurEnv(gym.Env[Observation, NDArray[np.float32]]):
    """Two-control simulator with a pursuer-only Gymnasium interface.

    ``step_controls(phi, psi)`` accepts both players' controls directly.
    ``step([phi])`` obtains psi from ``evader_policy(observation)``; the default
    is psi=0, a fixed relative heading, not an optimal evasion strategy.
    Reward is deliberately 0.0 at this mechanics-only stage.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self, config: GameConfig | None = None, evader_policy: EvaderPolicy | None = None
    ) -> None:
        super().__init__()
        self.config = config or GameConfig()
        self.evader_policy = evader_policy
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, -1, -1], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self._state: NDArray[np.float64] | None = None
        self.elapsed_time = 0.0
        self.step_count = 0
        self.termination_reason: str | None = None

    @property
    def state(self) -> NDArray[np.float64]:
        """A copy of the internal (x, y) state, in double precision."""
        if self._state is None:
            raise gym.error.ResetNeeded("Call reset() before accessing state or stepping")
        return self._state.copy()

    def _observation(self) -> Observation:
        x, y = self.state
        r = np.hypot(x, y)
        return np.array([x, y, r, x / r, y / r], dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "distance": float(np.linalg.norm(self.state)),
            "elapsed_time": self.elapsed_time,
            "step_count": self.step_count,
            "termination_reason": self.termination_reason,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset randomly outside capture, or to options={\"state\": (x, y)}.

        Explicit states may lie anywhere on the plane, but must be finite and
        strictly outside the capture circle. Random starts use the configured square.
        """
        super().reset(seed=seed)
        if options is not None and "state" in options:
            state = np.array(options["state"], dtype=np.float64, copy=True)
            if state.shape != (2,) or not np.all(np.isfinite(state)):
                raise ValueError("Initial state must be a finite vector of shape (2,)")
            if np.linalg.norm(state) <= self.config.beta:
                raise ValueError("Initial state must be outside the capture circle")
        else:
            while True:
                state = self.np_random.uniform(
                    self.config.initial_low, self.config.initial_high, size=2
                )
                if np.linalg.norm(state) > self.config.beta:
                    break
        self._state = state
        self.elapsed_time = 0.0
        self.step_count = 0
        self.termination_reason = None
        return self._observation(), self._info()

    @staticmethod
    def _control(value: ArrayLike, bound: float, name: str) -> float:
        control = np.asarray(value, dtype=np.float64)
        if control.shape not in ((), (1,)) or not np.all(np.isfinite(control)):
            raise ValueError(f"{name} must be a finite scalar or a vector of shape (1,)")
        return float(np.clip(control.item(), -bound, bound))

    def _require_active(self) -> None:
        if self._state is None or self.termination_reason is not None:
            raise gym.error.ResetNeeded("Call reset() before starting a new episode")

    def step(self, action: NDArray[np.float32]) -> Transition:
        self._require_active()
        phi = self._control(action, 1.0, "phi")
        psi = 0.0 if self.evader_policy is None else self.evader_policy(self._observation())
        return self.step_controls(phi, psi)

    def step_controls(self, phi: ArrayLike, psi: ArrayLike) -> Transition:
        """Advance both players; clip phi to [-1,1] and psi to [-pi,pi].

        Capture takes precedence if it coincides with the time limit. The last
        step is shortened to end exactly at the horizon or at numerical capture.
        """
        self._require_active()
        phi = self._control(phi, 1.0, "phi")
        psi = self._control(psi, np.pi, "psi")
        remaining = self.config.horizon - self.elapsed_time
        # Avoid an extra step caused by accumulated floating-point roundoff.
        at_horizon = remaining <= self.config.dt + 1e-12
        dt = remaining if at_horizon else self.config.dt
        state = self.state
        coefficients = _rk4_path(state, phi, psi, self.config.gamma, dt)
        # Radial speed is bounded by 1+gamma; distant states cannot be captured.
        fraction = None
        if np.linalg.norm(state) - (1 + self.config.gamma) * dt <= self.config.beta:
            fraction = _capture_fraction(coefficients, self.config.beta)
        terminated = fraction is not None
        fraction = 1.0 if fraction is None else fraction
        self._state = np.polynomial.polynomial.polyval(fraction, coefficients)
        self.elapsed_time = (
            self.config.horizon
            if at_horizon and not terminated
            else self.elapsed_time + dt * fraction
        )
        self.step_count += 1
        truncated = at_horizon and not terminated
        if terminated:
            # Remove root-solver roundoff at the capture boundary.
            self._state *= np.nextafter(self.config.beta, 0.0) / np.linalg.norm(self._state)
            self.termination_reason = "capture"
        elif truncated:
            self.termination_reason = "time_limit"
        info = self._info()
        info.update(phi=phi, psi=psi, dt=dt * fraction)
        return self._observation(), 0.0, terminated, truncated, info
