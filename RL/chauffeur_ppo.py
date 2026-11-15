"""Single-agent training environment for the homicidal-chauffeur simulator."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from RL.chauffeur_classical import DEFAULT_CONFIG, classical_evader
from RL.chauffeur_theory import BETA, GAMMA
from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv, Observation, Transition


class ChauffeurPPOEnv(HomicidalChauffeurEnv):
    """Control phi against fixed classical feedback with psi=0 outside its coverage."""

    def __init__(self, config: GameConfig = DEFAULT_CONFIG) -> None:
        if (config.gamma, config.beta) != (GAMMA, BETA):
            raise ValueError(f"The fixed evader requires gamma={GAMMA} and beta={BETA}")
        super().__init__(config)
        self._distance_scale = max(abs(config.initial_low), abs(config.initial_high), config.beta)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, 0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

    def _observation(self) -> Observation:
        x, y = self.state
        r = np.hypot(x, y)
        x_scaled, y_scaled, r_scaled = (2 / np.pi) * np.arctan2((x, y, r), self._distance_scale)
        time_left = max(0.0, self.config.horizon - self.elapsed_time) / self.config.horizon
        return np.array([x_scaled, y_scaled, r_scaled, x / r, y / r, time_left], dtype=np.float32)

    def step(self, action: Observation) -> Transition:
        """Hold both controls through the RK4 step, including across the axis."""
        self._require_active()
        x, y = self.state
        psi = classical_evader((float(x), float(y)), self.config)
        observation, _, terminated, truncated, info = self.step_controls(
            action, 0.0 if psi is None else psi
        )
        reward = float(terminated) - info["dt"] / self.config.horizon
        return observation, reward, terminated, truncated, info
