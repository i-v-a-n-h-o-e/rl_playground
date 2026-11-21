"""Single-agent training environment for the homicidal-chauffeur simulator."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, classical_feedback, step_feedback
from chauffeur_evader.chauffeur_theory import BETA, GAMMA
from chauffeur_evader.homicidal_chauffeur import (
    GameConfig,
    HomicidalChauffeurEnv,
    Observation,
    Transition,
)


class ChauffeurPPOEnv(HomicidalChauffeurEnv):
    """Control phi against the same classical pedestrian used in evaluation."""

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
        """End at the shared decision event or the finite-horizon task deadline."""
        self._require_active()
        state = tuple(map(float, self.state))
        observation, _, terminated, truncated, info = step_feedback(
            self, action, feedback=classical_feedback(state, self.config)
        )
        reward = float(info["termination_reason"] == "capture") - info["dt"] / self.config.horizon
        return observation, reward, terminated or truncated, False, info
