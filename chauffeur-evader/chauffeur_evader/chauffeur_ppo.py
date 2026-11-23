"""Single-agent training environment for the homicidal-chauffeur simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, FeedbackEpisode, requested_radius
from chauffeur_evader.chauffeur_theory import BETA, GAMMA
from chauffeur_evader.homicidal_chauffeur import (
    GameConfig,
    HomicidalChauffeurEnv,
    Observation,
    Transition,
)


class ChauffeurPPOEnv(HomicidalChauffeurEnv):
    """Control phi against the same classical pedestrian used in evaluation."""

    def __init__(self, config: GameConfig = DEFAULT_CONFIG, *, clearance: float = 0.0) -> None:
        if (config.gamma, config.beta) != (GAMMA, BETA):
            raise ValueError(f"The fixed evader requires gamma={GAMMA} and beta={BETA}")
        super().__init__(config)
        self._requested_radius = requested_radius(config, clearance)
        self._distance_scale = max(abs(config.initial_low), abs(config.initial_high), config.beta)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, 0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        result = super().reset(seed=seed, options=options)
        self._episode = FeedbackEpisode(self._requested_radius)
        return result

    def _observation(self) -> Observation:
        x, y = self.state
        r = np.hypot(x, y)
        x_scaled, y_scaled, r_scaled = (2 / np.pi) * np.arctan2((x, y, r), self._distance_scale)
        time_left = max(0.0, self.config.horizon - self.elapsed_time) / self.config.horizon
        return np.array([x_scaled, y_scaled, r_scaled, x / r, y / r, time_left], dtype=np.float32)

    def step(self, action: Observation) -> Transition:
        """End at the shared decision event or the finite-horizon task deadline."""
        self._require_active()
        observation, _, terminated, truncated, info = self._episode.step(self, action)
        reward = float(info["termination_reason"] == "capture") - info["dt"] / self.config.horizon
        return observation, reward, terminated or truncated, False, info
