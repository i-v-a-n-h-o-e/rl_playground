from __future__ import annotations

import unittest
from math import hypot

import numpy as np

from chauffeur_evader.chauffeur_ppo import ChauffeurPPOEnv
from chauffeur_evader.chauffeur_task10.common import PotentialRewardEnv
from chauffeur_evader.homicidal_chauffeur import GameConfig


class PotentialRewardTests(unittest.TestCase):
    def test_discounted_bonus_telescopes_at_capture_and_timeout(self) -> None:
        config = GameConfig(horizon=0.12)
        action = np.zeros(1, dtype=np.float32)
        for start, outcome in (((0, 0.34), "capture"), ((2, -2), "time_limit")):
            with (
                self.subTest(outcome=outcome),
                ChauffeurPPOEnv(config) as base,
                PotentialRewardEnv(config) as shaped,
            ):
                for env in (base, shaped):
                    env.reset(options={"state": start})
                discounted_bonus = 0.0
                for step in range(3):
                    observation, reward, terminated, truncated, info = base.step(action)
                    shaped_obs, shaped_reward, end, trunc, shaped_info = shaped.step(action)
                    np.testing.assert_array_equal(shaped_obs, observation)
                    self.assertEqual((end, trunc), (terminated, truncated))
                    self.assertEqual(shaped_info["termination_reason"], info["termination_reason"])
                    discounted_bonus += 0.99**step * (shaped_reward - reward)
                    if terminated:
                        break
                self.assertTrue(terminated)
                self.assertEqual(info["termination_reason"], outcome)
                initial_potential = 0.1 / (1 + (hypot(*start) - 0.3) / 4)
                self.assertAlmostEqual(discounted_bonus, -initial_potential)
