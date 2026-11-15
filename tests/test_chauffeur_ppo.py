from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env as check_gym_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as check_sb3_env

from RL.chauffeur_classical import classical_evader
from RL.chauffeur_ppo import ChauffeurPPOEnv
from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv


class ChauffeurPPOTests(unittest.TestCase):
    def test_scaled_observations_and_fixed_evader(self) -> None:
        env = ChauffeurPPOEnv()
        simulator = HomicidalChauffeurEnv()
        for start in ((1.105240, 2.278470), (6, -8)):
            with self.subTest(start=start):
                observation, _ = env.reset(options={"state": start})
                simulator.reset(options={"state": start})
                x, y = start
                r = np.hypot(x, y)
                scaled = (2 / np.pi) * np.arctan(np.array([x, y, r]) / 4)
                np.testing.assert_allclose(observation, [*scaled, x / r, y / r, 1])
                psi = classical_evader(start)
                if start == (6, -8):
                    self.assertIsNone(psi)
                else:
                    self.assertAlmostEqual(psi, 0.5, places=6)
                psi = 0.0 if psi is None else psi
                observation, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
                simulator.step_controls(1.0, psi)
                np.testing.assert_array_equal(info["state"], simulator.state)
                self.assertEqual((info["phi"], info["psi"]), (1.0, psi))
                self.assertTrue(env.observation_space.contains(observation))
                self.assertAlmostEqual(observation[-1], 1 - info["dt"] / env.config.horizon)

    def test_positive_axis_crossing_keeps_the_regular_step(self) -> None:
        env = ChauffeurPPOEnv()
        # turn_to_UL with a=0.025, y1=2.3 crosses the axis before dt=0.05.
        env.reset(options={"state": (0.057494026879201826, 2.311782589395382)})
        _, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        self.assertAlmostEqual(info["psi"], 0.025)
        self.assertLess(info["state"][0], 0.0)
        self.assertFalse(info["axis_crossing"])
        self.assertEqual((terminated, truncated), (False, False))
        self.assertAlmostEqual(info["dt"], env.config.dt)
        self.assertAlmostEqual(reward, -env.config.dt / env.config.horizon)

    def test_representable_large_distance_is_finite(self) -> None:
        env = ChauffeurPPOEnv()
        start = (1e155, 1e155)
        observation, info = env.reset(options={"state": start})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(np.isfinite(info["distance"]))
        np.testing.assert_allclose(info["distance"], np.sqrt(2) * 1e155)

    def test_reward_and_episode_endings(self) -> None:
        env = ChauffeurPPOEnv(GameConfig(dt=0.05, horizon=0.12))
        action = np.array([0.0], dtype=np.float32)
        for start, expected_time, expected_flags, reason, expected_return in (
            ((0, 0.34), 0.08, (True, False), "capture", 1 - 0.08 / 0.12),
            ((0, 0.36), 0.12, (True, False), "capture", 0.0),
            ((2, -2), 0.12, (False, True), "time_limit", -1.0),
        ):
            with self.subTest(start=start):
                env.reset(options={"state": start})
                episode_return = 0.0
                for index in range(3):
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    if terminated or truncated:
                        break
                    self.assertIsNone(info["termination_reason"])
                    self.assertAlmostEqual(reward, -env.config.dt / env.config.horizon)
                self.assertEqual(index + 1, 2 if expected_time < 0.12 else 3)
                self.assertEqual((terminated, truncated), expected_flags)
                self.assertEqual(info["termination_reason"], reason)
                self.assertAlmostEqual(info["elapsed_time"], expected_time)
                self.assertAlmostEqual(observation[-1], 1 - expected_time / env.config.horizon)
                self.assertAlmostEqual(episode_return, expected_return)
                with self.assertRaises(gym.error.ResetNeeded):
                    env.step(action)

    def test_capture_at_horizon_after_many_steps(self) -> None:
        env = ChauffeurPPOEnv(GameConfig(dt=0.02, horizon=4.0))
        action = np.array([0.0], dtype=np.float32)
        for start_y, expected_flags, reason, expected_return in (
            (2.3, (True, False), "capture", 0.0),
            (2.3 + 1e-12, (False, True), "time_limit", -1.0),
        ):
            with self.subTest(start_y=start_y):
                env.reset(options={"state": (0, start_y)})
                episode_return = 0.0
                for _ in range(200):
                    _, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                self.assertEqual((terminated, truncated), expected_flags)
                self.assertEqual(info["termination_reason"], reason)
                self.assertEqual(info["elapsed_time"], env.config.horizon)
                self.assertAlmostEqual(episode_return, expected_return)

    def test_gymnasium_and_sb3_contracts(self) -> None:
        check_gym_env(ChauffeurPPOEnv(), skip_render_check=True)
        check_sb3_env(ChauffeurPPOEnv())

    def test_random_rollouts_have_finite_transitions(self) -> None:
        env = ChauffeurPPOEnv()
        env.action_space.seed(42)
        for seed in range(42, 58):
            observation, _ = env.reset(seed=seed)
            self.assertTrue(env.observation_space.contains(observation))
            for _ in range(300):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertTrue(np.isfinite([reward, info["distance"], info["elapsed_time"]]).all())
                self.assertTrue(np.isfinite(info["state"]).all())
                self.assertFalse(terminated and truncated)
                if terminated or truncated:
                    reason = "capture" if terminated else "time_limit"
                    self.assertEqual(info["termination_reason"], reason)
                    break
                self.assertIsNone(info["termination_reason"])
            else:
                self.fail("Episode exceeded the horizon")

    def test_ppo_accepts_environment(self) -> None:
        env = ChauffeurPPOEnv()
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=42, device="cpu")
        model.learn(total_timesteps=128)
        self.assertEqual(model.num_timesteps, 128)
        self.assertTrue(np.isfinite(model.rollout_buffer.rewards).all())
        observation, _ = env.reset(seed=42)
        action, _ = model.predict(observation, deterministic=True)
        self.assertTrue(env.action_space.contains(action))


if __name__ == "__main__":
    unittest.main()
