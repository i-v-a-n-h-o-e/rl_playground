from __future__ import annotations

import unittest
from itertools import pairwise
from math import ceil, cos, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env as check_gym_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as check_sb3_env
from stable_baselines3.common.vec_env import DummyVecEnv

from chauffeur_evader.chauffeur_classical import classical_evader, rollout_classical
from chauffeur_evader.chauffeur_evaluation import _ppo_rollouts, evaluate_ppo
from chauffeur_evader.chauffeur_ppo import ChauffeurPPOEnv
from chauffeur_evader.chauffeur_theory import USABLE_PART_ANGLE, barrier_coordinates, geometry
from chauffeur_evader.homicidal_chauffeur import GameConfig


class ChauffeurPPOTests(unittest.TestCase):
    def assert_protected_replay(self, env, start, steering):
        decisions = []

        def record(state, config, observation):
            action = steering(observation, len(decisions))
            decisions.append((observation, action))
            return action

        (run,) = _ppo_rollouts(record, [start], env.config, clearance=0.001)
        obs, reset = env.reset(options={"state": start})
        self.assertEqual((reset["elapsed_time"], reset["step_count"]), (0.0, 0))
        for (before, after), (expected_obs, action) in zip(
            pairwise(run.trajectory), decisions, strict=True
        ):
            np.testing.assert_array_equal(obs, expected_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(info["state"], after.state)
            self.assertEqual((info["phi"], info["psi"]), (after.phi, after.psi))
            self.assertEqual(info["event"], after.event)
            self.assertAlmostEqual(info["dt"], after.time - before.time, places=13)
            self.assertEqual(info["elapsed_time"], after.time)
            last = after.time == run.trajectory[-1].time
            reason = run.outcome if last else None
            self.assertEqual(
                (terminated, truncated, info["termination_reason"]), (last, False, reason)
            )
            expected_reward = float(reason == "capture") - info["dt"] / env.config.horizon
            self.assertAlmostEqual(reward, expected_reward)
        return run

    def test_protected_steps_and_reset_across_barrier_and_capture(self) -> None:
        # Reuse one training env. Each evaluation rollout starts fresh.
        env = ChauffeurPPOEnv(clearance=0.001)
        x, y = geometry().b(1)
        p = USABLE_PART_ANGLE + 1
        near = x + 0.0005 * sin(p), y + 0.0005 * cos(p)
        narrow = self.assert_protected_replay(env, near, lambda *_: 1)
        hit = narrow.trajectory[1]
        self.assertEqual(hit.event, "barrier")
        self.assertAlmostEqual(barrier_coordinates(hit.state)[1], 0.30025, places=13)

        full = self.assert_protected_replay(env, (0.45, 1), lambda *_: 1)
        hit = next(pt for pt in full.trajectory if pt.event == "barrier")
        self.assertAlmostEqual(hit.time, 0.116569934, places=8)
        self.assertAlmostEqual(barrier_coordinates(hit.state)[1], 0.301, places=13)
        self.assertEqual((full.outcome, full.trajectory[-1].time), ("time_limit", 20.0))

        off_line = self.assert_protected_replay(env, (x, y), lambda *_: -2)
        self.assertEqual(off_line.trajectory[1].phi, -1)
        self.assertAlmostEqual(off_line.trajectory[1].psi, -1.41466618, places=8)
        tangent = self.assert_protected_replay(env, geometry().b(0.025), lambda *_: 2)
        self.assertEqual((tangent.outcome, tangent.trajectory[1].event), ("capture", "contact"))
        self.assertAlmostEqual(tangent.trajectory[1].time, 0.025)

        crossing = self.assert_protected_replay(env, (0, 0.31), lambda *_: 0)
        self.assertEqual(crossing.outcome, "capture")
        self.assertEqual(
            [pt.event for pt in crossing.trajectory], [None, "virtual_entry", "contact"]
        )
        np.testing.assert_allclose([pt.time for pt in crossing.trajectory], [0, 0.018, 0.020])

    def test_protected_p4_off_nominal_interval(self) -> None:
        start = (0.27991134764129716, 0.12293859570612813)  # P4 at t=1
        run = self.assert_protected_replay(
            ChauffeurPPOEnv(clearance=0.001),
            start,
            lambda obs, step: 0.9054176807403564 if step == 0 else 1,
        )
        entry, following = run.trajectory[1:3]
        self.assertEqual(entry.event, "virtual_entry")
        self.assertAlmostEqual(entry.time, 0.036148675, places=9)
        self.assertAlmostEqual(np.hypot(*entry.state), 0.301, places=13)
        self.assertAlmostEqual(following.psi, 0.469786958, places=8)

    def test_seed_43_checkpoint_replays_protected_p4(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "data/chauffeur-task10/length-wide-potential/seed-43/ppo-262144.zip"
        if not checkpoint.exists():
            self.skipTest("Local Task 10 checkpoint is required for the replay")
        env = ChauffeurPPOEnv(clearance=0.001)
        model = PPO.load(checkpoint, env=env, device="cpu")
        self.assertEqual((model.seed, model.num_timesteps), (43, 262144))
        run = self.assert_protected_replay(
            env, (0.45, 1), lambda obs, _: model.predict(obs, deterministic=True)[0].item()
        )
        before, hit = next((a, b) for a, b in pairwise(run.trajectory) if abs(a.time - 1) < 1e-12)
        self.assertAlmostEqual(hit.phi, 0.905418, places=5)
        self.assertEqual(hit.event, "virtual_entry")
        self.assertAlmostEqual(hit.time - before.time, 0.036148675, delta=1e-7)
        self.assertEqual((run.outcome, run.trajectory[-1].time), ("time_limit", 20.0))

    def test_scaled_observations_and_fixed_evader(self) -> None:
        env = ChauffeurPPOEnv()
        for start in ((1.105240, 2.278470), (6, -8)):
            with self.subTest(start=start):
                observation, _ = env.reset(options={"state": start})
                x, y = start
                r = np.hypot(x, y)
                scaled = (2 / np.pi) * np.arctan(np.array([x, y, r]) / 4)
                np.testing.assert_allclose(observation, [*scaled, x / r, y / r, 1])
                psi = classical_evader(start)
                self.assertIsNotNone(psi)
                if start != (6, -8):
                    self.assertAlmostEqual(psi, 0.5, places=6)
                observation, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
                baseline = rollout_classical(start, pursuer=lambda state, config: 1.0)
                np.testing.assert_array_equal(info["state"], baseline.trajectory[1].state)
                self.assertEqual((info["phi"], info["psi"]), (1.0, psi))
                self.assertTrue(env.observation_space.contains(observation))
                self.assertAlmostEqual(observation[-1], 1 - info["dt"] / env.config.horizon)

    def test_positive_axis_crossing_returns_a_new_decision_to_ppo(self) -> None:
        env = ChauffeurPPOEnv()
        # turn_to_UL with a=0.025, y1=2.3 crosses the axis before dt=0.05.
        env.reset(options={"state": (0.057494026879201826, 2.311782589395382)})
        _, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        self.assertAlmostEqual(info["psi"], 0.025)
        self.assertEqual(info["state"][0], 0.0)
        self.assertTrue(info["axis_crossing"])
        self.assertEqual((terminated, truncated), (False, False))
        self.assertAlmostEqual(info["dt"], 0.025)
        self.assertAlmostEqual(reward, -0.025 / env.config.horizon)
        _, _, _, _, next_info = env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual((next_info["phi"], next_info["psi"], next_info["state"][0]), (0, 0, 0))
        # The event also applies when the chauffeur deviates from the reference.
        env.reset(options={"state": (0.057494026879201826, 2.311782589395382)})
        _, _, _, _, other = env.step(np.array([0.75], dtype=np.float32))
        self.assertTrue(other["axis_crossing"])
        self.assertLess(0.025, other["dt"])
        self.assertLess(other["dt"], env.config.dt)

    def test_axis_pdl_pedestrian_sees_clipped_actual_steering(self) -> None:
        env = ChauffeurPPOEnv()
        for action, phi, psi in (
            (8, 1, -2.498091545),
            (-8, -1, 2.498091545),
            (0, 0, -2.498091545),
        ):
            env.reset(options={"state": (0, -3)})
            _, _, _, _, info = env.step(np.array([action], dtype=np.float32))
            self.assertEqual(info["phi"], phi)
            self.assertAlmostEqual(info["psi"], psi)

    def test_barrier_crossing_does_not_repeat_after_snapping_to_the_curve(self) -> None:
        env = ChauffeurPPOEnv()
        for side in (-1, 1):
            with self.subTest(side=side):
                # P1 just before the report's first barrier event.
                env.reset(options={"state": (side * 0.9377220624501995, 1.5130425617408336)})
                _, _, terminated, truncated, hit = env.step(-side * 0.22327621281147003)
                self.assertFalse(terminated or truncated)
                self.assertEqual(hit["event"], "barrier")
                self.assertAlmostEqual(hit["dt"], 0.004972689, places=8)

                _, _, terminated, truncated, after = env.step(-side * 0.222512885928154)
                self.assertFalse(terminated or truncated)
                self.assertIsNone(after["event"])
                self.assertEqual(after["dt"], env.config.dt)

    def test_unavailable_evader_is_a_separate_outcome_without_capture_bonus(
        self,
    ) -> None:
        env = ChauffeurPPOEnv()
        env.reset(options={"state": (0, -3)})
        with patch("chauffeur_evader.chauffeur_classical.classical_feedback", return_value=None):
            _, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        self.assertEqual((reward, terminated, truncated), (0.0, True, False))
        self.assertEqual(info["termination_reason"], "unsupported")
        self.assertEqual(info["elapsed_time"], 0.0)

    def test_sb3_does_not_bootstrap_at_the_finite_horizon(self) -> None:
        env = DummyVecEnv([lambda: ChauffeurPPOEnv(GameConfig(horizon=0.04), clearance=0.001)])
        env.set_options({"state": (2, -2)})
        env.reset()
        _, rewards, dones, infos = env.step(np.zeros((1, 1), dtype=np.float32))
        self.assertTrue(dones[0])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertEqual(infos[0]["termination_reason"], "time_limit")
        self.assertEqual(rewards[0], -1.0)
        self.assertEqual(infos[0]["terminal_observation"][-1], 0.0)
        env.close()

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
            ((2, -2), 0.12, (True, False), "time_limit", -1.0),
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
            (2.3 + 1e-12, (True, False), "time_limit", -1.0),
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
        step_limit = 2 * ceil(env.config.horizon / env.config.dt)
        env.action_space.seed(42)
        for seed in range(42, 58):
            observation, _ = env.reset(seed=seed)
            self.assertTrue(env.observation_space.contains(observation))
            for _ in range(step_limit):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertTrue(np.isfinite([reward, info["distance"], info["elapsed_time"]]).all())
                self.assertTrue(np.isfinite(info["state"]).all())
                self.assertFalse(terminated and truncated)
                if terminated or truncated:
                    self.assertIn(info["termination_reason"], ("capture", "time_limit"))
                    break
                self.assertIsNone(info["termination_reason"])
            else:
                self.fail("Episode exceeded the horizon")

    def test_ppo_accepts_environment(self) -> None:
        env = ChauffeurPPOEnv(clearance=0.001)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            seed=42,
            device="cpu",
        )
        model.learn(total_timesteps=128)
        self.assertEqual(model.num_timesteps, 128)
        self.assertTrue(np.isfinite(model.rollout_buffer.rewards).all())
        observation, _ = env.reset(seed=42)
        action, _ = model.predict(observation, deterministic=True)
        self.assertTrue(env.action_space.contains(action))
        with TemporaryDirectory() as directory:
            target = Path(directory) / "ppo.zip"
            model.save(target)
            reloaded = PPO.load(target, device="cpu")

            def policy(_state, _config, obs):
                action, _ = reloaded.predict(obs, deterministic=True)
                return float(action.item())

            result = evaluate_ppo(policy, [(0, -3), (0, 2.3)], GameConfig(horizon=0.12))
        self.assertEqual([r.outcome for r in result], ["time_limit", "time_limit"])


if __name__ == "__main__":
    unittest.main()
