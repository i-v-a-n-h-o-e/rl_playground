from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env as check_gym_env
from stable_baselines3.common.env_checker import check_env as check_sb3_env

from RL.chauffeur_demo import run_episode
from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv, dynamics


class ChauffeurTests(unittest.TestCase):
    def test_fixed_state_replays_an_entire_episode_exactly(self) -> None:
        config = GameConfig(horizon=1.0)
        first = run_episode(config, (1.0, 2.5), 42, 0.7, -0.4)
        self.assertEqual(first, run_episode(config, (1.0, 2.5), 42, 0.7, -0.4))
        self.assertEqual(first[-1]["termination_reason"], "time_limit")

    def test_seed_replays_random_resets_and_varying_controls(self) -> None:
        env = HomicidalChauffeurEnv()

        def replay():
            observations = [env.reset(seed=2026)[0]]
            for phi, psi in ((0.7, 0.2), (-0.9, -2.3), (0, np.pi)):
                observations.append(env.step_controls(phi, psi)[0])
            return np.array(observations)

        np.testing.assert_array_equal(replay(), replay())
        env.reset(seed=7)
        sequence = [env.reset()[0] for _ in range(10)]
        env.reset(seed=7)
        np.testing.assert_array_equal(sequence, [env.reset()[0] for _ in range(10)])
        self.assertTrue(all(np.linalg.norm(obs[:2]) > env.config.beta for obs in sequence))
        self.assertTrue(all(np.all(np.abs(obs[:2]) <= 4) for obs in sequence))
        self.assertFalse(np.array_equal(env.reset(seed=1)[0], env.reset(seed=2)[0]))

    def test_rk4_agrees_with_classical_formula_and_exact_turn(self) -> None:
        config = GameConfig()
        state = np.array([1.0, 2.5])
        phi, psi, h = 0.7, -0.4, config.dt

        def f(s):
            return dynamics(s, phi, psi, config.gamma)

        k1 = f(state)
        k2 = f(state + h * k1 / 2)
        k3 = f(state + h * k2 / 2)
        k4 = f(state + h * k3)
        expected_rk4 = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Exact solution rotates around the equilibrium of this affine ODE.
        center = np.array(
            [(1 - config.gamma * np.cos(psi)) / phi, config.gamma * np.sin(psi) / phi]
        )
        angle = phi * h
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        exact = center + rotation @ (state - center)
        env = HomicidalChauffeurEnv(config)
        env.reset(options={"state": state})
        env.step_controls(phi, psi)
        np.testing.assert_allclose(env.state, expected_rk4, rtol=0, atol=1e-14)
        np.testing.assert_allclose(env.state, exact, rtol=0, atol=2e-9)

    def test_benchmark_p1_capture_at_four_seconds(self) -> None:
        trajectory = run_episode(GameConfig(), (0.0, 2.3), 42, 0, 0)
        self.assertEqual(len(trajectory) - 1, 80)
        for row in trajectory:
            np.testing.assert_allclose(row["state"], [0, 2.3 - 0.5 * row["time"]], atol=1e-12)
        last = trajectory[-1]
        self.assertAlmostEqual(last["time"], 4.0, places=10)
        self.assertTrue(last["terminated"])
        self.assertFalse(last["truncated"])
        self.assertEqual(last["termination_reason"], "capture")

    def test_straight_motion_and_clockwise_heading_signs(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.5))
        for psi, x in ((np.pi / 2, 0.25), (-np.pi / 2, -0.25)):
            with self.subTest(psi=psi):
                env.reset(options={"state": (0, -1)})
                obs, _, terminated, truncated, info = env.step_controls(0, psi)
                r = np.hypot(x, -1.5)
                np.testing.assert_allclose(info["state"], [x, -1.5], rtol=0, atol=1e-14)
                np.testing.assert_allclose(obs, [x, -1.5, r, x / r, -1.5 / r], atol=1e-7)
                self.assertEqual((terminated, truncated), (False, False))

    def test_maximum_turn_and_step_halving(self) -> None:
        # With psi=0, the relative orbit is a unit circle around (phi/2, 0).
        # After a quarter turn from the top, it reaches (-phi/2, 0).
        for phi in (1, -1):
            with self.subTest(phi=phi):
                errors = []
                for dt in (0.05, 0.025):
                    trace = run_episode(
                        GameConfig(dt=dt, horizon=np.pi / 2), (phi / 2, 1), 42, phi, 0
                    )
                    times = np.array([row["time"] for row in trace])
                    states = np.array([row["state"] for row in trace])
                    exact = np.column_stack((phi * (0.5 - np.sin(times)), np.cos(times)))
                    errors.append(np.max(np.linalg.norm(states - exact, axis=1)))
                    self.assertEqual(trace[-1]["termination_reason"], "time_limit")
                    self.assertEqual(times[-1], np.pi / 2)
                    np.testing.assert_allclose(states[-1], [-phi / 2, 0], rtol=0, atol=1e-7)
                self.assertLessEqual(errors[1], errors[0] / 10 + 1e-12)

    def test_capture_between_step_endpoints(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.5, horizon=1.0))
        env.reset(options={"state": (0, 0.4)})
        # With psi=pi, the unconstrained endpoint is y=-0.35, outside capture.
        _, _, terminated, truncated, info = env.step_controls(0, np.pi)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["elapsed_time"], (0.4 - 0.3) / 1.5)
        self.assertLessEqual(info["distance"], env.config.beta)

    def test_curved_capture_between_step_endpoints(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.5, horizon=1.0))
        # The circle centered at (0.5, 0) passes closest to the origin
        # at t=0.25. Both endpoints t=0 and t=0.5 are outside the disk.
        radius = 0.79
        start = (0.5 - radius * np.cos(0.25), radius * np.sin(0.25))
        self.assertGreater(np.linalg.norm(start), env.config.beta)
        angle = np.arccos((0.5**2 + radius**2 - env.config.beta**2) / radius)
        expected_time = 0.25 - angle
        self.assertLess(0.25 + angle, env.config.dt)
        env.reset(options={"state": start})
        _, _, terminated, truncated, info = env.step_controls(1, 0)
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(info["termination_reason"], "capture")
        self.assertAlmostEqual(info["elapsed_time"], expected_time, delta=2e-6)
        self.assertAlmostEqual(info["distance"], env.config.beta, places=14)

    def test_capture_has_priority_at_the_horizon(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.05, horizon=0.05))
        env.reset(options={"state": (0, 0.325)})
        _, _, terminated, truncated, info = env.step_controls(0, 0)
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(info["termination_reason"], "capture")
        self.assertAlmostEqual(info["elapsed_time"], env.config.horizon, places=14)
        self.assertAlmostEqual(info["distance"], env.config.beta, places=14)

    def test_capture_after_horizon_is_excluded(self) -> None:
        config = GameConfig(beta=0.25, dt=0.5, horizon=0.25)
        env = HomicidalChauffeurEnv(config)
        for delay in (0.01, 2e-12):
            with self.subTest(delay=delay):
                # With closing speed 0.5, contact would occur at horizon + delay.
                start = config.beta + 0.5 * (config.horizon + delay)
                env.reset(options={"state": (0, start)})
                _, _, terminated, truncated, info = env.step_controls(0, 0)
                self.assertEqual((terminated, truncated), (False, True))
                self.assertEqual(info["termination_reason"], "time_limit")
                self.assertEqual(info["elapsed_time"], config.horizon)
                self.assertEqual(info["dt"], config.horizon)
                self.assertGreater(info["distance"], config.beta)

    def test_initial_state_on_or_inside_capture_disk_is_rejected(self) -> None:
        env = HomicidalChauffeurEnv()
        for state in ((0, 0), (0.1, -0.1), (0, env.config.beta)):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(ValueError, "outside the capture circle"),
            ):
                env.reset(options={"state": state})

    def test_tangent_contact_counts_as_capture(self) -> None:
        env = HomicidalChauffeurEnv()
        env.reset(options={"state": (0.3, 0.01)})
        _, _, terminated, truncated, info = env.step_controls(0, 0)
        self.assertEqual((terminated, truncated), (True, False))
        self.assertAlmostEqual(info["elapsed_time"], 0.02)
        np.testing.assert_allclose(env.state, [0.3, 0], atol=1e-12)

    def test_time_limit_shortens_last_step_and_prevents_further_steps(self) -> None:
        for config, expected_steps in ((GameConfig(horizon=0.12), 3), (GameConfig(), 400)):
            horizon = config.horizon
            with self.subTest(horizon=horizon):
                env = HomicidalChauffeurEnv(config)
                env.reset(options={"state": (2, -2)})
                while True:
                    _, reward, terminated, truncated, info = env.step_controls(0, 0)
                    self.assertEqual(reward, 0.0)
                    if terminated or truncated:
                        break
                self.assertEqual((terminated, truncated), (False, True))
                self.assertEqual(info["elapsed_time"], horizon)
                self.assertEqual(info["step_count"], expected_steps)
                self.assertEqual(info["termination_reason"], "time_limit")
                np.testing.assert_allclose(env.state, [2, -2 - 0.5 * horizon], atol=1e-12)
                with self.assertRaises(gym.error.ResetNeeded):
                    env.step_controls(0, 0)
                _, info = env.reset(options={"state": (1, 2)})
                self.assertIsNone(info["termination_reason"])
                self.assertEqual(info["elapsed_time"], 0)

    def test_controls_clip_and_invalid_controls_do_not_advance(self) -> None:
        env = HomicidalChauffeurEnv()
        env.reset(options={"state": (1, 2)})
        _, _, _, _, info = env.step_controls(99, -99)
        self.assertEqual((info["phi"], info["psi"]), (1, -np.pi))
        before = env.state
        for phi, psi in ((np.nan, 0), (0, np.inf), ([1, 2], 0), (0, [[1]])):
            with self.subTest(phi=phi, psi=psi), self.assertRaises(ValueError):
                env.step_controls(phi, psi)
            np.testing.assert_array_equal(env.state, before)

    def test_gym_step_calls_evader_with_current_observation(self) -> None:
        calls = []

        def evader(observation):
            calls.append(observation.copy())
            observation[:] = 0  # The policy cannot mutate the simulator state.
            return -0.4

        env = HomicidalChauffeurEnv(evader_policy=evader)
        observation, _ = env.reset(options={"state": (1, 2)})
        transition = env.step(np.array([0.7], dtype=np.float32))
        np.testing.assert_array_equal(calls[0], observation)
        reference = HomicidalChauffeurEnv()
        reference.reset(options={"state": (1, 2)})
        reference.step_controls(float(np.float32(0.7)), -0.4)
        np.testing.assert_array_equal(env.state, reference.state)
        self.assertEqual(transition[-1]["psi"], -0.4)

    def test_state_observation_and_plane_are_not_clipped(self) -> None:
        env = HomicidalChauffeurEnv()
        with self.assertRaises(gym.error.ResetNeeded):
            env.step_controls(0, 0)
        observation, info = env.reset(options={"state": (6, 8)})
        np.testing.assert_allclose(observation, [6, 8, 10, 0.6, 0.8])
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        info["state"][:] = 0
        copy = env.state
        copy[:] = 0
        np.testing.assert_array_equal(env.state, [6, 8])
        env.step_controls(0, 0)
        self.assertGreater(env.state[0], 4)

    def test_invalid_configuration_and_initial_states(self) -> None:
        for kwargs in (
            {"gamma": 0},
            {"gamma": 1},
            {"beta": 0},
            {"dt": -1},
            {"horizon": 0},
            {"dt": np.nan},
            {"initial_low": 4},
            {"beta": 10},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GameConfig(**kwargs)
        env = HomicidalChauffeurEnv()
        for state in ((np.inf, 1), (1,), ((1, 2),)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                env.reset(options={"state": state})

    def test_gymnasium_and_sb3_contracts(self) -> None:
        check_gym_env(HomicidalChauffeurEnv(), skip_render_check=True)
        check_sb3_env(HomicidalChauffeurEnv(), warn=True)


if __name__ == "__main__":
    unittest.main()
