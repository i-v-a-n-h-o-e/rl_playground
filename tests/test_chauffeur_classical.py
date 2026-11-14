from __future__ import annotations

import unittest
from math import cos, pi, sin

import numpy as np

from RL.chauffeur_classical import (
    classical_evader,
    classical_feedback,
    classical_pursuer,
    rollout_classical,
)
from RL.chauffeur_theory import BARRIER_END_TIME, retrograde_position, tangential_barrier
from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv


class ClassicalChauffeurTests(unittest.TestCase):
    def test_straight_pursuit(self) -> None:
        for dt in (0.05, 0.025):
            with self.subTest(dt=dt):
                config = GameConfig(dt=dt, horizon=15.0)
                self.assertEqual(classical_pursuer((0.0, 2.3), config), 0.0)
                self.assertEqual(classical_evader((0.0, 2.3), config), 0.0)
                run = rollout_classical((0.0, 2.3), config)
                self.assertEqual(run.outcome, "capture")
                self.assertAlmostEqual(run.trajectory[-1].time, 4.0, places=10)
                for point in run.trajectory[1:]:
                    self.assertEqual((point.phi, point.psi, point.axis_crossing), (0, 0, False))
                    np.testing.assert_allclose(point.state, (0, 2.3 - 0.5 * point.time), atol=1e-12)

    def test_turn_to_axis_under_feedback_and_its_reflection(self) -> None:
        start = (1.105240, 2.278470)
        for dt, expected_switch, expected_y, expected_capture in (
            (0.05, 0.502863, 2.298466, 4.499794),
            (0.025, 0.501435, 2.299257, 4.499949),
        ):
            config = GameConfig(dt=dt, horizon=15.0)
            for side in (1, -1):
                with self.subTest(dt=dt, side=side):
                    state = side * start[0], start[1]
                    self.assertEqual(classical_pursuer(state, config), side)
                    self.assertAlmostEqual(classical_evader(state, config), side * 0.5, places=6)
                    run = rollout_classical(state, config)
                    self.assertEqual(run.outcome, "capture")
                    self.assertAlmostEqual(run.trajectory[-1].time, expected_capture, places=6)
                    switches = [i for i, p in enumerate(run.trajectory) if p.axis_crossing]
                    self.assertEqual(len(switches), 1)
                    index = switches[0]
                    switch = run.trajectory[index]
                    self.assertEqual(switch.state[0], 0.0)
                    self.assertAlmostEqual(switch.state[1], expected_y, places=6)
                    self.assertAlmostEqual(switch.time, expected_switch, places=6)
                    self.assertLess(switch.time - run.trajectory[index - 1].time, dt)
                    turn = run.trajectory[1 : index + 1]
                    self.assertTrue(all(p.phi == side and 0 < side * p.psi < pi for p in turn))
                    self.assertTrue(
                        all(
                            (p.phi, p.psi, p.state[0]) == (0, 0, 0)
                            for p in run.trajectory[index + 1 :]
                        )
                    )
                    if dt == 0.025:
                        self.assertAlmostEqual(switch.time, 0.5, delta=2e-3)
                        self.assertAlmostEqual(run.trajectory[-1].time, 4.5, delta=1e-4)

    def test_prescribed_switch_from_the_source(self) -> None:
        # Check the open-loop mechanics replay separately from the state-based switch.
        for dt, expected in ((0.05, 4.499922), (0.025, 4.499980)):
            with self.subTest(dt=dt):
                turn = HomicidalChauffeurEnv(GameConfig(dt=dt, horizon=0.5))
                turn.reset(options={"state": (1.105240, 2.278470)})
                while turn.termination_reason is None:
                    turn.step_controls(1.0, 0.5 - turn.elapsed_time)
                self.assertEqual(turn.termination_reason, "time_limit")
                straight = HomicidalChauffeurEnv(GameConfig(dt=dt, horizon=14.5))
                straight.reset(options={"state": turn.state})
                while straight.termination_reason is None:
                    straight.step_controls(0.0, 0.0)
                self.assertEqual(straight.termination_reason, "capture")
                self.assertAlmostEqual(
                    turn.elapsed_time + straight.elapsed_time, expected, delta=2e-6
                )

    def test_primary_arc(self) -> None:
        theta, tau = 0.35, 0.05
        # Expanded formula from the specification, independent of the inversion.
        x = 1 - cos(tau) + 0.3 * sin(theta + tau) - 0.5 * tau * sin(theta + tau)
        y = sin(tau) + 0.3 * cos(theta + tau) - 0.5 * tau * cos(theta + tau)
        for dt, expected in ((0.05, 0.049978), (0.025, 0.049994)):
            for side in (1, -1):
                with self.subTest(dt=dt, side=side):
                    state = side * x, y
                    feedback = classical_feedback(state)
                    self.assertEqual(feedback.branch, "primary")
                    self.assertAlmostEqual(feedback.time_to_capture, tau)
                    self.assertEqual(classical_pursuer(state), side)
                    self.assertAlmostEqual(classical_evader(state), side * (theta + tau))
                    run = rollout_classical(state, GameConfig(dt=dt, horizon=15.0))
                    self.assertEqual(run.outcome, "capture")
                    self.assertAlmostEqual(run.trajectory[-1].time, expected, places=6)
                    self.assertTrue(all(not p.axis_crossing for p in run.trajectory))
                    target = side * 0.3 * sin(theta), 0.3 * cos(theta)
                    np.testing.assert_allclose(run.trajectory[-1].state, target, rtol=0, atol=7e-4)

    def test_partial_domain_and_mixed_rollout(self) -> None:
        start = (1.105240, 2.278470)
        for policy in (classical_pursuer, classical_evader):
            self.assertIsNone(policy((2, -2)))
            self.assertIsNone(policy((0, 0.3)))
            with self.assertRaisesRegex(ValueError, "gamma=0.5 and beta=0.3"):
                policy(start, GameConfig(beta=0.4))
        self.assertEqual(classical_feedback((0, 0.3)).branch, "capture")
        self.assertEqual(classical_feedback((1e-10, 2.0)).branch, "UL")
        self.assertEqual(classical_feedback(retrograde_position(0.2, 0.1)).branch, "primary")
        self.assertIsNone(classical_feedback(retrograde_position(0.51, 0.05)))
        self.assertEqual(rollout_classical((2, -2)).outcome, "unsupported")
        mixed = rollout_classical(start, pursuer=lambda state, config: 0.0)
        self.assertEqual(mixed.outcome, "unsupported")
        self.assertGreater(mixed.trajectory[-1].time, 0)
        self.assertIsNone(classical_feedback(mixed.trajectory[-1].state))
        limited = rollout_classical(start, GameConfig(horizon=0.12))
        self.assertEqual(limited.outcome, "time_limit")
        self.assertEqual(limited.trajectory[-1].time, 0.12)

    def test_feedback_uses_the_same_capture_boundary_as_reset(self) -> None:
        env = HomicidalChauffeurEnv()
        for state, outside_branch in (
            ((0.006282725965007086, 0.29993420504245366), None),
            ((0.06018871959163782, 0.2939001837936125), "primary"),
        ):
            with self.subTest(state=state):
                feedback = classical_feedback(state)
                # A norm at this scale may round to either side on different builds.
                if np.linalg.norm(state) <= env.config.beta:
                    with self.assertRaisesRegex(ValueError, "outside the capture circle"):
                        env.reset(options={"state": state})
                    self.assertEqual(feedback.branch, "capture")
                else:
                    env.reset(options={"state": state})
                    self.assertEqual(feedback.branch if feedback else None, outside_branch)
                    expected_outcome = "capture" if outside_branch == "primary" else "unsupported"
                    self.assertEqual(rollout_classical(state).outcome, expected_outcome)

    def test_tangential_barrier_geometry(self) -> None:
        self.assertAlmostEqual(BARRIER_END_TIME, 4 * pi / 3)
        contact = tangential_barrier(0)
        np.testing.assert_allclose(contact, (0.3 * sin(pi / 3), 0.15), atol=1e-12)
        self.assertAlmostEqual(0.5 - contact[1] / 0.3, 0.0)  # Radial speed at contact.
        end = tangential_barrier(BARRIER_END_TIME)
        np.testing.assert_allclose(end, (3.053992, -1.763223), rtol=0, atol=5e-7)
        self.assertEqual(tangential_barrier(BARRIER_END_TIME, side=-1), (-end[0], end[1]))
        self.assertIsNone(classical_feedback(end))

    def test_axis_event_before_horizon_allows_control_recalculation(self) -> None:
        config = GameConfig(dt=0.05, horizon=0.05)
        env = HomicidalChauffeurEnv(config)
        env.reset(options={"state": (0.01, 2.3)})
        _, _, terminated, truncated, info = env.step_controls(
            0, -pi / 2, stop_at_positive_axis=True
        )
        self.assertEqual((terminated, truncated, info["axis_crossing"]), (False, False, True))
        self.assertEqual(info["state"][0], 0.0)
        self.assertAlmostEqual(info["elapsed_time"], 0.02)
        self.assertIs(env.config, config)
        _, _, terminated, truncated, info = env.step_controls(0, 0, stop_at_positive_axis=True)
        self.assertEqual((terminated, truncated, info["axis_crossing"]), (False, True, False))
        self.assertEqual(info["elapsed_time"], config.horizon)

    def test_axis_event_at_step_endpoint(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.05))
        x_at_crossing = 0.11447027157116405
        for x, expected in (
            (x_at_crossing, True),
            (0.11447027157116409, True),
            (x_at_crossing + 1e-10, False),
        ):
            with self.subTest(x=x):
                env.reset(options={"state": (x, 2.3)})
                _, _, terminated, truncated, info = env.step_controls(
                    1, 0, stop_at_positive_axis=True
                )
                self.assertFalse(terminated or truncated)
                self.assertAlmostEqual(info["dt"], 0.05, places=14)
                self.assertEqual(info["axis_crossing"], expected)
                if expected:
                    self.assertEqual(info["state"][0], 0.0)
                else:
                    self.assertGreater(info["state"][0], 0.0)
