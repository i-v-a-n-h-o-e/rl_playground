from __future__ import annotations

import unittest
from math import atan2, cos, hypot, pi, sin, sqrt

import numpy as np

from chauffeur_evader.chauffeur_classical import (
    classical_evader,
    classical_feedback,
    classical_pursuer,
    rollout_classical,
)
from chauffeur_evader.chauffeur_theory import (
    BARRIER_END_TIME,
    USABLE_PART_ANGLE,
    arc,
    axis_arc,
    barrier_coordinates,
    flow,
    geometry,
    tangential_barrier,
)
from chauffeur_evader.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv


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
        for dt in (0.05, 0.025):
            expected_switch, expected_y = axis_arc(start, 1)
            expected_capture = expected_switch + (expected_y - 0.3) / 0.5
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
        for dt in (0.05, 0.025):
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
                    self.assertAlmostEqual(run.trajectory[-1].time, tau, places=9)
                    self.assertTrue(all(not p.axis_crossing for p in run.trajectory))
                    target = side * 0.3 * sin(theta), 0.3 * cos(theta)
                    np.testing.assert_allclose(run.trajectory[-1].state, target, rtol=0, atol=7e-4)

    def test_missing_policy_control_is_reported_without_substitution(self) -> None:
        start = (1.105240, 2.278470)
        for policy in (classical_pursuer, classical_evader):
            self.assertIsNone(policy((0, 0.3)))
            with self.assertRaisesRegex(ValueError, "gamma=0.5 and beta=0.3"):
                policy(start, GameConfig(beta=0.4))
        self.assertEqual(rollout_classical((0, 0.3)).outcome, "capture")
        self.assertEqual(classical_feedback((1e-10, 2.0)).branch, "UL")
        run = rollout_classical(
            start, pursuer=lambda state, config: 1.0 if state == start else None
        )
        self.assertEqual(run.outcome, "unsupported")
        self.assertEqual(run.trajectory[-1].time, 0.05)
        limited = rollout_classical(start, GameConfig(horizon=0.12))
        self.assertEqual(limited.outcome, "time_limit")
        self.assertEqual(limited.trajectory[-1].time, 0.12)

    def test_p2_through_p8_from_the_report_and_their_reflections(self) -> None:
        for start, phi, psi, capture, first_time, first_y in (
            ((1, 2.5), 1, 0.411517, 4.805609, 0.411517, 2.497046),
            ((0.2, 1), 1, 0.221314, 1.442629, 0.221314, 0.910657),
            ((0.45, 1), -1, -1.570796, 18.146612, 1.570796, -0.664602),
            ((0.5, 0), -1, -0.841069, 17.351089, 0.841069, -0.697500),
            ((2, -0.5), -1, -1.070617, 14.301691, 1.070617, -2.336973),
            ((3.2, -1.8), 1, 2.616018, 9.953690, 2.616018, 3.968836),
            ((0, -1), 0, 0, 15.905020, 3.401712, -2.700856),
        ):
            for side in (1, -1):
                with self.subTest(start=start, side=side):
                    z = side * start[0], start[1]
                    f = classical_feedback(z)
                    self.assertEqual(f.phi, side * phi)
                    self.assertAlmostEqual(f.psi, side * psi, delta=1e-6)
                    run = rollout_classical(z)
                    self.assertEqual(run.outcome, "capture")
                    self.assertAlmostEqual(run.trajectory[-1].time, capture, delta=1e-6)
                    event = next(p for p in run.trajectory if p.event)
                    self.assertEqual((event.event, event.state[0]), ("axis", 0))
                    self.assertAlmostEqual(event.time, first_time, delta=1e-6)
                    self.assertAlmostEqual(event.state[1], first_y, delta=1e-6)

    def test_el_departure_and_sliding_have_the_same_value(self) -> None:
        g = geometry()
        j = g.el.at(1.1)
        for side in (1, -1):
            with self.subTest(side=side):
                x, y = side * j[0], j[1]
                self.assertEqual(classical_feedback((x, y)).branch, "EL")
                glide = rollout_classical((x, y), slide_el=True)
                leave = rollout_classical((x, y))
                self.assertEqual((glide.outcome, leave.outcome), ("capture", "capture"))
                self.assertAlmostEqual(glide.trajectory[1].phi, side * 0.547600, delta=1e-6)
                self.assertAlmostEqual(leave.trajectory[1].psi, side * 3.067558, delta=1e-6)
                self.assertAlmostEqual(glide.trajectory[-1].time, leave.trajectory[-1].time)
                self.assertAlmostEqual(
                    next(p for p in glide.trajectory if p.event == "B").time, 1.1
                )
        incoming = rollout_classical((2.189032554, -2.566448322))
        self.assertEqual(incoming.trajectory[1].event, "EL")
        self.assertAlmostEqual(incoming.trajectory[1].time, 0.008930, delta=1e-6)
        self.assertEqual(incoming.trajectory[2].phi, 1.0)

    def test_throw_reaches_se_and_pdl_responds_to_the_chosen_turn(self) -> None:
        g = geometry()
        e, p = g.se_endpoint(0.0699)
        start = arc(e, p, 0.075, 1)
        k = arc(e, p, 0.134212, 1)
        for side in (1, -1):
            with self.subTest(side=side):
                run = rollout_classical((side * start[0], start[1]))
                event = next(i for i, point in enumerate(run.trajectory) if point.event)
                self.assertEqual(run.trajectory[event].event, "SE")
                self.assertAlmostEqual(run.trajectory[event].time, 0.075, delta=1e-6)
                self.assertEqual(run.trajectory[event + 1].phi, -side)
                self.assertAlmostEqual(run.trajectory[event + 1].psi, -side * 0.085626, delta=1e-6)
                z = side * k[0], k[1]
                self.assertEqual(classical_feedback(z).branch, "curved_PDL")
                self.assertAlmostEqual(classical_evader(z, phi=side), side * 1.476604, delta=1e-6)
                self.assertAlmostEqual(classical_evader(z, phi=-side), -side * 0.352567, delta=1e-6)
                for d, sign in ((-2e-5, side), (2e-5, -side)):
                    offset = side * (k[0] + d * 0.99892), k[1] + d * 0.04646
                    self.assertEqual(classical_pursuer(offset), sign)
        follow = rollout_classical(e, follow_se=True)
        event = next(p for p in follow.trajectory if p.event)
        self.assertEqual(event.event, "G")
        self.assertAlmostEqual(event.time, 0.0699)

    def test_safe_contact_continues_only_in_merz_or_with_positive_clearance(
        self,
    ) -> None:
        g = geometry()
        q = g.b(1)
        p = USABLE_PART_ANGLE + 1
        slow = q[0] + 0.01 * sin(p), q[1] + 0.01 * cos(p)
        fast = q[0] - 0.01 * sin(p), q[1] - 0.01 * cos(p)
        self.assertEqual(classical_feedback(fast).branch, "turn_to_UL")
        self.assertEqual(classical_feedback(slow).branch, "turn_to_contact")
        self.assertEqual(classical_feedback(q).phi, 1)
        closed = rollout_classical(slow)
        opened = rollout_classical(slow, mode="merz")
        self.assertEqual((closed.outcome, opened.outcome), ("capture", "capture"))
        self.assertAlmostEqual(closed.trajectory[-1].time, 0.976822, delta=1e-6)
        self.assertAlmostEqual(opened.trajectory[-1].time, 18.454940, delta=1e-6)
        self.assertTrue(any(pt.event == "M" for pt in opened.trajectory))
        prior = 0.0
        for rho in (0.301, 0.3005, 0.30025):
            expanded = rollout_classical(slow, virtual_radius=rho)
            self.assertEqual(expanded.outcome, "capture")
            point = next(pt for pt in expanded.trajectory if pt.event == "safe_contact")
            self.assertAlmostEqual(np.linalg.norm(point.state), rho)
            self.assertLess(prior, expanded.trajectory[-1].time)
            prior = expanded.trajectory[-1].time
        self.assertAlmostEqual(prior, opened.trajectory[-1].time, delta=0.0011)

    def test_off_nominal_p4_contact_and_departure(self) -> None:
        # Seed 43 at t=1, before its next sampled steering decision.
        z = (0.27991134764129716, 0.12293859570612813)
        rho, phi = 0.301, 0.9054176807403564
        initial = classical_feedback(z, virtual_radius=rho)
        self.assertEqual((initial.branch, initial.phi), ("turn_to_contact", -1))
        run = rollout_classical(
            z,
            GameConfig(horizon=1.5),
            virtual_radius=rho,
            pursuer=lambda state, _: phi if state == z else 1,
        )
        hit, following = run.trajectory[1:3]
        self.assertEqual(hit.event, "virtual_entry")
        self.assertAlmostEqual(hit.time, 0.036148675, places=9)
        self.assertAlmostEqual(hypot(*hit.state), rho, places=13)
        self.assertAlmostEqual(following.psi, 0.469786958, places=8)
        self.assertEqual(run.outcome, "time_limit")
        m = next(i for i, pt in enumerate(run.trajectory) if pt.event == "M")
        for pt in run.trajectory[1 : m + 1]:
            self.assertAlmostEqual(hypot(*pt.state), rho, places=13)
        self.assertAlmostEqual(run.trajectory[m + 1].psi, -geometry(rho).a_m)
        self.assertGreater(hypot(*run.trajectory[m + 1].state), rho)
        self.assertTrue(all(b.time > a.time for a, b in zip(run.trajectory, run.trajectory[1:])))

    def test_circle_angular_speed_for_applied_steering(self) -> None:
        theta, rho, dt = 1.45, 0.301, 1e-6
        for side in (-1, 1):
            for phi in (-1, 0, 1):
                z = side * rho * sin(theta), rho * cos(theta)
                run = rollout_classical(
                    z,
                    GameConfig(dt=dt, horizon=dt),
                    pursuer=lambda *_, turn=phi: turn,
                    virtual_radius=rho,
                )
                (end,) = run.trajectory[1:]
                vx, vy = flow(z, phi, end.psi)
                self.assertAlmostEqual(z[0] * vx + z[1] * vy, 0, places=13)
                self.assertAlmostEqual(hypot(*end.state), rho, places=13)
                omega = (sin(theta) - sqrt(0.5**2 - cos(theta) ** 2)) / rho - side * phi
                self.assertAlmostEqual(
                    (atan2(abs(end.state[0]), end.state[1]) - theta) / dt, omega, delta=1e-5
                )

    def test_virtual_tangency_transverse_entry_and_competing_axis(self) -> None:
        rho = 0.301
        tangent = rollout_classical(
            (rho, 0.01),
            GameConfig(horizon=0.04),
            pursuer=lambda *_: 0,
            evader=lambda *_: 0,
            virtual_radius=rho,
        )
        self.assertEqual(tangent.trajectory[1].event, "safe_contact")
        self.assertAlmostEqual(tangent.trajectory[1].time, 0.02)
        self.assertEqual(len(tangent.trajectory), 3)

        axis = rollout_classical(
            (-0.002, 0.32),
            GameConfig(horizon=0.01),
            pursuer=lambda *_: 0,
            evader=lambda *_: pi / 2,
            virtual_radius=rho,
        )
        self.assertEqual(axis.trajectory[1].event, "axis")
        self.assertAlmostEqual(axis.trajectory[1].time, 0.004)
        transverse = rollout_classical((0, 0.31), virtual_radius=rho)
        self.assertEqual(
            [pt.event for pt in transverse.trajectory], [None, "virtual_entry", "contact"]
        )
        self.assertAlmostEqual(transverse.trajectory[1].time, 0.018)
        self.assertAlmostEqual(transverse.trajectory[-1].time, 0.020)

    def test_first_of_two_grazing_entries_inside_one_step(self) -> None:
        # The distance has two minima below rho and an intervening maximum above it.
        start = (0.3009351118084666, 0.006250722586164961)
        run = rollout_classical(
            start,
            GameConfig(horizon=0.05),
            virtual_radius=0.301,
            pursuer=lambda *_: 0.8306478377719341,
            evader=lambda *_: 0.020766195944298352,
        )
        self.assertEqual(run.trajectory[1].event, "virtual_entry")
        self.assertAlmostEqual(run.trajectory[1].time, 0.007060982, delta=2e-9)

    def test_virtual_barrier_from_both_sides_and_on_the_line(self) -> None:
        rho, s = 0.301, 1.0
        x, y = geometry(rho).b(s)
        pb = USABLE_PART_ANGLE + s
        config = GameConfig(horizon=0.03)
        for side in (-1, 1):
            for offset, phi in ((0.002, 1), (-0.002, -1), (0, -1), (0, 0), (0, 1)):
                z = side * (x + offset * sin(pb)), y + offset * cos(pb)
                run = rollout_classical(
                    z, config, pursuer=lambda *_, turn=side * phi: turn, virtual_radius=rho
                )
                if offset:
                    self.assertEqual(run.trajectory[1].event, "barrier")
                contact, end = run.trajectory[-2:]
                s_hit, radius = barrier_coordinates((side * contact.state[0], contact.state[1]))
                self.assertAlmostEqual(radius, rho, places=13)
                p = USABLE_PART_ANGLE + s_hit
                vx, vy = flow(contact.state, end.phi, end.psi)
                normal = side * sin(p) * vx + cos(p) * vy
                if phi == 1:
                    self.assertAlmostEqual(normal, 0, places=13)
                    ex, ey = geometry(rho).b(s_hit - (end.time - contact.time))
                    np.testing.assert_allclose(end.state, (side * ex, ey), atol=1e-13, rtol=0)
                else:
                    self.assertGreater(normal, 0)
                    self.assertTrue(geometry(rho).slow_side((side * end.state[0], end.state[1])))
            # Exact starts on the true barrier use the closed-target convention of §8.
            for s_true in (0.35, 1.0):
                bx, by = geometry().b(s_true)
                z = side * bx, by
                run = rollout_classical(z, config, virtual_radius=rho)
                self.assertAlmostEqual(run.trajectory[1].psi, classical_feedback(z).psi)

            x0, y0 = geometry().b(1.0)
            # This true slow-side start has only half the requested normal gap.
            near = side * (x0 + 0.0005 * sin(pb)), y0 + 0.0005 * cos(pb)
            run = rollout_classical(near, config, virtual_radius=rho, pursuer=lambda *_, p=side: p)
            hit = next(pt for pt in run.trajectory if pt.event == "barrier")
            _, chosen_radius = barrier_coordinates((side * hit.state[0], hit.state[1]))
            self.assertAlmostEqual(chosen_radius, 0.30025, places=13)
            for turn in (-1, 0):
                z = side * x0, y0
                run = rollout_classical(
                    z, config, virtual_radius=rho, pursuer=lambda *_, phi=side * turn: phi
                )
                vx, vy = flow(z, run.trajectory[1].phi, run.trajectory[1].psi)
                self.assertGreater(side * sin(pb) * vx + cos(pb) * vy, 0)

    def test_p4_heuristic_guard_reaches_virtual_tangency(self) -> None:
        rho = 0.301
        start = (0.45, 1)
        for radius in (0.3, rho):
            self.assertTrue(geometry(radius).slow_side(start))
        # The bearing is positive and saturated over the entire tested arc.
        run = rollout_classical(
            start, GameConfig(horizon=1.15), pursuer=lambda *_: 1, virtual_radius=rho
        )
        hit = next(pt for pt in run.trajectory if pt.event == "barrier")
        self.assertAlmostEqual(hit.time, 0.116569934, places=8)
        q = next(pt for pt in run.trajectory if pt.event == "safe_contact")
        s_hit, _ = barrier_coordinates(hit.state)
        self.assertAlmostEqual(q.time - hit.time, s_hit, places=12)
        np.testing.assert_allclose(q.state, geometry(rho).b(0), atol=1e-13, rtol=0)
        self.assertAlmostEqual(hypot(*run.trajectory[-1].state), rho, places=13)

    def test_feedback_uses_the_same_capture_boundary_as_reset(self) -> None:
        env = HomicidalChauffeurEnv()
        for state, outside_branch in (
            ((0.006282725965007086, 0.29993420504245366), "primary"),
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
        self.assertEqual(classical_feedback(end).branch, "B")

    def test_barrier_junctions_and_critical_contact(self) -> None:
        g = geometry()
        for t, branch, merz_phi in (
            (0.2, "barrier", -1),
            (g.s_g, "G", -1),
            (0.35, "barrier", 1),
            (g.s_h, "H", 1),
            (0.6, "A", -1),
            (0.61, "barrier", -1),
            (BARRIER_END_TIME, "B", -1),
        ):
            z = g.b(t)
            with self.subTest(t=t):
                f = classical_feedback(z)
                self.assertEqual((f.branch, f.phi), (branch, 1))
                self.assertEqual(classical_feedback(z, mode="merz").phi, merz_phi)
        self.assertAlmostEqual(rollout_classical(g.b(0.6)).trajectory[-1].time, 0.6)
        self.assertAlmostEqual(
            rollout_classical(g.b(BARRIER_END_TIME)).trajectory[-1].time, 9.7481295
        )
        for delta, branch in (
            (-0.01, "turn_to_lower_UL"),
            (0.0, "turn_to_contact"),
            (0.01, "turn_to_contact"),
        ):
            self.assertEqual(classical_feedback(arc((0, g.y_m + delta), 0, 0.9, -1)).branch, branch)
        critical = rollout_classical(arc((0, g.y_m), 0, 0.9, -1))
        self.assertAlmostEqual(critical.trajectory[-1].time, 0.9 - g.a_m)
        for z in (g.b(0), g.contact(g.alpha_m)[0], (0, 0.3), (0, -0.3)):
            self.assertEqual(len(rollout_classical(z).trajectory), 1)

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

    def test_axis_event_near_an_interior_stationary_point(self) -> None:
        env = HomicidalChauffeurEnv(GameConfig(dt=0.05))
        x, y = 1.999999333335103e-06, 0.501999998666668
        for start_x, expected_time in (
            (x, 0.002),
            (x + 2e-20, 0.002),  # Residual is at roundoff scale near the minimum.
            (x - 1e-14, 0.00199985858),  # Two intersections within the step.
        ):
            with self.subTest(start_x=start_x):
                env.reset(options={"state": (start_x, y)})
                _, _, terminated, truncated, info = env.step_controls(
                    1, pi / 2, stop_at_positive_axis=True
                )
                self.assertFalse(terminated or truncated)
                self.assertTrue(info["axis_crossing"])
                self.assertAlmostEqual(info["elapsed_time"], expected_time, delta=1e-9)
                self.assertEqual(info["state"][0], 0.0)
                self.assertAlmostEqual(info["state"][1], 0.5, delta=2e-7)

        env.reset(options={"state": (x + 1e-18, y)})
        _, _, _, _, info = env.step_controls(1, pi / 2, stop_at_positive_axis=True)
        self.assertFalse(info["axis_crossing"])
        self.assertEqual(info["elapsed_time"], env.config.dt)
