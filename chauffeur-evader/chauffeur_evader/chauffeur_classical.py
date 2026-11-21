"""Classical Region II feedback for Merz and the closed capture disk in the project."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import acos, atan2, ceil, cos, hypot, isfinite, pi, sin
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from chauffeur_evader.chauffeur_theory import (
    A_TIME,
    BARRIER_END_TIME,
    BETA,
    GAMMA,
    TOL,
    USABLE_PART_ANGLE,
    Geometry,
    Point,
    arc,
    axis_arc,
    bisect,
    el_controls,
    flow,
    geometry,
    radial_roots,
    rotate,
    upper_value,
    wrap,
)
from chauffeur_evader.homicidal_chauffeur import (
    GameConfig,
    HomicidalChauffeurEnv,
    Observation,
    Transition,
    is_captured,
)

DEFAULT_CONFIG = GameConfig(gamma=GAMMA, beta=BETA)
AXIS_TOLERANCE = 1e-9
State = Point
Policy = Callable[[State, GameConfig], float | None]
Evader = Callable[[State, GameConfig, float], float | None]
RolloutPolicy = Callable[[State, GameConfig, Observation], float | None]
Path = Callable[[float], State]
Mode = Literal["first_contact", "merz"]


@dataclass(frozen=True)
class Feedback:
    branch: str
    phi: float | None
    psi: float | None
    time_to_capture: float
    # Next junction along the characteristic (time from this state, endpoint).
    junction: tuple[float, State] | None = None
    # At a dispersal line the other steering direction has its own course.
    alternate_psi: float | None = None

    def response(self, phi: float) -> float | None:
        if self.alternate_psi is not None and (phi < 0) != (self.phi < 0):
            return self.alternate_psi
        return self.psi


def _check_parameters(config: GameConfig) -> None:
    if (config.gamma, config.beta) != (GAMMA, BETA):
        raise ValueError(f"Classical feedback requires gamma={GAMMA} and beta={BETA}")


def _upper(z: State, g: Geometry, branch: str = "turn_to_UL") -> Feedback | None:
    root = axis_arc(z, 1)
    if root is not None:
        a, y = root
        if y >= g.radius - TOL:
            return Feedback(branch, 1, wrap(a), a + (y - g.radius) / (1 - GAMMA), (a, (0.0, y)))
    return None


def _primary(z: State, r: float) -> Feedback | None:
    x, y = z
    # All primary characteristics lie in this box, and F is convex here.
    cutoff = r / GAMMA
    if x > max(r * sin(USABLE_PART_ANGLE), 1 - cos(cutoff)) + TOL or not 0 < y <= sin(cutoff) + r:
        return None

    def residual(a: float) -> float:
        return (x - 1 + cos(a)) ** 2 + (y - sin(a)) ** 2 - (r - GAMMA * a) ** 2

    def slope(a: float) -> float:
        return (1 - x) * sin(a) - y * cos(a) + GAMMA * (r - GAMMA * a)

    if slope(0) >= 0:
        return None
    hi = bisect(slope, 0, cutoff) if slope(cutoff) > 0 else cutoff
    if residual(hi) > 1e-14:
        return None
    a = bisect(residual, 0, hi) if residual(hi) < 0 else hi
    theta = atan2(x - 1 + cos(a), y - sin(a)) - a
    if theta < -TOL or theta > USABLE_PART_ANGLE + TOL:
        return None
    q = r * sin(theta), r * cos(theta)
    return Feedback("primary", 1, wrap(theta + a), a, (a, q))


def _contact(z: State, g: Geometry) -> Feedback | None:
    choices = []
    for alpha, a, p in g.to_contact(z):
        q, _ = g.contact(alpha)
        # S/|P| along a left turn: -r*sin(alpha) + cos(p-a) - cos(p).
        # Check its maximum on the full interval, including a course of zero.
        most = 1.0 if p - a < 0 < p else max(cos(p), cos(p - a))
        if most - cos(p) > g.radius * sin(alpha) + TOL:
            continue
        choices.append(
            Feedback(
                "turn_to_contact",
                -1,
                wrap(p - a),
                a + g.value_at_contact(alpha),
                (a, q),
            )
        )
    return min(choices, key=lambda f: f.time_to_capture, default=None)


def _away(z: State, g: Geometry) -> Feedback | None:
    root = axis_arc(z, -1)
    if root is not None:
        a, y = root
        if g.y_l - TOL <= y <= g.y_m + TOL:
            if abs(y - g.y_m) <= 1e-10 and a > g.a_m:
                q = arc((0.0, g.y_m), 0, g.a_m, -1)
                return Feedback(
                    "turn_to_contact",
                    -1,
                    -a,
                    a + (g.y_m - g.y_l) / (1 - GAMMA) + upper_value((0, g.y_l), g.radius),
                    (a - g.a_m, q),
                )
            return Feedback(
                "turn_to_lower_UL",
                -1,
                -a,
                a + (y - g.y_l) / (1 - GAMMA) + upper_value((0, g.y_l), g.radius),
                (a, (0.0, y)),
            )
    if root is not None and root[1] < g.y_l:

        def endpoint(t: float) -> tuple[State, float]:
            q = g.el.at(t)
            return q, el_controls(q)[1]

        for t, a, p in radial_roots(z, -1, endpoint, [(0, g.el.times[-1])]):
            q = g.el.at(t)
            return Feedback("turn_to_EL", -1, wrap(p - a), a + upper_value(q, g.radius), (a, q))
    contact = _contact(z, g)
    if contact is not None:
        return contact
    # After passing the tangent point, this part of the arc reaches the axis
    # without another contact, even when its continuation backwards would hit.
    if root is not None:
        a, y = root
        if g.y_l - TOL <= y < -g.radius:
            return Feedback(
                "turn_to_lower_UL",
                -1,
                -a,
                a + (y - g.y_l) / (1 - GAMMA) + upper_value((0, g.y_l), g.radius),
                (a, (0.0, y)),
            )
    return None


def _throw(z: State, g: Geometry, away: Feedback) -> Feedback:
    # The envelope and the curved dispersal line bound a small wedge.
    if not 0.16 < z[0] < 0.20 or not 0.31 < z[1] < 0.57:
        return away
    pd = g.pdl.near(z, 2e-7)
    se = g.se.near(z, 2e-7)
    if se is not None:
        return replace(away, branch="D" if se == g.se.times[-1] else "SE")
    if (
        pd is None
        and (g.wedge_barrier.ray_hits(*z) + g.se.ray_hits(*z) + g.pdl.ray_hits(*z)) % 2 == 0
    ):
        return away
    # The wedge is so narrow that the value comparison is needed as well:
    # extending a characteristic beyond its first intersection with the PDL
    # does not give an admissible positive turn.
    candidates = []
    for t, a, p in radial_roots(z, 1, g.se_endpoint, [(0.0, g.se.times[-1])], splits=8):
        if a <= TOL or a > g.s_h - g.s_g + TOL:
            continue
        q = g.se.at(t)
        end = _contact(q, g)
        if end is not None:
            candidates.append(
                Feedback("throw_to_SE", 1, wrap(p + a), a + end.time_to_capture, (a, q))
            )
    throw = min(candidates, key=lambda f: f.time_to_capture, default=None)
    if throw is None:
        return away
    if abs(throw.time_to_capture - away.time_to_capture) <= 2e-7:
        return replace(throw, branch="curved_PDL", alternate_psi=away.psi)
    return throw if throw.time_to_capture < away.time_to_capture else away


def _barrier_parameter(z: State, g: Geometry, tol: float = 2e-8) -> float | None:
    s = g.barrier.near(z, 2e-7)
    if s is None:
        return None

    # Refine the projection on the exact characteristic, independent of the mesh.
    def residual(t: float) -> float:
        x, y = g.b(t)
        p = USABLE_PART_ANGLE + t
        return (z[0] - x) * cos(p) - (z[1] - y) * sin(p)

    lo, hi = max(0.0, s - 0.002), min(BARRIER_END_TIME, s + 0.002)
    if residual(lo) * residual(hi) < 0:
        s = bisect(residual, lo, hi)
    elif s < 1e-6:
        s = 0.0
    elif BARRIER_END_TIME - s < 1e-6:
        s = BARRIER_END_TIME
    x, y = g.b(s)
    return s if hypot(z[0] - x, z[1] - y) <= tol else None


def classical_feedback(
    state: State,
    config: GameConfig = DEFAULT_CONFIG,
    *,
    mode: Mode = "first_contact",
    slide_el: bool = False,
    follow_se: bool = False,
    virtual_radius: float | None = None,
) -> Feedback | None:
    """Return branch, actions and continuous characteristic time; None if unresolved.

    For a nominal safe contact, the value includes Merz's continuation even
    when the simulator ends at first contact. An optional virtual radius keeps
    the evader at a positive distance from the true capture disk.
    """
    _check_parameters(config)
    x, y = map(float, state)
    if not (isfinite(x) and isfinite(y)):
        raise ValueError("Classical feedback requires finite coordinates")
    if mode == "first_contact" and is_captured((x, y), config.beta):
        return Feedback("capture", None, None, 0.0)

    radius = BETA if virtual_radius is None else virtual_radius
    if hypot(x, y) < radius - TOL and virtual_radius is not None:
        radius = BETA  # Finish a transverse entry against the true target.
    g = geometry(radius)
    side = -1 if x < 0 else 1
    z = abs(x), y
    if mode == "merz" or virtual_radius is not None:
        distance = hypot(*z)
        if distance < radius - TOL:
            return Feedback("capture", None, None, 0.0)
        if abs(distance - radius) <= TOL:
            theta = atan2(*z)
            if theta < USABLE_PART_ANGLE - TOL:
                if radius == BETA:
                    return Feedback("capture", None, None, 0.0)
                return classical_feedback(state, config)
            if theta <= g.alpha_m - g.a_m + TOL:
                alpha = acos(max(-1.0, min(1.0, cos(theta) / GAMMA)))
                if abs(alpha - g.alpha_m) <= 1e-7:
                    q = (0.0, g.y_m)
                    return Feedback(
                        "M",
                        -side,
                        -side * g.a_m,
                        g.value_at_contact(g.alpha_m),
                        (g.a_m, q),
                    )
                _, p = g.contact(alpha)
                return Feedback("safe_contact", -side, wrap(side * p), g.value_at_contact(alpha))

    if z[0] <= AXIS_TOLERANCE:
        if y > radius:
            return Feedback("UL", 0, 0, (y - radius) / (1 - GAMMA))
        if g.y_l + TOL < y < -radius:
            a = (y - g.y_l) / (1 - GAMMA)
            return Feedback(
                "lower_UL", 0, 0, a + upper_value((0, g.y_l), radius), (a, (0.0, g.y_l))
            )
        if y <= g.y_l + TOL:
            f = _upper((0.0, y), g, "L" if abs(y - g.y_l) <= TOL else "axis_PDL")
            if f is not None:
                if slide_el and f.branch == "L":
                    return replace(f, phi=0.5, psi=0.0)
                return replace(f, alternate_psi=wrap(-f.psi))
        return None

    f = None
    s = _barrier_parameter(z, g)
    if s is not None:
        label = next(
            (
                name
                for value, name in (
                    (g.s_g, "G"),
                    (g.s_h, "H"),
                    (radius / GAMMA, "A"),
                    (BARRIER_END_TIME, "B"),
                )
                if abs(s - value) <= 1e-6
            ),
            "barrier",
        )
        if mode == "first_contact" and virtual_radius is None:
            if s <= A_TIME + 1e-8:
                f = Feedback(label, 1, wrap(USABLE_PART_ANGLE + s), s, (s, g.b(0)))
            else:
                f = _upper(z, g, label)
        elif label == "B":
            f = Feedback("B", -1, wrap(USABLE_PART_ANGLE + s), upper_value(z, radius))
        elif g.s_g + 1e-8 < s <= g.s_h + 1e-6:
            a = s - g.s_g
            other = _away(z, g)
            f = Feedback(
                label,
                1,
                wrap(USABLE_PART_ANGLE + s),
                a + _away(g.b(g.s_g), g).time_to_capture,
                (a, g.b(g.s_g)),
                other.psi if label == "H" and other else None,
            )
        else:
            other = _away(z, g)
            if other is not None:
                f = replace(other, branch=label)
    elif g.el.near(z) is not None:
        f = _upper(z, g, "EL")
        if slide_el and f is not None:
            f = replace(f, phi=el_controls(z)[0], psi=el_controls(z)[1], junction=None)
    elif g.slow_side(z):
        f = _away(z, g)
        if f is not None:
            f = _throw(z, g, f)
            if follow_se and f.branch in ("SE", "D"):
                t = g.se.near(z, 2e-7)
                _, p = g.se_endpoint(t)
                f = replace(f, branch="SE_follow", phi=1, psi=p, junction=None)
    else:
        f = _primary(z, radius) or _upper(z, g)

    if f is not None and side == -1:
        p = wrap(-f.psi)
        alt = None if f.alternate_psi is None else wrap(-f.alternate_psi)
        junction = f.junction
        if junction is not None:
            a, (qx, qy) = junction
            junction = a, (-qx, qy)
        f = replace(f, phi=-f.phi, psi=p, alternate_psi=alt, junction=junction)
    return f


def classical_pursuer(state: State, config: GameConfig = DEFAULT_CONFIG) -> float | None:
    f = classical_feedback(state, config)
    return f.phi if f is not None else None


def classical_evader(
    state: State, config: GameConfig = DEFAULT_CONFIG, phi: float | None = None
) -> float | None:
    f = classical_feedback(state, config)
    return (
        f.response(f.phi if phi is None else phi) if f is not None and f.phi is not None else None
    )


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    state: State
    # Controls on arrival at this point, absent at reset.
    phi: float | None = None
    psi: float | None = None
    axis_crossing: bool = False
    event: str | None = None


@dataclass(frozen=True)
class Rollout:
    outcome: Literal["capture", "time_limit", "unsupported"]
    trajectory: tuple[TrajectoryPoint, ...]


def _path(state: State, phi: float, psi: float) -> Path:
    """Exact sampled move: constant steering and constant inertial heading."""
    dx, dy = GAMMA * sin(psi), GAMMA * cos(psi)
    x, y = state
    if phi == 0:
        return lambda t: (x + t * dx, y + t * (dy - 1))
    return lambda t: rotate(
        x + t * dx - 2 * sin(phi * t / 2) ** 2 / phi,
        y + t * dy - sin(phi * t) / phi,
        -phi * t,
    )


def _capture_time(path: Path, phi: float, psi: float, length: float) -> float | None:
    if hypot(*path(0)) > BETA + (1 + GAMMA) * length + 1e-12:
        return None

    def clearance(t: float) -> float:
        return hypot(*path(t)) - BETA

    def radial(t: float) -> float:
        x, y = path(t)
        p = psi - phi * t
        return GAMMA * (x * sin(p) + y * cos(p)) - y

    # Split at a minimum as well as testing the endpoints: a tangency does
    # not change the sign of the distance to the circle.
    slices = max(1, ceil(length / (0.05 + 1e-12)))
    for i in range(slices):
        lo, hi = i * length / slices, (i + 1) * length / slices
        if radial(lo) < 0 < radial(hi):
            hi = bisect(radial, lo, hi)
        gap = clearance(hi)
        if gap < -5e-14:
            return bisect(clearance, lo, hi)
        if gap <= 5e-14:
            return hi
    return None


def _line_event(
    path: Path,
    dt: float,
    g: Geometry,
    phi: float,
    psi: float,
) -> tuple[float, str, State] | None:
    slices = max(1, ceil(dt / (0.05 + 1e-12)))
    for i in range(slices):
        left = i * dt / slices
        event = _line_event_slice(
            path if i == 0 else lambda t, left=left: path(left + t),
            dt / slices,
            g,
            phi,
            psi - phi * left,
        )
        if event is not None:
            t, label, point = event
            return left + t, label, point
    return None


def _line_event_slice(
    path: Path,
    dt: float,
    g: Geometry,
    phi: float,
    psi: float,
) -> tuple[float, str, State] | None:
    """Localize the first axis or exceptional-line crossing by y slices."""
    start, end = path(0), path(dt)
    events = []
    if abs(start[0]) > AXIS_TOLERANCE:

        def lateral(t: float) -> float:
            return flow(path(t), phi, psi - phi * t)[0]

        cut = dt
        if start[0] * end[0] > 0 and start[0] * lateral(0) < 0 < start[0] * lateral(dt):
            cut = bisect(lateral, 0, dt)
        x = path(cut)[0]
        if start[0] * x <= 0 or abs(x) <= 5e-14:
            t = cut if abs(x) <= 5e-14 else bisect(lambda t: path(t)[0], 0, cut)
            y = path(t)[1]
            if abs(y) > BETA:
                events.append((t, "axis", (0.0, y)))
    curves = (("barrier", g.barrier), ("EL", g.el), ("SE", g.se), ("curved_PDL", g.pdl))
    for label, curve in curves:
        for side in (1, -1):
            xlo, xhi = sorted((side * start[0], side * end[0]))
            ylo, yhi = sorted((start[1], end[1]))
            xmin, xmax, ymin, ymax = curve.bounds
            if xhi < xmin - 1e-5 or xlo > xmax + 1e-5 or yhi < ymin or ylo > ymax:
                continue
            for section in curve.sections:
                if max(start[1], end[1]) < section[0, 1] or min(start[1], end[1]) > section[-1, 1]:
                    continue

                def gap(t: float, side=side, section=section) -> float:
                    x, y = path(t)
                    return side * x - float(np.interp(y, section[:, 1], section[:, 0]))

                before, after = gap(0), gap(dt)
                if abs(before) <= 2e-7:
                    continue
                cut = dt
                if before * after > 0:
                    # Near SE a tangency can occur with the same sign at both ends.
                    if label != "SE" or min(abs(before), abs(after)) > 0.001:
                        continue
                    lo, hi = 0.0, dt
                    for _ in range(32):
                        a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
                        if before * gap(a) < before * gap(b):
                            hi = b
                        else:
                            lo = a
                    cut = (lo + hi) / 2
                    after = gap(cut)
                    if before * after > 0 and abs(after) > 5e-10:
                        continue
                if label == "barrier" and section[0, 1] <= start[1] <= section[-1, 1]:
                    # A snapped start is on the exact barrier, not necessarily on
                    # its polyline. Match the local section by normal distance:
                    # a steep segment in x(y) magnifies its mesh error in gap().
                    i = max(1, int(section[:, 1].searchsorted(start[1])))
                    dx, dy = section[i] - section[i - 1]
                    if abs(before) * dy <= 2e-7 * hypot(dx, dy) and _barrier_parameter(
                        (side * start[0], start[1]), g
                    ) is not None:
                        continue
                t = cut if abs(after) <= 5e-10 else bisect(gap, 0, cut)
                x, y = path(t)
                if section[0, 1] <= y <= section[-1, 1] and t > 1e-9:
                    if (
                        label == "barrier"
                        and (s := _barrier_parameter((side * x, y), g, 2e-7)) is not None
                    ):
                        x, y = g.b(s)
                        x *= side
                    events.append((t, label, (x, y)))
    return min(events, key=lambda event: event[0], default=None)


def step_feedback(
    env: HomicidalChauffeurEnv,
    action: ArrayLike | None,
    *,
    feedback: Feedback | None,
    evader: Evader | None = None,
    mode: Mode = "first_contact",
    slide_el: bool = False,
    virtual_radius: float | None = None,
) -> Transition:
    """Shared policy step. The clipped steering is visible to the pedestrian."""
    env._require_active()
    state = tuple(map(float, env.state))
    config = env.config
    phi = env._control(action, 1.0, "phi") if action is not None else None
    psi = None
    if phi is not None:
        if evader is not None:
            psi = evader(state, config, phi)
        elif feedback is not None:
            psi = feedback.response(phi)
    if phi is None or psi is None:
        env.termination_reason = (
            "capture" if feedback is not None and feedback.branch == "capture" else "unsupported"
        )
        return (
            env._observation(),
            0.0,
            True,
            False,
            dict(
                env._info(),
                dt=0.0,
                phi=phi,
                psi=None,
                event=None,
                axis_crossing=False,
            ),
        )
    psi = env._control(psi, pi, "psi")
    remaining = config.horizon - env.elapsed_time
    dt = min(config.dt, remaining)
    if remaining <= config.dt + 1e-12:
        dt = remaining

    rho = (
        virtual_radius
        if virtual_radius is not None and hypot(*state) >= virtual_radius - TOL
        else BETA
    )
    g = geometry(rho)
    nominal = feedback is not None and phi == feedback.phi and psi == feedback.response(phi)
    path = _path(state, phi, psi)
    junction = feedback.junction if nominal else None
    event = None
    endpoint = None
    contact_mode = mode == "first_contact" or virtual_radius is not None

    if nominal and feedback.branch == "safe_contact":
        theta = atan2(abs(state[0]), state[1])
        alpha = acos(max(-1.0, min(1.0, cos(theta) / GAMMA)))
        remaining_on_circle = g.circle_duration(alpha)
        if remaining_on_circle <= dt:
            dt, event = remaining_on_circle, "M"
            end_alpha = g.alpha_m
        else:
            end_alpha = bisect(
                lambda a: g.circle_duration(a) - (remaining_on_circle - dt), alpha, g.alpha_m
            )
        endpoint = g.contact(end_alpha)[0]
        if state[0] < 0:
            endpoint = -endpoint[0], endpoint[1]
    elif nominal and feedback.branch == "SE_follow":
        _, t = g.se.project((abs(state[0]), state[1]))
        if t <= dt:
            dt, event = t, "G"
        endpoint = g.se.at(max(0.0, t - dt))
        if state[0] < 0:
            endpoint = -endpoint[0], endpoint[1]
    elif nominal and slide_el and feedback.branch in ("EL", "L"):
        _, t = g.el.project((abs(state[0]), state[1]))
        if t <= dt:
            dt, event = t, "B"
        endpoint = g.el.at(max(0.0, t - dt))
        if state[0] < 0:
            endpoint = -endpoint[0], endpoint[1]
    elif nominal and mode == "merz" and feedback.branch == "B":
        # Merz's short turn away from B before taking the outgoing arc.
        dt = min(dt, config.dt / 20)
        endpoint = path(dt)
    elif junction is not None and 0 < junction[0] <= dt + 1e-10:
        dt = min(dt, junction[0])
        endpoint = junction[1]
        event = (
            "axis"
            if endpoint[0] == 0
            else {
                "turn_to_EL": "EL",
                "throw_to_SE": "SE",
                "curved_PDL": "SE",
            }.get(
                feedback.branch,
                "contact" if hypot(*endpoint) <= rho + TOL else "barrier",
            )
        )
    if endpoint is None:
        # For nominal characteristics all intervening events are known from
        # the inversion. Alternate controls can cross other special lines.
        if not nominal or (junction is None and feedback.branch != "UL"):
            crossing = _line_event(path, dt, g, phi, psi)
            if crossing is not None:
                dt, event, endpoint = crossing
        if endpoint is None:
            endpoint = path(dt)

    terminal = False
    if contact_mode and not (nominal and feedback.branch == "safe_contact"):
        if event == "contact" and rho == BETA:
            terminal = True
        else:
            capture = _capture_time(path, phi, psi, dt)
            if capture is not None:
                dt, endpoint, terminal, event = capture, path(capture), True, "contact"
    elif (event != "contact" and hypot(*endpoint) < BETA - TOL) or (
        hypot(*endpoint) <= BETA + 5e-14
        and atan2(abs(endpoint[0]), endpoint[1]) < USABLE_PART_ANGLE - TOL
    ):
        capture = _capture_time(path, phi, psi, dt)
        dt = dt if capture is None else capture
        endpoint, terminal, event = path(dt), True, "contact"
    if event == "contact" and not terminal:
        event = "safe_contact"
    return env._finish_step(
        endpoint,
        phi,
        psi,
        dt,
        terminal,
        event,
        remaining - dt <= 1e-12,
    )


def rollout_feedback(
    start: State,
    env: HomicidalChauffeurEnv,
    *,
    pursuer: RolloutPolicy | None = None,
    evader: Evader | None = None,
    mode: Mode = "first_contact",
    slide_el: bool = False,
    follow_se: bool = False,
    virtual_radius: float | None = None,
) -> Rollout:
    """Play either the reference or a supplied policy with the same decision clock."""
    config = env.config
    _check_parameters(config)
    if (mode == "first_contact" and is_captured(start, config.beta)) or hypot(*start) < BETA - TOL:
        return Rollout("capture", (TrajectoryPoint(0.0, start),))
    if virtual_radius is not None and not config.beta < virtual_radius < hypot(*start):
        raise ValueError("The virtual radius must be between beta and the initial distance")
    observation, info = env.reset(options={"state": start, "allow_contact": mode == "merz"})
    state = tuple(map(float, info["state"]))
    trajectory = [TrajectoryPoint(0.0, state)]
    while True:
        f = classical_feedback(
            state,
            config,
            mode=mode,
            slide_el=slide_el,
            follow_se=follow_se,
            virtual_radius=virtual_radius,
        )
        action = (f.phi if f else None) if pursuer is None else pursuer(state, config, observation)
        observation, _, terminated, truncated, info = step_feedback(
            env,
            action,
            feedback=f,
            evader=evader,
            mode=mode,
            slide_el=slide_el,
            virtual_radius=virtual_radius,
        )
        if info["dt"] > 0:
            state = tuple(map(float, info["state"]))
            trajectory.append(
                TrajectoryPoint(
                    info["elapsed_time"],
                    state,
                    info["phi"],
                    info["psi"],
                    info["axis_crossing"],
                    info["event"],
                )
            )
        if terminated or truncated:
            return Rollout(info["termination_reason"], tuple(trajectory))


def rollout_classical(
    start: State,
    config: GameConfig = DEFAULT_CONFIG,
    *,
    pursuer: Policy | None = None,
    evader: Evader | None = None,
    mode: Mode = "first_contact",
    slide_el: bool = False,
    follow_se: bool = False,
    virtual_radius: float | None = None,
) -> Rollout:
    policy = None if pursuer is None else lambda state, config, _: pursuer(state, config)
    return rollout_feedback(
        start,
        HomicidalChauffeurEnv(config),
        pursuer=policy,
        evader=evader,
        mode=mode,
        slide_el=slide_el,
        follow_se=follow_se,
        virtual_radius=virtual_radius,
    )
