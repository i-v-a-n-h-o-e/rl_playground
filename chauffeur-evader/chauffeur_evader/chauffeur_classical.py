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
    barrier_coordinates,
    bisect,
    circle_segment,
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
CONTACT_EPS = 5e-14
State = Point
Policy = Callable[[State, GameConfig], float | None]
Evader = Callable[[State, GameConfig, float], float | None]
RolloutPolicy = Callable[[State, GameConfig, Observation], float | None]
Path = Callable[[float], State]
Mode = Literal["first_contact", "merz"]


def _barrier_normal(p: float, phi: float, psi: float) -> float:
    """Velocity toward the slow side, in right-half-plane coordinates."""
    return (1 - phi) * (GAMMA - cos(p)) - 2 * GAMMA * sin((psi - p) / 2) ** 2


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
    # Reflection and parameter on the protective (positive-clearance) barrier.
    guard: tuple[int, float] | None = None

    def response(self, phi: float) -> float | None:
        psi = self.psi
        if self.alternate_psi is not None and (phi < 0) != (self.phi < 0):
            psi = self.alternate_psi
        if self.guard is not None:
            side, s = self.guard
            p = USABLE_PART_ANGLE + s
            if _barrier_normal(p, side * phi, side * psi) <= 0:
                return wrap(side * p)
        return psi


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
    coords = barrier_coordinates(z)
    if coords is None:
        return None
    s, radius = coords
    if -tol <= s <= BARRIER_END_TIME + tol and abs(radius - g.radius) <= tol:
        return min(BARRIER_END_TIME, max(0.0, s))
    return None


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
    if hypot(x, y) < radius - CONTACT_EPS and virtual_radius is not None:
        radius = BETA  # Finish a transverse entry against the true target.
    g = geometry(radius)
    side = -1 if x < 0 else 1
    z = abs(x), y
    s = _barrier_parameter(z, g, CONTACT_EPS if radius > BETA else 2e-8)
    rear_circle = False
    if mode == "merz" or radius > BETA:
        distance = hypot(*z)
        if distance < radius - TOL:
            return Feedback("capture", None, None, 0.0)
        if abs(distance - radius) <= CONTACT_EPS:
            theta = atan2(*z)
            if theta < USABLE_PART_ANGLE - CONTACT_EPS:
                if radius == BETA:
                    return Feedback("capture", None, None, 0.0)
                if s is None:
                    return classical_feedback(state, config)
                # Just before Q, r-rho is quadratic and can round to zero.
            elif theta <= g.alpha_m - g.a_m + CONTACT_EPS:
                alpha = acos(max(-1.0, min(1.0, cos(theta) / GAMMA)))
                if abs(alpha - g.alpha_m) <= 1e-12:
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
            else:
                rear_circle = True  # Continue on the outgoing side of the critical arc.

    if z[0] <= AXIS_TOLERANCE:
        if y > radius:
            return Feedback("UL", 0, 0, (y - radius) / (1 - GAMMA))
        if g.y_l + TOL < y and (y < -radius or rear_circle):
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
        if mode == "first_contact" and radius == BETA:
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
    elif rear_circle or g.slow_side(z):
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
    if f is not None and s is not None and radius > BETA and s < BARRIER_END_TIME:
        f = replace(f, guard=(side, s))
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


def _circle_hit(
    path: Path, phi: float, psi: float, length: float, radius: float
) -> tuple[float, str] | None:
    """First positive-time circle event on the applied inertial course."""
    if abs(hypot(*path(0)) - radius) > (1 + GAMMA) * length + CONTACT_EPS:
        return None

    def clearance(t: float) -> float:
        return hypot(*path(t)) - radius

    def radial(t: float) -> float:
        x, y = path(t)
        p = psi - phi * t
        return GAMMA * (x * sin(p) + y * cos(p)) - y

    def cuts(lo: float, hi: float):
        # D = (r²/2)'. A grazing turn can have two minima and a maximum
        # inside one decision, even with equal endpoint clearances.
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        x, y = path(mid)
        p = psi - phi * mid
        slope = 1 + GAMMA**2 - 2 * GAMMA * cos(p) - phi * x
        curvature = phi**2 * y - 3 * GAMMA * phi * sin(p)
        max_r = hypot(x, y) + (1 + GAMMA) * half
        third = phi**2 * (abs(phi) * max_r + 1 + 4 * GAMMA)
        variation = half * (abs(curvature) + half * third / 2)
        if abs(slope) > variation:
            if radial(lo) * radial(hi) < 0:
                yield bisect(radial, lo, hi)
        elif abs(radial(mid)) <= half * (abs(slope) + variation):
            if half**2 * (abs(slope) + variation) <= CONTACT_EPS * radius / 4:
                yield mid
            else:
                yield from cuts(lo, mid)
                yield from cuts(mid, hi)
                return
        yield hi

    slices = max(1, ceil(length / (0.05 + 1e-12)))
    for i in range(slices):
        lo, hi = i * length / slices, (i + 1) * length / slices
        for end in cuts(lo, hi):
            before, after = clearance(lo), clearance(end)
            departing = abs(before) <= CONTACT_EPS and not (radius == BETA and before > 0)
            if not departing and before * after < 0 and abs(after) > CONTACT_EPS:
                return bisect(clearance, lo, end), "entry" if before > 0 else "exit"
            if abs(before) > CONTACT_EPS and abs(after) <= CONTACT_EPS:
                speed = radial(end)
                kind = "tangent" if abs(speed) < 1e-12 else ("entry" if speed < 0 else "exit")
                return end, kind
            lo = end
    return None


def _guard_crossing(
    path: Path, dt: float, g: Geometry, side: int, phi: float, psi: float
) -> tuple[float, str, State] | None:
    """First intersection with the smooth parallel barrier in a <= 0.05 slice."""
    turn = side * phi
    x, y = path(dt / 2)
    d = hypot(1 - side * x, y)
    # The distance to the turn center changes at speed at most 1-turn+gamma.
    reach = (1 - turn + GAMMA) * dt / 2
    h0 = sin(USABLE_PART_ANGLE) - g.radius
    if d + reach < hypot(h0, GAMMA) or d - reach > hypot(h0 + GAMMA * BARRIER_END_TIME, GAMMA):
        return None

    def coordinates(t: float) -> Point:
        x, y = path(t)
        return barrier_coordinates((side * x, y))

    def gap(t: float) -> float:
        return coordinates(t)[1] - g.radius

    # The normal chart's angle wraps outside the physical barrier segment.
    start, end = coordinates(0), coordinates(dt)
    if abs(start[0] - end[0]) > pi:
        return None

    # In normal coordinates h²=d²-gamma². For u=1-turn and a=psi-p,
    # N' = -turn*u*sin(p) - (u*sin(p)+gamma*sin(a))²/h.
    h_min = ((d - reach) ** 2 - GAMMA**2) ** 0.5

    def first(lo: float, hi: float):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        p = USABLE_PART_ANGLE + coordinates(mid)[0]
        course = side * (psi - phi * mid)
        speed = _barrier_normal(p, turn, course)
        sin_bound = min(1, abs(sin(course - p)) + half * (1 - turn + GAMMA) / h_min)
        slope_bound = abs(turn) * (1 - turn) + (1 - turn + GAMMA * sin_bound) ** 2 / h_min
        spread = half * (abs(speed) + half * slope_bound / 2)
        if abs(gap(mid)) > spread + CONTACT_EPS:
            return None
        if turn != 1 and abs(speed) <= half * slope_bound and spread > CONTACT_EPS:
            return first(lo, mid) or first(mid, hi)
        left, right = gap(lo), gap(hi)
        if abs(left) > CONTACT_EPS and (left * right < 0 or abs(right) <= CONTACT_EPS):
            t = hi if abs(right) <= CONTACT_EPS else bisect(gap, lo, hi)
            s = coordinates(t)[0]
            if 0 <= s <= BARRIER_END_TIME:
                x, y = g.b(s)
                return t, "barrier", (side * x, y)
        return None

    return first(0, dt)


def _line_event(
    path: Path,
    dt: float,
    g: Geometry,
    phi: float,
    psi: float,
    *,
    guard_only: bool,
) -> tuple[float, str, State] | None:
    slices = max(1, ceil(dt / (0.05 + 1e-12)))
    for i in range(slices):
        left = i * dt / slices
        segment = path if i == 0 else lambda t, left=left: path(left + t)
        length, course = dt / slices, psi - phi * left
        events = []
        if not guard_only:
            events.append(_line_event_slice(segment, length, g, phi, course))
        if g.radius > BETA:
            for side in (1, -1):
                events.append(_guard_crossing(segment, length, g, side, phi, course))
        event = min((e for e in events if e is not None), key=lambda e: e[0], default=None)
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
        if label == "barrier" and g.radius > BETA:
            continue  # The protective barrier is intersected analytically.
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
        if virtual_radius is not None and hypot(*state) >= virtual_radius - CONTACT_EPS
        else BETA
    )
    g = geometry(rho)
    nominal = feedback is not None and phi == feedback.phi and psi == feedback.psi
    path = _path(state, phi, psi)
    junction = feedback.junction if nominal else None
    event = None
    endpoint = None
    contact_mode = mode == "first_contact" or virtual_radius is not None

    side = -1 if state[0] < 0 else 1
    on_circle = feedback is not None and feedback.branch == "safe_contact" and evader is None
    on_guard = (
        feedback is not None and feedback.guard is not None and side * phi == 1 and evader is None
    )
    slide = nominal and (
        feedback.branch == "SE_follow" or (slide_el and feedback.branch in ("EL", "L"))
    )
    if on_circle:
        theta = atan2(abs(state[0]), state[1])
        alpha = acos(max(-1.0, min(1.0, cos(theta) / GAMMA)))

        def travel(end: float) -> float:
            if side * phi == -1:
                return g.circle_duration(alpha) - g.circle_duration(end)
            # Simpson quadrature in alpha is regular even at Q.
            n = max(1, ceil((end - alpha) / 0.02))
            h = (end - alpha) / n
            return sum(
                circle_segment(alpha + i * h, alpha + (i + 1) * h, rho, side * phi)
                for i in range(n)
            )

        duration = travel(g.alpha_m)
        if duration <= dt:
            dt, event, end_alpha = duration, "M", g.alpha_m
        else:
            end_alpha = bisect(lambda a: travel(a) - dt, alpha, g.alpha_m)
        x, y = g.contact(end_alpha)[0]
        endpoint = side * x, y
    elif on_guard:
        # The only viable course for this steering is p_b. Its inertial heading
        # is constant and its exact forward characteristic is b(s-t).
        _, s = feedback.guard
        for target, label in sorted(
            ((g.s_h, "H"), (rho / GAMMA, "A"), (g.s_g, "G"), (0, "contact")), reverse=True
        ):
            if (target == 0 or s - target > 1e-12) and 0 < s - target <= dt:
                dt, event = s - target, label
                break
        x, y = g.b(max(0.0, s - dt))
        endpoint = side * x, y
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
    # Nominal inversions supply their own junction. Check the protective
    # barrier on every free move, including a nominal fast-side course.
    known_junction = nominal and (junction is not None or feedback.branch == "UL")
    search_lines = endpoint is None and not known_junction
    if search_lines or (rho > BETA and not (on_circle or on_guard or slide)):
        crossing = _line_event(path, dt, g, phi, psi, guard_only=not search_lines)
        if crossing is not None:
            dt, event, endpoint = crossing
    if endpoint is None:
        endpoint = path(dt)

    terminal = False
    if not on_circle and not slide:
        if virtual_radius is not None and not on_guard:
            hit = _circle_hit(path, phi, psi, dt, virtual_radius)
            if hit is not None:
                dt, kind = hit
                endpoint = path(dt)
                event = {
                    "entry": "virtual_entry",
                    "exit": "virtual_exit",
                    "tangent": "safe_contact",
                }[kind]
        if contact_mode:
            terminal = event == "contact" and rho == BETA
            if not terminal:
                hit = _circle_hit(path, phi, psi, dt, BETA)
                if hit is not None:
                    dt = hit[0]
                    endpoint, terminal, event = path(dt), True, "contact"
        elif (event != "contact" and hypot(*endpoint) < BETA - TOL) or (
            hypot(*endpoint) <= BETA + CONTACT_EPS
            and atan2(abs(endpoint[0]), endpoint[1]) < USABLE_PART_ANGLE - TOL
        ):
            hit = _circle_hit(path, phi, psi, dt, BETA)
            dt = dt if hit is None else hit[0]
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


def requested_radius(config: GameConfig, clearance: float) -> float | None:
    if not isfinite(clearance) or clearance < 0:
        raise ValueError("Clearance must be finite and nonnegative")
    return config.beta + clearance if clearance else None


class FeedbackEpisode:
    """Keep the selected pedestrian radius for one episode."""

    def __init__(self, requested_radius: float | None):
        if requested_radius is not None and (
            not isfinite(requested_radius) or requested_radius <= BETA
        ):
            raise ValueError("The virtual radius must exceed beta")
        self.requested_radius = requested_radius
        self.virtual_radius: float | None = None

    def step(
        self,
        env: HomicidalChauffeurEnv,
        action: ArrayLike | None,
        *,
        use_reference: bool = False,
        evader: Evader | None = None,
        mode: Mode = "first_contact",
        slide_el: bool = False,
        follow_se: bool = False,
    ) -> Transition:
        env._require_active()
        state = tuple(map(float, env.state))
        config = env.config
        # Choose rho once there is room for it. An exact true-barrier start
        # uses the closed-target rule until the first move off that line.
        true_barrier = None
        side = -1 if state[0] < 0 else 1
        if self.requested_radius is not None and self.virtual_radius is None:
            coords = barrier_coordinates((abs(state[0]), state[1]))
            room = hypot(*state) - BETA
            if coords is not None and -CONTACT_EPS <= coords[0] <= BARRIER_END_TIME + CONTACT_EPS:
                gap = coords[1] - BETA
                if abs(gap) <= CONTACT_EPS:
                    true_barrier = side, min(BARRIER_END_TIME, max(0.0, coords[0]))
                    room = 0.0
                elif gap > 0:
                    room = min(room, gap)
            if self.requested_radius - BETA <= room + CONTACT_EPS:
                self.virtual_radius = self.requested_radius
            elif room > 2 * CONTACT_EPS:
                self.virtual_radius = BETA + room / 2
        f = classical_feedback(
            state,
            config,
            mode=mode,
            slide_el=slide_el,
            follow_se=follow_se,
            virtual_radius=self.virtual_radius,
        )
        if use_reference:
            action = f.phi if f else None
        if (
            true_barrier is not None
            and 0 < true_barrier[1] < BARRIER_END_TIME
            and action is not None
            and side * env._control(action, 1.0, "phi") < 1
            and mode == "first_contact"
            and evader is None
        ):
            slow = classical_feedback(state, config, mode="merz")
            if slow is not None:
                f = replace(slow, guard=true_barrier)
        return step_feedback(
            env,
            action,
            feedback=f,
            evader=evader,
            mode=mode,
            slide_el=slide_el,
            virtual_radius=self.virtual_radius,
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
    episode = FeedbackEpisode(virtual_radius)
    observation, info = env.reset(options={"state": start, "allow_contact": mode == "merz"})
    state = tuple(map(float, info["state"]))
    trajectory = [TrajectoryPoint(0.0, state)]
    while True:
        action = None if pursuer is None else pursuer(state, config, observation)
        observation, _, terminated, truncated, info = episode.step(
            env,
            action,
            use_reference=pursuer is None,
            evader=evader,
            mode=mode,
            slide_el=slide_el,
            follow_se=follow_se,
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
