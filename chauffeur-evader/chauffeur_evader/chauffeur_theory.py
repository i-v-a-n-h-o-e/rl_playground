"""Analytical tangential barrier in Merz §2.3 for gamma=0.5, beta=0.3.

These curves describe the continuous game. They are independent of the
simulator, the rollout horizon, and the domain of the feedback policies.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from math import acos, atan2, cos, hypot, pi, sin, sqrt

import numpy as np

GAMMA, BETA = 0.5, 0.3
USABLE_PART_ANGLE = acos(GAMMA)
BARRIER_END_TIME = 2 * (pi - USABLE_PART_ANGLE)


def retrograde_position(theta: float, tau: float) -> tuple[float, float]:
    """Relative position along a right turn from terminal angle theta (Merz §2.1)."""
    q = BETA - GAMMA * tau
    return 1 - cos(tau) + q * sin(theta + tau), sin(tau) + q * cos(theta + tau)


def tangential_barrier(tau: float, *, side: int = 1) -> tuple[float, float]:
    """Point on the open barrier, from tangency at tau=0 to B at BARRIER_END_TIME."""
    if not 0 <= tau <= BARRIER_END_TIME:
        raise ValueError("tau must lie between 0 and BARRIER_END_TIME")
    if side not in (-1, 1):
        raise ValueError("side must be +1 (right) or -1 (left)")
    x, y = retrograde_position(USABLE_PART_ANGLE, tau)
    return side * x, y


# §6–§8 of the strategy report. Angles are measured from +y throughout.
Point = tuple[float, float]
TOL = 1e-8
LINE_TOL = 1e-5
A_TIME = BETA / GAMMA


def wrap(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def direction(p: float) -> Point:
    return sin(p), cos(p)


def barrier_coordinates(z: Point) -> Point | None:
    """Normal coordinates (s, radius) of the parallel family b_radius(s)."""
    x, y = z
    d = hypot(1 - x, y)
    if d <= GAMMA:
        return None
    # (1-x, -y) = h e(p_b) + gamma e'(p_b) on this sheet.
    h = sqrt((d - GAMMA) * (d + GAMMA))
    p = (atan2(1 - x, -y) - atan2(GAMMA, h)) % (2 * pi)
    s = p - USABLE_PART_ANGLE
    return s, sin(USABLE_PART_ANGLE) + GAMMA * s - h


def rotate(x: float, y: float, angle: float) -> Point:
    c, s = cos(angle), sin(angle)
    return x * c + y * s, y * c - x * s


def arc(q: Point, p: float, a: float, sign: int) -> Point:
    """Characteristic in backward time; q is its endpoint and p the heading there."""
    dx, dy = direction(p)
    x, y = rotate(q[0] - GAMMA * a * dx, q[1] - GAMMA * a * dy, sign * a)
    return x + sign * 2 * sin(a / 2) ** 2, y + sin(a)


def flow(z: Point, phi: float, psi: float) -> Point:
    x, y = z
    return -phi * y + GAMMA * sin(psi), phi * x - 1 + GAMMA * cos(psi)


def axis_arc(z: Point, sign: int) -> Point | None:
    """Return time and ordinate at the first axis intersection of this arc."""
    x, y = z
    u = sign * x
    r = hypot(u - 1, y)
    if r < 1 - TOL / 2:
        return None
    h = sign * sqrt(max(0.0, (r - 1) / r)) * sqrt((r + 1) / r) * r
    scale = max(1.0, abs(h))
    a = atan2(y / scale - (h / scale) * (1 - u), (1 - u) / scale + (h / scale) * y) % (2 * pi)
    return a, h + GAMMA * a


def upper_value(z: Point, radius: float = BETA) -> float:
    a, y = axis_arc(z, 1)
    return a + (y - radius) / (1 - GAMMA)


def bisect(fn: Callable[[float], float], lo: float, hi: float) -> float:
    """A bracketed scalar zero (endpoints can themselves be roots)."""
    left = fn(lo)
    if left == 0:
        return lo
    if fn(hi) == 0:
        return hi
    while hi - lo > 2e-14:
        mid = (lo + hi) / 2
        value = fn(mid)
        if value == 0:
            return mid
        if (value > 0) == (left > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def el_controls(z: Point) -> Point:
    x, y = z
    r = hypot(x, y)
    h = sqrt(max(0.0, (r * r - 2 * x)))
    return 1 - GAMMA * (r * r - x + h * r) / (r * (r - y)), atan2(-x, -y)


@dataclass
class Curve:
    """Piecewise linear geometry and its monotonically parameterized inverse slices."""

    nodes: np.ndarray
    times: np.ndarray
    sections: tuple[np.ndarray, ...]
    bounds: tuple[float, float, float, float]

    @classmethod
    def from_nodes(cls, nodes: list[Point], times: list[float]) -> Curve:
        points = np.asarray(nodes)
        slopes = np.diff(points[:, 1]) > 0
        breaks = np.flatnonzero(slopes[:-1] != slopes[1:]) + 1
        cuts = [0, *breaks, len(points) - 1]
        sections = []
        for lo, hi in pairwise(cuts):
            section = points[lo : hi + 1]
            sections.append(section if slopes[lo] else section[::-1])
        return cls(
            points,
            np.asarray(times),
            tuple(sections),
            (
                float(points[:, 0].min()),
                float(points[:, 0].max()),
                float(points[:, 1].min()),
                float(points[:, 1].max()),
            ),
        )

    def at(self, t: float) -> Point:
        i = min(max(1, bisect_left(self.times, t)), len(self.times) - 1)
        w = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        x, y = self.nodes[i - 1] + w * (self.nodes[i] - self.nodes[i - 1])
        return float(x), float(y)

    def ray_hits(self, x: float, y: float) -> int:
        return sum(
            x < np.interp(y, section[:, 1], section[:, 0])
            for section in self.sections
            if section[0, 1] <= y < section[-1, 1]
        )

    def project(self, z: Point) -> tuple[float, float]:
        """Distance and parameter of the closest point on the polyline."""
        start, end = self.nodes[:-1], self.nodes[1:]
        delta = end - start
        w = np.clip(np.sum((z - start) * delta, axis=1) / np.sum(delta**2, axis=1), 0, 1)
        err = z - (start + w[:, None] * delta)
        d2 = np.sum(err**2, axis=1)
        i = int(np.argmin(d2))
        return sqrt(d2[i]), self.times[i] + w[i] * (self.times[i + 1] - self.times[i])

    def near(self, z: Point, tol: float = LINE_TOL) -> float | None:
        x, y = z
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin - tol <= x <= xmax + tol and ymin - tol <= y <= ymax + tol):
            return None
        # A y slice and its two neighbors cheaply rule out distant points.
        for section in self.sections:
            ys = section[:, 1]
            i = bisect_left(ys, y - tol)
            j = bisect_left(ys, y + tol)
            sub = section[max(0, i - 1) : j + 1]
            if len(sub) >= 2 and min(sub[:, 0]) - tol <= x <= max(sub[:, 0]) + tol:
                distance, t = self.project(z)
                if distance <= tol:
                    return float(t)
                break
        return None


def rk4(z: Point, step: float, derivative: Callable[[Point], Point]) -> Point:
    def add(v: Point, scale: float) -> Point:
        return z[0] + scale * v[0], z[1] + scale * v[1]

    a = derivative(z)
    b = derivative(add(a, step / 2))
    c = derivative(add(b, step / 2))
    d = derivative(add(c, step))
    return tuple(q + step * (i + 2 * j + 2 * k + l) / 6 for q, i, j, k, l in zip(z, a, b, c, d))


def integrate(
    start: Point,
    derivative: Callable[[Point], Point],
    stop: Callable[[Point], float],
    step: float,
) -> tuple[list[Point], list[float]]:
    nodes, times = [start], [0.0]
    while True:
        end = rk4(nodes[-1], step, derivative)
        if stop(end) <= 0:
            last = bisect(lambda dt: stop(rk4(nodes[-1], dt, derivative)), 0, step)
            nodes.append(rk4(nodes[-1], last, derivative))
            times.append(times[-1] + last)
            return nodes, times
        nodes.append(end)
        times.append(len(times) * step)


def radial_roots(
    z: Point,
    sign: int,
    endpoint: Callable[[float], tuple[Point, float]],
    intervals: list[tuple[float, float]],
    splits: int = 1,
) -> list[tuple[float, float, float]]:
    """Invert arcs by slicing at fixed distance from the center of the turn.

    Intervals are split at extrema of the perpendicular radius, where the two
    quadratic roots for time coalesce. Bracket the remaining angular equation.
    """
    x, y = z
    r = hypot(x - sign, y)
    angle = atan2(x - sign, y)

    def components(u: float) -> tuple[float, float, float]:
        (qx, qy), p = endpoint(u)
        return p, (qx - sign) * sin(p) + qy * cos(p), (qx - sign) * cos(p) - qy * sin(p)

    def clearance(u: float) -> float:
        return r - abs(components(u)[2])

    roots = []
    for lo, hi in intervals:
        left, right = clearance(lo), clearance(hi)
        if max(left, right) < -TOL:
            continue
        if left < 0 < right:
            lo = bisect(clearance, lo, hi)
        elif right < 0 < left:
            hi = bisect(clearance, lo, hi)
        for orient in (1, -1):

            def residual(u: float, orient=orient) -> Point:
                p, b, c = components(u)
                dot = orient * sqrt(max(0.0, (r - abs(c)) * (r + abs(c))))
                a = (b - dot) / GAMMA
                return wrap(p + atan2(c, dot) + sign * a - angle), a

            left_u = lo
            left = residual(lo)[0]
            for k in range(1, splits + 1):
                right_u = lo + (hi - lo) * k / splits
                right = residual(right_u)[0]
                candidates = []
                if abs(left) <= TOL:
                    candidates.append(left_u)
                if left * right < 0 and abs(right - left) < pi:
                    candidates.append(bisect(lambda u: residual(u)[0], left_u, right_u))
                if k == splits and abs(right) <= TOL:
                    candidates.append(right_u)
                for u in candidates:
                    q, p = endpoint(u)
                    a = residual(u)[1]
                    if (
                        -TOL <= a < 2 * pi
                        and hypot(*(np.subtract(arc(q, p, a, sign), z))) < 4 * TOL
                    ):
                        roots.append((u, max(a, 0.0), p))
                left_u, left = right_u, right
    return roots


@dataclass
class Geometry:
    radius: float
    barrier: Curve
    el: Curve
    se: Curve
    se_angles: np.ndarray
    pdl: Curve
    wedge_barrier: Curve
    a_m: float
    y_m: float
    alpha_m: float
    circle_times: np.ndarray
    s_g: float
    s_h: float

    @property
    def y_l(self) -> float:
        return float(self.el.nodes[-1, 1])

    def b(self, s: float) -> Point:
        return arc(
            tuple(self.radius * v for v in direction(USABLE_PART_ANGLE)),
            USABLE_PART_ANGLE,
            s,
            1,
        )

    def contact(self, alpha: float) -> tuple[Point, float]:
        theta = acos(GAMMA * cos(alpha))
        return (self.radius * sin(theta), self.radius * cos(theta)), theta - alpha

    def to_contact(self, z: Point) -> list[tuple[float, float, float]]:
        return contact_roots(z, self.contact, self.alpha_m)

    def circle_duration(self, alpha: float) -> float:
        return circle_duration(alpha, self.radius, self.circle_times)

    def value_at_contact(self, alpha: float) -> float:
        return float(
            self.circle_duration(alpha)
            + self.a_m
            + (self.y_m - self.y_l) / (1 - GAMMA)
            + upper_value((0, self.y_l), self.radius)
        )

    def slow_side(self, z: Point) -> bool:
        x, y = z
        r = self.radius
        if x >= self.barrier.bounds[1] or y < self.y_l - 1:
            return False
        hits = self.barrier.ray_hits(x, y) + self.el.ray_hits(x, y)
        if -r <= y < r * GAMMA:
            hits += x < sqrt(max(0, r * r - y * y))
        return bool(hits % 2)

    def se_endpoint(self, t: float) -> tuple[Point, float]:
        return self.se.at(t), float(np.interp(t, self.se.times, self.se_angles))


def contact_roots(z: Point, endpoint: Callable, alpha_m: float):
    return radial_roots(z, -1, endpoint, [(0.0, pi / 2), (pi / 2, alpha_m)], splits=4)


def circle_segment(lo: float, hi: float, radius: float, phi: float = -1) -> float:
    # alpha = acos(cos(theta)/gamma) removes the square-root singularity at Q.
    def integrand(alpha: float) -> float:
        theta = acos(GAMMA * cos(alpha))
        dtheta = GAMMA * sin(alpha) / sin(theta)
        return radius * dtheta / (sin(theta) - GAMMA * sin(alpha) - phi * radius)

    return (hi - lo) * (integrand(lo) + 4 * integrand((lo + hi) / 2) + integrand(hi)) / 6


def circle_duration(alpha: float, radius: float, table: np.ndarray) -> float:
    i = min(bisect_left(table[:, 0], alpha), len(table) - 1)
    hi, tail = table[i]
    return float(tail + circle_segment(alpha, hi, radius))


@lru_cache(maxsize=4)
def geometry(radius: float = BETA) -> Geometry:
    """Build the Region II curves for the report's parameters and nearby radii."""
    a_m = bisect(
        lambda a: (
            (1 - cos(a)) * sqrt(1 + GAMMA**2 - 2 * GAMMA * cos(a)) - radius * (cos(a) - GAMMA)
        ),
        0,
        USABLE_PART_ANGLE,
    )
    y_m = GAMMA * a_m + (1 - GAMMA) * sin(a_m) / (GAMMA - cos(a_m))
    m = arc((0, y_m), 0, a_m, -1)
    alpha_m = atan2(*m) + a_m

    def contact(alpha: float) -> tuple[Point, float]:
        theta = acos(GAMMA * cos(alpha))
        return (radius * sin(theta), radius * cos(theta)), theta - alpha

    def barrier(s: float) -> Point:
        return arc(
            tuple(radius * v for v in direction(USABLE_PART_ANGLE)),
            USABLE_PART_ANGLE,
            s,
            1,
        )

    ss = np.linspace(0, BARRIER_END_TIME, 4097).tolist()
    barrier_curve = Curve.from_nodes([barrier(s) for s in ss], ss)

    def el_back(z: Point) -> Point:
        return tuple(-v for v in flow(z, *el_controls(z)))

    el_nodes, el_times = integrate(
        barrier(BARRIER_END_TIME),
        el_back,
        lambda z: min(z[0], hypot(*z) - radius),
        0.001,
    )
    el_nodes[-1] = (0.0, el_nodes[-1][1])
    el_curve = Curve.from_nodes(el_nodes, el_times)

    alphas = np.linspace(0, alpha_m, 1025)
    increments = [circle_segment(l, r, radius) for l, r in pairwise(alphas)]
    circle_times = np.column_stack((alphas, np.r_[np.cumsum(increments[::-1])[::-1], 0]))

    def slow(z: Point) -> tuple[float, float, float]:
        roots = contact_roots(z, contact, alpha_m)
        return min(roots, key=lambda k: k[1] + circle_duration(k[0], radius, circle_times))

    def start_condition(s: float) -> float:
        z = barrier(s)
        alpha, a, p = slow(z)
        px, py = direction(p - a)
        vx, vy = flow(z, 1, USABLE_PART_ANGLE + s)
        theta = p + alpha
        return px * vx + py * vy + sin(alpha) * (radius + sin(theta) - GAMMA * sin(alpha))

    s_g = bisect(start_condition, 0.25, 0.35)
    alpha_g, a_g, _ = slow(barrier(s_g))

    def se_data(params: Point) -> tuple[Point, float, Point, float]:
        alpha, a = params
        q, p0 = contact(alpha)
        theta = p0 + alpha
        u = GAMMA * sin(alpha) / sin(theta)
        z = arc(q, p0, a, -1)
        p = p0 - a
        sx, sy = direction(p)
        fx, fy = flow(z, -1, p)
        xa, ya = rotate(
            radius * u * cos(theta) - GAMMA * a * (u - 1) * cos(p0),
            -radius * u * sin(theta) + GAMMA * a * (u - 1) * sin(p0),
            -a,
        )
        s_hat = sx * z[1] - sy * z[0]
        arg = 1 + 2 * s_hat / GAMMA
        p_se = p + acos(arg)
        vx, vy = flow(z, 1, p_se)
        det = -xa * fy + ya * fx
        alpha_dot = (vx * fy - vy * fx) / det
        a_dot = (ya * vx - xa * vy) / det
        p_dot = (u - 1) * alpha_dot - a_dot
        s_dot = p_dot * (sy * z[1] + sx * z[0]) - sx * vy + sy * vx
        return (
            z,
            p_se,
            (alpha_dot, a_dot),
            p_dot - 2 * s_dot / (GAMMA * sqrt(1 - arg * arg)),
        )

    se_params, se_times = integrate(
        (alpha_g, a_g),
        lambda params: se_data(params)[2],
        lambda params: 1 - se_data(params)[3],
        0.00025,
    )
    se_points = [se_data(params) for params in se_params]
    se_curve = Curve.from_nodes([data[0] for data in se_points], se_times)
    se_angles = np.asarray([data[1] for data in se_points])

    def slow_time(z: Point) -> float:
        alpha, a, _ = slow(z)
        return a + circle_duration(alpha, radius, circle_times)

    s_h = bisect(lambda s: slow_time(barrier(s_g)) + s - s_g - slow_time(barrier(s)), 0.55, 0.66)
    pdl_nodes = [barrier(s_h)]
    # The nonzero PDL root tends to zero at D. Continue it from H, staying
    # away from the trivial t=0 root shared by all points of the envelope.
    prior = s_h - s_g
    for t in np.linspace(0, se_times[-1], 129)[1:-1]:
        q = se_curve.at(t)
        p = float(np.interp(t, se_times, se_angles))
        v = slow_time(q)

        def delta(a: float, v=v, q=q, p=p) -> float:
            return v + a - slow_time(arc(q, p, a, 1))

        lo, hi = prior / 16, prior * 1.25
        if delta(lo) < 0 < delta(hi):
            prior = bisect(delta, lo, hi)
            pdl_nodes.append(arc(q, p, prior, 1))
    pdl_nodes.append(tuple(se_curve.nodes[-1]))
    pdl_curve = Curve.from_nodes(pdl_nodes, list(range(len(pdl_nodes))))

    wedge_times = np.linspace(s_g, s_h, 401).tolist()
    wedge = Curve.from_nodes([barrier(s) for s in wedge_times], wedge_times)
    return Geometry(
        radius,
        barrier_curve,
        el_curve,
        se_curve,
        se_angles,
        pdl_curve,
        wedge,
        a_m,
        y_m,
        alpha_m,
        circle_times,
        s_g,
        s_h,
    )
