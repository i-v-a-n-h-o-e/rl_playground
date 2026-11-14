"""Compare time steps against exact trajectories and save a static plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from RL.chauffeur_demo import run_episode
from RL.homicidal_chauffeur import GameConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("chauffeur_diagnostics.png")
    )
    args = parser.parse_args()
    config = GameConfig()
    # For psi=0 and phi=1, the exact orbit starts at the top of a circle
    # centered at (1-gamma, 0). The first minimum distance occurs at pi/2.
    center = 1 - config.gamma
    cases = (
        ("Straight pursuit (P1)", (0.0, 2.3), 0, config.horizon),
        ("Maximum turn · clear miss", (center, 1.0), 1, np.pi / 2),
        ("Maximum turn · capture", (center, 0.6), 1, 2.3),
        ("Near tangency · false capture", (center, center + config.beta + 1e-9), 1, 2.3),
    )

    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4), layout="constrained")
    fig.suptitle("Homicidal chauffeur · time-step diagnostics", fontsize=16)
    steps = (
        (1, "#174e78", "-", "o"),
        (2, "#c4571a", "--", "x"),
        (4, None, None, None),
    )
    disk_style = {"facecolor": "#d9e5ed", "edgecolor": "#778b9c"}
    keys = [Circle((0, 0), 1, **disk_style, label=f"Capture disk (r ≤ {config.beta:g})")]
    for divisor, color, linestyle, marker in steps[:2]:
        keys.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=f"dt = {config.dt / divisor:g}",
            )
        )
    fig.legend(handles=keys, loc="outside lower center", ncols=3, frameon=False)

    for ax, (name, start, phi, horizon) in zip(axes.flat, cases, strict=True):
        ax.add_patch(Circle((0, 0), config.beta, **disk_style))
        ax.set(title=name, xlabel="Relative x", ylabel="Relative y")
        ax.set_aspect("equal", adjustable="box")
        if phi == 0:
            ax.set(xlim=(-0.55, 0.55), xticks=(-0.5, 0, 0.5))
        ax.grid(alpha=0.2)
        print(name)
        end_times = []
        # The third resolution helps diagnose outcome changes near tangency.
        for divisor, color, linestyle, marker in steps:
            dt = config.dt / divisor
            trace = run_episode(GameConfig(dt=dt, horizon=horizon), start, 42, phi, 0)
            times = np.array([row["time"] for row in trace])
            states = np.array([row["state"] for row in trace])
            if phi == 0:
                exact = np.column_stack((np.zeros_like(times), start[1] - center * times))
            else:
                exact = np.column_stack(
                    (center - start[1] * np.sin(times), start[1] * np.cos(times))
                )
            error = np.max(np.linalg.norm(states - exact, axis=1))
            reason = trace[-1]["termination_reason"]
            end_times.append(times[-1])
            print(f"  dt={dt:g}  {reason:10}  t={times[-1]:.9f}  max state error={error:.3e}")
            if color is not None:
                ax.plot(*states.T, color=color, linestyle=linestyle, linewidth=1.8)
                ax.plot(*states[-1], marker=marker, color=color, markersize=6)
        print(f"  |Δt_end| (dt vs dt/2) = {abs(end_times[0] - end_times[1]):.3e}")
        if phi:
            clearance = abs(start[1] - center) - config.beta
            print(f"  Exact clearance at pi/2 = {clearance:+.3e}")
            if clearance < 0:
                exact_capture_time = np.arcsin(
                    (center**2 + start[1] ** 2 - config.beta**2) / (2 * center * start[1])
                )
                print(f"  Exact capture time = {exact_capture_time:.9f}")
        ax.plot(*start, marker=".", color="#253641", markersize=6)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Plot: {args.output.resolve()}")


if __name__ == "__main__":
    main()
