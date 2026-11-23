"""Trajectory exports for the P1–P8 starts from the chauffeur report."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from chauffeur_evader.chauffeur_classical import Rollout, State
from chauffeur_evader.homicidal_chauffeur import GameConfig

# The t = 0 rows of data/chauffeur-report/benchmarks.csv.
BENCHMARKS: dict[str, State] = {
    "P1": (0.0, 2.3),
    "P2": (1.0, 2.5),
    "P3": (0.2, 1.0),
    "P4": (0.45, 1.0),
    "P5": (0.5, 0.0),
    "P6": (2.0, -0.5),
    "P7": (3.2, -1.8),
    "P8": (0.0, -1.0),
}
DRIVER_STYLES = {
    "classical": ("Classical", "#24679a", "-"),
    "heuristic": ("Heuristic", "#bb4e17", "--"),
    "ppo": ("PPO", "#17806a", "-."),
}


def _save_figure(
    target: Path, name: str, start: State, runs: dict[str, Rollout], config: GameConfig
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplot_mosaic(
        [["x", "xy"], ["y", "xy"], ["phi", "psi"]],
        figsize=(10, 8),
        layout="constrained",
    )
    fig.suptitle(f"{name} · Start ({start[0]:g}, {start[1]:g})", fontsize=16)
    axes["xy"].add_patch(Circle((0, 0), config.beta, facecolor="#e5e9ef", edgecolor="#8794a2"))
    axes["xy"].set_aspect("equal", adjustable="box")
    for panel, xlabel, ylabel, title in (
        ("x", "t", "Relative x", "x(t)"),
        ("y", "t", "Relative y", "y(t)"),
        ("xy", "Relative x", "Relative y", "y(x)"),
        ("phi", "t", "Steering rate (rad per unit time)", "φ(t)"),
        ("psi", "t", "Relative heading (rad)", "ψ(t)"),
    ):
        ax = axes[panel]
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        if panel not in ("x", "xy"):
            ax.sharex(axes["x"])

    for driver, run in runs.items():
        label, color, linestyle = DRIVER_STYLES[driver]
        style = {"color": color, "linestyle": linestyle, "linewidth": 1.5, "label": label}
        times = np.array([point.time for point in run.trajectory])
        x, y = np.array([point.state for point in run.trajectory]).T
        phi = np.array([point.phi for point in run.trajectory[1:]])
        psi_start = np.array([point.psi for point in run.trajectory[1:]])
        axes["x"].plot(times, x, **style)
        axes["y"].plot(times, y, **style)
        axes["xy"].plot(x, y, **style)
        axes["phi"].stairs(phi, times, baseline=None, **style)
        # The evader's inertial heading is fixed during each decision interval.
        psi_end = psi_start - phi * np.diff(times)
        axes["psi"].plot(
            np.repeat(times, 2)[1:-1], np.column_stack((psi_start, psi_end)).ravel(), **style
        )

    axes["xy"].plot(*start, marker="o", markersize=4, color="#253641")
    fig.legend(handles=axes["x"].lines, loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_benchmarks(output: Path, runs: dict[str, list[Rollout]], config: GameConfig) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (output / "benchmarks.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((
            "benchmark", "pursuer", "x0", "y0", "outcome", "time", "x", "y",
            "phi_on_arrival", "psi_at_interval_start", "event",
        ))
        for (name, start), *episodes in zip(BENCHMARKS.items(), *runs.values(), strict=True):
            games = dict(zip(runs, episodes, strict=True))
            for driver, run in games.items():
                writer.writerows(
                    (name, driver, *start, run.outcome, point.time, *point.state,
                     point.phi, point.psi, point.event)
                    for point in run.trajectory
                )
            _save_figure(figures / f"{name}.png", name, start, games, config)
