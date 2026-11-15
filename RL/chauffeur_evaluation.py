"""Reproducible finite-horizon evaluation against the classical evader."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import atan2
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np

from RL.chauffeur_classical import (
    DEFAULT_CONFIG,
    Policy,
    Rollout,
    State,
    classical_pursuer,
    rollout_classical,
)
from RL.homicidal_chauffeur import GameConfig, is_captured

GRID_MIN, GRID_MAX, GRID_SCALE = -40, 40, 10
TUNING_SEED, TUNING_COUNT = 42, 512
Outcome = Literal["capture", "time_limit", "unsupported"]


def heuristic_pursuer(state: State, config: GameConfig) -> float:
    """Turn toward the current bearing, at most one radian per unit time."""
    x, y = state
    bearing = atan2(x, y) if x != 0 or y > 0 else np.pi
    return max(-1.0, min(1.0, bearing / config.dt))


def final_grid() -> list[State]:
    """Return the 0.1-spaced grid on [-4, 4]^2, outside the capture disk."""
    axis = [i / GRID_SCALE for i in range(GRID_MIN, GRID_MAX + 1)]
    return [(x, y) for y in axis for x in axis if not is_captured((x, y), DEFAULT_CONFIG.beta)]


def tuning_starts() -> list[State]:
    """Sample a fixed set from the same square, disjoint from the final grid."""
    rng = np.random.default_rng(TUNING_SEED)
    excluded = set(final_grid())
    starts = []
    while len(starts) < TUNING_COUNT:
        x, y = rng.uniform(GRID_MIN / GRID_SCALE, GRID_MAX / GRID_SCALE, size=2)
        state = float(x), float(y)
        if state not in excluded and not is_captured(state, DEFAULT_CONFIG.beta):
            starts.append(state)
            excluded.add(state)
    return starts


@dataclass(frozen=True)
class EpisodeResult:
    start: State
    outcome: Outcome
    capture_time: float | None


def evaluate_pursuer(
    pursuer: Policy,
    starts: Iterable[State],
    config: GameConfig = DEFAULT_CONFIG,
    *,
    rollout: Callable[[Policy, State, GameConfig], Rollout] | None = None,
) -> list[EpisodeResult]:
    """Apply one episode protocol to a feedback policy on each start.

    By default, use the classical evader and its positive-axis event rule,
    with out-of-domain decisions marked unsupported. For another protocol,
    pass a rollout accepting the pursuer, start and config and returning the
    trajectory from reset. Use the same rollout for every compared policy.
    """
    return summarize_rollouts(
        (
            rollout_classical(start, config, pursuer=pursuer)
            if rollout is None
            else rollout(pursuer, start, config)
        )
        for start in starts
    )


def summarize_rollouts(runs: Iterable[Rollout]) -> list[EpisodeResult]:
    results = []
    for run in runs:
        capture_time = run.trajectory[-1].time if run.outcome == "capture" else None
        results.append(EpisodeResult(run.trajectory[0].state, run.outcome, capture_time))
    return results


def save_episodes(target: Path, episodes: Mapping[str, Iterable[EpisodeResult]]) -> None:
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("pursuer", "x", "y", "outcome", "capture_time"))
        for name, results in episodes.items():
            writer.writerows(
                (name, *result.start, result.outcome, result.capture_time) for result in results
            )


def compare(candidate: Sequence[EpisodeResult], reference: Sequence[EpisodeResult]) -> dict:
    """Summarize paired outcomes, with whole-grid bounds for unknown results."""
    pairs = list(zip(candidate, reference, strict=True))
    if not pairs or any(a.start != b.start for a, b in pairs):
        raise ValueError("Both policies must be evaluated on the same nonempty ordered starts")

    total = len(pairs)
    candidate_counts = Counter(result.outcome for result in candidate)
    reference_counts = Counter(result.outcome for result in reference)
    both_resolved = [(a, b) for a, b in pairs if "unsupported" not in (a.outcome, b.outcome)]
    capture_times = [
        (a.capture_time, b.capture_time)
        for a, b in both_resolved
        if a.capture_time is not None and b.capture_time is not None
    ]
    map_disagreement = sum(a.outcome != b.outcome for a, b in pairs)
    outcome_disagreement = sum(a.outcome != b.outcome for a, b in both_resolved)
    capture_delta = candidate_counts["capture"] - reference_counts["capture"]
    summary = {
        "starts": total,
        "both_resolved": len(both_resolved),
        "outcome_disagreement_count": outcome_disagreement,
        "outcome_disagreement_rate": (
            outcome_disagreement / len(both_resolved) if both_resolved else None
        ),
        "map_status_disagreement_count": map_disagreement,
        "map_status_disagreement_rate": map_disagreement / total,
        "capture_rate_delta_bounds": [
            (capture_delta - reference_counts["unsupported"]) / total,
            (capture_delta + candidate_counts["unsupported"]) / total,
        ],
        "both_captured": len(capture_times),
        "mean_capture_time_delta": mean(a - b for a, b in capture_times) if capture_times else None,
    }
    for index, (name, counts) in enumerate(
        (("candidate", candidate_counts), ("reference", reference_counts))
    ):
        captures = counts["capture"]
        paired_captures = sum(pair[index].outcome == "capture" for pair in both_resolved)
        summary[name] = {
            "counts": {
                outcome: counts[outcome] for outcome in ("capture", "time_limit", "unsupported")
            },
            "capture_rate_bounds": [captures / total, (captures + counts["unsupported"]) / total],
            "capture_rate_on_both_resolved": (
                paired_captures / len(both_resolved) if both_resolved else None
            ),
            "mean_capture_time_on_both_captured": (
                mean(pair[index] for pair in capture_times) if capture_times else None
            ),
        }
    return summary


def save_maps(
    target: Path,
    candidate: Sequence[EpisodeResult],
    reference: Sequence[EpisodeResult],
    config: GameConfig,
    *,
    labels: tuple[str, str] = ("Heuristic", "Classical"),
    opponent: str = "the classical evader",
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = {
        "outside evaluation": "#ffffff",
        "capture": "#116f70",
        "time_limit": "#324b70",
        "unsupported": "#e1e6eb",
        "one unsupported": "#f0a325",
        "capture vs. time limit": "#bf416d",
    }
    codes = {label: index for index, label in enumerate(colors)}
    size = GRID_MAX - GRID_MIN + 1
    maps = np.zeros((3, size, size), dtype=int)
    for a, b in zip(candidate, reference, strict=True):
        x, y = a.start
        col, row = round(x * GRID_SCALE) - GRID_MIN, round(y * GRID_SCALE) - GRID_MIN
        if a.outcome == b.outcome:
            comparison = a.outcome
        elif "unsupported" in (a.outcome, b.outcome):
            comparison = "one unsupported"
        else:
            comparison = "capture vs. time limit"
        maps[:, row, col] = [codes[a.outcome], codes[b.outcome], codes[comparison]]

    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.25, top=0.84, wspace=0.12)
    fig.suptitle(
        f"Capture by T = {config.horizon:g} against {opponent}",
        fontsize=17,
        fontweight="bold",
    )
    step = 1 / GRID_SCALE
    lower, upper = GRID_MIN / GRID_SCALE - step / 2, GRID_MAX / GRID_SCALE + step / 2
    for ax, values, title in zip(axes, maps, (*labels, "Paired outcomes")):
        ax.imshow(
            values,
            origin="lower",
            extent=(lower, upper, lower, upper),
            cmap=ListedColormap(list(colors.values())),
            vmin=0,
            vmax=len(colors) - 1,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", xticks=range(-4, 5, 2), yticks=range(-4, 5, 2))
    axes[0].set_ylabel("y")
    legend_labels = {
        "outside evaluation": "Inside capture disk",
        "capture": "Capture",
        "time_limit": "Time limit",
        "unsupported": "Unsupported",
        "one unsupported": "One unsupported",
        "capture vs. time limit": "Capture vs. time limit",
    }
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#75808c", label=legend_labels[label])
            for label, color in colors.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=3,
        frameon=False,
    )
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/chauffeur_baseline"))
    output = parser.parse_args().output

    import matplotlib

    matplotlib.use("Agg")
    config = DEFAULT_CONFIG
    starts = final_grid()
    heuristic = evaluate_pursuer(heuristic_pursuer, starts, config)
    classical = evaluate_pursuer(classical_pursuer, starts, config)
    summary = compare(heuristic, classical)
    output.mkdir(parents=True, exist_ok=True)
    save_episodes(output / "episodes.csv", {"heuristic": heuristic, "classical": classical})
    with (output / "tuning_starts.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("x", "y"))
        writer.writerows(tuning_starts())
    report = {
        "config": asdict(config),
        "grid": {
            "min": GRID_MIN / GRID_SCALE,
            "max": GRID_MAX / GRID_SCALE,
            "step": 1 / GRID_SCALE,
        },
        "tuning": {"seed": TUNING_SEED, "count": TUNING_COUNT},
        "candidate": "heuristic_pursuer",
        "reference": "classical_pursuer",
        "evader": "classical_evader",
        "metrics": summary,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_maps(output / "maps.png", heuristic, classical, config)
    print(f"Evaluated {len(starts)} starts per policy. Results saved to {output}")


if __name__ == "__main__":
    main()
