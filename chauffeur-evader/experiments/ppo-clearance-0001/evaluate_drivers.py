"""Evaluate task 09 and the controls on the clearance=0.001 protocol."""

import argparse
import csv
import json
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path

from chauffeur_evader.chauffeur_benchmarks import BENCHMARKS
from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, classical_pursuer
from chauffeur_evader.chauffeur_evaluation import (
    _episode_result,
    _ppo_rollouts,
    _pursuer_rollouts,
    final_grid,
    heuristic_pursuer,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ppo-clearance-0001"
FREEZE = ROOT / "experiments/chauffeur-task10-v1"
SERIES = DATA / "task09-pdl/full-after-07"
CONTROL = DATA / "full"
SEEDS = (17, 29, 43)
BUDGETS = (131072, 262144)
REPORT_SEED = 43
CLEARANCE = 0.001
GRAY = (-2.3, -0.2)
REFERENCES = ("classical", "heuristic")
DRIVERS = (*REFERENCES, "archive", "new")
EVALUATION = {
    "environment": asdict(DEFAULT_CONFIG),
    "clearance": CLEARANCE,
    "mode": "first_contact",
    "evader": "classical",
    "deterministic": True,
}
EPISODE_FIELDS = ("dataset", "start", "policy", "x", "y", "outcome", "capture_time")
POINT_FIELDS = (
    "dataset",
    "start",
    "policy",
    "decision",
    "time",
    "x",
    "y",
    "phi_on_arrival",
    "psi_at_interval_start",
    "event",
)


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def policy(group, seed, budget):
    return f"{group}-seed-{seed}-ppo-{budget}"


def cases(smoke):
    spec = json.loads((FREEZE / "protocol.json").read_text())["evaluation"]
    path = FREEZE / spec["tuning"]["file"]
    if digest(path) != spec["tuning"]["sha256"]:
        raise ValueError(f"Tuning starts differ from protocol: {path}")
    with path.open(newline="") as file:
        tuning = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(file)]
    grid = final_grid()
    if len(tuning) != 512 or len(set(tuning)) != 512 or len(grid) != 6532 or GRAY not in grid:
        raise ValueError("Unexpected tuning starts or final grid")
    grid_cases = {
        str(i): (x, y)
        for i, (x, y) in enumerate(grid, 1)
        if not smoke or (abs(x - GRAY[0]) < 0.21 and abs(y - GRAY[1]) < 0.21)
    }
    return {
        "tuning": (
            {str(i): start for i, start in enumerate(tuning[:8] if smoke else tuning, 1)},
            (
                *REFERENCES,
                "archive",
                *(
                    policy(group, seed, budget)
                    for group in ("09", "08")
                    for budget in BUDGETS
                    for seed in SEEDS
                ),
            ),
        ),
        "grid": (grid_cases, DRIVERS),
        "benchmarks": (BENCHMARKS, DRIVERS),
    }


def verify_series(directory, *, current):
    summary = json.loads((directory / "summary.json").read_text())
    frozen_path = FREEZE / "protocol.json"
    frozen = json.loads(frozen_path.read_text())
    if (
        (summary["status"], summary["mode"], summary["starts"])
        != ("complete", "full", {"tuning": 512, "benchmarks": 8})
        or summary["evaluation"] != EVALUATION
        or summary["inputs"]["protocol_sha256"] != digest(frozen_path)
        or summary["inputs"]["tuning"] != frozen["evaluation"]["tuning"]
        or digest(directory / "task10-protocol.json") != digest(frozen_path)
        or digest(directory / "tuning_starts.csv") != frozen["evaluation"]["tuning"]["sha256"]
        or summary["training"]
        != {
            **{k: v for k, v in frozen["control"].items() if k != "checkpoints"},
            "checkpoints": list(BUDGETS),
            "seeds": list(SEEDS),
            "clearance": CLEARANCE,
        }
    ):
        raise ValueError(f"Series protocol mismatch: {directory}")
    if current:
        sources = json.loads((directory / "source-sha256.json").read_text())
        for name in (
            "chauffeur_classical",
            "chauffeur_theory",
            "homicidal_chauffeur",
            "chauffeur_ppo",
            "chauffeur_evaluation",
            "chauffeur_ppo_tuning",
            "chauffeur_task10/common",
            "chauffeur_task10/clearance",
        ):
            source = f"chauffeur_evader/{name}.py"
            if sources[source] != digest(ROOT / source):
                raise ValueError(f"Saved 09 uses different code: {source}")
    manifest = {
        path: sha
        for sha, path in (
            line.split() for line in (directory / "models.sha256").read_text().splitlines()
        )
    }
    if manifest != {m["path"]: m["sha256"] for m in summary["models"].values()}:
        raise ValueError(f"Model manifest mismatch: {directory}")
    return summary


def load_models(series):
    import torch
    from chauffeur_evader.chauffeur_task10.clearance import load_checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models, identities = {}, {}
    for group, directory in (("09", series), ("08", CONTROL)):
        summary = verify_series(directory, current=group == "09")
        training = summary["training"]
        for seed in SEEDS:
            settings = directory / f"seed-{seed}/settings.json"
            recorded = json.loads(settings.read_text())
            expected = {
                key: training[key]
                for key in (
                    "architecture",
                    "reward",
                    "ppo",
                    "environment",
                    "clearance",
                    "torch_threads",
                )
            }
            expected.update(seed=seed, budgets=list(BUDGETS), tuning_seed=42)
            expected.update(
                {key: summary["runtime"][key] for key in ("torch", "stable_baselines3")}
            )
            if any(recorded[key] != value for key, value in expected.items()):
                raise ValueError(f"Training settings mismatch: {settings}")
            for budget in BUDGETS:
                label = policy(group, seed, budget)
                entry = summary["models"][f"seed-{seed}-ppo-{budget}"]
                path = directory / entry["path"]
                if (
                    (entry["seed"], entry["decisions"], entry["training_budget"])
                    != (seed, budget, BUDGETS[-1])
                    or path != settings.with_name(f"ppo-{budget}.zip")
                    or directory / entry["settings"] != settings
                ):
                    raise ValueError(f"Unexpected checkpoint identity: {path}")
                models[label], identity = load_checkpoint(
                    path,
                    entry["sha256"],
                    seed,
                    budget,
                    BUDGETS[-1],
                )
                identities[label] = {
                    **identity,
                    "loaded_path": str(path.relative_to(ROOT)),
                    "settings": str(settings.relative_to(ROOT)),
                    "settings_sha256": digest(settings),
                    "training_clearance": CLEARANCE,
                    "series": str(directory.relative_to(ROOT)),
                    "series_summary_sha256": digest(directory / "summary.json"),
                }
    fixed_path = FREEZE / "protocol.json"
    fixed = json.loads(fixed_path.read_text())["control"]["checkpoints"][str(REPORT_SEED)]
    archive = ROOT / fixed["path"]
    models["archive"], identity = load_checkpoint(
        archive,
        fixed["sha256"],
        REPORT_SEED,
        BUDGETS[-1],
        BUDGETS[-1],
    )
    identities["archive"] = {
        **identity,
        "loaded_path": fixed["path"],
        "settings": str(fixed_path.relative_to(ROOT)),
        "settings_sha256": digest(fixed_path),
        "training_clearance": 0.0,
    }
    report_policy = policy("09", REPORT_SEED, BUDGETS[-1])
    models["new"] = models[report_policy]
    identities["new"] = {**identities[report_policy], "tuning_policy": report_policy}
    return models, identities


def model_rollouts(model, starts):
    def predict(_state, _config, observation):
        action, _ = model.predict(observation, deterministic=True)
        return float(action.item())

    return _ppo_rollouts(predict, starts, DEFAULT_CONFIG, clearance=CLEARANCE)


def evaluate(output, series, *, smoke):
    datasets = cases(smoke)
    models, identities = load_models(series)
    output.mkdir(parents=True)
    baselines = {"classical": classical_pursuer, "heuristic": heuristic_pursuer}
    with (
        (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
        (output / "trajectories.csv").open("w", newline="", encoding="utf-8") as point_file,
    ):
        episodes, points = csv.writer(episode_file), csv.writer(point_file)
        episodes.writerow(EPISODE_FIELDS)
        points.writerow(POINT_FIELDS)
        for dataset, (starts, drivers) in datasets.items():
            for driver in drivers:
                print(f"{dataset}: {driver}, {len(starts)} starts", flush=True)
                rollouts = (
                    _pursuer_rollouts(
                        baselines[driver],
                        starts.values(),
                        DEFAULT_CONFIG,
                        clearance=CLEARANCE,
                    )
                    if driver in baselines
                    else model_rollouts(models[driver], starts.values())
                )
                for (label, start), run in zip(starts.items(), rollouts, strict=True):
                    result = _episode_result(run)
                    if result.start != start:
                        raise ValueError(f"Unexpected start: {dataset}/{label}/{driver}")
                    episodes.writerow(
                        (dataset, label, driver, *start, result.outcome, result.capture_time)
                    )
                    if dataset == "benchmarks" or (dataset == "grid" and start == GRAY):
                        points.writerows(
                            (
                                dataset,
                                label,
                                driver,
                                i,
                                point.time,
                                *point.state,
                                point.phi,
                                point.psi,
                                point.event,
                            )
                            for i, point in enumerate(run.trajectory)
                        )

    manifest = {
        "status": "complete",
        "mode": "smoke" if smoke else "full",
        "evaluation": EVALUATION,
        "starts": {name: len(starts) for name, (starts, _) in datasets.items()},
        "report_seed": REPORT_SEED,
        "protocol_sha256": digest(FREEZE / "protocol.json"),
        "models": identities,
        "sha256": {name: digest(output / name) for name in ("episodes.csv", "trajectories.csv")},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation complete: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory under data/ppo-clearance-0001")
    parser.add_argument("--series", type=Path, default=SERIES, help="completed 09 series")
    parser.add_argument("--smoke", action="store_true", help="8 tuning, 25 grid, all 8 benchmarks")
    args = parser.parse_args()
    output, series = args.output.resolve(), args.series.resolve()
    if not output.is_relative_to(DATA) or output == DATA or output.exists():
        parser.error(f"choose a new subdirectory of {DATA}")
    if not series.is_relative_to(DATA):
        parser.error(f"choose a series under {DATA}")
    evaluate(output, series, smoke=args.smoke)


if __name__ == "__main__":
    main()
