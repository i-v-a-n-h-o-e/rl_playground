"""Train and evaluate the two PPO snapshots for the task 09 handoff."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO

from chauffeur_evader.chauffeur_benchmarks import BENCHMARKS
from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, Rollout, State, classical_pursuer
from chauffeur_evader.chauffeur_evaluation import (
    EpisodeResult,
    _episode_result,
    _outcome_summary,
    _ppo_rollouts,
    _pursuer_rollouts,
    compare,
    heuristic_pursuer,
)
from chauffeur_evader.chauffeur_task10.common import (
    NETWORKS,
    POTENTIAL_SCALE,
    PPO_SETTINGS,
    ROOT,
    SEEDS,
    STEPS,
    train,
)

CLEARANCE = 0.001
PROTOCOL = ROOT / "experiments/chauffeur-task10-v1/protocol.json"
OUTPUT_ROOT = ROOT / "data/ppo-clearance-0001"
ARCHIVE_SEED = 43


def load_checkpoint(
    path: Path, expected_hash: str, seed: int, decisions: int, training_budget: int
) -> tuple[PPO, dict]:
    path = path.resolve(strict=True)
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {path}")
    model = PPO.load(BytesIO(payload), device="cpu")
    updates = decisions // PPO_SETTINGS["n_steps"] * PPO_SETTINGS["n_epochs"]
    if (
        (model.seed, model.num_timesteps, model._total_timesteps, model._n_updates)
        != (seed, decisions, training_budget, updates)
        or model.n_envs != 1
        or model.policy_kwargs["net_arch"] != NETWORKS["wide"]
        or model.policy_class is not PPO.policy_aliases["MlpPolicy"]
        or str(model.device) != "cpu"
        or any(
            getattr(model, key) != value for key, value in PPO_SETTINGS.items() if key != "device"
        )
    ):
        raise ValueError(f"Loaded PPO metadata does not match the protocol: {path}")
    return model, {
        "loaded_path": str(path),
        "sha256": digest,
        "seed": model.seed,
        "decisions": model.num_timesteps,
        "training_budget": model._total_timesteps,
        "optimizer_epochs": model._n_updates,
    }


def write_comparisons(
    output: Path,
    starts: dict[str, dict[str, State]],
    results: dict[str, dict[str, list[EpisodeResult]]],
) -> dict:
    comparisons = {}
    with (output / "paired_capture_times.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("dataset", "start", "candidate", "reference", "time_delta"))
        for dataset, policies in results.items():
            comparisons[dataset] = []
            for reference, candidate in combinations(policies, 2):
                a, b = policies[candidate], policies[reference]
                comparisons[dataset].append(
                    {**compare(a, b), "candidate_policy": candidate, "reference_policy": reference}
                )
                writer.writerows(
                    (dataset, label, candidate, reference, left.capture_time - right.capture_time)
                    for label, left, right in zip(starts[dataset], a, b, strict=True)
                    if left.capture_time is not None and right.capture_time is not None
                )
    return comparisons


def run(output: Path, *, smoke: bool) -> None:
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    control = protocol["control"]
    environment = asdict(DEFAULT_CONFIG)
    ppo_settings = {**PPO_SETTINGS, "policy": "MlpPolicy", "net_arch": NETWORKS["wide"]}
    frozen_budgets = (STEPS, 2 * STEPS)
    expected = {
        "architecture": "wide",
        "reward": "potential",
        "potential_scale": POTENTIAL_SCALE,
        "decisions": frozen_budgets[-1],
        "ppo": ppo_settings,
        "environment": environment,
        "torch_threads": 1,
    }
    if any(control[key] != value for key, value in expected.items()) or (
        tuple(protocol["training_seeds"]["task10"]) != SEEDS
    ):
        raise ValueError(f"Training settings differ from {PROTOCOL}")
    rollout_steps = PPO_SETTINGS["n_steps"]
    budgets = (rollout_steps, 2 * rollout_steps) if smoke else frozen_budgets

    tuning = protocol["evaluation"]["tuning"]
    tuning_path = PROTOCOL.parent / tuning["file"]
    tuning_bytes = tuning_path.read_bytes()
    if sha256(tuning_bytes).hexdigest() != tuning["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {tuning_path}")
    tuning_starts = [
        (float(row["x"]), float(row["y"]))
        for row in csv.DictReader(tuning_bytes.decode("utf-8").splitlines())
    ]
    if len(tuning_starts) != tuning["count"]:
        raise ValueError(f"Wrong number of tuning starts: {tuning_path}")
    starts = {
        "tuning": {
            str(i): state
            for i, state in enumerate(tuning_starts[:8] if smoke else tuning_starts, 1)
        },
        "benchmarks": BENCHMARKS,
    }

    archive = control["checkpoints"][str(ARCHIVE_SEED)]
    archive_source = (ROOT / archive["path"]).resolve(strict=True)
    archive_bytes = archive_source.read_bytes()
    if sha256(archive_bytes).hexdigest() != archive["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {archive_source}")

    torch.set_num_threads(expected["torch_threads"])
    torch.set_num_interop_threads(expected["torch_threads"])
    output.mkdir(parents=True)
    archive_path = output / "archive" / f"seed-{ARCHIVE_SEED}" / "ppo-262144.zip"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive_bytes)
    (output / "task10-protocol.json").write_bytes(protocol_bytes)
    (output / "tuning_starts.csv").write_bytes(tuning_bytes)

    models = {}
    results = {dataset: {} for dataset in starts}
    with (
        (output / "episodes.csv").open("w", encoding="utf-8", newline="") as episode_file,
        gzip.open(
            output / "trajectories.csv.gz", "wt", compresslevel=3, encoding="utf-8", newline=""
        ) as trajectory_file,
    ):
        episodes, trajectories = csv.writer(episode_file), csv.writer(trajectory_file)
        episodes.writerow(("dataset", "start", "policy", "x", "y", "outcome", "capture_time"))
        trajectories.writerow(
            (
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
        )

        def record(policy: str, dataset: str, rollouts: Iterator[Rollout]) -> None:
            summaries = results[dataset][policy] = []
            for (label, start), rollout in zip(starts[dataset].items(), rollouts, strict=True):
                result = _episode_result(rollout)
                if result.start != start:
                    raise ValueError(f"Unexpected start: {dataset}/{label} for {policy}")
                summaries.append(result)
                episodes.writerow(
                    (dataset, label, policy, *start, result.outcome, result.capture_time)
                )
                trajectories.writerows(
                    (
                        dataset,
                        label,
                        policy,
                        step,
                        point.time,
                        *point.state,
                        point.phi,
                        point.psi,
                        point.event,
                    )
                    for step, point in enumerate(rollout.trajectory)
                )

        def evaluate_checkpoint(
            policy: str,
            path: Path,
            digest: str,
            seed: int,
            decisions: int,
            training_budget: int,
            settings_path: Path,
        ) -> None:
            model, identity = load_checkpoint(path, digest, seed, decisions, training_budget)
            models[policy] = {
                **identity,
                "path": str(path.relative_to(output)),
                "settings": str(settings_path.relative_to(output)),
            }
            print(f"Evaluate {policy}: {identity['loaded_path']} SHA-256 {digest}", flush=True)

            def predict(_state, _config, observation):
                action, _ = model.predict(observation, deterministic=True)
                return float(action.item())

            for dataset, cases in starts.items():
                record(
                    policy,
                    dataset,
                    _ppo_rollouts(predict, cases.values(), DEFAULT_CONFIG, clearance=CLEARANCE),
                )

        for policy, pursuer in (("classical", classical_pursuer), ("heuristic", heuristic_pursuer)):
            print(f"Evaluate {policy}", flush=True)
            for dataset, cases in starts.items():
                record(
                    policy,
                    dataset,
                    _pursuer_rollouts(pursuer, cases.values(), DEFAULT_CONFIG, clearance=CLEARANCE),
                )

        archive_budget = control["decisions"]
        evaluate_checkpoint(
            "archive-seed-43",
            archive_path,
            archive["sha256"],
            ARCHIVE_SEED,
            archive_budget,
            archive_budget,
            output / "task10-protocol.json",
        )
        models["archive-seed-43"]["source_path"] = str(archive_source)
        models["archive-seed-43"]["training_clearance"] = 0.0

        for seed in SEEDS:
            print(f"Train seed {seed} from scratch for {budgets[-1]} decisions", flush=True)
            directory = output / f"seed-{seed}"
            train(
                directory,
                seed,
                control["architecture"],
                control["reward"],
                budgets,
                clearance=CLEARANCE,
            )
            for decisions in budgets:
                path = directory / f"ppo-{decisions}.zip"
                evaluate_checkpoint(
                    f"seed-{seed}-ppo-{decisions}",
                    path,
                    sha256(path.read_bytes()).hexdigest(),
                    seed,
                    decisions,
                    budgets[-1],
                    directory / "settings.json",
                )

    comparisons = write_comparisons(output, starts, results)
    (output / "models.sha256").write_text(
        "".join(f"{model['sha256']}  {model['path']}\n" for model in models.values()),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "mode": "smoke" if smoke else "full",
        "training": {
            **expected,
            "decisions": budgets[-1],
            "checkpoints": budgets,
            "seeds": SEEDS,
            "clearance": CLEARANCE,
        },
        "evaluation": {
            "environment": environment,
            "clearance": CLEARANCE,
            "mode": "first_contact",
            "evader": "classical",
            "deterministic": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "inputs": {"protocol_sha256": sha256(protocol_bytes).hexdigest(), "tuning": tuning},
        "starts": {dataset: len(cases) for dataset, cases in starts.items()},
        "models": models,
        "outcomes": {
            dataset: {
                policy: _outcome_summary(Counter(e.outcome for e in runs), len(runs))
                for policy, runs in policies.items()
            }
            for dataset, policies in results.items()
        },
        "comparisons": comparisons,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {report['mode']} run: {output / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="train for two rollouts per seed")
    parser.add_argument("--output", type=Path, help="new directory under data/ppo-clearance-0001")
    args = parser.parse_args()
    output = (args.output or OUTPUT_ROOT / ("smoke" if args.smoke else "full")).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        parser.error(f"choose a subdirectory of {OUTPUT_ROOT}")
    try:
        run(output, smoke=args.smoke)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
