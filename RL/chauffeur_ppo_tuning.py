"""Train a chauffeur PPO or evaluate a saved checkpoint on the tuning starts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from RL.chauffeur_classical import DEFAULT_CONFIG, rollout_feedback
from RL.chauffeur_evaluation import (
    TUNING_SEED,
    EpisodeResult,
    compare,
    heuristic_pursuer,
    tuning_starts,
)
from RL.chauffeur_ppo import ChauffeurPPOEnv

STEPS = 131_072
SHORT_STEP = 0.001


class TrainingDiagnostics(BaseCallback):
    def __init__(self):
        super().__init__()
        self.time = 0.0
        self.short_steps = 0
        self.outcomes = Counter()

    def _on_step(self) -> bool:
        (info,) = self.locals["infos"]
        dt = info["dt"]
        self.time += dt
        self.short_steps += 0 < dt < SHORT_STEP
        if info["termination_reason"] is not None:
            self.outcomes[info["termination_reason"]] += 1
        return True


def train(output: Path, seed: int, discount: float) -> Path:
    diagnostics = TrainingDiagnostics()
    checkpoint = output / "ppo.zip"
    with Monitor(ChauffeurPPOEnv(), filename=str(output / "training")) as env:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            seed=seed,
            n_steps=1024,
            gamma=discount,
            policy_kwargs={"net_arch": [32, 32]},
        )
        model.set_logger(configure(str(output), ["csv"]))
        started = perf_counter()
        model.learn(total_timesteps=STEPS, callback=diagnostics)
        seconds = perf_counter() - started
        model.save(checkpoint)
    record = {
        "seconds": seconds,
        "simulated_time": diagnostics.time,
        "short_steps": diagnostics.short_steps,
        "finished_episodes": dict(diagnostics.outcomes),
    }
    (output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Trained {STEPS} steps in {seconds:.1f}s: {record}", flush=True)
    return checkpoint


def evaluate(output: Path, checkpoint: Path) -> None:
    model = PPO.load(checkpoint, device="cpu")

    def predict(_state, _config, observation):
        action, _ = model.predict(observation, deterministic=True)
        return float(action.item())

    def heuristic(state, config, _observation):
        return heuristic_pursuer(state, config)

    starts = tuning_starts()
    results = {}
    diagnostics = {}
    with ChauffeurPPOEnv() as env, (output / "episodes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("policy", "x", "y", "outcome", "end_time", "steps", "short_steps"))
        for name, policy in (("heuristic", heuristic), ("ppo", predict)):
            episodes = []
            totals = Counter()
            for start in starts:
                run = rollout_feedback(start, env, pursuer=policy)
                end = run.trajectory[-1].time
                steps = len(run.trajectory) - 1
                short = sum(0 < b.time - a.time < SHORT_STEP for a, b in pairwise(run.trajectory))
                episodes.append(
                    EpisodeResult(start, run.outcome, end if run.outcome == "capture" else None)
                )
                totals.update(steps=steps, short_steps=short)
                writer.writerow((name, *start, run.outcome, end, steps, short))
            results[name] = episodes
            diagnostics[name] = dict(totals)
            print(f"Evaluated {name} on {len(starts)} tuning starts", flush=True)

    summary = {
        "checkpoint": str(checkpoint),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": 1,
        "seed": model.seed,
        "timesteps": model.num_timesteps,
        "ppo": {
            "policy": type(model.policy).__name__,
            "network": model.policy_kwargs["net_arch"],
            "discount": model.gamma,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate,
            "gae_lambda": model.gae_lambda,
            "ent_coef": model.ent_coef,
        },
        "environment": asdict(DEFAULT_CONFIG),
        "tuning_seed": TUNING_SEED,
        "diagnostics": diagnostics,
        "ppo_vs_heuristic": compare(results["ppo"], results["heuristic"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory for the results")
    parser.add_argument(
        "--checkpoint", type=Path, help="evaluate an existing model without training"
    )
    parser.add_argument("--seed", type=int, default=17, help="training seed")
    parser.add_argument("--undiscounted", action="store_true", help="train with PPO discount 1.0")
    args = parser.parse_args()
    if args.checkpoint is not None and (args.undiscounted or args.seed != 17):
        parser.error("--seed and --undiscounted are only used when training")

    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    checkpoint = args.checkpoint or train(
        args.output, args.seed, 1.0 if args.undiscounted else 0.99
    )
    evaluate(args.output, checkpoint)


if __name__ == "__main__":
    main()
