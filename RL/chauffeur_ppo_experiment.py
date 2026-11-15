"""Train, reload and compare the first CPU PPO chauffeur on held-out starts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import gymnasium
import matplotlib
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from RL.chauffeur_classical import (
    DEFAULT_CONFIG,
    Policy,
    Rollout,
    State,
    TrajectoryPoint,
    classical_pursuer,
)
from RL.chauffeur_evaluation import (
    TUNING_SEED,
    compare,
    final_grid,
    heuristic_pursuer,
    save_episodes,
    save_maps,
    summarize_rollouts,
    tuning_starts,
)
from RL.chauffeur_ppo import ChauffeurPPOEnv
from RL.homicidal_chauffeur import GameConfig, Observation

TRAIN_SEED = 17
N_STEPS = 1024
POLICY_LAYERS = [32, 32]
BENCHMARKS = ((0.0, 2.3), (0.5, 0.0), (0.0, -1.0), (1.105240, 2.278470))
RolloutPolicy = Callable[[State, GameConfig, Observation], float | None]


def train(output: Path, steps: int) -> dict:
    torch.set_num_threads(1)
    env = Monitor(ChauffeurPPOEnv(), filename=str(output / "training"))
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=TRAIN_SEED,
        n_steps=N_STEPS,
        policy_kwargs={"net_arch": POLICY_LAYERS},
    )
    model.set_logger(configure(str(output), ["csv"]))
    model.save(output / "untrained.zip")
    started = perf_counter()
    model.learn(total_timesteps=steps)
    duration = perf_counter() - started
    model.save(output / "ppo.zip")
    env.close()
    return {
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_threads": 1,
        "seed": TRAIN_SEED,
        "initial_states": "uniform random starts outside the capture disk",
        "environment": asdict(DEFAULT_CONFIG),
        "policy": "MlpPolicy",
        "policy_layers": POLICY_LAYERS,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": model.ent_coef,
        "timesteps": model.num_timesteps,
        "seconds": duration,
        "steps_per_second": model.num_timesteps / duration,
    }


def rollout_in_env(pursuer: RolloutPolicy, start: State, *, env: ChauffeurPPOEnv) -> Rollout:
    observation, info = env.reset(options={"state": start})
    state = tuple(info["state"].tolist())
    trajectory = [TrajectoryPoint(0.0, state)]
    while True:
        phi = pursuer(state, env.config, observation)
        if phi is None:
            return Rollout("unsupported", tuple(trajectory))
        observation, _, captured, timed_out, info = env.step(np.array([phi], dtype=np.float32))
        state = tuple(info["state"].tolist())
        trajectory.append(TrajectoryPoint(info["elapsed_time"], state, info["phi"], info["psi"]))
        if captured or timed_out:
            return Rollout("capture" if captured else "time_limit", tuple(trajectory))


def checkpoint_policy(model: PPO) -> RolloutPolicy:
    def predict(_state: State, _config: GameConfig, observation: Observation) -> float:
        action, _ = model.predict(observation, deterministic=True)
        return float(action.item())

    return predict


def state_policy(policy: Policy) -> RolloutPolicy:
    def predict(state: State, config: GameConfig, _observation: Observation) -> float | None:
        return policy(state, config)

    return predict


def save_trajectories(output: Path, runs: dict[str, list[Rollout]]) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    colors = {"ppo": "#116f70", "heuristic": "#bf416d", "classical": "#324b70"}
    styles = {"ppo": "-", "heuristic": "--", "classical": ":"}
    runs_by_start = {
        name: {run.trajectory[0].state: run for run in rollouts} for name, rollouts in runs.items()
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    with (output / "trajectories.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("pursuer", "x0", "y0", "outcome", "time", "x", "y", "phi", "psi"))
        for ax, start in zip(axes.flat, BENCHMARKS, strict=True):
            ax.add_patch(Circle((0, 0), DEFAULT_CONFIG.beta, fill=False, ls=":", color="#262626"))
            for name, rollouts in runs_by_start.items():
                run = rollouts[start]
                writer.writerows(
                    (name, *start, run.outcome, point.time, *point.state, point.phi, point.psi)
                    for point in run.trajectory
                )
                x, y = zip(*(point.state for point in run.trajectory))
                ax.plot(x, y, color=colors[name], ls=styles[name], label=f"{name} · {run.outcome}")
                ax.scatter(x[-1], y[-1], color=colors[name], s=18)
            ax.set(title=f"Start ({start[0]:g}, {start[1]:g})", xlabel="x", ylabel="y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Relative trajectories under the shared PPO environment")
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)


def evaluate(output: Path) -> dict:
    config = DEFAULT_CONFIG
    env = ChauffeurPPOEnv(config)
    rollout = partial(rollout_in_env, env=env)
    model = PPO.load(output / "ppo.zip", device="cpu")
    untrained = checkpoint_policy(PPO.load(output / "untrained.zip", device="cpu"))
    policies = {
        "ppo": checkpoint_policy(model),
        "heuristic": state_policy(heuristic_pursuer),
        "classical": state_policy(classical_pursuer),
    }
    tuning = tuning_starts()
    before = summarize_rollouts(rollout(untrained, start) for start in tuning)
    after = summarize_rollouts(rollout(policies["ppo"], start) for start in tuning)
    print(f"Compared both checkpoints on {len(tuning)} tuning starts", flush=True)

    starts = final_grid()
    results = {}
    for name, pursuer in policies.items():
        episodes = summarize_rollouts(rollout(pursuer, start) for start in starts)
        results[name] = episodes
        print(f"Evaluated {name} on {len(episodes)} grid starts", flush=True)
    save_episodes(output / "episodes.csv", results)

    save_trajectories(
        output,
        {
            name: [rollout(pursuer, start) for start in BENCHMARKS]
            for name, pursuer in policies.items()
        },
    )
    summary = {
        "evader": "classical_evader, psi=0 outside supported branches",
        "tuning_seed": TUNING_SEED,
        "trained_vs_untrained": compare(after, before),
    }
    for reference in ("heuristic", "classical"):
        save_maps(
            output / f"ppo_vs_{reference}.png",
            results["ppo"],
            results[reference],
            config,
            labels=("PPO", reference.title()),
            opponent="the fixed evader (classical or ψ = 0)",
        )
        summary[f"ppo_vs_{reference}"] = compare(results["ppo"], results[reference])
    env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/chauffeur_ppo"))
    parser.add_argument("--steps", type=int, default=131_072)
    parser.add_argument(
        "--pilot", action="store_true", help="train and measure CPU throughput only"
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % N_STEPS:
        parser.error(f"--steps must be a positive multiple of {N_STEPS}")
    args.output.mkdir(parents=True)
    matplotlib.use("Agg")
    training = train(args.output, args.steps)
    (args.output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    print(
        f"Trained {training['timesteps']} steps at {training['steps_per_second']:.0f} steps/s",
        flush=True,
    )
    if not args.pilot:
        summary = evaluate(args.output)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
