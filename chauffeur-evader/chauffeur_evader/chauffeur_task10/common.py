"""Training, paired evaluation and reports shared by the four experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from math import hypot
from pathlib import Path
from statistics import mean
from time import perf_counter

import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, State, classical_pursuer
from chauffeur_evader.chauffeur_evaluation import (
    TUNING_SEED,
    EpisodeResult,
    compare,
    evaluate_ppo,
    evaluate_pursuer,
    heuristic_pursuer,
    tuning_starts,
)
from chauffeur_evader.chauffeur_ppo import ChauffeurPPOEnv
from chauffeur_evader.chauffeur_ppo_tuning import STEPS, TrainingDiagnostics
from chauffeur_evader.homicidal_chauffeur import Observation, Transition

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (17, 29, 43)
DISCOUNT = 0.99
NETWORKS = {"baseline": [32, 32], "wide": {"pi": [64, 64], "vf": [32, 32]}}
REWARDS = ("base", "potential")
PPO_SETTINGS = {
    "device": "cpu",
    "n_steps": 1024,
    "gamma": DISCOUNT,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
}


class PotentialRewardEnv(ChauffeurPPOEnv):
    def _potential(self, distance: float) -> float:
        return 0.1 / (1 + (distance - self.config.beta) / 4)

    def step(self, action: Observation) -> Transition:
        before = self._potential(hypot(*self.state))
        observation, reward, terminated, truncated, info = super().step(action)
        after = 0.0 if terminated or truncated else self._potential(info["distance"])
        return observation, reward + (DISCOUNT * after - before), terminated, truncated, info


class TrainingSnapshots(TrainingDiagnostics):
    def __init__(self, checkpoints: dict[int, Path]):
        super().__init__()
        self.checkpoints = checkpoints

    def _on_rollout_start(self) -> None:
        # The previous rollout has already been used for an optimizer update.
        checkpoint = self.checkpoints.get(self.model.num_timesteps)
        if checkpoint is not None:
            self.model.save(checkpoint)


def train(
    output: Path,
    seed: int,
    architecture: str,
    reward: str,
    budgets: tuple[int, ...],
    *,
    clearance: float = 0.0,
) -> None:
    output.mkdir()
    net_arch = NETWORKS[architecture]
    settings = {
        "seed": seed,
        "budgets": budgets,
        "architecture": architecture,
        "reward": reward,
        "ppo": {**PPO_SETTINGS, "policy": "MlpPolicy", "net_arch": net_arch},
        "environment": asdict(DEFAULT_CONFIG),
        "clearance": clearance,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "tuning_seed": TUNING_SEED,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    diagnostics = TrainingSnapshots({steps: output / f"ppo-{steps}.zip" for steps in budgets[:-1]})
    env_type = PotentialRewardEnv if reward == "potential" else ChauffeurPPOEnv
    env = env_type(clearance=clearance)
    with Monitor(env, filename=str(output / "training")) as monitored:
        model = PPO(
            "MlpPolicy", monitored, seed=seed, policy_kwargs={"net_arch": net_arch}, **PPO_SETTINGS
        )
        model.set_logger(configure(str(output), ["csv"]))
        started = perf_counter()
        model.learn(total_timesteps=budgets[-1], callback=diagnostics)
        model.save(output / f"ppo-{budgets[-1]}.zip")
    record = {
        "seconds": perf_counter() - started,
        "simulated_time": diagnostics.time,
        "short_steps": diagnostics.short_steps,
        "finished_episodes": dict(diagnostics.outcomes),
    }
    (output / "training.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def evaluate(
    checkpoint: Path, starts: list[State], *, clearance: float = 0.0
) -> list[EpisodeResult]:
    return evaluate_model(PPO.load(checkpoint, device="cpu"), starts, clearance=clearance)


def evaluate_model(
    model: PPO, starts: list[State], *, clearance: float = 0.0
) -> list[EpisodeResult]:
    def predict(_state, _config, observation):
        action, _ = model.predict(observation, deterministic=True)
        return float(action.item())

    return evaluate_ppo(predict, starts, clearance=clearance)


def save_episodes(path: Path, episodes: list[EpisodeResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("x", "y", "outcome", "capture_time"))
        writer.writerows((*e.start, e.outcome, e.capture_time) for e in episodes)


def read_episodes(path: Path, starts: list[State]) -> list[EpisodeResult]:
    with path.open(encoding="utf-8", newline="") as file:
        episodes = [
            EpisodeResult(
                (float(row["x"]), float(row["y"])),
                row["outcome"],
                float(row["capture_time"]) if row["capture_time"] else None,
            )
            for row in csv.DictReader(file)
        ]
    if [e.start for e in episodes] != starts:
        raise ValueError(f"Набор стартов не совпадает в {path}")
    return episodes


def report(architecture: str, reward: str, count: int, groups: dict, control: str | None) -> str:
    lines = [
        "# Задача 10 · Сравнение PPO",
        "",
        (
            f"Обучение: архитектура `{architecture}`, награда `{reward}`. "
            f"Оценка: {count} фиксированных стартов (seed {TUNING_SEED}), "
            f"захват к T = {DEFAULT_CONFIG.horizon:g}."
        ),
        "",
        (
            "Разности считаются как кандидат минус контроль. Время сравнивается на "
            "стартах с захватом у обеих политик; отрицательное значение означает ускорение."
        ),
    ]
    if control is not None:
        lines += ["", f"Результаты контроля: [{control}](../{control}/report.md)."]
    for (candidate, reference), results in groups.items():
        lines += [
            "",
            f"## {candidate} / {reference}",
            "",
            (
                "| Seed | Захваты кандидата | Захваты контроля | Δ захваты | Δ п. п. | "
                "Общие захваты | Δ время |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        captures, deltas = [], []
        for seed, result in results:
            a = result["candidate"]["counts"]["capture"]
            b = result["reference"]["counts"]["capture"]
            delta = a - b
            captures.append(a)
            deltas.append(delta)
            time = result["mean_capture_time_delta"]
            time_text = "—" if time is None else f"{time:+.3f}"
            lines.append(
                f"| {seed} | {a} | {b} | {delta:+d} | {100 * delta / count:+.2f} | "
                f"{result['both_captured']} | {time_text} |"
            )
        lines += [
            "",
            (
                f"Захваты кандидата: в среднем {mean(captures):.1f} из {count}, "
                f"диапазон {min(captures)}…{max(captures)}. "
                f"Прирост: в среднем {mean(deltas):+.1f}, "
                f"диапазон {min(deltas):+d}…{max(deltas):+d}. "
                f"Кандидат поймал чаще на {sum(d > 0 for d in deltas)}/{len(deltas)} seed."
            ),
        ]
    return "\n".join(lines) + "\n"


def run(
    output: Path,
    *,
    architecture: str,
    reward: str = "base",
    reference: Path | None = None,
    budgets: tuple[int, ...] = (STEPS,),
    seeds: tuple[int, ...] = SEEDS,
    frozen_controls: dict[int, Path] | None = None,
    clearance: float = 0.0,
) -> None:
    if output.exists():
        raise SystemExit(f"Каталог уже существует: {output}. Выберите новую серию через --output.")
    starts = tuning_starts()
    controls = {}
    if reference is not None:
        if not (reference / "report.md").is_file():
            raise SystemExit(f"Сначала завершите контрольный эксперимент: {reference}")
        for seed in seeds:
            controls[seed] = read_episodes(reference / f"seed-{seed}/episodes-{STEPS}.csv", starts)
        heuristic = read_episodes(reference / "heuristic.csv", starts)
    else:
        heuristic = evaluate_ppo(
            lambda state, config, _: heuristic_pursuer(state, config), starts, clearance=clearance
        )
    classical = evaluate_pursuer(classical_pursuer, starts, clearance=clearance)

    output.mkdir(parents=True)
    save_episodes(output / "heuristic.csv", heuristic)
    save_episodes(output / "classical.csv", classical)
    torch.set_num_threads(1)
    frozen_controls = frozen_controls or {}
    if frozen_controls:
        (output / "control-checkpoints.json").write_text(
            json.dumps({seed: str(path) for seed, path in frozen_controls.items()}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    groups = defaultdict(list)
    for seed in seeds:
        directory = output / f"seed-{seed}"
        print(f"Seed {seed}: train for {budgets[-1]} decisions", flush=True)
        train_budgets = budgets[1:] if seed in frozen_controls else budgets
        train(directory, seed, architecture, reward, train_budgets, clearance=clearance)
        results = {}
        for steps in budgets:
            print(f"Seed {seed}: evaluate checkpoint at {steps} decisions", flush=True)
            checkpoint = directory / f"ppo-{steps}.zip"
            if steps == budgets[0] and seed in frozen_controls:
                checkpoint = frozen_controls[seed]
            episodes = evaluate(checkpoint, starts, clearance=clearance)
            save_episodes(directory / f"episodes-{steps}.csv", episodes)
            results[steps] = episodes
            label = f"PPO ({steps} решений)"
            groups[label, "Эвристика"].append((seed, compare(episodes, heuristic)))
            groups[label, "Классическая стратегия"].append((seed, compare(episodes, classical)))
        if reference is not None:
            groups[output.name, reference.name].append(
                (seed, compare(results[STEPS], controls[seed]))
            )
        if len(budgets) > 1:
            groups[f"PPO ({budgets[-1]} решений)", f"PPO ({budgets[0]} решений)"].append(
                (seed, compare(results[budgets[-1]], results[budgets[0]]))
            )
    path = output / "report.md"
    path.write_text(
        report(architecture, reward, len(starts), groups, reference.name if reference else None),
        encoding="utf-8",
    )
    print(f"Saved {path}", flush=True)


def parser(description: str) -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/chauffeur-task10",
        help="корневой каталог серии экспериментов",
    )
    return cli
