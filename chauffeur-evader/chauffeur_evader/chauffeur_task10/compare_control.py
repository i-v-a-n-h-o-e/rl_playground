"""Сравнить сохранённый PPO с контролем задачи 10 и обеими стратегиями."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from chauffeur_evader.chauffeur_classical import classical_pursuer
from chauffeur_evader.chauffeur_evaluation import compare, evaluate_pursuer, heuristic_pursuer
from chauffeur_evader.chauffeur_task10.common import evaluate_model, save_episodes

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "experiments/chauffeur-task10-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="чекпойнт нового PPO")
    parser.add_argument("--output", type=Path, required=True, help="новый каталог сравнения")
    parser.add_argument(
        "--holdout", action="store_true", help="использовать независимые 2048 стартов"
    )
    parser.add_argument(
        "--control", type=Path, help="контрольный чекпойнт для нового seed 59 или 71"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"каталог уже существует: {args.output}")

    protocol = json.loads((FREEZE / "protocol.json").read_text(encoding="utf-8"))
    frozen = protocol["control"]
    torch.set_num_threads(frozen["torch_threads"])
    candidate = PPO.load(args.candidate, device="cpu")
    seed = candidate.seed
    entry = frozen["checkpoints"].get(str(seed))
    if entry is not None:
        if args.control is not None:
            parser.error(f"для seed {seed} используется зафиксированный чекпойнт")
        control_path = ROOT / entry["path"]
        if sha256(control_path) != entry["sha256"]:
            parser.error(f"SHA-256 контрольного чекпойнта не совпадает: {control_path}")
    elif seed in protocol["training_seeds"]["confirmation"] and args.control is not None:
        control_path = args.control
        settings = json.loads((control_path.parent / "settings.json").read_text(encoding="utf-8"))
        for key in ("architecture", "reward", "ppo", "environment", "torch_threads"):
            if settings.get(key) != frozen[key]:
                parser.error(f"параметр контроля {key} отличается от зафиксированного")
    else:
        parser.error(f"для seed {seed} нужен парный контроль; новые seed — 59 и 71")

    control = PPO.load(control_path, device="cpu")
    if control.seed != seed or control.num_timesteps != frozen["decisions"]:
        parser.error("у контроля должен быть тот же seed и бюджет 262144 решения")
    ppo = frozen["ppo"]
    if control.policy_kwargs["net_arch"] != ppo["net_arch"] or any(
        getattr(control, key) != ppo[key]
        for key in (
            "n_steps",
            "gamma",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gae_lambda",
            "ent_coef",
        )
    ):
        parser.error("параметры контрольной модели не совпадают с протоколом")

    dataset = "holdout" if args.holdout else "tuning"
    dataset_spec = protocol["evaluation"][dataset]
    path = FREEZE / dataset_spec["file"]
    if sha256(path) != dataset_spec["sha256"]:
        parser.error(f"SHA-256 стартов не совпадает: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        starts = [(float(row["x"]), float(row["y"])) for row in csv.DictReader(file)]

    args.output.mkdir(parents=True)
    references = {
        "control": evaluate_model(control, starts),
        "heuristic": evaluate_pursuer(heuristic_pursuer, starts, clearance=0),
        "classical": evaluate_pursuer(classical_pursuer, starts, clearance=0),
    }
    episodes = evaluate_model(candidate, starts)
    for name, results in {"candidate": episodes, **references}.items():
        save_episodes(args.output / f"{name}.csv", results)
    summary = {
        "seed": seed,
        "dataset": dataset,
        "starts_sha256": dataset_spec["sha256"],
        "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        "control": {"path": str(control_path), "sha256": sha256(control_path)},
        "comparisons": {name: compare(episodes, results) for name, results in references.items()},
    }
    report = args.output / "summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Сохранено: {report}")


if __name__ == "__main__":
    main()
