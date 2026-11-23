"""Replay the seed-43 grid's unsupported episode and save its benchmark figure."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from stable_baselines3 import PPO

from chauffeur_evader import chauffeur_classical as classical
from chauffeur_evader.chauffeur_benchmarks import _save_figure
from chauffeur_evader.chauffeur_evaluation import DEFAULT_CONFIG, _ppo_rollouts, heuristic_pursuer


def test_gray_cell_replay(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "data/chauffeur-task10/length-wide-potential/seed-43/ppo-262144.zip"
    if not checkpoint.exists():
        pytest.skip("Local Task 10 checkpoint is required for the replay")
    audit = json.loads((root / "experiments/chauffeur-task10-v1/protocol.json").read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == audit["control"]["checkpoints"]["43"]["sha256"]
    model = PPO.load(checkpoint, device="cpu")
    assert (model.seed, model.num_timesteps) == (43, 262144)
    actions = []

    def predict(state, config, observation):
        action, _ = model.predict(observation, deterministic=True)
        actions.append((state, float(action.item())))
        return actions[-1][1]

    start = (-2.3, -0.2)
    ppo, = _ppo_rollouts(predict, [start], DEFAULT_CONFIG, clearance=0)
    runs = {
        "classical": classical.rollout_classical(start, DEFAULT_CONFIG),
        "heuristic": classical.rollout_classical(start, DEFAULT_CONFIG, pursuer=heuristic_pursuer),
        "ppo": ppo,
    }
    for name, time in [("classical", 14.0085907), ("heuristic", 2.7375897)]:
        assert runs[name].outcome == "capture"
        assert runs[name].trajectory[-1].time == pytest.approx(time, abs=1e-6)
    last = ppo.trajectory[-1]
    assert ppo.outcome == "unsupported"
    assert last.time == pytest.approx(2.6099716, abs=1e-6)
    assert last.event == "SE"
    assert last.state == pytest.approx((0.19326453, 0.31884189), abs=1e-7)
    assert actions[-1] == (last.state, -1.0)
    assert math.hypot(*last.state) > DEFAULT_CONFIG.beta
    assert classical.classical_feedback(last.state, DEFAULT_CONFIG) is None

    _save_figure(tmp_path / "gray-cell.png", "PPO unsupported", start, runs, DEFAULT_CONFIG)
