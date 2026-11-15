from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from stable_baselines3 import PPO

from RL.chauffeur_classical import Rollout, TrajectoryPoint
from RL.chauffeur_ppo import ChauffeurPPOEnv
from RL.chauffeur_ppo_experiment import (
    BENCHMARKS,
    checkpoint_policy,
    rollout_in_env,
    save_trajectories,
)
from RL.homicidal_chauffeur import GameConfig


class ChauffeurPPOExperimentTests(unittest.TestCase):
    def test_checkpoint_receives_observations_returned_by_environment(self) -> None:
        env = Mock(spec=ChauffeurPPOEnv, config=GameConfig(horizon=0.1))
        start = (0.0, 2.3)

        def info(time: float, state: tuple[float, float]) -> dict:
            return {"elapsed_time": time, "state": np.array(state), "phi": 0.0, "psi": 0.0}

        initial, following, terminal = np.eye(6, dtype=np.float32)[:3]
        env.reset.return_value = initial, info(0.0, start)
        env.step.side_effect = [
            (following, 0.0, False, False, info(0.05, (0.0, 2.275))),
            (terminal, 0.0, False, True, info(0.1, (0.0, 2.25))),
        ]
        model = Mock(spec=PPO)
        model.predict.return_value = np.array([0.0], dtype=np.float32), None

        run = rollout_in_env(checkpoint_policy(model), start, env=env)

        self.assertEqual(run.outcome, "time_limit")
        env._observation.assert_not_called()
        for call, expected in zip(model.predict.call_args_list, (initial, following), strict=True):
            self.assertIs(call.args[0], expected)
            self.assertEqual(call.kwargs, {"deterministic": True})

    def test_trajectory_export_uses_start_to_match_reordered_runs(self) -> None:
        matplotlib.use("Agg")
        runs = [Rollout("unsupported", (TrajectoryPoint(0.0, start),)) for start in BENCHMARKS]
        with (
            TemporaryDirectory() as directory,
            patch.object(Figure, "savefig", autospec=True) as save,
        ):
            output = Path(directory)
            save_trajectories(output, {"ppo": runs, "classical": runs[::-1]})
            with (output / "trajectories.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2 * len(BENCHMARKS))
            for row in rows:
                self.assertEqual((row["x0"], row["y0"]), (row["x"], row["y"]))
            figure = save.call_args.args[0]
            for axis, (x, y) in zip(figure.axes, BENCHMARKS, strict=True):
                self.assertEqual(axis.get_title(), f"Start ({x:g}, {y:g})")
                self.assertEqual([tuple(line.get_xydata()[0]) for line in axis.lines], [(x, y)] * 2)
