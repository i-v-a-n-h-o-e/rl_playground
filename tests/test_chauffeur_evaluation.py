from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from RL.chauffeur_classical import Policy, Rollout, State, TrajectoryPoint, classical_pursuer
from RL.chauffeur_evaluation import (
    EpisodeResult,
    compare,
    evaluate_pursuer,
    final_grid,
    heuristic_pursuer,
    save_episodes,
    tuning_starts,
)
from RL.homicidal_chauffeur import GameConfig


class ChauffeurEvaluationTests(unittest.TestCase):
    def test_episode_csv_preserves_schema_and_missing_capture_times(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "episodes.csv"
            save_episodes(
                target,
                {
                    "ppo": [EpisodeResult((0.0, 1.0), "capture", 1.4)],
                    "classical": [EpisodeResult((0.0, -1.0), "unsupported", None)],
                },
            )
            with target.open(encoding="utf-8", newline="") as file:
                self.assertEqual(
                    list(csv.reader(file)),
                    [
                        ["pursuer", "x", "y", "outcome", "capture_time"],
                        ["ppo", "0.0", "1.0", "capture", "1.4"],
                        ["classical", "0.0", "-1.0", "unsupported", ""],
                    ],
                )

    def test_heuristic_turns_toward_the_bearing_with_saturation(self) -> None:
        config = GameConfig(dt=0.05)
        for state, expected in (
            ((0, 2), 0),
            ((2, 1), 1),
            ((-2, 1), -1),
            ((-0.0, -1), 1),
            ((0.01, 1), 0.1999933337333),
        ):
            with self.subTest(state=state):
                self.assertAlmostEqual(heuristic_pursuer(state, config), expected)

    def test_fixed_disjoint_start_sets(self) -> None:
        grid = final_grid()
        tuning = tuning_starts()
        self.assertEqual(len(set(grid)), 6532)
        self.assertEqual(len(set(tuning)), 512)
        self.assertEqual(tuning, tuning_starts())
        self.assertTrue(set(grid).isdisjoint(tuning))
        self.assertIn((4.0, -4.0), grid)
        self.assertNotIn((0.0, 0.3), grid)

    def test_same_driver_classifies_capture_timeout_and_unsupported(self) -> None:
        starts = [(0.0, 0.4), (0.0, 2.3), (2.0, -2.0)]
        config = GameConfig(horizon=0.4)
        candidate = evaluate_pursuer(heuristic_pursuer, iter(starts), config)
        reference = evaluate_pursuer(classical_pursuer, starts, config)
        self.assertEqual(candidate, reference)
        self.assertEqual([result.start for result in candidate], starts)
        self.assertEqual(
            [result.outcome for result in candidate], ["capture", "time_limit", "unsupported"]
        )
        self.assertAlmostEqual(candidate[0].capture_time, 0.2)
        self.assertIsNone(candidate[1].capture_time)
        self.assertIsNone(candidate[2].capture_time)

    def test_heuristic_nearly_zero_steering_during_capture(self) -> None:
        # On this start, steering decays toward zero until the event polynomial
        # has tiny leading terms. The capture solver must still find the root.
        (result,) = evaluate_pursuer(heuristic_pursuer, [(0.1, 2.3)])
        self.assertEqual(result.outcome, "capture")
        self.assertAlmostEqual(result.capture_time, 4.00436, places=5)

    def test_custom_rollout_preserves_start_and_capture_time(self) -> None:
        starts = [(2.0, -2.0), (0.0, 0.4)]
        config = GameConfig(horizon=0.4)

        def rollout(pursuer: Policy, start: State, settings: GameConfig) -> Rollout:
            self.assertIs(pursuer, heuristic_pursuer)
            self.assertIs(settings, config)
            initial = TrajectoryPoint(0.0, start)
            if start == starts[0]:
                return Rollout("time_limit", (initial, TrajectoryPoint(settings.horizon, (2, -3))))
            return Rollout("capture", (initial, TrajectoryPoint(0.2, (0, 0.3))))

        self.assertEqual(
            evaluate_pursuer(heuristic_pursuer, iter(starts), config, rollout=rollout),
            [
                EpisodeResult(starts[0], "time_limit", None),
                EpisodeResult(starts[1], "capture", 0.2),
            ],
        )

    def test_paired_metrics_distinguish_unknown_results_and_use_common_captures(self) -> None:
        candidate = [
            EpisodeResult((0, 1), "capture", 1.0),
            EpisodeResult((0, 2), "capture", 9.0),
            EpisodeResult((0, 3), "unsupported", None),
            EpisodeResult((0, 4), "time_limit", None),
        ]
        reference = [
            EpisodeResult((0, 1), "capture", 3.0),
            EpisodeResult((0, 2), "time_limit", None),
            EpisodeResult((0, 3), "capture", 4.0),
            EpisodeResult((0, 4), "time_limit", None),
        ]
        summary = compare(candidate, reference)
        self.assertEqual(summary["both_resolved"], 3)
        self.assertEqual(summary["outcome_disagreement_rate"], 1 / 3)
        self.assertEqual(summary["map_status_disagreement_rate"], 0.5)
        self.assertEqual(summary["capture_rate_delta_bounds"], [0.0, 0.25])
        self.assertEqual(summary["candidate"]["capture_rate_on_both_resolved"], 2 / 3)
        self.assertEqual(summary["reference"]["capture_rate_on_both_resolved"], 1 / 3)
        self.assertEqual(summary["both_captured"], 1)
        self.assertEqual(summary["mean_capture_time_delta"], -2.0)
        self.assertIsNone(compare(candidate[2:3], reference[2:3])["mean_capture_time_delta"])
        self.assertIsNone(compare(candidate[2:3], reference[2:3])["outcome_disagreement_rate"])
        with self.assertRaisesRegex(ValueError, "same nonempty ordered starts"):
            compare(candidate, reference[::-1])
