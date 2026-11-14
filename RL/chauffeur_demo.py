"""Run a reproducible episode with prescribed controls; no training or GUI."""

from __future__ import annotations

import argparse
import json

import numpy as np

from RL.homicidal_chauffeur import GameConfig, HomicidalChauffeurEnv


def run_episode(
    config: GameConfig, state: tuple[float, float], seed: int, phi: float, psi: float
) -> list[dict]:
    env = HomicidalChauffeurEnv(config)
    _, info = env.reset(seed=seed, options={"state": state})
    trajectory = [{"time": 0.0, "state": info["state"].tolist()}]
    while True:
        _, _, terminated, truncated, info = env.step_controls(phi, psi)
        trajectory.append(
            {
                "time": info["elapsed_time"],
                "state": info["state"].tolist(),
                "phi": info["phi"],
                "psi": info["psi"],
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info["termination_reason"],
            }
        )
        if terminated or truncated:
            break
    env.close()
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", nargs=2, type=float, default=(0.0, 2.3), metavar=("X", "Y"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=15.0)
    parser.add_argument("--trace", action="store_true", help="Print all states as JSON lines")
    args = parser.parse_args()
    config = GameConfig(gamma=args.gamma, beta=args.beta, dt=args.dt, horizon=args.horizon)
    trajectory = run_episode(config, tuple(args.state), args.seed, args.phi, args.psi)
    repeated = run_episode(config, tuple(args.state), args.seed, args.phi, args.psi)
    if trajectory != repeated:
        raise RuntimeError("Episode replay differs")
    if args.trace:
        for row in trajectory:
            print(json.dumps(row))
    last = trajectory[-1]
    print(
        f"steps={len(trajectory) - 1} time={last['time']:.6f} "
        f"reason={last['termination_reason']} reproducible=True"
    )
    print(f"state={np.array(last['state'])} phi={last['phi']} psi={last['psi']}")


if __name__ == "__main__":
    main()
