#!/usr/bin/env python3
"""Smoke-test the RL/visualization stack and report PyTorch acceleration."""

from __future__ import annotations

import platform

import cv2
import gym as legacy_gym
import gym_maze  # noqa: F401  # Registers the original gym-maze environments.
import gymnasium as gym
import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch
import torchvision


def best_torch_device() -> torch.device:
    """Prefer CUDA, then Apple Metal (MPS), and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    device = best_torch_device()

    # Exercise the packages without opening windows or writing generated files.
    tensor = torch.arange(9, dtype=torch.float32, device=device).reshape(3, 3)
    product = tensor @ tensor.T
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    figure = go.Figure(data=go.Scatter(y=[0, 1, 0]))
    matplotlib_figure = plt.figure()
    plt.plot([0, 1, 0])
    environment = gym.make("CartPole-v1")
    observation, _ = environment.reset(seed=42)
    environment.step(environment.action_space.sample())
    environment.close()
    maze_environment = legacy_gym.make(
        "maze-sample-5x5-v0",
        apply_api_compatibility=True,
        disable_env_checker=True,
        enable_render=False,
    )
    maze_observation, _ = maze_environment.reset(seed=42)
    maze_transition = maze_environment.step(maze_environment.action_space.sample())
    maze_environment.close()
    maze_environment.unwrapped.env.maze_view.quit_game()
    plt.close(matplotlib_figure)

    assert product.shape == (3, 3)
    assert gray.shape == (8, 8)
    assert len(figure.data) == 1
    assert observation.shape == (4,)
    assert maze_observation.shape == (2,)
    assert len(maze_transition) == 5

    print(f"platform={platform.system()} {platform.machine()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"accelerator={device.type}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print("gym-maze=ok")
    print("smoke=ok")


if __name__ == "__main__":
    main()
