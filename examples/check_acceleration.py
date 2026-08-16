#!/usr/bin/env python3
"""Smoke-test the RL/visualization stack and report PyTorch acceleration."""

from __future__ import annotations

import platform

import cv2
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
    plt.close(matplotlib_figure)

    assert product.shape == (3, 3)
    assert gray.shape == (8, 8)
    assert len(figure.data) == 1
    assert observation.shape == (4,)

    print(f"platform={platform.system()} {platform.machine()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"accelerator={device.type}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print("smoke=ok")


if __name__ == "__main__":
    main()
