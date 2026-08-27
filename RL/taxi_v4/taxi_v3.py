from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go


class Trajectory(TypedDict):
    states: list[int]
    actions: list[int]
    rewards: list[float]


def plot_training_rewards(
    mean_rewards: list[float],
    max_rewards: list[float],
) -> None:
    """Build and save reward curves for all training iterations."""
    iterations = list(range(1, len(mean_rewards) + 1))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=mean_rewards,
            mode="lines+markers",
            name="Средняя награда",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=max_rewards,
            mode="lines+markers",
            name="Максимальная награда",
        )
    )
    figure.update_layout(
        title="Награда в процессе обучения",
        xaxis_title="Итерация",
        yaxis_title="Награда",
        template="plotly_white",
        hovermode="x unified",
    )

    output_dir = Path(__file__).resolve().parent
    figure.write_html(output_dir / "taxi_training_rewards.html")
    figure.write_image(
        output_dir / "taxi_training_rewards.jpeg",
        format="jpeg",
        scale=2,
    )


def make_env(render_mode: str | None) -> gym.Env:
    """Create deterministic Taxi-v4, matching classic Taxi-v3 dynamics."""
    return gym.make(
        "Taxi-v4",
        render_mode=render_mode,
        # Taxi-v4 differs from classic Taxi-v3 only when these optional
        # stochastic modes are enabled.
        is_rainy=False,
        fickle_passenger=False,
    )


class RandomAgent:
    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        self.action_space = action_space

    def get_action(self, state: int) -> int:
        del state
        return int(self.action_space.sample())


class CrossEntropyAgent:
    def __init__(self, state_n, action_n) -> None:
        self.action_n = action_n
        self.state_n = state_n
        self.model = np.ones((self.state_n, self.action_n)) / action_n

    def get_action(self, state) -> int:
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def unzip_state(i: int) -> dict[str, int]:
    """Unzip Taxi-v4 state into (taxi_row, taxi_col, passenger_loc, destination)."""
    taxi_row = (i // 4) // 5 // 5
    taxi_col = (i // 4) // 5 % 5
    passenger_loc = (i // 4) % 5
    destination = i % 4
    return {"taxi_row": taxi_row, "taxi_col": taxi_col, "passenger_loc": passenger_loc, "destination": destination}


def get_trajectory(
    env: gym.Env,
    agent: RandomAgent,
    max_steps: int = 1000) -> Trajectory:
    trajectory: Trajectory = {"states": [], "actions": [], "rewards": []}

    observation, _ = env.reset(seed=42)
    state = int(observation)

    env.action_space.seed(42)

    for step in range(max_steps):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        observation, reward, terminated, truncated, _ = env.step(action)
        trajectory["rewards"].append(float(reward))

        state = int(observation)
        # print(f"step={step}\t{unzip_state(state)}\taction={action}\treward={reward}\tterminated={terminated}\ttruncated={truncated}")
        if terminated or truncated:
            break

    return trajectory

visual_env = make_env(render_mode="human")
print(
    f"environment=Taxi-v4 states={visual_env.observation_space.n} "
    f"actions={visual_env.action_space.n}"
)


train_env = make_env(render_mode=None)

agent = CrossEntropyAgent(train_env.observation_space.n, train_env.action_space.n)
q_param = 0.9
trajectory_n = 50
iteration_n = 10
mean_rewards_by_iteration: list[float] = []
max_rewards_by_iteration: list[float] = []

for iteration in range(iteration_n):

    #policy evaluation
    trajectories = [get_trajectory(train_env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    mean_reward = float(np.mean(total_rewards))
    max_reward = float(np.max(total_rewards))
    mean_rewards_by_iteration.append(mean_reward)
    max_rewards_by_iteration.append(max_reward)
    print("iteration:", iteration, "mean total reward:", mean_reward, "max reward:", max_reward)

    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards']) 
        if total_reward >= quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

plot_training_rewards(mean_rewards_by_iteration, max_rewards_by_iteration)

trajectory =get_trajectory(visual_env, agent, max_steps=200)
print("total reward:", sum(trajectory['rewards']))
print("model:", agent.model)
