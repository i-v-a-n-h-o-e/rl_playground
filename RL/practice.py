"""Run the original MattChanTK/gym-maze 5x5 sample environment."""

from __future__ import annotations

import time

import gym
import gym_maze  # noqa: F401  # Registers the legacy maze environments in Gym.
import numpy as np
import random


visual_env = gym.make(
    "maze-sample-5x5-v0",
    apply_api_compatibility=True,
    disable_env_checker=True,
    render_mode="human",
)

training_env = gym.make(
    "maze-sample-5x5-v0",
    apply_api_compatibility=True,
    disable_env_checker=True,
    render_mode=None,
    enable_render=False,
)

state_n = 25
action_n = 4

class RandomAgent:
    def __init__(self, action_space_n) -> None:
        self.action_space_n = action_space_n

    def get_action(self, state) -> int:
        return np.random.randint(self.action_space_n)

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

def get_state(observation):
    return int(np.sqrt(state_n) * observation[0] + observation[1])

def get_trajectory(env, agent, max_steps=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    observation, _ = env.reset(seed=42)
    state = get_state(observation)

    for _ in range(max_steps):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(obs)

        if visualize:
            print(visualize)
            time.sleep(0.5)
            env.render()
        
        if done:
            break

    return trajectory

agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.9
trajectory_n = 100
iteration_n = 10

for iteration in range(iteration_n):

    #policy evaluation
    trajectories = [get_trajectory(training_env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print("iteration:", iteration, "mean toral reward:", np.mean(total_rewards), "max reward:", np.max(total_rewards))

    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards']) 
        if total_reward >= quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)


trajectory =get_trajectory(visual_env, agent, max_steps=100, visualize=True)
print("total reward:", sum(trajectory['rewards']))
print("model:", agent.model)


