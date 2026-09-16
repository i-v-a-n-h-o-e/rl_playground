import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


visual_env = gym.make(
    "CartPole-v1",
    disable_env_checker=True,
    render_mode="human",
)

training_env = gym.make(
    "CartPole-v1",
    disable_env_checker=True,
    render_mode=None,
)

state_dim = training_env.observation_space.shape[0]
action_dim = training_env.action_space.n

class CEMAgent(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim))
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state) -> int:
        state_tensor = torch.FloatTensor(state)
        action_probs = self.softmax(self.forward(state_tensor))
        action = np.random.choice(self.action_dim, p=action_probs.detach().numpy())
        return action

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        pred_action = self.forward(elite_states)
        loss = self.loss(pred_action, elite_actions)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        

def get_trajectory(env, agent, max_steps=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state, _ = env.reset()

    for _ in range(max_steps):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _x, _q = env.step(action)
        trajectory['rewards'].append(reward)

        if visualize:
            print(visualize)
            time.sleep(0.01)
            env.render()
        
        if done:
            break

    return trajectory


agent = CEMAgent(state_dim, action_dim)
q_param = 0.9
iteration_n = 100
trajectory_len = 500
trajectory_n = 20

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


trajectory =get_trajectory(visual_env, agent, max_steps=1000, visualize=True)
print("total reward:", sum(trajectory['rewards']))