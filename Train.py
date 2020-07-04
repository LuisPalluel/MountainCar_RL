import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Utils import select_action, optimize_model, plot_durations
from ReplayMemory import ReplayMemory
from DQNet import DQNet
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('MountainCar-v0')
init_screen = env.reset()

args = {
    "BATCH_SIZE" : 50,
    "GAMMA" : 0.999,
    "EPS_START" : 0.9,
    "EPS_END" : 0.05,
    "EPS_DECAY" : 200,
    "TARGET_UPDATE" : 10
}
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQNet(n_obs, n_actions).to(device)
target_net = DQNet(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 3000
for i_episode in range(num_episodes):
    print("Episode", i_episode)
    state = torch.Tensor(env.reset()).to(device)
    next_state = torch.Tensor(env.reset()).to(device)

    for t in range(3000):
        if i_episode%1 == 0:
            env.render(mode='rgb_array')

        state = next_state.to(device)

        action = select_action(state, steps_done, policy_net, n_actions, args, device)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        reward += next_state[0]
        next_state = torch.Tensor(next_state).to(device)

        """
        if not done:
            next_state = next_state - state
        else:
            next_state = None
        """

        memory.push(state, action, next_state, reward)

        optimize_model(policy_net, target_net, memory, optimizer, args, device)
        if done:
            episode_durations.append(t + 1)
            #plot_durations(episode_durations)
            #break

    if i_episode % args["TARGET_UPDATE"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()