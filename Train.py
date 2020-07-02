import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Utils import get_screen, select_action, optimize_model, plot_durations
from ReplayMemory import ReplayMemory
from DQNet import DQNet
from itertools import count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('MountainCar-v0')
env.reset()

args = {
    "BATCH_SIZE" : 1,
    "GAMMA" : 0.999,
    "EPS_START" : 0.9,
    "EPS_END" : 0.05,
    "EPS_DECAY" : 200,
    "TARGET_UPDATE" : 10
}

init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQNet(screen_height, screen_width, n_actions, init_screen.shape).to(device)
target_net = DQNet(screen_height, screen_width, n_actions, init_screen.shape).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 50
for i_episode in range(num_episodes):

    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    for t in count():

        action = select_action(state, steps_done, policy_net, n_actions, args, device)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = get_screen(env, device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model(policy_net, target_net, memory, optimizer, args, device)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

    if i_episode % args["TARGET_UPDATE"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()