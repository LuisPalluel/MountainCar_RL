import torch.nn as nn
import torch

DQN = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=5, stride=2),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=5, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=5, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
)

class DQNet(nn.Module):

    def __init__(self, state_space, action_space):
        super(DQNet, self).__init__()

        hidden_units = 200
        self.input_layer = nn.Linear(state_space, hidden_units, bias=False)
        self.output_layer = nn.Linear(hidden_units, action_space, bias=False)

    def forward(self, x):
        x = self.input_layer(x)
        return self.output_layer(x)
