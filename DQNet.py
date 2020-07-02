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

    def __init__(self, h, w, outputs, shape):
        super(DQNet, self).__init__()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.conv = DQN.to('cpu')
        self.flatten = nn.Flatten()

        ones = torch.ones(shape)
        x = self.conv(torch.ones(shape))
        x = self.flatten(x)
        self.head = nn.Linear(x.shape[1], outputs)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.head(x)
