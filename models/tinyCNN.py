import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNNRegressor(nn.Module):
    def __init__(self):
        super(TinyCNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # reduce spatial size

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # output x and y

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # -> [32, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # -> [64, 8, 8]

        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # [batch, 2]
        return out
