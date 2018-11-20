from __future__ import print_function, division
import torch

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class simple_net(torch.nn.Module):
    def __init__(self, input_channels, output_channels, device):
        super(simple_net, self).__init__()
        self.device = device

        self.conv1 = torch.nn.Conv2d(input_channels, 32, 8, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.full1 = 0
        self.full2 = torch.nn.Linear(512, output_channels)
        self.relu = torch.nn.ReLU()

        self.init = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(self.bn1(x)))
        x = self.conv3(self.relu(self.bn2(x)))
        x = self.bn3(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        if not self.init:
            self.full1 = torch.nn.Linear(x.shape[1], 512).to(self.device)
            self.init = True

        x = self.full1(self.relu(x))
        x = self.full2(self.relu(x))
        return x

