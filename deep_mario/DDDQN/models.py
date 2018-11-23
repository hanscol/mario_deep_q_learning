from __future__ import print_function, division
import torch

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class simple_net(torch.nn.Module):
    def __init__(self, input_channels, output_channels, device):
        super(simple_net, self).__init__()
        self.device = device

        self.conv1 = torch.nn.Conv2d(input_channels, 32, 8, stride=4, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.adv_fc = torch.nn.Linear(12800, 512)
        self.adv = torch.nn.Linear(512, output_channels)
        self.val_fc = torch.nn.Linear(12800, 512)
        self.val = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(self.bn1(x)))
        x = self.conv3(self.relu(self.bn2(x)))
        x = self.bn3(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        a = self.adv_fc(self.relu(x))
        a = self.adv(self.relu(a))
        v = self.val_fc(self.relu(x))
        v = self.val(self.relu(v))

        x = torch.add(v, torch.sub(a.permute([1,0]), torch.mean(a, dim=1)).permute([1,0]))

        return x
