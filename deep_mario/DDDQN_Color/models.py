from __future__ import print_function, division
import torch

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class simple_net(torch.nn.Module):
    def __init__(self, input_channels, output_channels, device):
        super(simple_net, self).__init__()
        self.device = device

        self.conv1 = torch.nn.Conv2d(input_channels, 64, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.drop = torch.nn.Dropout2d(p=0.3)

        self.adv_fc = torch.nn.Linear(131072, 1024)
        self.adv = torch.nn.Linear(1024, output_channels)

        self.val_fc = torch.nn.Linear(131072, 1024)
        self.val = torch.nn.Linear(1024, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x3 = x[:, 6:9, :, :]
        x4 = x[:, 9:, :, :]

        x1 = self.conv1(x1)
        x1 = self.conv2(self.relu(self.bn1(x1)))
        x1 = self.conv3(self.relu(self.bn2(x1)))
        #x1 = self.conv4(self.relu(self.bn3(x1)))
        #x1 = self.conv5(self.relu(self.bn4(x1)))
        x1 = self.bn3(x1)
        x1 = self.drop(x1)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3])

        x2 = self.conv1(x2)
        x2 = self.conv2(self.relu(self.bn1(x2)))
        x2 = self.conv3(self.relu(self.bn2(x2)))
        #x2 = self.conv4(self.relu(self.bn3(x2)))
        #x2 = self.conv5(self.relu(self.bn4(x2)))
        x2 = self.bn3(x2)
        x2 = self.drop(x2)
        x2 = x2.view(-1, x2.shape[1] * x2.shape[2] * x2.shape[3])

        x3 = self.conv1(x3)
        x3 = self.conv2(self.relu(self.bn1(x3)))
        x3 = self.conv3(self.relu(self.bn2(x3)))
        #x3 = self.conv4(self.relu(self.bn3(x3)))
        #x3 = self.conv5(self.relu(self.bn4(x3)))
        x3 = self.bn3(x3)
        x3 = self.drop(x3)
        x3 = x3.view(-1, x3.shape[1] * x3.shape[2] * x3.shape[3])

        x4 = self.conv1(x4)
        x4 = self.conv2(self.relu(self.bn1(x4)))
        x4 = self.conv3(self.relu(self.bn2(x4)))
        #x4 = self.conv4(self.relu(self.bn3(x4)))
        #x4 = self.conv5(self.relu(self.bn4(x4)))
        x4 = self.bn3(x4)
        x4 = self.drop(x4)
        x4 = x4.view(-1, x4.shape[1] * x4.shape[2] * x4.shape[3])

        x = torch.cat((x1, x2, x3, x4), dim=1)

        a = self.adv_fc(self.relu(x))
        a = self.adv(self.relu(a))

        v = self.val_fc(self.relu(x))
        v = self.val(self.relu(v))

        x = torch.add(v, torch.sub(a.permute([1,0]), torch.mean(a, dim=1)).permute([1,0]))

        return x
