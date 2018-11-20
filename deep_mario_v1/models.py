from __future__ import print_function, division
import torch

import torch.nn.functional as F
import warnings
from torchvision import models
warnings.filterwarnings("ignore")


def inNet(num_classes, pretrain):
    model = models.inception_v3(pretrained=pretrain)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.aux_logits= False
    return model

def denseNet(num_classes, layers, pretrain):
    if layers == 161:
        model = models.densenet161(pretrained=pretrain)
    if layers == 169:
        model = models.densenet169(pretrained=pretrain)
    if layers == 201:
        model = models.densenet201(pretrained=pretrain)
    else:
        model = models.densenet121(pretrained=pretrain)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    return model

def resnet(num_classes, layers, pretrain):
    if layers == 34:
        model = models.resnet34(pretrained=pretrain)
    elif layers == 50:
        model = models.resnet50(pretrained=pretrain)
    elif layers == 101:
        model = models.resnet101(pretrained=pretrain)
    elif layers == 152:
        model = models.resnet152(pretrained=pretrain)
    else:
        model = models.resnet18(pretrained=pretrain)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


class simple_net(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(simple_net, self).__init__()
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

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        if not self.init:
            self.full1 = torch.nn.Linear(x.shape[1], 512)
            self.init = True

        x = self.full1(self.relu(self.bn3(x)))
        x = self.full2(self.relu(x))
        return x

