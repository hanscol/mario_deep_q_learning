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
