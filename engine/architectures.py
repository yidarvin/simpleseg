from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets, models, transforms,utils
from torchvision.transforms import functional as func
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from .fpn import *
from .google import *

class densenet101(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, pretrained=False):
        super(densenet101, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, pretrained_backbone=pretrained)
        self.model.classifier = DeepLabHead(2048, out_chan)
        #self.classifier = nn.Conv2d(21, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        if in_chan != 3:
            self.model.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.model(x)['out']#self.classifier(self.model(x)['out'])

class densenet50(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, pretrained=False):
        super(densenet50, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, pretrained_backbone=pretrained)
        self.model.classifier = DeepLabHead(2048, out_chan)
        if in_chan != 3:
            self.model.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.model(x)['out']

class FCN50(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, pretrained=False):
        super(FCN50, self).__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=pretrained)
        self.model.classifier = FCNHead(2048, out_chan)
        if in_chan != 3:
            self.model.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.model(x)

class FPN101(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, pretrained=None):
        super(FPN101, self).__init__()
        self.model = FPN(in_chan=in_chan, out_chan=out_chan, dropout=0.3)
    def forward(self,x):
        return self.model(x)

class GoogLe(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, dropout=0.0, pretrained=None):
        super(GoogLe, self).__init__()
        self.model = GoogLeNet(in_chan, out_chan, dropout)
    def forward(self,x):
        return self.model(x)
