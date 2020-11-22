import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Inception(nn.Module):
  def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, dropout=0.3):
    super(Inception, self).__init__()
    #1x1 conv branch
    self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),
                            nn.BatchNorm2d(n1x1),
                            nn.ReLU(True),
                           )
    #1x1 conv -> 3x3 conv branch
    self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                            nn.BatchNorm2d(n3x3red),
                            nn.ReLU(True),
                            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n3x3),
                            nn.ReLU(True),
                           )
    self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                            nn.BatchNorm2d(n5x5red),
                            nn.ReLU(True),
                            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n5x5),
                            nn.ReLU(True),
                            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n5x5),
                            nn.ReLU(True),
                           )
    self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                            nn.BatchNorm2d(pool_planes),
                            nn.ReLU(True),
                           )
    self.do = nn.Dropout2d(p=dropout)
  def forward(self, x):
    y1 = self.b1(x)
    y2 = self.b2(x)
    y3 = self.b3(x)
    y4 = self.b4(x)
    y  = torch.cat([y1, y2, y3, y4], 1)
    y  = self.do(y)
    return y

class GoogLeNet(nn.Module):
  def __init__(self, in_chan=3, out_chan=1, dropout=0.3):
    super(GoogLeNet, self).__init__()
    self.pre_layers = nn.Sequential(nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Dropout2d(p=dropout),
                                   )
    self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, dropout=dropout)
    self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, dropout=dropout)

    self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, dropout=dropout)
    self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, dropout=dropout)
    self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, dropout=dropout)
    self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, dropout=dropout)
    self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, dropout=dropout)

    self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, dropout=dropout)
    self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, dropout=dropout)

    self.mp = nn.MaxPool2d(3, stride=2, padding=1)

    self.cT5 = nn.ConvTranspose2d(1024, out_chan, kernel_size=8, stride=4, padding=2)
    self.cT4 = nn.ConvTranspose2d(832,  out_chan, kernel_size=4,  stride=2, padding=1)
    #self.cT3 = nn.ConvTranspose2d(480,  out_chan, kernel_size=4,  stride=2, padding=1)
    self.cT3 = nn.Conv2d(480, out_chan, kernel_size=3, stride=1, padding=1)

    self.mp = nn.MaxPool2d(3, stride=2, padding=1)

  def forward(self, x):
    l2 = self.pre_layers(x)
    l3 = self.b3(self.a3(l2))
    l4 = self.e4(self.d4(self.c4(self.b4(self.a4(self.mp(l3))))))
    l5 = self.b5(self.a5(self.mp(l4)))
    out3 = self.cT3(l3)
    out4 = self.cT4(l4)
    out5 = self.cT5(l5)
    return out3+out4+out5
