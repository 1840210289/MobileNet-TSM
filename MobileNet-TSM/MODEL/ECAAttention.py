import numpy as np
from torch.nn import init
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from math import log


class ECAAttention(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    eca = ECAAttention(kernel_size=3)
    output=eca(input)
    print(output.shape)

    