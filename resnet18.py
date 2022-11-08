import torch.nn as nn
import torch
from torch import Tensor

from typing import Type


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1,
                 downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion #multiplicative factor for subsequentconv2d layers, 1 for resnet18 and 34
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, img_channels: int, num_layers: int, block: Type[BasicBlock], num_classes: int = 1000) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # the following variable defines how many 'BasicBlock's should be used and how they are staked together
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # ResNets (18 to 152) contain: Conv2d->BN->ReLU
        self.conv1 = nn.Conv2d(in_channels=img_channels,
                               out_channels=self.in_channels,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)


