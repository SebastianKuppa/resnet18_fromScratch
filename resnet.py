import torch.nn as nn
import torch
import argparse

from torch import Tensor
from typing import Type

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--num-layers', dest='num_layers', default=18,
    type=int,
    help='number of layer to build ResNet with',
    choices=[18, 34, 50, 101, 152]
)
args = vars(parser.parse_args())


class BasicBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1,
                 downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.num_layers = num_layers
        #
        # multiplicative factor for the subsequent conv2d layers output channels
        # it is 1 for resnet 18/35 and 4 for the rest
        self.expansion = expansion
        self.downsample = downsample
        # 1x1 conv2d layer for resnet >34
        if self.num_layers > 34:
            self.conv0 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
            self.bn0 = nn.BatchNorm2d(out_channels)
            in_channels = out_channels
        # common 3x3 conv
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 1x1 for resnet > 34
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels*self.expansion,
                kernel_size=1,
                stride=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        else:
            # 3x3 for resnet <= 34
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor)->Tensor:
        identity = x
        # 1x1 conv for resnet > 34
        if self.num_layers > 34:
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.relu(out)
        if self.num_layers > 34:
            out = self.conv1(out)
        else:
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
            layers = [2, 2, 2, 2]
            self.expansion = 1
        if num_layers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1
        if num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        if num_layers == 101:
            layers = [3, 4, 23, 3]
            self.expansion = 4
        if num_layers == 152:
            layers = [3, 8, 36, 3]
            self.expansion = 4

        self.in_channels = 64

        # every ResNet contains a Conv2D(kernel_size=7), BN, Relu at the beginning
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )