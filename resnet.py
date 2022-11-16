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
    def __init__(self, num_layers:int, in_channels:int, out_channels:int, stride:int = 1, expansion:int = 1,
                 downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.num_layers = num_layers
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
