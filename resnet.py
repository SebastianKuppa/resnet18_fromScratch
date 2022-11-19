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
    def __init__(self, num_layers: int,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expansion: int = 1,
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
            padding=1,
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

    def forward(self, x: Tensor) -> Tensor:
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
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], num_layers=num_layers)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_layers=num_layers)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_layers=num_layers)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_layers=num_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1,
                    num_layers: int = 18) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            # section 3.3 of paper (https://arxiv.org/pdf/1512.03385v1.pdf)
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels*self.expansion),
            )
        layers = []
        layers.append(
            block(
                num_layers,
                self.in_channels,
                out_channels,
                stride,
                self.expansion,
                downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    num_layers,
                    self.in_channels,
                    out_channels,
                    expansion=self.expansion
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(f'Tensor shape: {x.shape}')

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    tensor = torch.rand([1, 3, 224, 224])
    model = ResNet(
        img_channels=3,
        num_layers=args['num_layers'],
        block=BasicBlock,
        num_classes=1000
    )
    print(model)
    # total params and trainable params
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')

    output = model(tensor)
