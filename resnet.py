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
