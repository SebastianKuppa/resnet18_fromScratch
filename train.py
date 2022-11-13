import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import get_data, save_plots

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--model', default='scratch',
                    help='choose model build from scratch or the Torchvision model',
                    choices=['scratch', 'torchvision']
                    )
args = vars(parser.parse_args())

# set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

