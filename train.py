import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

import utils

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

# learning and training params
epochs = 20
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_data(batch_size=batch_size)

if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch..')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    print('[INFO]: Training the Torchvision Resnet18 model..')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10)
    plot_name = 'resnet_torchvision'

#print total params and number of trainable params
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters')

# init optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss function
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # create lists for losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # start the training
    for epoch in range(epochs):
        print(f'[INFO]: Epoch {epoch} out of {epochs}')
        train_epoch_loss, train_epoch_acc = train(model=model,
                                                  trainloader=train_loader,
                                                  optimizer=optimizer,
                                                  criterion=criterion,
                                                  device=device)
        valid_epoch_loss, valid_epoch_acc = validate(model=model,
                                                     testloader=valid_loader,
                                                     criterion=criterion,
                                                     device=device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f'Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}')
        print(f'Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}')
        print('-'*50)
    # save plots to disk
    save_plots(train_acc,
               valid_acc,
               train_loss,
               valid_loss,
               plot_name)
    print('Training finished..')
