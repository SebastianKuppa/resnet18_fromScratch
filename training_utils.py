import torch

from tqdm import tqdm


# training function
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training..')
