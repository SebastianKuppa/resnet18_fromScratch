import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

plt.style.use('ggplot')


def get_data(batch_size=64):
    # CIFAR10 dataset for training
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=False,
        transform=ToTensor(),
        )
    # CIFAR10 dataset for validation
    dataset_validate = datasets.CIFAR10(
        root='data',
        train=False,
        download=False,
        transform=ToTensor(),
    )
    # init training dataloader
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    # init validation dataloader
    validation_loader = DataLoader(
        dataset_validate,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader
