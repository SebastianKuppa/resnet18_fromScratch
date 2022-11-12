import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

plt.style.use('ggplot')


def get_data(batch_size=64):
    #CIFAR10 dataset
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=False,
        transform=ToTensor(),
        )

