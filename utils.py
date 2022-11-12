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

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Saves training and validation accuracy/loss on disk
    :param train_acc: training accuracy
    :param valid_acc: validation accuracy
    :param train_loss: training loss
    :param valid_loss: validation loss
    :param name: file name
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='train_accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-',
        label='validation_accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_accuracy.png'))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train_loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validation_loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_loss.png'))
