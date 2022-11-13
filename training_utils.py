import torch

from tqdm import tqdm


# network training
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Starting network training..')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calc loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calc accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update weights
        optimizer.step()

    # loss and acc for whole epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct/len(trainloader.dataset))
    return epoch_loss, epoch_acc


# network validation
def validate(model, testloader, criterion, device):
    model.eval()
    print('Starting network validation..')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(image)
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        # calc accuracy
        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()

    # loss and acc for whole epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct/len(testloader.dataset))
    return epoch_loss, epoch_acc
