import torch

from tqdm import tqdm


# training function
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training..')
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
