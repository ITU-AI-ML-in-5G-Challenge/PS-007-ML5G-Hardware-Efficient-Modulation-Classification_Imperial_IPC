import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader import radioml_18_dataset
from models import FullModelWithMoments
import os


def train(model, train_loader, optimizer, criterion):
    losses = []
    # ensure model is in training mode
    model.train()
    tbar = tqdm(train_loader, desc="Batches", disable=False)
    for (inputs, target, snr) in tbar:

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()

        # forward pass
        output = model(inputs)

        # We simply consider the sum of all classifier sub-losses
        loss = criterion(output[0], target) + criterion(output[1], target) + criterion(output[2], target)

        # backward pass + run optimizer to update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())
        tbar.set_postfix_str(np.round(np.mean(losses), 4))

    return losses


def test(model, test_loader):
    # ensure model is in eval mode
    model.eval()
    y_true = []
    y_pred = [[], [], []]

    with torch.no_grad():
        for (inputs, target, snr) in test_loader:

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            y_true.extend(target.tolist())
            for i in range(3):
                pred = output[i].argmax(dim=1, keepdim=True)
                y_pred[i].extend(pred.reshape(-1).tolist())

    # Test function here outputs the accuracies of all the classifiers
    return [accuracy_score(y_true, y_pred[i]) for i in range(3)]


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Dataset loading, please change the dataset_path!
    dataset_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError("Cannot find the data file, make sure to provide correct directory!")
    dataset = radioml_18_dataset(dataset_path)
    batch_size = 1024
    num_epochs = 20

    data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler, num_workers=1)
    data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler, num_workers=1)

    # Instantiating the model
    model = FullModelWithMoments(a_bits=6, w_bits=6, filters_conv=64, filters_dense=128)
    if torch.cuda.is_available():
        model = model.cuda()

    # Other training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    running_loss = []
    running_test_acc = []

    for epoch in range(num_epochs):
        loss_epoch = train(model, data_loader_train, optimizer, criterion)
        test_acc = test(model, data_loader_test)
        test_acc = test_acc[0]
        print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)
        lr_scheduler.step()

    torch.save(model.state_dict(), "models/initial_model.pth")
