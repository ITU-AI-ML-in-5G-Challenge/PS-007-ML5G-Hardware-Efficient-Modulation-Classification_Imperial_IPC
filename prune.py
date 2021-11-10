import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader import radioml_18_dataset
from models import FullModelWithMoments
from torch.nn.utils import prune
import brevitas.nn as qnn
from torch.nn import functional as F
import os


"""
In this part of the code we perform pruning. We consider only L1-norm pruning, implemented with standard PyTorch library.
We show, that despite its simplicity, this strategy, combined with statistics-based-side-information and quantization
can achieve satisfactory results in terms of memory and computation.
"""


def calculate_nonzero_params_conv(net):
    # Calculates the number of nonzero parameters in the convolutional layers
    zeros = 0
    total = 0
    for i in [2, 5, 8, 11, 14, 17, 20]:
        zeros += float(torch.sum(net.features[i].quant_weight().value == 0))
        total += float(net.features[i].weight.nelement())
    print("CONV number of zeros: {:}, total number of params: {:}, fraction of pruned params: {:}".format(zeros, total,
                                                                                                          zeros/total))


def calculate_nonzero_params_linear(net):
    # Calculates the number of nonzero parameters in the linear layers
    zeros = 0
    total = 0

    if isinstance(net.features[25], qnn.QuantLinear):
        zeros += float(torch.sum(net.features[25].weight.value == 0))
        total += float(net.features[25].weight.value.nelement())
    for module in [net.classifier[0], net.classifier[3]]:
        if isinstance(module, qnn.QuantLinear):
            zeros += float(torch.sum(module.quant_weight().value == 0))
            total += float(module.weight.nelement())
    if isinstance(net.momentum_features[1], qnn.QuantLinear):
        zeros += float(torch.sum(net.momentum_features[1].quant_weight().value == 0))
        total += float(net.momentum_features[1].weight.nelement())
    print("LINEAR number of zeros: {:}, total number of params: {:}, fraction of pruned params: {:}".format(zeros, total,
                                                                                                          zeros/total))


def train(model, teacher, train_loader, optimizer, criterion):
    losses = []
    # Ensure model is in training mode
    model.train()
    tbar = tqdm(train_loader, desc="Batches", disable=False)
    for (inputs, target, snr) in tbar:

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()

        # Forward pass
        output = model(inputs)

        # Get soft labels
        with torch.no_grad():
            soft_target = F.softmax(teacher(inputs)[0] / 2., dim=1)

        # Here we consider knowledge distillation term in loss function
        loss = 0.5 * (criterion(output[0], target) + criterion(output[1], target) + criterion(output[2], target)) + \
               0.5 * (4. * F.kl_div(F.log_softmax(output[0] / 2., dim=1), soft_target, reduction='batchmean'))

        # Backward pass + run optimizer to update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep track of loss value
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

    # Instantiating the models
    model = FullModelWithMoments(a_bits=6, w_bits=6, filters_conv=64, filters_dense=128)
    model.load_state_dict(torch.load("models/initial_model.pth", map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model = model.cuda()

    teacher = FullModelWithMoments(a_bits=6, w_bits=6, filters_conv=64, filters_dense=128)
    teacher.load_state_dict(torch.load("models/initial_model.pth", map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        teacher = teacher.cuda()

    # Defining parameters we would like to prune. We prune convolutional and linear layers separately
    conv_parameters_to_prune = (
        (model.features[2], 'weight'),
        (model.features[5], 'weight'),
        (model.features[8], 'weight'),
        (model.features[11], 'weight'),
        (model.features[14], 'weight'),
        (model.features[17], 'weight'),
        (model.features[20], 'weight')
    )

    linear_parameters_to_prune = (
        (model.classifier[0], 'weight'),
        (model.classifier[3], 'weight'),
        (model.features[25], 'weight'),
        (model.momentum_features[1], 'weight'),
    )

    # Other training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    # First pruning iteration, 20 % of remaining weights pruned at each iteration
    prune.global_unstructured(parameters=conv_parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.15)
    prune.global_unstructured(parameters=linear_parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.15)

    # Keeping track of pruning steps
    idx = 1

    pruning_epochs = 180
    for epoch in range(pruning_epochs):
        loss_epoch = train(model, teacher, data_loader_train, optimizer, criterion)
        lr_scheduler.step()

        if (epoch + 1) % 20 == 0:
            test_acc = test(model, data_loader_test)[0]
            # Reporting number of parameters and the accuracy
            calculate_nonzero_params_conv(model)
            calculate_nonzero_params_linear(model)
            print("Test acc: " + str(test_acc))

            # Terminate the pruning process if we are below the performance threshold (+ some safety margin to account
            # for the operation replacement, which happens in final model.
            if test_acc < 0.563:
                exit()

            # Saving the intermediate model
            torch.save(model.state_dict(), "models/final_model.pth")

            # Next pruning step
            idx += 1
            prune.global_unstructured(parameters=conv_parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.15)
            prune.global_unstructured(parameters=linear_parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.15)
