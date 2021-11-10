import os.path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader import radioml_18_dataset
from models import FinalModelWithMoments
import brevitas.nn as qnn
import json


"""
This file is used to evaluate the accuracy of FinalModelWithMoments, which is the model with all the side-branches
discarded, and some of the operations altered. It also ensures, that the network weights were correctly pruned.
It further exports the ONNX model and performs computational and memory cost estimation performed by finn.
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


def test(model, test_loader):
    # ensure model is in eval mode
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (inputs, target, snr) in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            y_true.extend(target.tolist())
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.reshape(-1).tolist())

    return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Data loading, please ensure to provide your path to the dataset!
    dataset_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError("Cannot find the data file, make sure to provide correct directory!")
    dataset = radioml_18_dataset(dataset_path)

    # Smaller batch size, because new operations are quire memory-inefficient
    data_loader_test = DataLoader(dataset, batch_size=128, sampler=dataset.test_sampler, num_workers=1)

    # Instantiate the model, we aim for lower bits, because our pruned model had significant safety margin above
    # the required accuracy.
    model = FinalModelWithMoments(a_bits=5, w_bits=5, filters_conv=64, filters_dense=128)

    # Loading pruned model
    dict_pruned = torch.load("models/final_model.pth", map_location=torch.device('cpu'))

    # Performing actual pruning of the weights, because PyTorch, does not save module.weight as a dictionary entry,
    # instead it saves weight masks and original (unpruned) weights
    for key in ["momentum_features.1", "features.2", "features.5", "features.8",  "features.11", "features.14",
                "features.17", "features.20", "features.25", "classifier.0", "classifier.3"]:
        dict_pruned[key + '.weight'] = dict_pruned[key + '.weight_orig'] * dict_pruned[key + '.weight_mask']
        del dict_pruned[key + '.weight_orig']
        del dict_pruned[key + '.weight_mask']

    # Testing the model
    model.load_state_dict(dict_pruned, strict=False)
    calculate_nonzero_params_linear(model)
    calculate_nonzero_params_conv(model)
    if torch.cuda.is_available():
        model = model.cuda()
    test_acc = test(model, data_loader_test)

    # Calculate the inference cost
    cost_dict_path = "models/model_cost.json"
    with open(cost_dict_path, 'r') as f:
        inference_cost_dict = json.load(f)
    bops = int(inference_cost_dict["total_bops"])
    w_bits = int(inference_cost_dict["total_mem_w_bits"])

    bops_baseline = 807699904
    w_bits_baseline = 1244936

    score = 0.5 * (bops / bops_baseline) + 0.5 * (w_bits / w_bits_baseline)

    print("Accuracy of the pruned model: {:}, score of the pruned model: {:}".format(test_acc, score))

