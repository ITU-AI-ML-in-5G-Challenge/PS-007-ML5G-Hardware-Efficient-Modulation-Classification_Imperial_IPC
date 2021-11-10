from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from models import FinalModelWithMoments
import torch


# Instantiating model
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

model.load_state_dict(dict_pruned, strict=False)

# Exporting ONNX
export_onnx_path = "models/model_export.onnx"
BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path, opset_version=9)