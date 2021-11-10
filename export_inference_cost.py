from finn.util.inference_cost import inference_cost

final_onnx_path = "models/model_final.onnx"
cost_dict_path = "models/model_cost.json"
export_onnx_path = "models/model_export.onnx"

inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
               preprocess=True, discount_sparsity=True)
