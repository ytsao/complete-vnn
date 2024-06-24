from typing import List, Tuple

import onnx.model_container
from vnnlib import compat
import onnx
import onnxruntime as ort
import numpy as np

from .parameters_networks import NetworksStructure


def _read_onnx_model(onnx_file_path: str) -> onnx.ModelProto:
    return onnx.load(onnx_file_path)


def _get_num_inputs(onnx_model: onnx.ModelProto) -> int: 
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]

    graph = onnx_model.graph
    all_input = [n for n in graph.input if n.name == inputs[0]][0]

    num_inputs = 1
    for d in all_input.type.tensor_type.shape.dim:
        num_inputs *= d.dim_value

    return num_inputs


def _get_num_outputs(onnx_model: onnx.ModelProto) -> int:
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    outputs = [o.name for o in sess.get_outputs()]

    graph = onnx_model.graph
    all_output = [n for n in graph.output if n.name == outputs[0]][0]

    num_outputs = 1
    for d in all_output.type.tensor_type.shape.dim:
        num_outputs *= d.dim_value

    return num_outputs


def _get_layer_to_layer(onnx_model: onnx.ModelProto) -> List[Tuple[int, int]]:
    layer_to_layer: List[Tuple[int, int]] = []

    for initializer in onnx_model.graph.initializer:
        array: np.ndarray = onnx.numpy_helper.to_array(initializer)

        if "MatMul" in initializer.name:
            layer_to_layer.append(array.shape)
        elif "weight" in initializer.name:
            layer_to_layer.append(array.shape[1], array.shape[0])

    return layer_to_layer


def _get_weight_matrix(onnx_model: onnx.ModelProto) -> List[List[List[float]]]:
    weights: List[List[List[float]]] = []

    for initializer in onnx_model.graph.initializer:
        array: np.ndarray = onnx.numpy_helper.to_array(initializer)

        if "MatMul" in initializer.name:
            weights.append(array)
        elif "weight" in initializer.name:
            weights.append(array.transpose())

    return weights
     

def _get_bias_vector(onnx_model: onnx.ModelProto) -> List[List[float]]:
    biases: List[List[float]] = []

    for initializer in onnx_model.graph.initializer:
        array: np.ndarray = onnx.numpy_helper.to_array(initializer)

        if "Add" in initializer.name:
            biases.append(array)
        elif "bias" in initializer.name:
            biases.append(array)

    return biases


def _read_vnnlib_spec(vnnlib_file_path: str, inputs: int, outputs: int):
    vnnlib_spec = compat.read_vnnlib_simple(vnnlib_file_path, inputs, outputs)

    return vnnlib_spec


def extract_network_structure(onnx_file_path: str, vnnlib_file_path: str) -> NetworksStructure:
    onnx_model: onnx.ModelProto = _read_onnx_model(onnx_file_path)
    num_inputs: int = _get_num_inputs(onnx_model)
    num_outputs: int = _get_num_outputs(onnx_model)
    vnnlib_spec = _read_vnnlib_spec(vnnlib_file_path, num_inputs, num_outputs)
    pre_condition = vnnlib_spec[0][0]
    post_condition = vnnlib_spec[0][1]
    

    n = NetworksStructure()
    n.onnx_filename = onnx_file_path
    n.vnnlib_filename = vnnlib_file_path
    
    n.num_inputs = num_inputs
    n.num_outputs = num_outputs

    n.num_post_region = len(post_condition)

    n.layer_to_layer = _get_layer_to_layer(onnx_model)
    n.num_layers = len(n.layer_to_layer) + 1
    n.matrix_weights = _get_weight_matrix(onnx_model)
    n.vector_bias = _get_bias_vector(onnx_model)

    n.pre_condition = pre_condition
    n.post_condition = post_condition

    print(n)

    return n


def _test():
    onnx_model: onnx.ModelProto = _read_onnx_model("./benchmarks/onnx/ACASXU_run2a_1_1_batch_2000.onnx")
    num_inputs: int = _get_num_inputs(onnx_model)
    num_outputs: int = _get_num_outputs(onnx_model)
    print("num_inputs: ", num_inputs)
    print("num_outputs: ", num_outputs)
    vnnlib_spec = _read_vnnlib_spec("./benchmarks/vnnlib/prop_7.vnnlib", num_inputs, num_outputs)
    # vnnlib_spec = _read_vnnlib_spec("./benchmarks/vnnlib/test_small.vnnlib", num_inputs, num_outputs)

    input_region = vnnlib_spec[0][0]
    output_region = vnnlib_spec[0][1]

    # print information
    print(vnnlib_spec)
    print(len(vnnlib_spec[0]))
    print(vnnlib_spec[0][0])    # input region
    print(vnnlib_spec[0][1])    # output region
    print("number of output conditions:")
    print(len(vnnlib_spec[0][1]))
    print(type(vnnlib_spec[0][1]))

    # test
    print("test")
    print(type(vnnlib_spec[0][1][0]))
    print(len(vnnlib_spec[0][1][0]))
    print(vnnlib_spec[0][1][0][0]) # first condition's lhs
    print("===================================")
    print(vnnlib_spec[0][1][0][1])  # first condition's rhs

    print("-----------------------------------")
    print(vnnlib_spec[0][1][1][0])  # second condition's lhs
    print(vnnlib_spec[0][1][1][1])  # second condition's rhs
    print(type(vnnlib_spec[0][1][0][0]))
    print(len(vnnlib_spec[0][1][0][0]))
    print(type(vnnlib_spec[0][1][0][0][0]))
    print(len(vnnlib_spec[0][1][0][0][0]))
    print(type(vnnlib_spec[0][1][0][0][0][0]))


if __name__ == "__main__":
    # _test()
    networks: NetworksStructure = extract_network_structure("./benchmarks/onnx/ACASXU_run2a_1_1_batch_2000.onnx", "./benchmarks/vnnlib/prop_7.vnnlib")
    print(networks)
    print("Done!")