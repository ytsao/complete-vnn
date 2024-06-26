from typing import List, Tuple
from numpy import array as Array
from dataclasses import dataclass, field


@dataclass
class NetworksStructure:
    onnx_filename: str      = field(default="")
    vnnlib_filename: str    = field(default="")
    
    num_inputs: int     = field(default=0)
    num_outputs: int    = field(default=0)
    num_layers: int     = field(default=0)

    num_post_region: int = field(default=0) # if there is an 'or' condition in output region, then it is >= 2

    layer_to_layer: List[Tuple[int, int]]   = field(init=False) # from, to
    matrix_weights: List[List[List[float]]] = field(init=False) # layer, from, to
    vector_bias: List[List[float]]          = field(init=False) # layer, neuron

    # skip the type for pre_condition and post_condition    
    pre_condition: List = field(init=False) 
    post_condition: List = field(init=False)  # [num_post_region, lhs, rhs], mat * y <= rhs

@dataclass
class DataSet:
    train_images: Array = field(init=False)
    train_labels: Array = field(init=False)
    test_images: Array = field(init=False)
    test_labels: Array = field(init=False)
    num_labels: int = field(init=False)
    num_pixels: int = field(init=False)
    num_height: int = field(init=False)
    num_weight: int = field(init=False)
    num_channel: int = field(init=False)
