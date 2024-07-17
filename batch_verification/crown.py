import sys
import os
import argparse

import onnxruntime as ort
import onnx2pytorch
import onnx
import torch
import torch.nn as nn
import torchvision

# sys.path = ["complete_verifier"] + sys.path
# from arguments import ConfigHandler
from complete_verifier.abcrown import ABCROWN

# from utils.parameters_networks import DataSet
# from utils.read_dataset import load_dataset




if __name__ == "__main__":
    abcrown: ABCROWN = ABCROWN()
    print(abcrown)

    # dataset: DataSet = load_dataset("mnist")
    # onnx_filename: str = "./utils/benchmarks/onnx/mnist-net_256x2.onnx"
    # vnnlib_filename: str = "./utils/benchmarks/vnnlib/prop_7_0.03.vnnlib"

    # onnx_model: onnx.ModelProto = onnx.load(onnx_filename)

    # print(abcrown.incomplete_verifier(model_ori=onnx_model,
    #       data=dataset.test_images[0], vnnlib=vnnlib_filename))
