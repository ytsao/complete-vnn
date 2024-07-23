import os
import argparse

import onnxruntime as ort
import onnx2pytorch
import onnx
import torch
import torch.nn as nn
import torchvision

from util.read_results import Crown
from util.log import Logger

# from utils.parameters_networks import DataSet
# from utils.read_dataset import load_dataset

# from arguments import ConfigHandler
from complete_verifier.abcrown import ABCROWN
import complete_verifier.arguments as arguments


def crown_verifier(onnx_file_path: str, vnnlib_file_path: str) -> str:
    # * set configuration for abcrown
    parser = argparse.ArgumentParser("ABCROWN")
    parser.add_argument("--mode", type=str, default="debug")
    parser.add_argument("--solver", type=str, default="crown")
    parser.add_argument(
        "--config", type=str, default="./util/crown_config/default.yaml"
    )
    # parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--onnx_path", type=str, default=onnx_file_path)
    parser.add_argument("--vnnlib_path", type=str, default=vnnlib_file_path)
    args = parser.parse_args()
    Logger.debugging(messages=[f"args: {args}"])

    abcrown_args = [
        f"--config={args.config}",
        # f"--timeout={args.timeout}",
        f"--onnx_path={args.onnx_path}",
        f"--vnnlib_path={args.vnnlib_path}",
        "--num_outputs=1",
        "--spec_type=box",
        "--robustness_type=all-positive",
        "--enable_input_split",
        "--branching_method=sb",
        "--sb_coeff_thresh=0.01",
    ]

    abcrown: ABCROWN = ABCROWN(args=abcrown_args)
    Logger.info("Executing abcrown")
    abcrown.main()
    Logger.info("abcrown done")

    result: str = Crown.get_status("./out.txt")

    return result
