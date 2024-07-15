import sys
import os 
from jax import numpy as jnp

from .parameters_networks import NetworksStructure


def write_vnnlib(data: jnp.ndarray, data_id:int, num_classes: int, true_label:int, epsilon: float) -> str:
    """
    write the data to vnnlib file.

    support dataset: MNIST
    """
    filename: str = f"infinity_{data_id}_{true_label}_{epsilon}.vnnlib"
    directory: str = "./utils/benchmarks/vnnlib/"
    file_path: str = os.path.join(directory, filename)

    with open(file_path, "w+") as f:
        f.write("; robustness verification of neural network\n")
        # * declare the input and output variables 
        for id, each_pixel in enumerate(data): 
            f.write(f"(declare-const X_{id} Real)\n")
        f.write("\n")
        for each_class in range(num_classes):
            f.write(f"(declare-const Y_{each_class} Real)\n")
        f.write("\n")

        # * declare pre-conditions
        for id, each_pixel in enumerate(data):
            f.write(f"(assert (<= X_{id} {min(1, each_pixel+epsilon)}))\n")
            f.write(f"(assert (>= X_{id} {max(0, each_pixel-epsilon)}))\n\n")
        f.write("\n")

        # * declare post-conditions
        f.write("(assert (or \n")
        for each_class in range(num_classes):
            if each_class == true_label:
                continue
            else:
                f.write(f"\t(and (>= Y_{each_class} Y_{true_label}))\n")
        f.write("))\n")

    return file_path


def write_vnnlib_merge(networks:NetworksStructure ,data: jnp.ndarray, data_id:int, num_classes: int, true_label:int, epsilon: float) -> str:
    """
    * write the data to vnnlib file.
    * merge several property together into one file.
    
    support dataset: MNIST
    """
    filename: str = f"infinity_{data_id}_{true_label}_{epsilon}_merge.vnnlib"
    directory: str = "./utils/benchmarks/vnnlib/"
    file_path: str = os.path.join(directory, filename)

    with open(file_path, "w+") as f:
        f.write("; robustness verification of neural network (merged) \n")
        # * declare the input and output variables
        for id, each_pixel in enumerate(data):
            f.write(f"(declare-const X_{id} Real)\n")
        f.write("\n")
        for each_class in range(num_classes):
            f.write(f"(declare-const Y_{each_class} Real)\n")
        f.write("\n")

        # * declare pre-conditions
        for id, each_pixel in enumerate(data):
            ub: float = min(1, max(networks.pre_condition[id][1], each_pixel+epsilon))
            lb: float = max(0, min(networks.pre_condition[id][0], each_pixel-epsilon))
            f.write(f"(assert (<= X_{id} {ub}))\n")
            f.write(f"(assert (>= X_{id} {lb}))\n\n")
        f.write("\n")

        # * declare post-conditions
        f.write("(assert (or \n")
        for each_class in range(num_classes):
            if each_class == true_label:
                continue
            else:
                f.write(f"\t(and (>= Y_{each_class} Y_{true_label}))\n")
        f.write("))\n")


    return file_path


def write_vnnlib_meet(networks:NetworksStructure, data: jnp.ndarray, data_id:int, num_classes: int, true_label:int, epsilon: float) -> str:
    """
    * write the data to vnnlib file.
    * meet several property together into one file.

    support dataset: MNIST
    """
    filename: str = f"infinity_{data_id}_{true_label}_{epsilon}_meet.vnnlib"
    directory: str = "./utils/benchmarks/vnnlib/"
    file_path: str = os.path.join(directory, filename)

    with open(file_path, "w+") as f:
        f.write("; robustness verification of neural network (meet) \n")
        # * declare the input and output variables
        for id, each_pixel in enumerate(data):
            f.write(f"(declare-const X_{id} Real)\n")
        f.write("\n")
        for each_class in range(num_classes):
            f.write(f"(declare-const Y_{each_class} Real)\n")
        f.write("\n")

        # * declare pre-conditions
        for id, each_pixel in enumerate(data):
            ub: float = min(1, min(networks.pre_condition[id][1], each_pixel+epsilon))
            lb: float = max(0, max(networks.pre_condition[id][0], each_pixel-epsilon))
            f.write(f"(assert (<= X_{id} {ub}))\n")
            f.write(f"(assert (>= X_{id} {lb}))\n\n")
        f.write("\n")

        # * declare post-conditions
        f.write("(assert (or \n")
        for each_class in range(num_classes):
            if each_class == true_label:
                continue
            else:
                f.write(f"\t(and (>= Y_{each_class} Y_{true_label}))\n")
        f.write("))\n")

    
    return file_path


if __name__ == "__main__":
    write_vnnlib(jnp.array([1, 2, 3, 4]), 0, 10, 5, 0.1)