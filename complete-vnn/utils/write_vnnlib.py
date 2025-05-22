import sys
import os
from typing import List
import numpy as np

from .parameters_networks import NetworksStructure


def write_vnnlib(
    data: np.ndarray, num_classes: int, true_label: int, epsilon: float
) -> str:
    """
    write the data to vnnlib file.

    support dataset: MNIST
    """
    filename: str = f"infinity_{true_label}_{epsilon}.vnnlib"
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
        f.flush()

    return file_path


def export_vnnlib(
    lb: List[float],
    ub: List[float],
    num_data: int,
    num_classes: int,
    true_label: int,
    epsilon: float,
) -> str:
    """
    write the data to vnnlib file.

    support dataset: MNIST
    """
    filename: str = f"infinity_all_{true_label}_{epsilon}_num{num_data}.vnnlib"
    directory: str = "./util/benchmarks/vnnlib/"
    file_path: str = os.path.join(directory, filename)

    with open(file_path, "w+") as f:
        f.write("; robustness verification of neural network\n")
        # * declare the input and output variables
        for i in range(len(lb)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")
        for each_class in range(num_classes):
            f.write(f"(declare-const Y_{each_class} Real)\n")
        f.write("\n")

        # * declare pre-conditions
        for id, (l, u) in enumerate(zip(lb, ub)):
            f.write(f"(assert (<= X_{id} {u}))\n")
            f.write(f"(assert (>= X_{id} {l}))\n\n")
        f.write("\n")

        # * declare post-conditions
        f.write("(assert (or \n")
        for each_class in range(num_classes):
            if each_class == true_label:
                continue
            else:
                f.write(f"\t(and (>= Y_{each_class} Y_{true_label}))\n")
        f.write("))\n")
        f.flush()

    return file_path


if __name__ == "__main__":
    write_vnnlib(np.array([1, 2, 3, 4]), 0, 10, 5, 0.1)
