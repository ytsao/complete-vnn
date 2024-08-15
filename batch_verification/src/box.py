from typing import List
import itertools

from apronpy.box import PyBox, PyBoxDManager, PyBoxMPQManager, PyBoxMPFRManager
from apronpy.environment import PyEnvironment
from apronpy.interval import PyDoubleInterval, PyMPQInterval, PyMPFRInterval
from apronpy.manager import PyManager
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

import numpy as np

from util import NetworksStructure
from util import DataSet
from util.read_dataset import extract_network_structure, load_dataset
from util.log import Logger


def _create_real_vars(networks: NetworksStructure) -> List[List[PyVar]]:
    real_vars: List[List[PyBox]] = []

    for k, Nk in enumerate(networks.layer_to_layer):
        layer_real_vars: List[PyVar] = []
        for nk in range(Nk[0]):
            name: str = f"x{k}_{nk}"
            layer_real_vars.append(PyVar(name))
        if k > networks.num_layers - 3:
            for nk in range(Nk[1]):
                name: str = f"x{k + 1}_{nk}"
                layer_real_vars.append(PyVar(name))
        real_vars.append(layer_real_vars)

    return real_vars


def _create_box_vars(
    env: PyEnvironment, man: PyManager, real_vars: List[List[PyVar]]
) -> List[List[PyBox]]:
    box_vars: List[List[PyBox]] = [[PyBox.bottom(man, env)]]
    lb: float = -99999
    ub: float = 99999
    for layer_real_vars in real_vars:
        layer_box_vars: List[PyBox] = []
        for each_var in layer_real_vars:
            each_box = PyBox(
                man,
                env,
                variables=[each_var],
                intervals=[PyDoubleInterval(lb, ub)],
            )
            layer_box_vars.append(each_box)
        box_vars.append(layer_box_vars)
    box_vars.append([PyBox.top(man, env)])

    return box_vars


# TODO: Implement the following functions
def _pre_condition(
    env: PyEnvironment,
    man: PyManager,
    networks: NetworksStructure,
    real_vars: List[List[PyVar]],
    box_vars: List[List[PyBox]],
) -> None:
    for id, value in enumerate(networks.pre_condition):
        box_vars[1][id] = PyBox(
            man,
            env,
            variables=[real_vars[0][id]],
            intervals=[PyDoubleInterval(value[0], value[1])],
        )
        # box_vars[1][id].meet(tmp_box)
        print(box_vars[1][id])

    return


def _forward_propagation(
    networks: NetworksStructure, box_vars: List[List[PyBox]]
) -> None:
    # TODO: Implement forward propagation
    for k, Nk in enumerate(networks.layer_to_layer):
        for nk in range(Nk[0]):
            pass
        if k > networks.num_layers - 3:
            for nk in range(Nk[1]):
                pass
    return


def execute(networks: NetworksStructure) -> None:
    # create an environment
    env = PyEnvironment()
    man: PyManager = PyBoxDManager()

    # add real variables
    real_vars: List[List[PyVar]] = _create_real_vars(networks)
    env.add(real_vars=list(itertools.chain.from_iterable(real_vars)))

    # add box variables
    PyBox.bottom(man, env)
    box_vars: List[List[PyBox]] = _create_box_vars(env, man, real_vars)

    _pre_condition(env, man, networks=networks, real_vars=real_vars, box_vars=box_vars)

    return


def _main() -> None:
    networks: NetworksStructure = extract_network_structure(
        onnx_file_path="./util/benchmarks/onnx/mnist-net_256x2.onnx",
        vnnlib_file_path="./util/benchmarks/vnnlib/prop_0_0.03.vnnlib",
    )
    execute(networks=networks)

    return


if __name__ == "__main__":
    _main()
    print("Done")
