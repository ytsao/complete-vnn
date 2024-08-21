from typing import List
import itertools

from apronpy.box import PyBox, PyBoxDManager, PyBoxMPQManager, PyBoxMPFRManager
from apronpy.environment import PyEnvironment
from apronpy.coeff import PyDoubleScalarCoeff
from apronpy.interval import PyDoubleInterval, PyMPQInterval, PyMPFRInterval
from apronpy.manager import PyManager
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.lincons0 import ConsTyp
from apronpy.lincons1 import PyLincons1
from apronpy.linexpr1 import PyLinexpr1
from apronpy.texpr1 import PyTexpr1
from apronpy.tcons1 import PyTcons1
from apronpy.var import PyVar

import numpy as np

from util import NetworksStructure
from util import DataSet
from util.read_dataset import extract_network_structure, load_dataset
from util.log import Logger


def _create_real_vars(networks: NetworksStructure) -> List[List[PyVar]]:
    real_vars: List[List[PyVar]] = [[PyVar("bottom")]]

    for k, Nk in enumerate(networks.layer_to_layer):
        layer_real_vars: List[PyVar] = []
        for nk in range(Nk[0]):
            name: str = f"x{k}_{nk}"
            layer_real_vars.append(PyVar(name))
        real_vars.append(layer_real_vars)

        if k > networks.num_layers - 3:
            layer_real_vars: List[PyVar] = []
            for nk in range(Nk[1]):
                name: str = f"x{k + 1}_{nk}"
                layer_real_vars.append(PyVar(name))
            real_vars.append(layer_real_vars)

    real_vars.append([PyVar("top")])

    return real_vars


def _create_box_vars(
    env: PyEnvironment, man: PyManager, real_vars: List[List[PyVar]]
) -> List[List[PyBox]]:
    box_vars: List[List[PyBox]] = []
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

    return box_vars


def _pre_condition(
    env: PyEnvironment,
    man: PyManager,
    networks: NetworksStructure,
    real_vars: List[List[PyVar]],
    box_vars: List[List[PyBox]],
) -> List[List[PyBox]]:
    for id, value in enumerate(networks.pre_condition):
        box_vars[1][id] = PyBox(
            man,
            env,
            variables=[real_vars[1][id]],
            intervals=[PyDoubleInterval(value[0], value[1])],
        )
        # print(box_vars[1][id])

    return box_vars


def _forward_propagation(
    env: PyEnvironment,
    man: PyManager,
    networks: NetworksStructure,
    real_vars: List[List[PyVar]],
    box_vars: List[List[PyBox]],
) -> None:
    # TODO: Implement forward propagation
    for k, Nk in enumerate(networks.layer_to_layer):
        for j in range(Nk[1]):
            expr: PyLinexpr1 = PyLinexpr1(env)
            for i in range(Nk[0]):
                expr.set_coeff(
                    real_vars[k + 1][i],
                    PyDoubleScalarCoeff(float(networks.matrix_weights[k][i][j])),
                )
            expr.set_cst(PyDoubleScalarCoeff(float(networks.vector_bias[k][j])))
            box_vars[k + 2][j].bound_linexpr(expr)

    for bb in box_vars:
        for b in bb:
            print(str(b))

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

    box_vars = _pre_condition(
        env, man, networks=networks, real_vars=real_vars, box_vars=box_vars
    )
    box_vars = _forward_propagation(
        env, man, networks=networks, real_vars=real_vars, box_vars=box_vars
    )

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
