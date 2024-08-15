from typing import List

from apronpy.box import PyBox, PyBoxDManager, PyBoxMPQManager, PyBoxMPFRManager
from apronpy.environment import PyEnvironment
from apronpy.interval import PyDoubleInterval, PyMPQInterval, PyMPFRInterval
from apronpy.manager import PyManager
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from util import NetworksStructure
from util import DataSet
from util import read_dataset
from util.log import Logger


def execute() -> None:
    # create an environment
    env = PyEnvironment()
    man: PyManager = PyBoxDManager()

    # add real variables
    real_vars: List[PyVar] = []
    for i in range(3):
        real_vars.append(PyVar(f"x{i}"))
    env.add(real_vars=real_vars)

    print(PyBox.bottom(man, env).is_bottom())

    return


def _main() -> None:
    execute()

    return


if __name__ == "__main__":
    _main()
    print("Done")
