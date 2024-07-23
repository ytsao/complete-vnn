from io import TextIOWrapper
from typing import Tuple, List


from .gurobi_modeling import GurobiModel
from .scip_modeling import SCIPModel
from .parameters_networks import NetworksStructure
from .options import VerificationSolver
from .log import Logger


class Crown:
    @staticmethod
    def get_status(filename: str) -> str:
        try:
            f: TextIOWrapper = open(filename, "r")
            res: str = f.readline().strip()
            f.close()
        except OSError:
            Logger.error(f"Cannot find the file {filename}")
            return ""

        assert res == "sat" or res == "unsat", f"Unexpected result: {res}"

        return res.upper()  # either "SAT" or "UNSAT"

    @staticmethod
    def get_ce(filename: str) -> List[float]:
        try:
            f: TextIOWrapper = open(filename, "r")
            s = f.readlines()
            f.close()
        except OSError:
            Logger.error(f"Cannot find the file {filename}")
            return [], []

        characters_to_remove = ["(", ")", "\n"]
        translation_table = str.maketrans("", "", "".join(characters_to_remove))
        s = [x.translate(translation_table) for x in s]

        x: List[float] = [float(v.split()[1]) for v in s if "X" in v.split()[0]]
        y: List[float] = [float(v.split()[1]) for v in s if "Y" in v.split()[0]]
        Logger.debugging(f"x: {len(x)}")
        Logger.debugging(f"y: {len(y)}")

        return x


class MIP:
    @staticmethod
    def get_ce(m: GurobiModel | SCIPModel, networks: NetworksStructure) -> List[float]:
        x: List[float] = []
        for k, Nk in enumerate(networks.layer_to_layer):
            if k == 0:
                for nk in range(Nk[0]):
                    name: str = f"x_{k}_{nk}"
                    variable = m.solver.continue_variables[name]
                    x.append(m.get_primal_solution(variable))
            else:
                break

        return x


def get_ce(
    solver: VerificationSolver,
    networks: NetworksStructure,
    filename: str,
    m: GurobiModel | SCIPModel | None = None,
) -> List[float]:
    if solver == VerificationSolver.GUROBI or solver == VerificationSolver.SCIP:
        return MIP.get_ce(m, networks)
    elif solver == VerificationSolver.CROWN:
        return Crown.get_ce(filename)
    else:
        Logger.error(f"Solver {solver} not supported")
        return []


if __name__ == "__main__":
    # * Testing :: OK
    Logger.initialize()

    Logger.info(Crown.get_status("../out.txt"))
    Crown.get_ce("../test_cex1.txt")
