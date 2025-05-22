from io import TextIOWrapper
from typing import Tuple, List


from .gurobi_modeling import GurobiModel
from .scip_modeling import SCIPModel
from .parameters_networks import NetworksStructure
from .log import Logger

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
    solver: str,
    networks: NetworksStructure,
    filename: str,
    m: GurobiModel | SCIPModel | None = None,
) -> List[float]:
    if solver == "scip" or solver == "gurobi":
        return MIP.get_ce(m, networks)
    else:
        Logger.error(f"Solver {solver} not supported")
        return []


if __name__ == "__main__":
    # * Testing :: OK
    Logger.initialize()