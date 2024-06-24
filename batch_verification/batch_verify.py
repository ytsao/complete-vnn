from typing import List

from utils.mip_modeling import Model
from utils.scip_modeling import SCIPModel
from utils.gurobi_modeling import GurobiModel

def main() -> None:

    print("this is main function")

    m = SCIPModel(solver=Model())
    # m2 = GurobiModel(solver=Model(solver_name="gurobi"))
    m.solver.binary_variables["x"] = m.add_variable(lb=0, ub=1, vtype="B", name="x")    
    print("got it")

    return 


if __name__ == "__main__":
    print("batch verification is starting...")
    main()