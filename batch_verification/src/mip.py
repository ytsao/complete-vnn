from typing import Any

from ..utils import Model
from ..utils import SCIPModel
from ..utils import GurobiModel
from ..utils import NetworksStructure
from ..utils import read_dataset

# required >= 3.12 version
# type ModelType = SCIPModel | GurobiModel

class _ObjectiveFunction():
    @staticmethod 
    def robustness_property(m: SCIPModel | GurobiModel, epsilon: float) -> SCIPModel | GurobiModel:
        
        return 
    

    @staticmethod
    def safety_property(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return 
    

class _Constraints():
    @staticmethod
    def pre_condition(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for id, value in enumerate(networks.pre_condition):
            m.change_variable_lb(m.solver.continue_variables[f"x_{0}_{id}"], value[0])
            m.change_variable_ub(m.solver.continue_variables[f"x_{0}_{id}"], value[1])

        return m


    @staticmethod
    def post_condition(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        return m


    @staticmethod
    def feedforward_networks(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for k, Nk in enumerate(networks.layer_to_layer):
            for j in range(Nk[1]):
                expression = sum(networks.matrix_weights[k][i][j] * m.solver.continue_variables[f"x_{k}_{i}"] for i in range(Nk[0])) + networks.vector_bias[k][j] == m.solver.continue_variables[f"x_{k+1}_{j}"] - m.solver.continue_variables[f"s_{k+1}_{j}"]
                m.add_constraint(express=expression, name=f"feedforward_{k}_{j}")

        return m
    
    
    #TODO: implement the following functions
    @staticmethod
    def convolutional_networks(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    #TODO: implement the following functions
    @staticmethod
    def residual_networks(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    #TODO: implement the following functions
    @staticmethod
    def max_pooling(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    @staticmethod
    def relu(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for k, Nk in enumerate(networks.layer_to_layer):
            if k == 0: 
                continue

            for nk in range(Nk[0]):
                expression = m.solver.continue_variables[f"x_{k}_{nk}"] <= 99999 * (1 - m.solver.binary_variables[f"z_{k}_{nk}"])
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_1")

                expression = m.solver.continue_variables[f"s_{k}_{nk}"] <= 99999 * m.solver.binary_variables[f"z_{k}_{nk}"]
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_2")

        return m
    

    #TODO: implement the following functions
    @staticmethod
    def sigmoid(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m


def _create_decision_variables(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
    for k, Nk in enumerate(networks.layer_to_layer):
        if k == 0: # input layer
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=None, ub=None, vtype="C", name=name)
        elif k <= networks.num_layers - 3:
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
        else: # output layer
            print("output layer")
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)

                name = f"y_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

            for nk in range(Nk[1]):
                name = f"x_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k+1}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
            

            #TODO: 'or' condition in post-condition 


    return m

def build_mip_verifier (solver: str) -> SCIPModel | GurobiModel | None:
    m: SCIPModel | GurobiModel | None = None
    if solver == "scip":
        m = SCIPModel(solver=Model())
    elif solver == "gurobi":
        m = GurobiModel(solver=Model(solver_name="gurobi"))
    else:
        raise ValueError("Invalid solver type")
    
    # extract networks structure
    networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/ACASXU_run2a_1_1_batch_2000.onnx", "./batch_verification/utils/benchmarks/vnnlib/prop_7.vnnlib")

    # create decision variables
    m = _create_decision_variables(m, networks)
    
    # create objective function
    # _ObjectiveFunction.robustness_property(m, 0.1)

    # create constraints
    _Constraints.pre_condition(m=m, networks=networks)
    print("got pre-condition")
    _Constraints.feedforward_networks(m=m, networks=networks)
    # _Constraints.convolutional_networks(m)
    # _Constraints.residual_networks(m)
    # _Constraints.max_pooling(m)
    _Constraints.relu(m=m, networks=networks)
    # _Constraints.sigmoid(m)



    return m


def main() -> None:
    print("this is main function")
    build_mip_verifier("scip")
    print("scip got it")
    build_mip_verifier("gurobi")
    print("gurobi got it")
    return

if __name__ == "__main__":
    # python -m batch_verification.src.mip 
    print("batch verification is starting...")
    main()