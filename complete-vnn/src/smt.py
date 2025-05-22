import numpy as np 

import z3

from utils import SMTModel
from utils import NetworksStructure
from utils.log import Logger

class _Constraints:
    @staticmethod
    def pre_condition(m:SMTModel, networks: NetworksStructure) -> SMTModel:
        Logger.info(messages="Create pre-condition constraints")

        for id, value in enumerate(networks.pre_condition):
            m._model.add(m.continue_variables[f"x_{0}_{id}"] >= value[0])
            m._model.add(m.continue_variables[f"x_{0}_{id}"] <= value[1])

        Logger.info(messages="Pre-condition constraints are created")

        return 

    @staticmethod
    def post_condition(m: SMTModel, networks: NetworksStructure) -> SMTModel:
        Logger.info(messages="Create post-condition constraints")

        num_post_condition: int = len(networks.post_condition)
        last_layer_id: int = networks.num_layers - 1
        
        # Only consider Conjunctive constraints 
        for j in range(len(networks.post_condition[0][0])):
            m._model.add(z3.Sum(networks.post_condition[0][0][j][k] * m.continue_variables[f"x_{last_layer_id}_{k}"] - m.continue_variables[f"s_{last_layer_id}_{k}"] 
                                for k in range(num_post_condition))
                        <= networks.post_condition[0][1][j])

        Logger.info(messages="Post-condition constraints are created")
        return 

    @staticmethod
    def feedforward_networks(m: SMTModel, networks:NetworksStructure) -> SMTModel:
        Logger.info(messages="Create feedforward constraints")

        for k, Nk in enumerate(networks.layer_to_layer):
            for j in range(Nk[1]):
                m._model.add(z3.Sum(networks.matrix_weights[k][i][j] * m.continue_variables[f"x_{k}_{i}"] for i in range(Nk[0])
                                    + networks.vector_bias[k][j]) 
                             == m.continue_variables[f"x_{k+1}_{j}"])
        
        Logger.info(messages="Feedforward constraints are created")

        return m

    @staticmethod
    def relu(m: SMTModel, networks: NetworksStructure) -> SMTModel:
        Logger.info(messages="Create ReLU constraints")

        for k, Nk in enumerate(networks.layer_to_layer):
            if k == 0: continue 

            for nk in range(Nk[0]):
                m._model.add(m.continue_variables[f"s_{k}_{nk}"] == z3.If(
                    m.continue_variables[f"x_{k}_{nk}"] > 0,
                    m.continue_variables[f"x_{k}_{nk}"],
                    0
                ))

        return m

def _create_variables(m: SMTModel, networks: NetworksStructure) -> SMTModel:
    Logger.info(messages="Create variables")

    for k, Nk in  enumerate(networks.layer_to_layer):
        if k == 0: # input layer 
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.continue_variables[name] = z3.Real(name)
        elif k <= networks.num_layers - 2:
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.continue_variables[name] = z3.Real(name)

                name: str = f"s_{k}_{nk}"
                m.continue_variables[name] = z3.Real(name)
        else: # output layer  
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.continue_variables[name] = z3.Real(name)

                name: str = f"s_{k}_{nk}"
                m.continue_variables[name] = z3.Real(name)
            for nk in range(Nk[1]):
                name: str = f"x_{k+1}_{nk}"
                m.continue_variables[name] = z3.Real(name)

                name = f"s_{k+1}_{nk}"
                m.continue_variables[name] = z3.Real(name)

    Logger.info(messages="Variables are created")

    return m

def dump(m: SMTModel) -> None:
    return 

def smt_verifier(solver_name: str, networks: NetworksStructure) -> SMTModel:
    m: SMTModel | None = None 
    if solver_name == "z3":
        m = SMTModel(solver_name=solver_name)
    else:
        raise ValueError(f"Solver {solver_name} is not supported")
    
    _create_variables(m, networks)
    _Constraints.pre_condition(m, networks)
    _Constraints.post_condition(m, networks)
    _Constraints.feedforward_networks(m, networks)
    _Constraints.relu(m, networks)

    m._model.check()

    return m
