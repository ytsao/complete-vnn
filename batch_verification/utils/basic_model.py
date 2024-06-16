from typing import Any

from .mip_modeling import Model
from .scip_modeling import SCIPModel
from .gurobi_modeling import GurobiModel

class _ObjectiveFunction():
    @staticmethod
    def robustness_property(self, m: Any, epsilon: float) -> None:
        return 
    
    @staticmethod
    def safety_property(self, m: Any) -> None:
        return
    
class _Constraints():
    @staticmethod
    def feedforward_networks(self, m: Any) -> None:
        return 
    
    @staticmethod
    def convolutional_networks(self, m: Any) -> None:
        return 
    
    @staticmethod
    def residual_networks(self, m: Any) -> None:
        return

    @staticmethod
    def max_pooling(self, m: Any) -> None:
        return 

    @staticmethod
    def relu(self, m: Any) -> None:
        return 

def build(solver: str) -> None:
    print("start building the model")

    # create model object
    m: Any = None
    if solver == "scip":
        m = SCIPModel(solver=Model())
    else:
        m = GurobiModel(solver=Model(solver="gurobi"))


    # create decision variables


    # create objective function


    # create constraints

    return
