"""
This module defines the data class and abstract class for building MIP models.
"""

from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pyscipopt

try:
    import gurobipy as gp
except ImportError:
    print("no gurobi installed!!!!!!!!!!!!!!!!!!!!!!!")
    # sys.exit(1)


@dataclass
class MIPModel:
    """
    data class for MIP models, which can be used with different solvers like Gurobi or SCIP.
    """

    solver_name: str = field(default="scip")
    # from gurobi or scip module, only for gurobi_modeling or scip_modeling used
    _model: gp.Model | pyscipopt.Model = field(init=False)
    binary_variables: defaultdict[Dict] = field(
        default_factory=lambda: defaultdict(Dict)
    )
    integer_variables: defaultdict[Dict] = field(
        default_factory=lambda: defaultdict(Dict)
    )
    continue_variables: defaultdict[Dict] = field(
        default_factory=lambda: defaultdict(Dict)
    )

    timelimits: int = field(default=60)  # default: 1 minute


class MIPOptimizer(ABC):
    """
    abstract class for MIP optimizers, which can be used with different solvers like Gurobi or SCIP.
    """

    @abstractmethod
    def add_variable(
        self, lb: int | None, ub: int | None, vtype: str, name: str
    ) -> None:
        """
        add a variable to the MIP model.
        """

    @abstractmethod
    def add_objective_function(self, express: Any, sense: str) -> None:
        """
        add an objective function to the MIP model.
        """

    @abstractmethod
    def add_constraint(self, express: Any, name: str) -> None:
        """
        add a constraint to the MIP model.
        """

    @abstractmethod
    def add_max_constraint(
        self, max_variable: Any, variables: List[Any], name: str
    ) -> None:
        """
        add a max constraint to the MIP model.
        """

    @abstractmethod
    def change_variable_lb(self, variable: Dict[str, Any], lb: int) -> None:
        """
        change the lower bound of a variable in the MIP model.
        """

    @abstractmethod
    def change_variable_ub(self, variable: Dict[str, Any], ub: int) -> None:
        """
        change the upper bound of a variable in the MIP model.
        """

    @abstractmethod
    def export_lp_file(self, name: str) -> None:
        """
        export the MIP model to a .lp file.
        """

    @abstractmethod
    def optimize(self) -> None:
        """
        optimize the MIP model.
        """

    @abstractmethod
    def get_constraints(self) -> Any:
        """
        get the constraints of the MIP model.
        """

    @abstractmethod
    def get_constraint_name(self, constraint: Any) -> str:
        """
        get the name of a constraint in the MIP model.
        """

    @abstractmethod
    def get_primal_solution(self, variable: Any) -> float:
        """
        get the primal solution of a variable in the MIP model.
        """

    @abstractmethod
    def get_dual_solution(self, constraint: Any) -> float:
        """
        get the dual solution of a constraint in the MIP model.
        """

    @abstractmethod
    def get_solution_status(self) -> str:
        """
        get the solution status of the MIP model.
        """
