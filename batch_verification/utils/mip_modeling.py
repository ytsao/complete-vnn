from typing import List, Tuple, Dict, Any
from collections import defaultdict
from termcolor import cprint
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class Model:
    solver: str = field(default="scip")
    model: Any = field(default=scip.Model())
    binary_variables: defaultdict[Dict] = field(default_factory=lambda: defaultdict(Dict))
    integer_variables: defaultdict[Dict] = field(default_factory=lambda: defaultdict(Dict))
    continue_variables: defaultdict[Dict] = field(default_factory=lambda: defaultdict(Dict))

    timelimits: int = field(default=60) # default: 1 minute

class MIPOptimizer(ABC):

    @abstractmethod
    def add_variable(self, lb: int, ub: int, vtype: str, name: str) -> None:
        pass

    @abstractmethod
    def add_objective_function(self, express: Any, sense: str) -> None:
        pass

    @abstractmethod
    def add_constraint(self, express: Any, name: str) -> None:
        pass

    @abstractmethod
    def change_variable_lb(self, variable: Dict[str, Any], lb: int) -> None:
        pass

    @abstractmethod
    def change_variable_ub(self, variable: Dict[str, Any], ub: int) -> None:
        pass

    @abstractmethod
    def export_lp_file(self, name: str) -> None:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def get_constraints(self) -> Any:
        pass

    @abstractmethod
    def get_constraint_name(self, constraint: Any) -> str:
        pass

    @abstractmethod
    def get_primal_solution(self, variable: Any) -> float:
        pass

    @abstractmethod
    def get_dual_solution(self, constraint: Any) -> float:
        pass

    @abstractmethod
    def get_solution_status(self) -> str:
        pass