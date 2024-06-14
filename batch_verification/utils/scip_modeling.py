from typing import List, Tuple, Dict, Any

import pyscipopt as scip

from .mip_modeling import Model
from .mip_modeling import MIPOptimizer

class SCIPModel(MIPOptimizer):
    def __init__(self, solver: Model) -> None:
        self.solver: Model = solver
        self.solver.model = scip.Model()

        return

    def add_variable(self, lb: int, ub: int, vtype: str, name: str) -> None:
        """
        create single decision variable in MIP model.
        """
        if vtype == "B":
            self.solver.binary_variables[name] = self.solver.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
        elif vtype == "I":
            self.solver.integer_variables[name] = self.solver.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
        elif vtype == "C":
            self.solver.continue_variables[name] = self.solver.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)

        return 
    
    def add_objective_function(self, express: Any, sense: str) -> None:
        """
        create objective function in MIP model.
        """
        if type(express) == type(None): return 

        self.solver.model.setObjective(express, sense=sense)

        return 
    
    def add_constraint(self, express: Any, name: str) -> None:
        """
        create single constraint in MIP model.
        """
        self.solver.model.addCons(express, name=name)

        return 
    
    def change_variable_lb(self, variable: Any, lb: int) -> None:
        """
        change the lower bound for specific decision variable.
        """
        self.solver.model.chgVarLb(variable, lb)

        return
    
    def change_variable_ub(self, variable: Any, ub: int) -> None:
        """
        change the upper bound for specific decision variable.
        """
        self.solver.model.chgVarUb(variable, ub)

        return 
    
    def export_lp_file(self, name: str) -> None:
        """
        export the lp file to check whether model is correct or not.
        """
        self.solver.model.writeProblem(f"{name}.lp")

        return 
    
    def optimize(self) -> None:
        """
        solve MIP model
        """
        self.solver.model.optimize()

        return 
    
    def get_constraints(self) -> Any:
        """
        after building MIP model, we can retrive all of constraints from MIP model object.
        """
        self.solver.model.getConss()

        return
    
    def get_constraint_name(self, constraint: Any) -> str:
        """
        after building MIP model, we can retrive the name of specific constraint from MIP model object.
        """

        return constraint.name
    
    def get_primal_solution(self, variable: Any) -> float:
        """
        after solving MIP model, we can retrive the primal solution of specific decision variable from MIP model object.
        """
        
        return self.solver.model.getVal(variable)
    
    def get_dual_solution(self, constraint: Any) -> float:
        """
        after solving, we can get the dual solution from specific constraint.

        PS  your optimization model cannot contain any integer or binary variable.
            Otherwise, you cannot get dual solution.
        """

        return self.solver.model.getDualsolLinear(constraint) 

    def get_solution_status(self) -> str:
        """
        after solving, we can use this function to check the solution status.
        if the solution status is infeasible or unbounded, you might not get the primal/dual solutions.
        """        
        msgdict: dict = {"optimal": "Optimal",
                         "infeasible": "Infeasible", "unbounded": "Unbounded"}

        return msgdict[self.solver.model.getStatus()]