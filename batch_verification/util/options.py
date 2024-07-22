from enum import Enum

class InputMergedBy(Enum):
    MEET = 1
    JOIN = 2


class VerificationSolver(Enum):
    GUROBI = "gurobi"
    SCIP = "scip"
    CROWN = "crown"
