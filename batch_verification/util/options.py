from enum import Enum


class InputMergedBy(Enum):
    MEET = "meet"
    JOIN = "join"


class VerificationSolver(Enum):
    GUROBI = "gurobi"
    SCIP = "scip"
    CROWN = "crown"


class RobustnessType(Enum):
    LP_NORM = "linf"
    ROTATION = "rotation"
    BRIGHTNESS = "brightness"
