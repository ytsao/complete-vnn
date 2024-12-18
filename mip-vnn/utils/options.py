from enum import Enum

class VerificationSolver(Enum):
    GUROBI = "gurobi"
    SCIP = "scip"

class RobustnessType(Enum):
    LP_NORM = "linf"
    ROTATION = "rotation"  # TODO: implement rotation based robustness verification
    BRIGHTNESS = (
        "brightness"  # TODO: implement brightness based robustness verification
    )
