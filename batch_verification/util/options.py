from enum import Enum


class InputMergedBy(Enum):
    MEET = "meet"
    JOIN = "join"


class VerificationSolver(Enum):
    GUROBI = "gurobi"
    SCIP = "scip"
    CROWN = "crown"
    BOX = "box"  # TODO: implement box based verifier by apronpy
    ZONOTOPE = "zonotope"  # TODO: implement zonotope based verifier by apronpy


class RobustnessType(Enum):
    LP_NORM = "linf"
    ROTATION = "rotation"  # TODO: implement rotation based robustness verification
    BRIGHTNESS = (
        "brightness"  # TODO: implement brightness based robustness verification
    )
