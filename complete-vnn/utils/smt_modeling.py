"""
This module defines the SMT-based model for neural network verification.
"""

from typing import Dict
from collections import defaultdict
from dataclasses import dataclass, field

import z3


@dataclass
class SMTModel:
    """
    SMTModel class to create a Z3 solver model for verification.
    """

    solver_name: str = field(default="z3")
    model: z3.Solver = z3.Solver()
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
