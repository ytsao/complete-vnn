from typing import List, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field

import z3 

@dataclass 
class SMTModel:
    solver_name: str = field(default="z3")
    _model: z3.Solver = field(init=False)
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