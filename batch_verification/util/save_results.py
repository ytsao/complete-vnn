from typing import Dict, List

import jax.numpy as jnp 

class Results:
    counter_examples: Dict[int, List[jnp.ndarray]] = {}
    unsatisfiable_inputs: Dict[int, List[int]] = {}
    

    @staticmethod
    def add(ce_id: int, ce: jnp.ndarray, label: int) -> None:
        if label not in Results.counter_examples:
            Results.counter_examples[label] = []
            Results.unsatisfiable_inputs[label] = []
        Results.unsatisfiable_inputs[label].append(ce_id)
        Results.counter_examples[label].append(ce)
    

    @staticmethod
    def get_unsatisfiable_inputs(label: int) -> List[int]:
        if label not in Results.unsatisfiable_inputs:
            return []
        return Results.unsatisfiable_inputs[label]