from typing import Dict, List
from collections import defaultdict

import jax.numpy as jnp 

class Results:
    def __init__(self):
        self.counter_examples: Dict[int, List[jnp.ndarray]] = defaultdict
    
    @staticmethod
    def add(self, ce: jnp.ndarray, label: int) -> None:
        self.counter_examples[label].append(ce)