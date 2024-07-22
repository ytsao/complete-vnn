from typing import List, Tuple

import jax.numpy as jnp

class Checker:
    def __init__(self, all_images: List[jnp.ndarray], counter_example: jnp.ndarray, epsilon: float):
        self.epsilon = epsilon
        self.all_images = all_images
        self.ce = counter_example


    def check(self) -> Tuple[int, bool]:
        ce_id: int = -1
        result: bool = False

        for id, each_image in enumerate(self.all_images):
            if jnp.all(jnp.abs(each_image - self.ce) <= self.epsilon):
                result = True
                ce_id = id
                break

        return ce_id, result