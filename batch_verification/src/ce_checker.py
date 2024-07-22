from typing import List

import jax.numpy as jnp

class Checker:
    def __init__(self, all_images: List[jnp.ndarray], counter_example: jnp.ndarray, epsilon: float):
        self.epsilon = epsilon
        self.all_images = all_images
        self.ce = counter_example


    def check(self) -> bool:
        result: bool = False

        for each_image in self.all_images:
            if jnp.all(jnp.abs(each_image - self.ce) <= self.epsilon):
                result = True
                break

        return result