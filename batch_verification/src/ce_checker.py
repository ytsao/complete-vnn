from typing import List, Tuple

import jax.numpy as jnp


class Checker:
    def __init__(
        self,
        all_inputs: List[jnp.ndarray],
        counter_example: jnp.ndarray,
        epsilon: float,
    ):
        self.epsilon = epsilon
        self.all_inputs = all_inputs
        self.ce = counter_example

    def check(self) -> Tuple[int, bool]:
        ce_id: int = -1
        result: bool = False

        for id, each_input in enumerate(self.all_inputs):
            if jnp.all(
                jnp.round(jnp.abs(jnp.subtract(self.ce, each_input)), 5) <= self.epsilon
            ):
                result = True
                ce_id = id
                break

        # for id, each_input in enumerate(self.all_inputs):
        #     for i in range(len(each_input)):
        #         print(
        #             f"ce: {self.ce[i]}, each_input: {each_input[i]}, diff = {abs(self.ce[i] - each_input[i])}"
        #         )
        #         if abs(self.ce[i] - each_input[i]) > self.epsilon:
        #             result = False
        #             ce_id = -1
        #             break
        #         else:
        #             result = True
        #             ce_id = id

        return ce_id, result
