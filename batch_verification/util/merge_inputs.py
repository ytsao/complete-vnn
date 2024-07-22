from typing import List, Tuple

import jax.numpy as jnp

from .log import Logger


# ? There is a bug that will lead lb >= ub.
# ? if lb >= ub, that means the result from meet is empty!
# TODO: fix the bug
def meet(all_inputs: List[jnp.ndarray], num_dimensions: int, epsilon: float) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [max(0,v-epsilon) for v in all_inputs[0]]
    each_pixel_ub: List[float] = [min(1,v+epsilon) for v in all_inputs[0]]

    for id, each_input in enumerate(all_inputs):
        if id == 0:
            continue
        each_pixel_lb = [max(each_pixel_lb[i], each_input[i]-epsilon) for i in range(num_dimensions)]
        each_pixel_ub = [min(each_pixel_ub[i], each_input[i]+epsilon) for i in range(num_dimensions)]
    
    assert len(each_pixel_lb) == num_dimensions
    assert len(each_pixel_ub) == num_dimensions

    return each_pixel_lb, each_pixel_ub


def join(all_inputs: List[jnp.ndarray], removed_inputs: List[int] = [], num_dimensions: int = -1, epsilon: float = 0.01) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [99999 for _ in range(num_dimensions)]
    each_pixel_ub: List[float] = [-99999 for _ in range(num_dimensions)]

    for id, each_input in enumerate(all_inputs):
        if id in removed_inputs:
            continue

        each_pixel_lb = [max(0, min(each_pixel_lb[i], each_input[i]-epsilon)) for i in range(num_dimensions)]
        each_pixel_ub = [min(1, max(each_pixel_ub[i], each_input[i]+epsilon)) for i in range(num_dimensions)]

    assert len(each_pixel_lb) == num_dimensions
    assert len(each_pixel_ub) == num_dimensions

    return each_pixel_lb, each_pixel_ub
                                                                                                                               