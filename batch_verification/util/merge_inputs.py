from typing import List, Tuple

import jax.numpy as jnp

from .parameters_networks import DataSet
from .log import Logger


# ? There is a bug that will lead lb >= ub.
# ? if lb >= ub, that means the result from meet is empty!
# TODO: fix the bug
def meet(all_inputs: List[jnp.ndarray], dataset: DataSet, epsilon: float) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [max(0,v-epsilon) for v in all_inputs[0]]
    each_pixel_ub: List[float] = [min(1,v+epsilon) for v in all_inputs[0]]

    for id, each_input in enumerate(all_inputs):
        if id == 0:
            continue
        each_pixel_lb = [max(each_pixel_lb[i], each_input[i]-epsilon) for i in range(dataset.num_pixels)]
        each_pixel_ub = [min(each_pixel_ub[i], each_input[i]+epsilon) for i in range(dataset.num_pixels)]
    
    assert len(each_pixel_lb) == dataset.num_pixels
    assert len(each_pixel_ub) == dataset.num_pixels

    return each_pixel_lb, each_pixel_ub


def join(all_inputs: List[jnp.ndarray], dataset: DataSet, epsilon: float) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [99999 for _ in range(dataset.num_pixels)]
    each_pixel_ub: List[float] = [-99999 for _ in range(dataset.num_pixels)]

    for each_input in all_inputs:
        each_pixel_lb = [max(0, min(each_pixel_lb[i], each_input[i]-epsilon)) for i in range(dataset.num_pixels)]
        each_pixel_ub = [min(1, max(each_pixel_ub[i], each_input[i]+epsilon)) for i in range(dataset.num_pixels)]

    assert len(each_pixel_lb) == dataset.num_pixels
    assert len(each_pixel_ub) == dataset.num_pixels

    return each_pixel_lb, each_pixel_ub
                                                                                                                               