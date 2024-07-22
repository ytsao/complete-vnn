from typing import List, Tuple

import jax.numpy as jnp

from .parameters_networks import DataSet
from .log import Logger

def meet(all_inputs: List[jnp.ndarray], dataset: DataSet, epsilon: float) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [99999 for _ in range(dataset.num_pixels)]
    each_pixel_ub: List[float] = [-99999 for _ in range(dataset.num_pixels)]

    for each_input in all_inputs:
        each_pixel_lb = [min(each_pixel_lb[i], max(0, each_input[i]-epsilon)) for i in range(dataset.num_pixels)]
        each_pixel_ub = [max(each_pixel_ub[i], min(1, each_input[i]+epsilon)) for i in range(dataset.num_pixels)]

    return each_pixel_lb, each_pixel_ub


# ? There is a bug that will lead lb >= ub.
# TODO: fix the bug
def join(all_inputs: List[jnp.ndarray], dataset: DataSet, epsilon: float) -> Tuple[List[float], List[float]]:
    each_pixel_lb: List[float] = [99999 for _ in range(dataset.num_pixels)]
    each_pixel_ub: List[float] = [-99999 for _ in range(dataset.num_pixels)]

    for each_input in all_inputs:
        each_pixel_lb = [max(each_pixel_lb[i], each_input[i]-epsilon) if each_pixel_lb[i] != 99999 else each_input[i]-epsilon for i in range(dataset.num_pixels)]
        each_pixel_ub = [min(each_pixel_ub[i], each_input[i]+epsilon) if each_pixel_ub[i] != -99999 else each_input[i]+epsilon for i in range(dataset.num_pixels)]
            
    return each_pixel_lb, each_pixel_ub
                                                                                                                               