from typing import List, Tuple

import numpy as np
from jax import jit
import jax.numpy as jnp

from utils import read_dataset
from utils.parameters_networks import NetworksStructure
from utils.parameters_networks import DataSet



class Cluster:

    @staticmethod
    @jit
    def generate_distance_matrix(all_data: List[jnp.ndarray]) -> jnp.ndarray:
        """
        generate distance matrix between all data points.
        """

        num_data: int = len(all_data)
        all_data: jnp.ndarray = jnp.array(all_data)
        # distance_matrix: jnp.ndarray = jnp.zeros((num_data, num_data))

        # for id1, data1 in enumerate(all_data):
        #     for id2, data2 in enumerate(all_data):
        #         if id1 == id2:
        #             continue
        #         else:
        #             distance: float = 0
        #             match distance_measurement:
        #                 case "l1":
        #                     distance = jnp.sum(jnp.abs(data1 - data2))
        #                 case "l2":
        #                     distance = jnp.sqrt(jnp.sum(data1 - data2), 0.5)
        #                 case "linf":
        #                     distance = jnp.max(jnp.abs(data1 - data2))
        #                 case default:
        #                     raise ValueError("distance measurement not supported")
                    
        #             # distance_matrix.at[id1, id2].set(distance) # ! super slow 

        # ? one function to calculate distance matrix (L1 norm)
        # ? this is faster, but haven't checked if it's correct
        # TODO: check if this is correct
        difference: jnp.ndarray = all_data[:, None, :] - all_data[None, :, :]
        distance_matrix = jnp.sum(jnp.abs(difference), axis=-1)
        distance_matrix = distance_matrix.at[jnp.diag_indices(num_data)].set(0)

        return distance_matrix
    

    def mip():
        print("cluster MIP")
        return 
    
    def approximation():
        print("cluster approximation")
        return 
    
    def greedy():
        print("cluster greedy")
        return
    