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
    def generate_l1_distance_matrix(all_data: List[jnp.ndarray]) -> jnp.ndarray:
        """
        generate distance matrix between all data points.
        """

        num_data: int = len(all_data)
        all_data: jnp.ndarray = jnp.array(all_data)

        # ? one function to calculate distance matrix (L1 norm)
        # ? this is faster, but haven't checked if it's correct
        # TODO: check if this is correct
        # * all_data.shape() := (971, 784)
        # * all_data[:, None, :].shape := (971, 1, 784)
        # * all_data[None, :, :].shape := (1, 971, 784)
        difference: jnp.ndarray = all_data[:, None, :] - all_data[None, :, :]
        distance_matrix = jnp.sum(jnp.abs(difference), axis=-1)
        distance_matrix = distance_matrix.at[jnp.diag_indices(num_data)].set(0)

        # ! CANNOT PRINT distance_matrix
        # ? how to print the distance_matrix?
        print(type(distance_matrix))
        print(distance_matrix.at[0].get().at[0].get().addressable_data(0))
        print(distance_matrix.at[0].get().at[1].get())

        return distance_matrix
    

    @staticmethod
    @jit
    def generate_l2_distance_matrix(all_data: List[jnp.ndarray]) -> jnp.ndarray:
        """
        generate distance matrix between all data points.
        """

        num_data: int = len(all_data)
        all_data: jnp.ndarray = jnp.array(all_data)

        difference: jnp.ndarray = all_data[:, None, :] - all_data[None, :, :]
        distance_matrix = jnp.sum(jnp.abs(difference), axis=-1)
        distance_matrix = distance_matrix.at[jnp.diag_indices(num_data)].set(0)

        return distance_matrix
    

    @staticmethod
    @jit
    def generate_linf_distance_matrix(all_data: List[jnp.ndarray]) -> jnp.ndarray:
        """
        generate distance matrix between all data points.
        """

        num_data: int = len(all_data)
        all_data: jnp.ndarray = jnp.array(all_data)
        
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
    