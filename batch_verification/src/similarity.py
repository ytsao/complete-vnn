from typing import List, Tuple
from functools import partial

import numpy as np
from jax import jit, lax
import jax.numpy as jnp
from jax.experimental.host_callback import call

from util import read_dataset
from util.parameters_networks import NetworksStructure
from util.parameters_networks import DataSet
from util.log import Logger


class Similarity:

    # // Example of using jit and jnp.ndarray to compute distance matrix and print it out.
    # // [20240703] this function is not used in this project.
    # //            it is replaced by generate_distance_matrix() function.
    # @staticmethod
    # @jit
    # def _generate_l1_distance_matrix(all_data: List[jnp.ndarray]) -> jnp.ndarray:
    #     """
    #     generate L1-norm distance matrix between all data points.
    #     """

    #     num_data: int = len(all_data)
    #     all_data: jnp.ndarray = jnp.array(all_data)

    #     # # * all_data.shape() := (971, 784)
    #     # # * all_data[:, None, :].shape := (971, 1, 784)
    #     # # * all_data[None, :, :].shape := (1, 971, 784)
    #     # difference: jnp.ndarray = all_data[:, None, :] - all_data[None, :, :]
    #     # print("difference: ")
    #     # distance_matrix = jnp.sum(jnp.abs(difference), axis=-1)
    #     # print("sum")
    #     # # distance_matrix = distance_matrix.at[jnp.diag_indices(num_data)].set(0)
    #     A: jnp.ndarray = all_data[:, None, :]
    #     B: jnp.ndarray = all_data[None, :, :]
    #     # distance_matrix: jnp.ndarray = jnp.einsum('ijk', 'ijk->ij', A-B, A-B)

    #     # ? how to print the distance_matrix?
    #     # * solution : https://stackoverflow.com/questions/71548823/how-to-print-with-jax
    #     # call(lambda x: print(x), distance_matrix)

    #     # assert distance_matrix.shape == (num_data, num_data), "L1-norm distance matrix shape is wrong"

    #     distance_matrix: jnp.ndarray = jnp.zeros((len(all_data), len(all_data)))

    #     return distance_matrix

    @staticmethod
    def generate_distance_matrix(
        all_data: List[jnp.ndarray], distance_type: str = "l1", chunk_size: int = 100
    ) -> jnp.ndarray:
        @jit
        def compute_l0_distance(difference: jnp.ndarray) -> jnp.ndarray:
            return jnp.count_nonzero(difference, axis=-1)

        @jit
        def compute_l1_distance(difference: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(jnp.abs(difference), axis=-1)

        @jit
        def compute_l2_distance(difference: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(difference), axis=-1))

        @jit
        def compute_linf_distance(difference: jnp.ndarray) -> jnp.ndarray:
            return jnp.max(jnp.abs(difference), axis=-1)

        num_data: int = len(all_data)
        distance_matrix: np.ndarray = np.zeros((num_data, num_data))
        all_data: jnp.ndarray = jnp.array(all_data)

        for i in range(0, num_data, chunk_size):
            for j in range(0, num_data, chunk_size):
                chunk_dataA: jnp.ndarray = all_data[i : i + chunk_size]
                chunk_dataB: jnp.ndarray = all_data[j : j + chunk_size]

                differences: jnp.ndarray = (
                    chunk_dataA[:, None, :] - chunk_dataB[None, :, :]
                )

                if distance_type == "l1":
                    distance: jnp.ndarray = compute_l1_distance(differences)
                elif distance_type == "l2":
                    distance: jnp.ndarray = compute_l2_distance(differences)
                elif distance_type == "linf":
                    distance: jnp.ndarray = compute_linf_distance(differences)
                else:
                    raise ValueError("distance type is not supported")

                distance_matrix[i : i + chunk_size, j : j + chunk_size] = np.array(
                    distance
                )
        np.fill_diagonal(distance_matrix, 0)

        return jnp.array(distance_matrix)

    @staticmethod
    def greedy(distance_matrix: jnp.ndarray) -> List[int]:
        """
        Build the lattice for the given data points.

        this version:
            - find the closest two data points and merge them into one cluster.
            - default data is first one.
        """
        # TODO: build the lattice for the given data points.
        reference_data: int = 0
        similarity_data = jnp.argsort(distance_matrix[reference_data])
        Logger.debugging(f"Similarity Data: {similarity_data[1]}")

        reference_data: int = 0
        visited_data: List[int] = np.zeros(len(distance_matrix), dtype=bool)
        next_data: int = np.zeros(len(distance_matrix), dtype=int)
        while visited_data[reference_data] == False:
            visited_data[reference_data] = True
            minimum_distance: int = np.inf
            minimum_index: int = -1
            for id, value in enumerate(distance_matrix[reference_data]):
                if visited_data[id] == False and value < minimum_distance:
                    minimum_distance = value
                    minimum_index = id
            if minimum_index == -1:
                reference_data = -1
                for id, value in enumerate(visited_data):
                    if value == False:
                        reference_data = id
                        break
                if reference_data == -1:
                    break
            else:
                next_data[reference_data] = minimum_index
                reference_data = minimum_index


        return next_data

    @staticmethod
    def lexicgraphical_order(all_data: List[jnp.ndarray]) -> List[int]:
        """
        Build the lattice for the given data points.
        Sorting Algorithm: bubble sort (testing)

        Suppose:
            data is a square matrix.
        """

        def bubble_sort(lex_order_result: List[int]) -> List[int]:
            num_dims: int = len(all_data[0])
            for i in range(len(all_data)):
                for j in range(i + 1, len(all_data)):
                    for k in range(num_dims):
                        if all_data[i][k] == all_data[j][k]:
                            continue
                        elif all_data[i][k] > all_data[j][k]:
                            all_data[i], all_data[j] = all_data[j], all_data[i]
                            lex_order_result[i], lex_order_result[j] = (
                                lex_order_result[j],
                                lex_order_result[i],
                            )
                            break
                        else:
                            break
            return lex_order_result

        def heapify(lex_order_result: List[int], n: int, i: int):
            largest: int = i
            l: int = 2 * i + 1
            r: int = 2 * i + 2

            if l < n:
                for k in range(len(all_data[i])):
                    if all_data[largest][k] == all_data[l][k]:
                        continue
                    elif all_data[largest][k] < all_data[l][k]:
                        largest = l
                        break
                    else:
                        break
            if r < n:
                for k in range(len(all_data[i])):
                    if all_data[largest][k] == all_data[r][k]:
                        continue
                    elif all_data[largest][k] < all_data[r][k]:
                        largest = r
                        break
                    else:
                        break
            if largest != i:
                all_data[i], all_data[largest] = all_data[largest], all_data[i]
                lex_order_result[i], lex_order_result[largest] = (
                    lex_order_result[largest],
                    lex_order_result[i],
                )
                heapify(lex_order_result, n, largest)

            return

        def heap_sort(lex_order_result: List[int]) -> List[int]:
            n: int = len(all_data)
            for i in range(n // 2 - 1, -1, -1):
                heapify(lex_order_result, n, i)
            for i in range(n - 1, 0, -1):
                all_data[i], all_data[0] = all_data[0], all_data[i]
                lex_order_result[i], lex_order_result[0] = (
                    lex_order_result[0],
                    lex_order_result[i],
                )
                heapify(lex_order_result, i, 0)

            return lex_order_result

        # for a, b, c, d in zip(all_data[0], all_data[1], all_data[2], all_data[3]):
        #     if a == 0 and b == 0 and c == 0 and d == 0:
        #         continue
        #     Logger.debugging(f"Data1: {a}, \t Data2: {b}, \t Data3: {c}, \t Data4: {d}")

        lex_order_result: List[int] = bubble_sort(list(range(len(all_data))))
        lex_order_result2: List[int] = heap_sort(lex_order_result)

        for a, b in zip(lex_order_result, lex_order_result2):
            assert a == b, "Lexicographical order is wrong"

        assert len(all_data) == len(lex_order_result), "Lexicographical order is wrong"

        return lex_order_result
