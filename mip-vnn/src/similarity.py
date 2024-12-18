from typing import List, Any, Dict, Tuple, Set
from functools import partial
from itertools import compress

import numpy as np
import jax
from jax import jit, lax
import jax.numpy as jnp
from jax.experimental.host_callback import call

from utils import read_dataset
from utils.parameters_networks import NetworksStructure
from utils.parameters_networks import DataSet
from utils.log import Logger


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
        reference_data: int = 0
        similarity_data = jnp.argsort(distance_matrix[reference_data])
        Logger.debugging(f"Similarity data: {similarity_data}")

        return similarity_data

    @staticmethod
    def kmeans(
        group: List[int],
        dim_inputs: int,
        num_clusters: int = 2,
    ) -> List[List[int]]:
        result: List[List[int]] = []
        for _ in range(num_clusters):
            result.append([])

        centroids: List[jnp.ndarray] = []
        for _ in range(num_clusters):
            key = jax.random.PRNGKey(0)
            random_centroids: jnp.ndarray = jax.random.uniform(
                key=key, shape=(dim_inputs,)
            )
            centroids.append(random_centroids)

        Logger.debugging(f"Initial Centroids: {centroids}")

        isChanged: bool = True
        while isChanged:
            # compute distance between each data point and centroids
            for each_data in group:
                distance: int = 99999
                closest_centroid: int = -1
                for cid, centroid in enumerate(centroids):
                    d = jnp.linalg.norm(each_data - centroid)
                    if d < distance:
                        distance = d
                        closest_centroid = cid
                if each_data not in result[closest_centroid]:
                    isChanged = True
                    result[closest_centroid].append(each_data)
                    # TODO : remove

        return result

    # // [20240703] this function is not used in this project.
    # //            because I tested 200 data, there is no input will be overlapped for each.
    @staticmethod
    def meet_merging_rule(
        all_data: List[jnp.ndarray], dataset: DataSet
    ) -> List[Tuple[int]]:
        meet_merging_result: Set[List[int]] = set()
        for id, each_data in enumerate(all_data):
            # * What's the most efficient way to test if two ranges overlap?
            # * https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-if-two-ranges-overlap
            isSingleOverlapped: jnp.ndarray = jnp.array(
                [
                    jnp.all(
                        (
                            jnp.maximum(0, each_data2 - dataset.epsilon)
                            <= jnp.minimum(1, each_data + dataset.epsilon)
                        )
                        & (
                            jnp.maximum(0, each_data - dataset.epsilon)
                            <= jnp.minimum(1, each_data2 + dataset.epsilon)
                        )
                    )
                    for id2, each_data2 in enumerate(all_data)
                ],
            )
            # tmp_mergin_result.extend(list(compress(all_data, isSingleOverlapped)))
            tmp_mergin_result: Tuple[int] = tuple(
                jnp.where(isSingleOverlapped)[0].tolist()
            )
            meet_merging_result.add(tmp_mergin_result)

        Logger.debugging(f"Meet merging result: {list(meet_merging_result)}")

        return list(meet_merging_result)

    # // archived code
    # // in testing, lexicographical order is not working to measure the similarity between inputs.
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

    @staticmethod
    def output_vector_similarity(
        all_inference_result: List[Any], distance_matrix: jnp.ndarray
    ) -> Dict[Tuple[int], List[int]]:
        group: Dict[Tuple[int], List[int]] = dict()

        for id, each_inference_result in enumerate(all_inference_result):
            order_index: Tuple[int] = tuple(
                jnp.argsort(each_inference_result).tolist()[0]
            )
            if order_index in group:
                group[order_index].append(id)
            else:
                group[order_index] = [id]

        # * Custom sorting function: sum of distances to all other nodes
        # * order each group by the distance matrix
        for key, value in group.items():

            def sort_key(node):
                return sum(
                    distance_matrix[node][other] for other in value if other != node
                )

            value = sorted(value, key=sort_key)
            group[key] = value

        # * summary
        summary_table: Dict[int, int] = dict()
        for key, value in group.items():
            if len(value) in summary_table:
                summary_table[len(value)] += 1
            else:
                summary_table[len(value)] = 1

        for i in sorted(summary_table):
            Logger.debugging(f"The group size is = {i}, value = {summary_table[i]}")
        for key, value in group.items():
            if 0 in value or 1109 in value:
                Logger.debugging(f"Key: {key}, Value: {value}")

        return group
