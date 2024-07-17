from src.mip import mip_verifier
from src.cluster import Cluster
from utils.gurobi_modeling import GurobiModel
from utils.scip_modeling import SCIPModel
from utils.mip_modeling import Model
from utils.read_dataset import extract_network_structure
from utils.read_dataset import load_dataset
from utils.parameters_networks import NetworksStructure
from utils.parameters_networks import DataSet
from utils.write_vnnlib import write_vnnlib, write_vnnlib_merge, export_vnnlib
from matplotlib import pyplot as plt
import onnxruntime as ort
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from typing import List, Dict
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# * algorithms


# TODO: implement verification algorithm in different ways
def verify() -> None:
    return


def main() -> str:
    """
    Batch verification algorithm: 
        step 0: read the input files (onnx, image files)

        step 1: filter correct classification results from testing dataset.
        step 2: based on each label, separate into different groups.
        step 3: cluster analysis
                - goal: group similar data.
                        based on pre-defined metric.
        step 4: define abstract domain, 𝒜 for each cluster.
        step 5: Verify 𝒜 -> r.
                    if r is UNSAT: STOP
                    else: 
                        divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
                        back to step 5 to verify each 𝒜_i 

    Purpose:
        1. Reduce training data to train robust NN.
        2. Accelerate overall verification process.
        3. When we retrain NN, we have to verify the property again, so this might help to reduce the cost.
    """
    result: str = "UNSAT"

    # *  ************************  * #
    # *  step 0. read the input files
    # *  ************************  * #
    dataset: DataSet = load_dataset("mnist")
    onnx_filename: str = "./utils/benchmarks/onnx/mnist-net_256x2.onnx"
    vnnlib_filename: str = "./utils/benchmarks/vnnlib/prop_7_0.03.vnnlib"
    # networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
    print("step 0: read the input files")

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    session = ort.InferenceSession(
        onnx_filename, providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("step 1: filter correct classification results from testing dataset")

    # *  =======================  * #
    # *  training dataset part
    # *  =======================  * #
    # filter_train_images: List[jnp.ndarray] = []
    # filter_train_labels: List[int] = []
    # num_train_images: int = len(dataset.train_images)
    # print("number of images in training dataset: ", num_train_images)
    # for data_id in range(num_train_images):
    #     inference_result = session.run([output_name],
    #                                    {input_name: dataset.train_images[data_id].astype(jnp.float32).reshape(1, dataset.num_pixels, 1)})[0]
    #     inference_label = jnp.argmax(inference_result)
    #     true_label = dataset.train_labels[data_id]
    #     if inference_label == true_label:
    #         filter_train_images.append(dataset.train_images[data_id])
    #         filter_train_labels.append(true_label)
    # print("filter_train_images: ", len(filter_train_images))
    # print("filter_train_labels: ", len(filter_train_labels))

    # *  =======================  * #
    # *  testing dataset part
    # *  =======================  * #
    filterd_test_images: List[jnp.ndarray] = []
    filterd_test_labels: List[int] = []
    num_test_images: int = len(dataset.test_images)
    print("number of images in testing dataset: ", num_test_images)
    for data_id in range(num_test_images):
        inference_result = session.run([output_name],
                                       {input_name: dataset.test_images[data_id].astype(jnp.float32).reshape(1, dataset.num_pixels, 1)})[0]
        inference_label = jnp.argmax(inference_result)
        true_label = dataset.test_labels[data_id]
        if inference_label == true_label:
            filterd_test_images.append(dataset.test_images[data_id])
            filterd_test_labels.append(true_label)

    print("filterd_test_images: ", len(filterd_test_images))
    print("filterd_test_labels: ", len(filterd_test_labels))
    print("test accuracy: ", len(filterd_test_images) / num_test_images)

    # *  ************************  * #
    # *  step 2. based on each label, separate into different groups.
    # *  ************************  * #
    distribution_filtered_test_labels: Dict[int, List[jnp.ndarray]] = {}
    for label in range(dataset.num_labels):
        distribution_filtered_test_labels[label] = []
    for index, label in enumerate(filterd_test_labels):
        distribution_filtered_test_labels[label].append(
            filterd_test_images[index])
    for label in range(dataset.num_labels):
        print("label: ", label, " number of images: ", len(
            distribution_filtered_test_labels[label]))

    # *  ************************  * #
    # *  step 3. cluster analysis
    # *  ************************  * #
    type_of_property: str = "meet"
    vnnlib_filename: str = ""
    test_true_label: int = 0        # YES: 0,    Y3(1)
    epsilon: float = 0.03
    count: int = 0
    distance_matrix: jnp.ndarray
    print(
        f"number of testing images: {len(distribution_filtered_test_labels[test_true_label])}")
    distance_matrix = Cluster.generate_distance_matrix(
        all_data=distribution_filtered_test_labels[test_true_label], distance_type="l2", chunk_size=100)  # ! out of memory

    # * find the similar data
    similarity_data: List[int] = Cluster.greedy(
        distance_matrix=distance_matrix, num_clusters=2)
    print("similarity_data: ", similarity_data)

    # TODO: Test merge abstract domain if possible
    test_set_inputs: List[int] = similarity_data
    # test_set_inputs: List[int] = [similarity_data[0], similarity_data[1], similarity_data[2], similarity_data[3]]
    # test_set_inputs: List[int] = [similarity_data[3], similarity_data[len(similarity_data) - 1]]
    for id in test_set_inputs:
        if type_of_property == "meet":
            if id == test_set_inputs[0]:
                # if True:
                vnnlib_filename: str = write_vnnlib(data=distribution_filtered_test_labels[test_true_label][id],
                                                    data_id=id,
                                                    num_classes=dataset.num_labels,
                                                    true_label=test_true_label,
                                                    epsilon=epsilon)
            else:
                os.remove(vnnlib_filename)
                vnnlib_filename: str = write_vnnlib_merge(networks=networks,
                                                          data=distribution_filtered_test_labels[test_true_label][id],
                                                          data_id=id,
                                                          num_classes=dataset.num_labels,
                                                          true_label=test_true_label,
                                                          epsilon=epsilon)
        elif type_of_property == "join":
            if id == test_set_inputs[0]:
                # if True:
                vnnlib_filename: str = write_vnnlib(data=distribution_filtered_test_labels[test_true_label][id],
                                                    data_id=id,
                                                    num_classes=dataset.num_labels,
                                                    true_label=test_true_label,
                                                    epsilon=epsilon)
            else:
                os.remove(vnnlib_filename)
                vnnlib_filename: str = write_vnnlib_merge(networks=networks,
                                                          data=distribution_filtered_test_labels[test_true_label][id],
                                                          data_id=id,
                                                          num_classes=dataset.num_labels,
                                                          true_label=test_true_label,
                                                          epsilon=epsilon)

        # * update networks by new vnnlib
        networks: NetworksStructure = extract_network_structure(
            onnx_filename, vnnlib_filename)
        if id != test_set_inputs[len(test_set_inputs) - 1]:
            continue

        print("------------------------------------------------------")
        print("start verifying ...")
        m: SCIPModel | GurobiModel = mip_verifier(
            solver_name="scip", networks=networks)
        counter_example: List[float] = []
        if m.get_solution_status() == "Infeasible":
            print("UNSAT")
        else:
            for k, Nk in enumerate(networks.layer_to_layer):
                if k == 0:
                    for nk in range(Nk[0]):
                        name: str = f"x_{k}_{nk}"
                        variable = m.solver.continue_variables[name]
                        counter_example.append(m.get_primal_solution(variable))
                else:
                    break
            print("SAT")

        # plt.imshow(distribution_filtered_test_labels[test_true_label][id].reshape(28, 28), cmap='gray')
        counter_example_image = np.array(counter_example).reshape(28, 28)
        plt.imshow(counter_example_image, cmap='gray')
        plt.show()

    # for index, image in enumerate(distribution_filtered_test_labels[test_true_label]):
    #     if index == 0:
    #         vnnlib_filename: str = write_vnnlib(data=image,
    #                                     data_id=index,
    #                                     num_classes=dataset.num_labels,
    #                                     true_label=test_true_label,
    #                                     epsilon=epsilon)
    #         print("vnnlib_filename: ", vnnlib_filename)
    #     else:
    #         vnnlib_filename: str = write_vnnlib_merge(networks=networks,
    #                                                     data=image,
    #                                                     data_id=index,
    #                                                     num_classes=dataset.num_labels,
    #                                                     true_label=test_true_label,
    #                                                     epsilon=epsilon)
    #         print("merged_vnnlib_filename: ", vnnlib_filename)

    #     networks:NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
    #     m: SCIPModel | GurobiModel = mip_verifier(solver_name="gurobi", networks=networks)
    #     if m.get_solution_status() == "Infeasible":
    #         print("UNSAT")
    #     else:
    #         print("SAT")

    #     if count == 2:
    #         break
    #     count += 1

    # TODO: implement cluster analysis

    # *  ************************  * #
    # *  step 4. define abstract domain, 𝒜 for each cluster.
    # *  ************************  * #
    # TODO: implement abstract domain transformer for interval, zonotope.

    # *  ************************  * #
    # *  step 5. Verify 𝒜 -> r.
    # *     if r is UNSAT: STOP
    # *     else:
    # *         divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
    # *         back to step 5 to verify each 𝒜_i
    # *  ************************  * #
    # TODO: implement verification algorithm

    return result


def release() -> str:
    """
    Batch verification algorithm: 
        step 0: read the input files (onnx, image files)

        step 1: filter correct classification results from testing dataset.
        step 2: based on each label, separate into different groups.
        step 3: cluster analysis
                - goal: group similar data.
                        based on pre-defined metric.
        step 4: define abstract domain, 𝒜 for each cluster.
        step 5: Verify 𝒜 -> r.
                    if r is UNSAT: STOP
                    else: 
                        divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
                        back to step 5 to verify each 𝒜_i 

    Purpose:
        1. Reduce training data to train robust NN.
        2. Accelerate overall verification process.
        3. When we retrain NN, we have to verify the property again, so this might help to reduce the cost.
    """
    result: str = "UNSAT"

    # *  ************************  * #
    # *  step 0. read the input files
    # *  ************************  * #
    dataset: DataSet = load_dataset("mnist")
    onnx_filename: str = "./utils/benchmarks/onnx/mnist-net_256x2.onnx"
    vnnlib_filename: str = "./utils/benchmarks/vnnlib/prop_7_0.03.vnnlib"
    # networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
    print("step 0: read the input files")

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    session = ort.InferenceSession(
        onnx_filename, providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("step 1: filter correct classification results from testing dataset")

    # *  =======================  * #
    # *  training dataset part
    # *  =======================  * #
    # filter_train_images: List[jnp.ndarray] = []
    # filter_train_labels: List[int] = []
    # num_train_images: int = len(dataset.train_images)
    # print("number of images in training dataset: ", num_train_images)
    # for data_id in range(num_train_images):
    #     inference_result = session.run([output_name],
    #                                    {input_name: dataset.train_images[data_id].astype(jnp.float32).reshape(1, dataset.num_pixels, 1)})[0]
    #     inference_label = jnp.argmax(inference_result)
    #     true_label = dataset.train_labels[data_id]
    #     if inference_label == true_label:
    #         filter_train_images.append(dataset.train_images[data_id])
    #         filter_train_labels.append(true_label)
    # print("filter_train_images: ", len(filter_train_images))
    # print("filter_train_labels: ", len(filter_train_labels))

    # *  =======================  * #
    # *  testing dataset part
    # *  =======================  * #
    filterd_test_images: List[jnp.ndarray] = []
    filterd_test_labels: List[int] = []
    num_test_images: int = len(dataset.test_images)
    print("number of images in testing dataset: ", num_test_images)
    for data_id in range(num_test_images):
        inference_result = session.run([output_name],
                                       {input_name: dataset.test_images[data_id].astype(jnp.float32).reshape(1, dataset.num_pixels, 1)})[0]
        inference_label = jnp.argmax(inference_result)
        true_label = dataset.test_labels[data_id]
        if inference_label == true_label:
            filterd_test_images.append(dataset.test_images[data_id])
            filterd_test_labels.append(true_label)

    print("filterd_test_images: ", len(filterd_test_images))
    print("filterd_test_labels: ", len(filterd_test_labels))
    print("test accuracy: ", len(filterd_test_images) / num_test_images)

    # *  ************************  * #
    # *  step 2. based on each label, separate into different groups.
    # *  ************************  * #
    distribution_filtered_test_labels: Dict[int, List[jnp.ndarray]] = {}
    for label in range(dataset.num_labels):
        distribution_filtered_test_labels[label] = []
    for index, label in enumerate(filterd_test_labels):
        distribution_filtered_test_labels[label].append(
            filterd_test_images[index])
    for label in range(dataset.num_labels):
        print("label: ", label, " number of images: ", len(
            distribution_filtered_test_labels[label]))

    # *  ************************  * #
    # *  step 3. cluster analysis
    # *  ************************  * #
    type_of_property: str = "meet"
    vnnlib_filename: str = ""
    test_true_label: int = 0        # YES: 0,    Y3(1)
    epsilon: float = 0.03
    count: int = 0
    distance_matrix: jnp.ndarray
    print(
        f"number of testing images: {len(distribution_filtered_test_labels[test_true_label])}")
    distance_matrix = Cluster.generate_distance_matrix(
        all_data=distribution_filtered_test_labels[test_true_label], distance_type="l2", chunk_size=100)  # ! out of memory

    # * find the similar data
    similarity_data: List[int] = Cluster.greedy(
        distance_matrix=distance_matrix, num_clusters=2)
    print("similarity_data: ", similarity_data)

    # TODO: Test merge abstract domain if possible
    test_set_inputs: List[int] = similarity_data
    # test_set_inputs: List[int] = [similarity_data[0], similarity_data[1], similarity_data[2], similarity_data[3]]
    # test_set_inputs: List[int] = [similarity_data[3], similarity_data[len(similarity_data) - 1]]
    each_pixel_lb: List[float] = [99999 for _ in range(dataset.num_pixels)]
    each_pixel_ub: List[float] = [-99999 for _ in range(dataset.num_pixels)]
    for id in test_set_inputs:
        if type_of_property == "meet":
            each_pixel_lb = [min(each_pixel_lb[i], distribution_filtered_test_labels[test_true_label][id][i])
                             for i in range(dataset.num_pixels)]
            each_pixel_ub = [max(each_pixel_ub[i], distribution_filtered_test_labels[test_true_label][id][i])
                             for i in range(dataset.num_pixels)]
        elif type_of_property == "join":
            each_pixel_lb = [max(each_pixel_lb[i], distribution_filtered_test_labels[test_true_label][id][i]) if each_pixel_lb[i]
                             != 99999 else distribution_filtered_test_labels[true_label][id][i] - epsilon for i in range(dataset.num_pixels)]
            each_pixel_ub = [min(each_pixel_ub[i], distribution_filtered_test_labels[test_true_label][id][i]) if each_pixel_ub[i]
                             != -99999 else distribution_filtered_test_labels[true_label][id][i] + epsilon for i in range(dataset.num_pixels)]

    # * update networks by new vnnlib
    vnnlib_filename = export_vnnlib(lb=each_pixel_lb, ub=each_pixel_ub,
                                    num_classes=dataset.num_labels, true_label=test_true_label, epsilon=epsilon)
    networks: NetworksStructure = extract_network_structure(
        onnx_filename, vnnlib_filename)
    print(f"vnnlib_filename: {vnnlib_filename}")

    print("------------------------------------------------------")
    print("start verifying ...")
    m: SCIPModel | GurobiModel = mip_verifier(
        solver_name="scip", networks=networks)
    counter_example: List[float] = []
    if m.get_solution_status() == "Infeasible":
        print("UNSAT")
    else:
        for k, Nk in enumerate(networks.layer_to_layer):
            if k == 0:
                for nk in range(Nk[0]):
                    name: str = f"x_{k}_{nk}"
                    variable = m.solver.continue_variables[name]
                    counter_example.append(m.get_primal_solution(variable))
            else:
                break
        print("SAT")

    # plt.imshow(distribution_filtered_test_labels[test_true_label][id].reshape(28, 28), cmap='gray')
    counter_example_image = np.array(counter_example).reshape(28, 28)
    plt.imshow(counter_example_image, cmap='gray')
    plt.show()

    # TODO: implement cluster analysis

    # *  ************************  * #
    # *  step 4. define abstract domain, 𝒜 for each cluster.
    # *  ************************  * #
    # TODO: implement abstract domain transformer for interval, zonotope.

    # *  ************************  * #
    # *  step 5. Verify 𝒜 -> r.
    # *     if r is UNSAT: STOP
    # *     else:
    # *         divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
    # *         back to step 5 to verify each 𝒜_i
    # *  ************************  * #
    # TODO: implement verification algorithm

    return "released version"


if __name__ == "__main__":
    print("batch verification is starting...")
    # main()
    release()
    # verify()
