from typing import List, Dict
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import jax.numpy as jnp
import onnxruntime as ort

from utils.parameters_networks import DataSet
from utils.parameters_networks import NetworksStructure
from utils.read_dataset import load_dataset
from utils.read_dataset import extract_network_structure

from src.cluster import Cluster

from utils.mip_modeling import Model
from utils.scip_modeling import SCIPModel
from utils.gurobi_modeling import GurobiModel


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
    networks: NetworksStructure = extract_network_structure("./utils/benchmarks/onnx/mnist-net_256x6.onnx", "./utils/benchmarks/vnnlib/prop_7_0.03.vnnlib")

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    # session = ort.InferenceSession("./utils/benchmarks/onnx/mnist-net_256x2.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # session = ort.InferenceSession("./utils/benchmarks/onnx/mnist-net_256x2.onnx", providers=["CPUExecutionProvider"])
    session = ort.InferenceSession("./utils/benchmarks/onnx/mnist-net_256x6.onnx", providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

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
        distribution_filtered_test_labels[label].append(filterd_test_images[index])
    for label in range(dataset.num_labels):
        print("label: ", label, " number of images: ", len(distribution_filtered_test_labels[label]))

    # *  ************************  * #
    # *  step 3. cluster analysis
    # *  ************************  * #
    cluster: Cluster = Cluster()
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


if __name__ == "__main__":
    print("batch verification is starting...")
    main()