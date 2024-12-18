from collections import defaultdict
from typing import List, Dict, Tuple
import time
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # disable information and warning from tensorflow
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import argparse

from matplotlib import pyplot as plt
import onnxruntime as ort
import numpy as np
import jax.numpy as jnp

from src.mip import mip_verifier
from src.similarity import Similarity

from utils.gurobi_modeling import GurobiModel
from utils.scip_modeling import SCIPModel
from utils.mip_modeling import Model
from utils.read_dataset import extract_network_structure, load_dataset
from utils.read_results import get_ce
from utils.parameters_networks import NetworksStructure, DataSet
from utils.write_vnnlib import (
    write_vnnlib,
    export_vnnlib,
)
from utils.options import VerificationSolver, RobustnessType
from utils.save_results import Results
from utils.log import Logger


# global configuration
T: int = 0 # true label


# TODO: implement verification algorithm in different ways
def verify(
    solver: VerificationSolver,
    dataset: DataSet,
    input: jnp.ndarray,
) -> str:
    """
    Verification algorithm:

    Support: MIP (SCIP, Gurobi), CROWN
    """
    result: str = "UNSAT"
    vnnlib_filename: str = write_vnnlib(data=input, 
                                        num_classes=10,
                                        true_label=T,
                                        epsilon=dataset.epsilon)
    networks: NetworksStructure = extract_network_structure(
        onnx_file_path=dataset.onnx_filename, vnnlib_file_path=vnnlib_filename
    )
    
    if solver is VerificationSolver.SCIP or solver is VerificationSolver.GUROBI:
        Logger.info(messages=f"Verification Algorithm is MIP solver ({solver})")
        
        m: SCIPModel | GurobiModel = mip_verifier(solver_name=solver, networks=networks)
        result = "UNSAT" if m.get_solution_status() == "Infeasible" else "SAT"
    elif solver is VerificationSolver.BOX:
        Logger.info(messages="Verification Algorithm is BOX")
        Logger.info(messages="[ERROR] Box verifier is not able to use in this version.")
    if result == "UNSAT":
        Logger.info(messages="UNSAT, New template generated!")
        return "UNSAT"
    else:
        Logger.info(messages="SAT")
        # * Testing checker for counter-example found by verifier
        counter_example: List[float] = get_ce(
            solver=solver, networks=networks, filename="./test_cex.txt", m=m
        )
        counter_example: jnp.ndarray = jnp.array(counter_example)

    return result


def _execute(solver: VerificationSolver) -> None:
    """
    MIP verification
    
    Build a mixed-integer programming model to verify neural networks.
    
    step 0. read the input files.
    step 1. filter correct classification results from testing dataset.
    step 2. based on each label, separate into different groups.
    step 3. [Archived] similarity analysis.
    step 4. build a mixed-integer programming by either gurobi or scip optimizer.
    step 5. solve it.
    step 6. store the result into csv file.
    """
    
    # step 0.
    Logger.info(messages="step 0: read the input files")
    dataset: DataSet = load_dataset(
        dataset_name="mnist",
        onnx_filename="./utils/benchmarks/onnx/mnist-net_256x2.onnx",
        robustness_type=RobustnessType.LP_NORM,
        num_inputs=2,  # len(distribution_filtered_test_labels[test_true_label])
        distance_type="l2",
        epsilon=0.03,
    )

    # step 1.
    Logger.info(
        messages="step 1: filter correct classification results from testing dataset"
    )
    session = ort.InferenceSession(
        dataset.onnx_filename, providers=ort.get_available_providers()
    )
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
    all_inference_result = dict()
    num_test_images: int = len(dataset.test_images)
    Logger.debugging(messages=f"number of images in testing dataset: {num_test_images}")
    for data_id in range(num_test_images):
        inference_result = session.run(
            [output_name],
            {
                input_name: dataset.test_images[data_id]
                .astype(jnp.float32)
                .reshape(1, dataset.num_pixels, 1)
            },
        )[0]
        inference_label = jnp.argmax(inference_result)
        true_label = dataset.test_labels[data_id]
        if inference_label == true_label:
            filterd_test_images.append(dataset.test_images[data_id])
            filterd_test_labels.append(true_label)

            if true_label not in all_inference_result:
                all_inference_result[true_label] = [inference_result]
            else:
                all_inference_result[true_label].append(inference_result)

    Logger.debugging(messages=f"filterd_test_images: {len(filterd_test_images)}")
    Logger.debugging(messages=f"filterd_test_labels: {len(filterd_test_labels)}")
    Logger.debugging(
        messages=f"test accuracy: {len(filterd_test_images) / num_test_images}"
    )

    # *  ************************  * #
    # *  step 2. based on each label, separate into different groups.
    # *  ************************  * #
    distribution_filtered_test_labels: Dict[int, List[jnp.ndarray]] = (
        {}
    )  # * key: label, value: images
    for label in range(dataset.num_labels):
        distribution_filtered_test_labels[label] = []
    for index, label in enumerate(filterd_test_labels):
        distribution_filtered_test_labels[label].append(filterd_test_images[index])
    for label in range(dataset.num_labels):
        Logger.debugging(
            messages=f"label: {label}, number of images: {len(distribution_filtered_test_labels[label])}"
        )

    # # *  ************************  * #
    # # *  [Archived] step 3. similarity analysis
    # # *  ************************  * #
    # test_true_label: int = 1  # YES: 0,    Y3(1), label 1: 0 & 1109 可以結合
    # Logger.debugging(
    #     messages=f"number of testing images: {len(distribution_filtered_test_labels[test_true_label])}"
    # )
    # distance_matrix: jnp.ndarray = Similarity.generate_distance_matrix(
    #     all_data=distribution_filtered_test_labels[test_true_label],
    #     distance_type=dataset.distance_type,
    #     chunk_size=100,
    # )

    # # * find the similar data
    # all_inputs: List[jnp.ndarray] = []
    # similarity_data: List[int] = Similarity.greedy(distance_matrix=distance_matrix)
    # for id, value in enumerate(similarity_data):
    #     all_inputs.append(distribution_filtered_test_labels[test_true_label][value])
    
    # for i in all_inputs:
    #     verify(solver=solver, dataset=dataset, all_inputs=[i], true_label=test_true_label)
    
    # end_time = time.time()
    # Logger.info(
    #     messages=f"elapsed time for batch verification is : {end_time - start_time}"
    # )
    # Logger.info(messages=f"number of iterations: {COUNT}")


    # step 4. & step 5.
    all_inputs: List[jnp.ndarray] = distribution_filtered_test_labels[T]
    for each_input in all_inputs:
        verify(solver=solver,
               dataset=dataset,
               input=each_input)

    # step 6.
    # # * save the experiment result to csv file
    # Results.record_experiments(
    #     robustness_type="Lp",
    #     dataset="mnist",
    #     num_data=num_images,
    #     distance=dataset.distance_type,
    #     time=str(end_time - start_time),
    #     num_iterations=COUNT,
    #     epsilon=dataset.epsilon,
    # )

    return


def main(solver: VerificationSolver = VerificationSolver.SCIP) -> str:
    Logger.initialize(filename="log.txt", with_log_file=False)
    Logger.info(messages="batch verification is starting...")

    Logger.info(messages="release mode is enabled")
    return _execute(solver=solver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="scip")
    parser.add_argument("--mergedtype", type=str, default="join")

    args = parser.parse_args()
    solver: VerificationSolver = VerificationSolver(args.solver)

    main(solver=solver)

    Logger.info(messages="batch verification is finished!")
