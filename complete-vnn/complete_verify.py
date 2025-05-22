from typing import List, Dict, Any
import time
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # disable information and warning from tensorflow
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from matplotlib import pyplot as plt
import onnxruntime as ort
import numpy as np

from src.mip import mip_verifier
from src.smt import smt_verifier

from utils.gurobi_modeling import GurobiModel
from utils.scip_modeling import SCIPModel
from utils.smt_modeling import SMTModel
from utils.read_dataset import extract_network_structure, load_dataset
from utils.read_results import get_ce
from utils.parameters_networks import NetworksStructure, DataSet
from utils.write_vnnlib import (
    write_vnnlib,
    export_vnnlib,
)
from utils.save_results import Results
from utils.log import Logger
import utils.parser as parser

# global configuration
T: int = 0 # true label


# TODO: implement verification algorithm in different ways
def verify(
    args,
    dataset: DataSet,
    input: np.ndarray,
) -> str:
    """
    Verification algorithm:

    Support: MIP (SCIP, Gurobi), SMT 
    """
    result: str = "UNSAT"
    vnnlib_filename: str = write_vnnlib(data=input, 
                                        num_classes=10,
                                        true_label=T,
                                        epsilon=dataset.epsilon)
    networks: NetworksStructure = extract_network_structure(
        onnx_file_path=dataset.onnx_filename, vnnlib_file_path=vnnlib_filename
    )
    
    if args.solver == "scip" or args.solver == "gurobi":
        Logger.info(messages=f"Verification Algorithm is MIP solver ({args.solver})")
        m: SCIPModel | GurobiModel = mip_verifier(solver_name=args.solver, networks=networks)
        result = "UNSAT" if m.get_solution_status() == "Infeasible" else "SAT"
    elif args.solver == "z3":
        Logger.info(messages="Verification Algorithm is SMT solver (Z3)")
        m: SMTModel = smt_verifier(solver_name=args.solver, networks=networks)
        result = "UNSAT" if m.get_solution_status() == "UNSAT" else "SAT"
    if result == "UNSAT":
        Logger.info(messages="UNSAT")
        return "UNSAT"
    else:
        Logger.info(messages="SAT")
        # * Testing checker for counter-example found by verifier
        counter_example: List[float] = get_ce(
            solver=args.solver, networks=networks, filename="./test_cex.txt", m=m
        )
        counter_example: np.ndarray = np.array(counter_example)

    return result


def _execute(args) -> None:
    """
    Complete verification
    
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
        dataset_name=args.dataset,
        onnx_filename=f"./utils/benchmarks/onnx/{args.network}",
        robustness_type=args.perturbation_type,
        num_test=args.num_test,  # len(distribution_filtered_test_labels[test_true_label])
        epsilon=args.epsilon,
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
    filterd_test_images: List[np.ndarray] = []
    filterd_test_labels: List[int] = []
    all_inference_result = dict()
    num_test_images: int = len(dataset.test_images)
    Logger.debugging(messages=f"number of images in testing dataset: {num_test_images}")
    for data_id in range(num_test_images):
        inference_result = session.run(
            [output_name],
            {
                input_name: dataset.test_images[data_id]
                .astype(np.float32)
                .reshape(1, dataset.num_pixels, 1)
            },
        )[0]
        inference_label = np.argmax(inference_result)
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
    distribution_filtered_test_labels: Dict[int, List[np.ndarray]] = (
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

    # step 4. & step 5.
    all_inputs: List[np.ndarray] = distribution_filtered_test_labels[T]
    results: Results = Results()
    for i, each_input in enumerate(all_inputs):
        if i < dataset.num_inputs:
            start_time = time.time()
            status: str = verify(args, dataset=dataset, input=each_input)
            end_time = time.time()
            new_result: List[Any] = ["Lp", 
                                     "mnist", 
                                     i, 
                                     str(end_time - start_time),
                                     status,
                                     dataset.epsilon]
            results.add_result(new_result)

    return


def main(args) -> str:
    Logger.initialize(filename="log.txt", with_log_file=False)
    Logger.info(messages="complete verification is starting...")
    
    _execute(args)
    
    Logger.info(messages="complete verification is finished!")
    
    return 


if __name__ == "__main__":
    args = parser.parse()
    main(args)