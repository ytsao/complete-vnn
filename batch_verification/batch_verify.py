from collections import defaultdict
from typing import List, Dict
import time
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # disable information and warning from tensorflow
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

from matplotlib import pyplot as plt
import onnxruntime as ort
import numpy as np
import jax.numpy as jnp

from src.crown import crown_verifier
from src.mip import mip_verifier
from src.similarity import Similarity
from src.ce_checker import Checker

from util.gurobi_modeling import GurobiModel
from util.scip_modeling import SCIPModel
from util.mip_modeling import Model
from util.read_dataset import extract_network_structure, load_dataset
from util.read_results import get_ce
from util.parameters_networks import NetworksStructure, DataSet
from util.write_vnnlib import (
    write_vnnlib,
    write_vnnlib_join,
    write_vnnlib_meet,
    export_vnnlib,
)
from util.merge_inputs import merge_inputs
from util.options import InputMergedBy, VerificationSolver, Mode
from util.save_results import Results
from util.log import Logger


# * algorithms
COUNT: int = 0


# TODO: implement verification algorithm in different ways
def verify(
    solver: VerificationSolver,
    onnx_filename: str,
    dataset: DataSet,
    all_inputs: List[jnp.ndarray],
    mergedtype: InputMergedBy,
    true_label: int,
    epsilon: float,
) -> str:
    """
    Verification algorithm:

    Support: MIP (SCIP, Gurobi), CROWN
    """
    # *  ************************  * #
    # *  step 4. define abstract domain, 𝒜 for each cluster.
    # *  ************************  * #
    # * merge abstract domain if possible
    # * update networks by new vnnlib
    vnnlib_filename: str = merge_inputs(
        all_inputs=all_inputs,
        num_input_dimensions=dataset.num_pixels,
        num_output_dimension=dataset.num_labels,
        true_label=true_label,
        epsilon=epsilon,
        mergedtype=mergedtype,
    )

    global COUNT
    COUNT += 1
    result: str = "UNSAT"
    networks: NetworksStructure = extract_network_structure(
        onnx_file_path=onnx_filename, vnnlib_file_path=vnnlib_filename
    )
    m: SCIPModel | GurobiModel | None = None

    if solver is VerificationSolver.SCIP or solver is VerificationSolver.GUROBI:
        Logger.info(messages=f"Verification Algorithm is MIP solver ({solver})")

        m: SCIPModel | GurobiModel = mip_verifier(solver_name=solver, networks=networks)
        result = "UNSAT" if m.get_solution_status() == "Infeasible" else "SAT"
    elif solver is VerificationSolver.CROWN:
        Logger.info(messages="Verification Algorithm is CROWN")

        result = crown_verifier(
            onnx_file_path=onnx_filename, vnnlib_file_path=vnnlib_filename
        )

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

        ce_checker: Checker = Checker(
            all_images=all_inputs,
            counter_example=counter_example,
            epsilon=epsilon,
        )

        ce_id: int
        res_ce_check: bool
        ce_id, res_ce_check = ce_checker.check()
        Logger.debugging(messages=f"counter example check: {res_ce_check}")

        if res_ce_check is False:
            Logger.error(messages="Counter example is not correct")
            # ? How to divide the input domain? -> binary search
            # ? What is the goal? -> generate template for reusing in testing dataset
            # ? What is the termination condition? -> generate template or valid counter example

            # * remove vnnlib file, since it is not correct
            Logger.debugging(messages=f"remove the vnnlib file: {vnnlib_filename}")
            os.remove(vnnlib_filename)

            if len(all_inputs) == 1:
                Logger.error(messages="IMPOSSIBLE")
                return "IMPOSSIBLE"

            # * binary search
            left: List[jnp.ndarray] = all_inputs[: len(all_inputs) // 2]
            right: List[jnp.ndarray] = all_inputs[len(all_inputs) // 2 :]
            verify(
                solver=solver,
                onnx_filename=onnx_filename,
                dataset=dataset,
                all_inputs=left,
                mergedtype=mergedtype,
                true_label=true_label,
                epsilon=epsilon,
            )
            verify(
                solver=solver,
                onnx_filename=onnx_filename,
                dataset=dataset,
                all_inputs=right,
                mergedtype=mergedtype,
                true_label=true_label,
                epsilon=epsilon,
            )
        else:
            Logger.info(messages="Counter example from MIP is a real counter example")
            assert ce_id != -1

            Results.add(ce=counter_example, ce_id=ce_id, label=true_label)
            all_inputs.remove(ce_id)

            # * updating input domain
            vnnlib_filename: str = merge_inputs(
                all_inputs=all_inputs,
                num_input_dimensions=networks.num_inputs,
                num_output_dimension=networks.num_outputs,
                true_label=true_label,
                epsilon=epsilon,
                mergedtype=mergedtype,
            )
            all_inputs.remove(ce_id)
            Logger.info(messages="updated the input domain")
            Logger.debugging(
                messages=f"removed: {Results.get_unsatisfiable_inputs(label=true_label)}"
            )
            verify(
                solver=solver,
                onnx_filename=onnx_filename,
                vnnlib_filename=vnnlib_filename,
                all_images=all_inputs,
                mergedtype=mergedtype,
                true_label=true_label,
                epsilon=epsilon,
            )

    return result


def debug(solver: VerificationSolver, mergedtype: InputMergedBy) -> str:
    """
    Batch verification algorithm:
        step 0: read the input files (onnx, image files)

        step 1: filter correct classification results from testing dataset.
        step 2: based on each label, separate into different groups.
        step 3: similarity analysis
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

    Logger:
        debugging: # * for show the intermediate results.
        info: # * for show the main results.
        error: # * for show the error messages.
    """
    # *  ************************  * #
    # *  step 0. read the input files
    # *  ************************  * #
    Logger.info(messages="step 0: read the input files")
    dataset: DataSet = load_dataset("mnist")
    onnx_filename: str = "./util/benchmarks/onnx/mnist-net_256x2.onnx"
    vnnlib_filename: str = "./util/benchmarks/vnnlib/prop_7_0.03.vnnlib"
    # networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    Logger.info(
        messages="step 1: filter correct classification results from testing dataset"
    )
    session = ort.InferenceSession(
        onnx_filename, providers=ort.get_available_providers()
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # *  =======================  * #
    # *  training dataset part
    # *  =======================  * #
    # filter_train_images: List[jnp.ndarray] = []
    # filter_train_labels: List[int] = []
    # num_train_images: int = len(dataset.train_images)
    # Logger.debugging(messages=f"number of images in training dataset: {len(dataset.train_images)}")
    # for data_id in range(num_train_images):
    #     inference_result = session.run([output_name],
    #                                    {input_name: dataset.train_images[data_id].astype(jnp.float32).reshape(1, dataset.num_pixels, 1)})[0]
    #     inference_label = jnp.argmax(inference_result)
    #     true_label = dataset.train_labels[data_id]
    #     if inference_label == true_label:
    #         filter_train_images.append(dataset.train_images[data_id])
    #         filter_train_labels.append(true_label)
    # Logger.debugging(messages=f"filter_train_images: {len(filter_train_images)}")
    # Logger.debugging(messages=f"filter_train_labels: {len(filter_train_labels)}")

    # *  =======================  * #
    # *  testing dataset part
    # *  =======================  * #
    filterd_test_images: List[jnp.ndarray] = []
    filterd_test_labels: List[int] = []
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

    Logger.debugging(messages=f"filterd_test_images: {len(filterd_test_images)}")
    Logger.debugging(messages=f"filterd_test_labels: {len(filterd_test_labels)}")
    Logger.debugging(
        messages=f"test accuracy: {len(filterd_test_images) / num_test_images}"
    )

    # *  ************************  * #
    # *  step 2. based on each label, separate into different groups.
    # *  ************************  * #
    distribution_filtered_test_labels: Dict[int, List[jnp.ndarray]] = {}
    for label in range(dataset.num_labels):
        distribution_filtered_test_labels[label] = []
    for index, label in enumerate(filterd_test_labels):
        distribution_filtered_test_labels[label].append(filterd_test_images[index])
    for label in range(dataset.num_labels):
        Logger.debugging(
            messages=f"label: {label}, number of images: {len(distribution_filtered_test_labels[label])}"
        )

    # *  ************************  * #
    # *  step 3. similarity analysis
    # *  ************************  * #
    type_of_property: InputMergedBy = InputMergedBy.JOIN
    vnnlib_filename: str = ""
    test_true_label: int = 0  # YES: 0,    Y3(1)
    epsilon: float = 0.03
    count: int = 0
    distance_matrix: jnp.ndarray
    Logger.debugging(
        messages=f"number of testing images: {len(distribution_filtered_test_labels[test_true_label])}"
    )
    distance_matrix = Similarity.generate_distance_matrix(
        all_data=distribution_filtered_test_labels[test_true_label],
        distance_type="l2",
        chunk_size=100,
    )  # ! out of memory

    # * find the similar data
    similarity_data: List[int] = Similarity.greedy(distance_matrix=distance_matrix)
    Logger.debugging(messages=f"similarity_data: {similarity_data}")

    # TODO: Test merge abstract domain if possible
    # *  ************************  * #
    # *  step 4. define abstract domain, 𝒜 for each cluster.
    # *  ************************  * #
    test_set_inputs: List[int] = similarity_data
    # test_set_inputs: List[int] = [similarity_data[0], similarity_data[1], similarity_data[2], similarity_data[3]]
    # test_set_inputs: List[int] = [similarity_data[3], similarity_data[len(similarity_data) - 1]]
    for id in test_set_inputs:
        if type_of_property is InputMergedBy.JOIN:
            if id == test_set_inputs[0]:
                vnnlib_filename: str = write_vnnlib(
                    data=distribution_filtered_test_labels[test_true_label][id],
                    data_id=id,
                    num_classes=dataset.num_labels,
                    true_label=test_true_label,
                    epsilon=epsilon,
                )
            else:
                os.remove(vnnlib_filename)
                vnnlib_filename: str = write_vnnlib_join(
                    networks=networks,
                    data=distribution_filtered_test_labels[test_true_label][id],
                    data_id=id,
                    num_classes=dataset.num_labels,
                    true_label=test_true_label,
                    epsilon=epsilon,
                )
        elif type_of_property is InputMergedBy.MEET:
            if id == test_set_inputs[0]:
                # if True:
                vnnlib_filename: str = write_vnnlib(
                    data=distribution_filtered_test_labels[test_true_label][id],
                    data_id=id,
                    num_classes=dataset.num_labels,
                    true_label=test_true_label,
                    epsilon=epsilon,
                )
            else:
                os.remove(vnnlib_filename)
                vnnlib_filename: str = write_vnnlib_meet(
                    networks=networks,
                    data=distribution_filtered_test_labels[test_true_label][id],
                    data_id=id,
                    num_classes=dataset.num_labels,
                    true_label=test_true_label,
                    epsilon=epsilon,
                )

        # * update networks by new vnnlib
        networks: NetworksStructure = extract_network_structure(
            onnx_filename, vnnlib_filename
        )

    # *  ************************  * #
    # *  step 5. Verify 𝒜 -> r.
    # *     if r is UNSAT: STOP
    # *     else:
    # *         divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
    # *         back to step 5 to verify each 𝒜_i
    # *  ************************  * #
    Logger.info(messages="start verifying ...")
    result: str = verify(
        solver=solver,
        onnx_filename=onnx_filename,
        vnnlib_filename=vnnlib_filename,
        all_images=distribution_filtered_test_labels[test_true_label],
        mergedtype=mergedtype,
        true_label=test_true_label,
        epsilon=epsilon,
    )

    # m: SCIPModel | GurobiModel = mip_verifier(solver_name=solver, networks=networks)
    # counter_example: List[float] = []
    # if m.get_solution_status() == "Infeasible":
    #     Logger.info(messages="UNSAT")
    # else:
    #     for k, Nk in enumerate(networks.layer_to_layer):
    #         if k == 0:
    #             for nk in range(Nk[0]):
    #                 name: str = f"x_{k}_{nk}"
    #                 variable = m.solver.continue_variables[name]
    #                 counter_example.append(m.get_primal_solution(variable))
    #         else:
    #             break
    #     Logger.info(messages="SAT")

    # # plt.imshow(distribution_filtered_test_labels[test_true_label][id].reshape(28, 28), cmap='gray')
    # counter_example_image = np.array(counter_example).reshape(28, 28)
    # plt.imshow(counter_example_image, cmap='gray')
    # plt.show()

    return result


def release(solver: VerificationSolver, mergedtype: InputMergedBy) -> str:
    """
    Batch verification algorithm:
        step 0: read the input files (onnx, image files)

        step 1: filter correct classification results from testing dataset.
        step 2: based on each label, separate into different groups.
        step 3: similarity analysis
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

    Logger:
        debugging: # * for show the intermediate results.
        info: # * for show the main results.
        error: # * for show the error messages.
    """
    # *  ************************  * #
    # *  step 0. read the input files
    # *  ************************  * #
    Logger.info(messages="step 0: read the input files")
    dataset: DataSet = load_dataset("mnist")
    onnx_filename: str = "./util/benchmarks/onnx/mnist-net_256x2.onnx"
    vnnlib_filename: str = "./util/benchmarks/vnnlib/prop_7_0.03.vnnlib"
    # networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    Logger.info(
        messages="step 1: filter correct classification results from testing dataset"
    )
    session = ort.InferenceSession(
        onnx_filename, providers=ort.get_available_providers()
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

    # *  ************************  * #
    # *  step 3. similarity analysis
    # *  ************************  * #
    test_true_label: int = 1  # YES: 0,    Y3(1)
    epsilon: float = 0.03
    # num_images: int = len(distribution_filtered_test_labels[test_true_label])
    num_images: int = 10  # testing small sized instance
    distance_matrix: jnp.ndarray
    Logger.debugging(
        messages=f"number of testing images: {len(distribution_filtered_test_labels[test_true_label])}"
    )
    distance_matrix = Similarity.generate_distance_matrix(
        all_data=distribution_filtered_test_labels[test_true_label],
        distance_type="l2",
        chunk_size=100,
    )

    # * find the similar data
    similarity_data: List[int] = Similarity.greedy(distance_matrix=distance_matrix)
    all_inputs: List[jnp.ndarray] = []
    # for id, value in enumerate(similarity_data):
    #     if id < num_images:
    #         all_inputs.append(distribution_filtered_test_labels[test_true_label][value])
    #         Logger.debugging(f"similarity_data: {value}")
    #     else:
    #         break
    next_data: int = 0
    while len(all_inputs) < num_images:
        Logger.debugging(f"similarity_data: {next_data}")
        all_inputs.append(distribution_filtered_test_labels[test_true_label][next_data])
        next_data = similarity_data[next_data]

    # lex_order_result: List[int] = Similarity.lexicgraphical_order(all_data=all_inputs)
    # all_inputs = [all_inputs[i] for i in lex_order_result]

    # *  ************************  * #
    # *  step 5. Verify 𝒜 -> r.
    # *     if r is UNSAT: STOP
    # *     else:
    # *         divide 𝒜 into 𝒜_1, 𝒜_2,..., 𝒜_n
    # *         back to step 5 to verify each 𝒜_i
    # *  ************************  * #
    Logger.info(messages="start verifying ...")

    start_time = time.time()
    result: str = verify(
        solver=solver,
        onnx_filename=onnx_filename,
        dataset=dataset,
        all_inputs=all_inputs,
        mergedtype=mergedtype,
        true_label=test_true_label,
        epsilon=epsilon,
    )
    end_time = time.time()
    Logger.info(messages=f"elapsed time for batch verification is : {end_time - start_time}")
    Logger.info(messages=f"number of iterations: {COUNT}")

    # * individual verification
    # start_time = time.time()
    # for each_input in all_inputs:
    #     data: List[jnp.ndarray] = [(each_input)]
    #     verify(
    #         solver=solver,
    #         onnx_filename=onnx_filename,
    #         dataset=dataset,
    #         all_inputs=data,
    #         mergedtype=mergedtype,
    #         true_label=test_true_label,
    #         epsilon=epsilon,
    #     )
    # end_time = time.time()
    # Logger.info(messages=f"elapsed time for normal verification is : {end_time - start_time}")


    # m: SCIPModel | GurobiModel = mip_verifier(solver_name=solver, networks=networks)
    # counter_example: List[float] = []
    # if m.get_solution_status() == "Infeasible":
    #     Logger.info(messages="UNSAT")
    # else:
    #     for k, Nk in enumerate(networks.layer_to_layer):
    #         if k == 0:
    #             for nk in range(Nk[0]):
    #                 name: str = f"x_{k}_{nk}"
    #                 variable = m.solver.continue_variables[name]
    #                 counter_example.append(m.get_primal_solution(variable))
    #         else:
    #             break
    #     Logger.info(messages="SAT")

    # # plt.imshow(distribution_filtered_test_labels[test_true_label][id].reshape(28, 28), cmap='gray')
    # counter_example_image = np.array(counter_example).reshape(28, 28)
    # plt.imshow(counter_example_image, cmap='gray')
    # plt.show()

    return result


def main(
    mode: Mode = Mode.DEBUG,
    solver: VerificationSolver = VerificationSolver.SCIP,
    mergedtype: InputMergedBy = InputMergedBy.JOIN,
) -> str:
    Logger.initialize(filename="log.txt", with_log_file=False)
    Logger.info(messages="batch verification is starting...")

    if mode is Mode.DEBUG:
        Logger.info(messages="debug mode is enabled")
        return debug(solver=solver, mergedtype=mergedtype)
    elif mode is Mode.RELEASE:
        Logger.info(messages="release mode is enabled")
        return release(solver=solver, mergedtype=mergedtype)
    else:
        Logger.error(messages="mode is not supported")

    return "main version"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="debug")
    parser.add_argument("--solver", type=str, default="scip")
    parser.add_argument("--mergedtype", type=str, default="join")

    args = parser.parse_args()
    mode: Mode = Mode(args.mode)
    solver: VerificationSolver = VerificationSolver(args.solver)
    mergedtype: InputMergedBy = InputMergedBy(args.mergedtype)

    main(mode=mode, solver=solver, mergedtype=mergedtype)

    Logger.info(messages="batch verification is finished!")