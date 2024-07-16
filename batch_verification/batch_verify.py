from typing import List, Dict
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import onnx2pytorch
import onnx
import torch 
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

from utils.write_vnnlib import write_vnnlib, write_vnnlib_merge
from utils.parameters_networks import DataSet
from utils.parameters_networks import NetworksStructure
from utils.read_dataset import load_dataset
from utils.read_dataset import extract_network_structure

from utils.mip_modeling import Model
from utils.scip_modeling import SCIPModel
from utils.gurobi_modeling import GurobiModel

# * algorithms 
from src.cluster import Cluster
from src.mip import mip_verifier

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


# TODO: implement verification algorithm in different ways
# TODO: Test auto_LiRPA
def verify() -> None: 
    def mnist_model():
        model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32*7*7,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        return model
    

    onnx_filename: str = "./utils/benchmarks/onnx/mnist-net_256x2.onnx"
    vnnlib_filename: str = "./utils/benchmarks/vnnlib/prop_7_0.03.vnnlib"
    onnx_model = onnx.load(onnx_filename)
    model = onnx2pytorch.ConvertModel(onnx_model, experimental=True)
    # model = mnist_model()
    

    ## Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST(
        './dataset', train=False, download=True,
        transform=torchvision.transforms.ToTensor())
    # For illustration we only use 2 image from dataset
    N = 2
    n_classes = 10
    image = test_data.data[:N].view(N,1,28,28)
    true_label = test_data.targets[:N]
    # Convert to float
    image = image.to(torch.float32) / 255.0
    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter is for constructing the trace of the computational graph,
    # and its content is not important.
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    print('Running on', image.device)

    ## Step 4: Compute bounds using LiRPA given a perturbation
    eps = 0.3
    norm = float("inf")
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = lirpa_model(image)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()
    print('Demonstration 1: Bound computation and comparisons of different methods.\n')

    ## Step 5: Compute bounds for final output
    for method in [
            'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)',
            'CROWN-Optimized (alpha-CROWN)']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(N):
            print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
            for j in range(n_classes):
                indicator = '(ground-truth)' if j == true_label[i] else ''
                print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                    j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
        print()

    print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.\n')
    # There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
    # of the output layer, with respect to the input layer (the image).
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])

    for method in [
            'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN',
            'CROWN-Optimized (alpha-CROWN)']:
        print("Bounding method:", method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], return_A=True, needed_A_dict=required_A)
        lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
        upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
        print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
        print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
        print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
        print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
        print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
        print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
        print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
        print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
        print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')

    ## An example for computing margin bounds.
    # In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
    # For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
    # This generally yields tighter bounds.
    # Here we compute the margin between groundtruth class and groundtruth class + 1.
    # If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    C = torch.zeros(size=(N, 1, n_classes), device=image.device)
    groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
    target_label = (groundtruth + 1) % n_classes
    C.scatter_(dim=2, index=groundtruth, value=1.0)
    C.scatter_(dim=2, index=target_label, value=-1.0)
    print('Demonstration 3: Computing bounds with a specification matrix.\n')
    print('Specification matrix:\n', C)

    for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
        for i in range(N):
            print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
            print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
                j=true_label[i], target=(true_label[i] + 1) % n_classes, l=lb[i][0].item(), u=ub[i][0].item()))
        print()

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
    networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)

    # *  ************************  * #
    # *  step 1. filter correct classification results from testing dataset.
    # *  ************************  * #
    session = ort.InferenceSession(onnx_filename, providers=ort.get_available_providers())
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
    type_of_property: str = "meet"
    vnnlib_filename: str = ""
    test_true_label: int = 0        # YES: 0,    Y3(1)
    epsilon: float = 0.03
    count: int = 0
    distance_matrix: jnp.ndarray
    distance_matrix = Cluster.generate_distance_matrix(all_data=distribution_filtered_test_labels[test_true_label], distance_type="l2", chunk_size=100)  # ! out of memory

    # * find the similar data
    similarity_data: List[int] = Cluster.greedy(distance_matrix=distance_matrix, num_clusters=2)
    print("similarity_data: ", similarity_data)

    # TODO: Test merge abstract domain if possible
    test_set_inputs: List[int] = [similarity_data[0], similarity_data[1]]
    # test_set_inputs: List[int] = [similarity_data[0], similarity_data[1], similarity_data[2], similarity_data[3]]
    # test_set_inputs: List[int] = [similarity_data[3], similarity_data[len(similarity_data) - 1]]
    for id in test_set_inputs:

        if type_of_property == "meet":
            print("meet")
            if id != test_set_inputs[len(test_set_inputs) - 1]:
            # if True:
                vnnlib_filename: str = write_vnnlib(data=distribution_filtered_test_labels[test_true_label][id], 
                                                    data_id=id, 
                                                    num_classes=dataset.num_labels, 
                                                    true_label=test_true_label, 
                                                    epsilon=epsilon)
                print("vnnlib_filename: ", vnnlib_filename)
                print("------------------------------------------------------")
                networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
                continue
            else:
                print("id: ", id)
                vnnlib_filename: str = write_vnnlib_merge(networks=networks,
                                                        data=distribution_filtered_test_labels[test_true_label][id], 
                                                        data_id=id, 
                                                        num_classes=dataset.num_labels, 
                                                        true_label=test_true_label, 
                                                        epsilon=epsilon)
                print("merged_vnnlib_filename: ", vnnlib_filename)
        elif type_of_property == "join":
            print("join")
            if id != test_set_inputs[len(test_set_inputs) - 1]:
            # if True:
                vnnlib_filename: str = write_vnnlib(data=distribution_filtered_test_labels[test_true_label][id], 
                                                    data_id=id, 
                                                    num_classes=dataset.num_labels, 
                                                    true_label=test_true_label, 
                                                    epsilon=epsilon)
                print("vnnlib_filename: ", vnnlib_filename)
                print("------------------------------------------------------")
                networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
                continue
            else:
                print("id: ", id)
                vnnlib_filename: str = write_vnnlib_merge(networks=networks,
                                                        data=distribution_filtered_test_labels[test_true_label][id], 
                                                        data_id=id, 
                                                        num_classes=dataset.num_labels, 
                                                        true_label=test_true_label, 
                                                        epsilon=epsilon)
                print("merged_vnnlib_filename: ", vnnlib_filename)
        
        
        networks: NetworksStructure = extract_network_structure(onnx_filename, vnnlib_filename)
        m: SCIPModel | GurobiModel = mip_verifier(solver_name="scip", networks=networks)
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
                else: break
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


if __name__ == "__main__":
    print("batch verification is starting...")
    # main()
    verify()
