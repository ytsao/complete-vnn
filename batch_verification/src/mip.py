from typing import Any

from ..utils import Model
from ..utils import SCIPModel
from ..utils import GurobiModel
from ..utils import NetworksStructure
from ..utils import DataSet
from ..utils import read_dataset

# required >= 3.12 version
# type ModelType = SCIPModel | GurobiModel

class _ObjectiveFunction():
    @staticmethod 
    def robustness_property(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:        
        robustness_variable = m.solver.continue_variables["robustness_property"]
        max_variable = m.solver.continue_variables["max_value"]
        m.add_objective_function(express=robustness_variable + max_variable, sense="minimize")

        return m
    

    #TODO: implement the following functions
    @staticmethod
    def safety_property(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return 
    

class _Constraints():
    @staticmethod
    def adversarial_distance(m: SCIPModel | GurobiModel, dataset: DataSet, data_id: int, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for each_pixel in range(dataset.num_pixels):
            value = dataset.test_images[data_id][each_pixel]
            variable_x = m.solver.continue_variables[f"x_0_{each_pixel}"]
            expression = m.solver.continue_variables["robustness_property"] >= value - variable_x
            m.add_constraint(express=expression, name=f"adversarial_distance1_{data_id}_{each_pixel}")

            expression = m.solver.continue_variables["robustness_property"] >= variable_x - value
            m.add_constraint(express=expression, name=f"adversarial_distance2_{data_id}_{each_pixel}")


        return m


    @staticmethod
    def pre_condition(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for id, value in enumerate(networks.pre_condition):
            m.change_variable_lb(m.solver.continue_variables[f"x_{0}_{id}"], value[0])
            m.change_variable_ub(m.solver.continue_variables[f"x_{0}_{id}"], value[1])

        return m


    @staticmethod
    def post_condition(m: SCIPModel | GurobiModel, dataset: DataSet, data_id: int, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        last_layer_id: int = networks.num_layers - 1
        true_label: int = dataset.test_labels[data_id].argmax()

        all_variables = [v_var for k_var, v_var in m.solver.continue_variables.items() if f"x_{last_layer_id}" in k_var]
        target_variable = m.solver.continue_variables[f"x_{last_layer_id}_{true_label}"]
        m.add_max_constraint(max_variable=m.solver.continue_variables["max_value"], variables=all_variables, name=f"post_condition_{data_id}")
        m.add_constraint(express=m.solver.continue_variables["max_value"] >= target_variable + 0.000001, name=f"post_condition_{data_id}")
            

        return m


    @staticmethod
    def feedforward_networks(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for k, Nk in enumerate(networks.layer_to_layer):
            for j in range(Nk[1]):
                expression = sum(networks.matrix_weights[k][i][j] * m.solver.continue_variables[f"x_{k}_{i}"] for i in range(Nk[0])) + networks.vector_bias[k][j] == m.solver.continue_variables[f"x_{k+1}_{j}"] - m.solver.continue_variables[f"s_{k+1}_{j}"]
                m.add_constraint(express=expression, name=f"feedforward_{k}_{j}")

        return m
    
    
    #TODO: implement the following functions
    @staticmethod
    def convolutional_networks(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    #TODO: implement the following functions
    @staticmethod
    def residual_networks(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    #TODO: implement the following functions
    @staticmethod
    def max_pooling(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return m
    

    @staticmethod
    def relu(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        for k, Nk in enumerate(networks.layer_to_layer):
            if k == 0: 
                continue

            for nk in range(Nk[0]):
                expression = m.solver.continue_variables[f"x_{k}_{nk}"] <= 99999 * (1 - m.solver.binary_variables[f"z_{k}_{nk}"])
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_1")

                expression = m.solver.continue_variables[f"s_{k}_{nk}"] <= 99999 * m.solver.binary_variables[f"z_{k}_{nk}"]
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_2")

        return m
    

def _create_decision_variables(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:

    m.add_variable(lb=0, ub=None, vtype="C", name="robustness_property")

    for k, Nk in enumerate(networks.layer_to_layer):
        if k == 0: # input layer
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=None, ub=None, vtype="C", name=name)
        elif k <= networks.num_layers - 3:
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
        else: # output layer
            print("output layer")
            for nk in range(Nk[0]):
                name = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)

                name = f"y_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

            for nk in range(Nk[1]):
                name = f"x_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"s_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k+1}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
            

            #TODO: 'or' condition in post-condition 
    

    # max function
    m.add_variable(lb=0, ub=None, vtype="C", name="max_value")


    return m

def mip_verifier (solver_name: str) -> SCIPModel | GurobiModel | None:
    m: SCIPModel | GurobiModel | None = None
    if solver_name == "scip":
        m = SCIPModel(solver=Model())
    elif solver_name == "gurobi":
        m = GurobiModel(solver=Model(solver_name="gurobi"))
    else:
        raise ValueError("Invalid solver type")
    
    # load dataset
    dataset: DataSet = read_dataset.load_dataset(dataset_name="mnist")
    # print("++++++++++++++++++++++++++++++++++++++++++++++")
    # print('Train:', dataset.train_images.shape, dataset.train_labels.shape)
    # print('Test:', dataset.test_images.shape, dataset.test_labels.shape)
    # print("num_labels: ", dataset.num_labels)
    # print("num_pixels: ", dataset.num_pixels)
    # print("num_height: ", dataset.num_height)
    # print("num_weight: ", dataset.num_weight)
    # print("num_channel: ", dataset.num_channel)
    # print("++++++++++++++++++++++++++++++++++++++++++++++")

    # extract networks structure
    networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/ACASXU_run2a_1_1_batch_2000.onnx", "./batch_verification/utils/benchmarks/vnnlib/prop_7.vnnlib")
    
    # create decision variables
    _create_decision_variables(m=m, networks=networks)
    
    # create objective function
    _ObjectiveFunction.robustness_property(m=m)

    # create constraints
    _Constraints.pre_condition(m=m, networks=networks)
    _Constraints.post_condition(m=m, dataset=dataset, data_id=0, networks=networks)
    # _Constraints.adversarial_distance(m=m, dataset=dataset, data_id=0, networks=networks) # it should work
    _Constraints.feedforward_networks(m=m, networks=networks)
    # _Constraints.convolutional_networks(m=m)
    # _Constraints.residual_networks(m=m)
    # _Constraints.max_pooling(m=m)
    _Constraints.relu(m=m, networks=networks)
    # _Constraints.sigmoid(m=m)


    # export lp file 
    m.export_lp_file("mip")

    # solve 
    m.optimize()

    # print results
    for k, v in m.solver.continue_variables.items():
        primal_solution = m.get_primal_solution(v)
        if primal_solution > 0:
            print(f"{k}: {primal_solution}")
    for k, v in m.solver.binary_variables.items():
        primal_solution = m.get_primal_solution(v)
        if primal_solution > 0.5:
            print(f"{k}: {primal_solution}")

    return m


def main() -> None:
    print("this is main function")
    mip_verifier(solver_name="scip")
    print("scip got it")
    mip_verifier(solver_name="gurobi")
    print("gurobi got it")
    return

if __name__ == "__main__":
    # python -m batch_verification.src.mip 
    print("batch verification is starting...")
    main()