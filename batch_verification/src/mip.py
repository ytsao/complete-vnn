from typing import Any, List

from jax import numpy as jnp

from utils import Model
from utils import SCIPModel
from utils import GurobiModel
from utils import NetworksStructure
from utils import DataSet
from utils import read_dataset

# required >= 3.12 version
# type ModelType = SCIPModel | GurobiModel

class _ObjectiveFunction():
    @staticmethod 
    def robustness_property(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:        
        robustness_variable = m.solver.continue_variables["robustness_property"]
        max_variable = m.solver.continue_variables["max_value"]

        expression = robustness_variable + max_variable
        for key, value in m.solver.binary_variables.items():
            expression += 2 * value 

        m.add_objective_function(express=expression, sense="minimize")

        return m
    

    #TODO: implement the following functions
    @staticmethod
    def safety_property(m: SCIPModel | GurobiModel) -> SCIPModel | GurobiModel:
        return 
    

class _Constraints():
    @staticmethod
    def pre_condition(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        """
        build pre-condition constraints based on vnnlib definition.
        """
        for id, value in enumerate(networks.pre_condition):
            m.change_variable_lb(m.solver.continue_variables[f"x_{0}_{id}"], value[0])
            m.change_variable_ub(m.solver.continue_variables[f"x_{0}_{id}"], value[1])

        return m

    # @staticmethod
    # def pre_condition(m: SCIPModel | GurobiModel, current_data: jnp.ndarray, epsilon: float) -> SCIPModel | GurobiModel:
    #     """
    #     build pre-condition constraints based on the input data (L-infinity norm).
    #     """
    #     for each_pixel in current_data:
    #         value = current_data[each_pixel]
    #         m.change_variable_lb(m.solver.continue_variables[f"x_{0}_{each_pixel}"], value - epsilon)
    #         m.change_variable_ub(m.solver.continue_variables[f"x_{0}_{each_pixel}"], value + epsilon)
        
    #     return m

    # @staticmethod
    # def post_condition(m: SCIPModel | GurobiModel, dataset: DataSet, data_id: int, networks: NetworksStructure) -> SCIPModel | GurobiModel:
    #     last_layer_id: int = networks.num_layers - 1
    #     true_label: int = dataset.test_labels[data_id].argmax()

    #     all_variables = [v_var for k_var, v_var in m.solver.continue_variables.items() if f"x_{last_layer_id}" in k_var]
    #     target_variable = m.solver.continue_variables[f"x_{last_layer_id}_{true_label}"]
    #     m.add_max_constraint(max_variable=m.solver.continue_variables["max_value"], variables=all_variables, name=f"post_condition_{data_id}")
    #     m.add_constraint(express=m.solver.continue_variables["max_value"] >= target_variable + 0.000001, name=f"post_condition_{data_id}")
            
    #     return m

    @staticmethod
    def post_condition(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:
        """
        we don't need to consider negative value in post-condition.
        this function is used to build post-condition constraints based on vnnlib definition.
        """
        
        num_post_condition: int = len(networks.post_condition)
        last_layer_id: int = networks.num_layers - 1

        if num_post_condition > 1:
            expression = sum(v_var for k_var, v_var in m.solver.binary_variables.items() if "y_pos" in k_var) >= 1
            m.add_constraint(express=expression, name="post_condition1")

            for i in range(num_post_condition):
                for j in range(len(networks.post_condition[i][0])):

                    rhs = networks.post_condition[i][1][j]
                    if rhs > 0: 
                        lhs = sum(networks.post_condition[i][0][j][k] * m.solver.continue_variables[f"x_{last_layer_id}_{k}"] for k in range(networks.layer_to_layer[-1][1]))
                        m.add_constraint(express=lhs >= -rhs * m.solver.binary_variables[f"y_pos_{i}"], name=f"post_condition2_{i}_{j}")
                        m.add_constraint(express=lhs <= rhs * (1 - m.solver.binary_variables[f"y_pos_{i}"]), name=f"post_condition2_{i}_{j}")

                        # lhs = sum(networks.post_condition[i][0][j][k] * m.solver.continue_variables[f"s_{last_layer_id}_{k}"] for k in range(networks.layer_to_layer[-1][1]))
                        # m.add_constraint(express=lhs >= -rhs * (1 - m.solver.binary_variables[f"y_neg_{i}"]), name=f"post_condition2_{i}_{j}")
                        # m.add_constraint(express=lhs <= rhs * m.solver.binary_variables[f"y_neg_{i}"], name=f"post_condition2_{i}_{j}")
                    else:
                        lhs = sum(networks.post_condition[i][0][j][k] * m.solver.continue_variables[f"x_{last_layer_id}_{k}"] for k in range(networks.layer_to_layer[-1][1]))
                        m.add_constraint(express=lhs >= -9999 * m.solver.binary_variables[f"y_pos_{i}"], name=f"post_condition2_{i}_{j}")
                        m.add_constraint(express=lhs <= 9999 * (1 - m.solver.binary_variables[f"y_pos_{i}"]), name=f"post_condition2_{i}_{j}")

                        # lhs = sum(networks.post_condition[i][0][j][k] * m.solver.continue_variables[f"s_{last_layer_id}_{k}"] for k in range(networks.layer_to_layer[-1][1]))
                        # m.add_constraint(express=lhs >= -9999 * (1 - m.solver.binary_variables[f"y_neg_{i}"]), name=f"post_condition2_{i}_{j}")
                        # m.add_constraint(express=lhs <= 9999 * m.solver.binary_variables[f"y_neg_{i}"], name=f"post_condition2_{i}_{j}")

        else: # num_post_condition == 1
            for j in range(len(networks.post_condition[0][0])):
                lhs = sum(networks.post_condition[0][0][j][k] * ( m.solver.continue_variables[f"x_{last_layer_id}_{k}"] - m.solver.continue_variables[f"s_{last_layer_id}_{k}"] ) for k in range(networks.layer_to_layer[-1][1]))
                rhs = networks.post_condition[0][1][j]
                m.add_constraint(express=lhs <= rhs, name=f"post_condition2_{j}")

        return 

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
                expression = m.solver.continue_variables[f"x_{k}_{nk}"] <= 9999 * (1 - m.solver.binary_variables[f"z_{k}_{nk}"])
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_1")

                expression = m.solver.continue_variables[f"s_{k}_{nk}"] <= 9999 * m.solver.binary_variables[f"z_{k}_{nk}"]
                m.add_constraint(express=expression, name=f"relu_{k}_{nk}_2")
            
            # last layer, even the output layer didn't need activation, these constraints still required.
            # because there are several post-conditions have to be considered.
            # includes OR and AND operators.
            if k == networks.num_layers - 2:
                for nk in range(Nk[1]):
                    expression = m.solver.continue_variables[f"x_{k+1}_{nk}"] <= 9999 * (1 - m.solver.binary_variables[f"z_{k+1}_{nk}"])
                    m.add_constraint(express=expression, name=f"relu_{k+1}_{nk}_1")

                    expression = m.solver.continue_variables[f"s_{k+1}_{nk}"] <= 9999 * m.solver.binary_variables[f"z_{k+1}_{nk}"]
                    m.add_constraint(express=expression, name=f"relu_{k+1}_{nk}_2")

        return m
    

def _create_decision_variables(m: SCIPModel | GurobiModel, networks: NetworksStructure) -> SCIPModel | GurobiModel:

    m.add_variable(lb=0, ub=None, vtype="C", name="robustness_property")

    for k, Nk in enumerate(networks.layer_to_layer):
        if k == 0: # input layer
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.add_variable(lb=None, ub=None, vtype="C", name=name)
        elif k <= networks.num_layers - 3:
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name: str = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
        else: # output layer
            for nk in range(Nk[0]):
                name: str = f"x_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name: str = f"s_{k}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name: str = f"z_{k}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)

                # 20240627: not sure, why do I need this.
                # name: str = f"y_{k}_{nk}"
                # m.add_variable(lb=0, ub=None, vtype="C", name=name)

            for nk in range(Nk[1]):
                name: str = f"x_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name: str = f"s_{k+1}_{nk}"
                m.add_variable(lb=0, ub=None, vtype="C", name=name)

                name: str = f"z_{k+1}_{nk}"
                m.add_variable(lb=0, ub=1, vtype="B", name=name)
            

            # 'or' condition in post-condition 
            # 1 binary variable to indicate 1 disjunctive term
            num_post_condition: int = len(networks.post_condition)
            if num_post_condition > 1:
                for i in range(num_post_condition):
                    name: str = f"y_pos_{i}"
                    m.add_variable(lb=0, ub=1, vtype="B", name=name)

                    name: str = f"y_neg_{i}"
                    m.add_variable(lb=0, ub=1, vtype="B", name=name)


    # max function
    m.add_variable(lb=0, ub=None, vtype="C", name="max_value")


    return m


def _print_results(m: SCIPModel | GurobiModel) -> None:
    solution_status: str = m.get_solution_status()
    if solution_status == "Infeasible":
        print("UNSAT")
    else:
        print("SAT")
        for k, v in m.solver.continue_variables.items():
            primal_solution = m.get_primal_solution(v)
            if primal_solution > 0:
                print(f"{k}: {primal_solution}")
        for k, v in m.solver.binary_variables.items():
            primal_solution = m.get_primal_solution(v)
            if primal_solution > 0.5 and "y" in k:
                print(f"{k}: {primal_solution}")

    return


def mip_verifier (solver_name: str, networks: NetworksStructure) -> SCIPModel | GurobiModel | None:
    """
    
    """
    m: SCIPModel | GurobiModel | None = None
    if solver_name == "scip":
        m = SCIPModel(solver=Model())
    elif solver_name == "gurobi":
        m = GurobiModel(solver=Model(solver_name="gurobi"))
    else:
        raise ValueError("Invalid solver type")
    
    # * load dataset
    # * we don't need to load dataset in this function.
    # //dataset: DataSet = read_dataset.load_dataset(dataset_name="mnist")
    # //print("++++++++++++++++++++++++++++++++++++++++++++++")
    # //print('Train:', dataset.train_images.shape, dataset.train_labels.shape)
    # //print('Test:', dataset.test_images.shape, dataset.test_labels.shape)
    # //print("num_labels: ", dataset.num_labels)
    # //print("num_pixels: ", dataset.num_pixels)
    # //print("num_height: ", dataset.num_height)
    # //print("num_weight: ", dataset.num_weight)
    # //print("num_channel: ", dataset.num_channel)
    # //print("++++++++++++++++++++++++++++++++++++++++++++++")

    # * extract networks structure
    # ! There is the problem in mnist-net_256x6.onnx
    #
    # networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/mnist-net_256x2.onnx", "./batch_verification/utils/benchmarks/vnnlib/prop_0_0.03.vnnlib") # UNSAT
    # networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/mnist-net_256x2.onnx", "./batch_verification/utils/benchmarks/vnnlib/prop_7_0.03.vnnlib")  # SAT
    # networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/test_sat.onnx", "./batch_verification/utils/benchmarks/vnnlib/test_prop.vnnlib")
    # networks: NetworksStructure = read_dataset.extract_network_structure("./batch_verification/utils/benchmarks/onnx/mnist-net_256x4.onnx", "./batch_verification/utils/benchmarks/vnnlib/test2.vnnlib")
    # networks: NetworksStructure = read_dataset.extract_network_structure(onnx_file, vnnlib_file)
    
    # create decision variables
    _create_decision_variables(m=m, networks=networks)
    
    # create objective function
    _ObjectiveFunction.robustness_property(m=m)

    # create constraints
    _Constraints.pre_condition(m=m, networks=networks)
    # _Constraints.post_condition(m=m, dataset=dataset, data_id=0, networks=networks)
    _Constraints.post_condition(m=m, networks=networks)
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
    _print_results(m=m)

    return m