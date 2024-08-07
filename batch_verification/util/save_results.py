from typing import Dict, List

import jax.numpy as jnp
import pandas as pd


class Results:
    counter_examples: Dict[int, List[jnp.ndarray]] = {}
    unsatisfiable_inputs: Dict[int, List[int]] = {}

    @staticmethod
    def add(ce_id: int, ce: jnp.ndarray, label: int) -> None:
        if label not in Results.counter_examples:
            Results.counter_examples[label] = []
            Results.unsatisfiable_inputs[label] = []
        Results.unsatisfiable_inputs[label].append(ce_id)
        Results.counter_examples[label].append(ce)

    @staticmethod
    def get_unsatisfiable_inputs(label: int) -> List[int]:
        if label not in Results.unsatisfiable_inputs:
            return []
        return Results.unsatisfiable_inputs[label]

    @staticmethod
    def record_experiments(
        robustness_type: str,
        dataset: str,
        num_data: int,
        distance: str,
        time: str,
        num_iterations: int,
        epsilon: float = 0.01,
        degree: float = 5,
        brightness: float = 0.05,
    ):
        df = pd.DataFrame()
        df["Robustness Type"] = [robustness_type]
        df["Dataset"] = [dataset]
        df["Num Data"] = [num_data]
        df["Distance"] = [distance]
        df["Time"] = [time]
        df["Num Iterations"] = [num_iterations]

        if robustness_type == "Lp":
            df["Epsilon"] = [epsilon]
            df["Degree"] = ["-"]
            df["Brightness"] = ["-"]
        elif robustness_type == "Rotation":
            df["Epsilon"] = ["-"]
            df["Degree"] = [degree]
            df["Brightness"] = ["-"]
        elif robustness_type == "Brightness":
            df["Epsilon"] = ["-"]
            df["Degree"] = ["-"]
            df["Brightness"] = [brightness]

        df.to_csv("./results.csv", mode="a", header=False, index=False)

        return
