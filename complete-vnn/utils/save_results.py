from typing import Any, List

import pandas as pd


class Results:
    def __init__(self):
        self.COLUMN_NAMES: List[str] = ["Robustness Type",
                                        "Dataset",
                                        "Data ID",
                                        "Time",
                                        "Status",
                                        "Epsilon"]
        self.df = pd.DataFrame(columns=self.COLUMN_NAMES)

    def add_result(self, new_result: List[Any]):
        self.df.loc[len(self.df)] = new_result
        return 
    
    def save_to_csv(self):
        self.df.to_csv("./results.csv", mode="w", header=True, index=False)
        return

    @staticmethod
    def record_experiments(
        robustness_type: str,
        dataset: str,
        num_data: int,
        inputs: List[int],
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
        df["Input ID"] = [inputs]
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
