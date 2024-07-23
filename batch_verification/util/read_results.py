from io import TextIOWrapper
from typing import Tuple, List

from log import Logger


class Crown:
    @staticmethod
    def get_status(filename: str) -> str:
        f: TextIOWrapper = open(filename, "r")
        res: str = f.readline()

        assert res == "sat" or res == "unsat"

        return res  # either "SAT" or "UNSAT"

    @staticmethod
    def get_ce(filename: str) -> Tuple[List[float], List[float]]:
        try:
            f: TextIOWrapper = open(filename, "r")
        except OSError:
            Logger.error(f"Cannot find the file {filename}")
            return [], []

        s = f.readlines()

        characters_to_remove = ["(", ")", "\n"]
        translation_table = str.maketrans("", "", "".join(characters_to_remove))
        s = [x.translate(translation_table) for x in s]

        x: List[float] = [v.split()[1] for v in s if "X" in v.split()[0]]
        y: List[float] = [v.split()[1] for v in s if "Y" in v.split()[0]]
        Logger.debugging(f"x: {len(x)}")
        Logger.debugging(f"y: {len(y)}")

        return x, y


if __name__ == "__main__":
    # * Testing :: OK
    Logger.initialize()

    Logger.info(Crown.get_status("../out.txt"))
    Crown.get_ce("../test_cex1.txt")
