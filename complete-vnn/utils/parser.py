import argparse 

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="scip")
    parser.add_argument("--timelimit", type=float, default=3600)
    parser.add_argument("--perturbation_type", type=str, default="Linf")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--network", type=str, default="None")
    parser.add_argument("--num_test", type=int, default=1)

    args = parser.parse_args()

    return args