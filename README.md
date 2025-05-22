# complete-vnn 

This repository implements different complete methods to verify neural networks.


## Dependencies
* MIP: scip \& gurobi python interface.
* SMT: z3

## How to use?
```bash
git clone git@github.com:ytsao/complete-vnn.git
cd complete_vnn
conda env create -f environment.yml
conda activate complete_verify
```

## Reference
- [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356)
- [Deep Neural Networks and Mixed Integer Linear Optimization](https://link.springer.com/article/10.1007/s10601-018-9285-6)
- [Z3 Python Tutorial](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)