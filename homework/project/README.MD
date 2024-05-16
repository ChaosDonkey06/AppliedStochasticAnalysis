# Applied Stochastic Analysis project
Author: Jaime Cascante Vega

email: jc12343@nyu.edu

## How to re-run the results.
I used the greene cluster and the file to run the Iterated filtering using the EAKF and the SMC are [eakf](eakf.sh) and [pf](if.sh) respectively.

For running one single inference on a random trajectory of the Lorenz-63 SDE use the code below.

    python3 run_eakf.py --i 101
    python3 run_smc.py --i 101

The code reproducing the figures is in the jupyter notebook [viz_inference](visualize_inference.ipynb).