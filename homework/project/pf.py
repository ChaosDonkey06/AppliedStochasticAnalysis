from plot_utils import *
import pandas as pd
import numpy as np
import datetime
import os

import sys
sys.path.insert(0, "pompjax/pompjax")

from diagnostic_plots import convergence_plot
from utils import create_df_response
from ifeakf import ifeakf

from model import simulate_em_sde01, euler_maruyama_sde_01, simulate_inference_trajectory

####-####-####-####-####-####
import argparse
parser = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--i', type=int, help='microbial pathogen',
       default=0)

idx_save_infer = parser.parse_args().i

