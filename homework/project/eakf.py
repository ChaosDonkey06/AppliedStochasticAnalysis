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
####-####-####-####-####-####

model_settings = {
    "m"           : 300,                        # number of ensembles
    "p"           : 3,                          # number of parameters
    "k"           : 3,                          # number of observations
    "n"           : 3,                          # number of state variables / dimension of the state space
    "dt"          : 1,                          # time step
    "param_name"  : ["α", "ρ", "β"],            # name of the parameters
    "param_truth" : [10, 28, 8/3]               # true parameter values (not required - just for the example)
    }

if_settings = {
   "Nif"                : 30,                     # number of iterations of the IF
   "type_cooling"       : "geometric",            # type of cooling schedule
   "shrinkage_factor"   : 0.9,                    # shrinkage factor for the cooling schedule
   "inflation"          : 1.01,                   # inflation factor for spreading the variance after the EAKF step
}

αmin = 1
αmax = 50

ρmin = 20
ρmax = 50

βmin = 2
βmax = 4

state_space_range = np.array([-30, 30])
parameters_range  = np.array([[αmin, αmax],
                              [ρmin, ρmax],
                              [βmin, βmax]])

σ_perturb = np.array([(αmax - αmin)/10,
                      (ρmax - ρmin)/10,
                      (βmax - βmin)/10])



δt = 1e-3

def f(t, x, α, ρ, β):
    params = {"α": α, "ρ": ρ, "β": β}
    return euler_maruyama_sde_01(x, t, δt, params)

def g(t, x, θ):
    return x

def f0(m):
    x0 = np.array([[-5.91652, -5.52332, 24.57231]]).T * np.ones((1, m))
    return x0

# Function to be used for the ikeafk function.
f_if  = lambda t, x, θ: f(t, x, θ[0, :], θ[1, :], θ[2, :])
g_if  = lambda t, x, θ: g(t, x, θ)
f0_if = lambda θ: f0(model_settings["m"])

path_to_results = os.path.join("results", "eakf")
if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)

infer_df, tsim, xsim, id_infer = simulate_inference_trajectory(h=δt, tmax=10)
model_settings["dates"]           = infer_df["date"].values
if_settings["assimilation_dates"] = infer_df["date"].values

θmle, θpost = ifeakf(process_model            = f_if,
                    observational_model       = g_if,
                    state_space_initial_guess = f0_if,
                    observations_df           = infer_df.set_index("date"), # resample so assimilitation dates are weekly
                    parameters_range          = parameters_range,
                    state_space_range         = state_space_range,
                    model_settings            = model_settings,
                    if_settings               = if_settings,
                    perturbation              = σ_perturb)

θpost = θpost.mean(-2)
np.savez_compressed(os.path.join(path_to_results, f"{idx_save_infer}".zfill(3)+"_infer.npz"),
                θmle  = θmle,
                θpost = θpost,
                δt    = δt,
                tsim  = tsim,
                xsim  = xsim,
                id_infer = id_infer)
