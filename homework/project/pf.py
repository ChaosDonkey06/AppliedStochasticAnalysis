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

model_settings = {
    "m"           : 1000,                       # number of particles
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
   "inflation"          : 0,                   # inflation factor for spreading the variance after the EAKF step
}

def normal_loglikelihood(real_world_observations, model_observations, error_variance=None, A=0.1, num_times=100):
    if not error_variance:
        error_variance = 1 + (0.2*real_world_observations)**2

    nll =  A * np.exp(-0.5 * (real_world_observations - model_observations)**2 / error_variance) # Normal LL
    return - np.sum(nll, 0)

def simulate_inference_trajectory(h=1e-3, tmax = 10):
    num_sims = 100
    δt       = h

    α = 10
    ρ = 28
    β = 8/3

    param = {"α": α, "ρ": ρ, "β": β}

    tsim, xsim = simulate_em_sde01(tmax, δt,  m=num_sims, params=param)
    id_infer   = np.random.choice(num_sims)

    infer_df         = pd.DataFrame(xsim[:, id_infer, :].T, columns=['y1', 'y2', 'y3'])
    infer_df["oev1"] = np.maximum(np.max(infer_df["y1"].values), 1 +( 0.2 * infer_df["y1"].values)**2)
    infer_df["oev2"] = np.maximum(np.max(infer_df["y2"].values), 1 +( 0.2 * infer_df["y2"].values)**2)
    infer_df["oev3"] = np.maximum(np.max(infer_df["y3"].values), 1 +( 0.2 * infer_df["y3"].values)**2)
    infer_df["date"] = pd.date_range(start=datetime.datetime(1997, 3, 12), periods=len(tsim), freq='D')
    infer_df         = infer_df.iloc[7:]

    return infer_df, tsim, xsim, id_infer

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

infer_df, tsim, xsim, id_infer = simulate_inference_trajectory(h=δt, tmax=10)

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

from inference_local import if_pf

infer_df, tsim, xsim, id_infer    = simulate_inference_trajectory(h=δt, tmax=10)
model_settings["dates"]           = infer_df["date"].values
if_settings["assimilation_dates"] = infer_df["date"].values

θmle, θpost = if_pf(process_model             = f_if,
                    observational_model       = g_if,
                    state_space_initial_guess = f0_if,
                    measure_density           = normal_loglikelihood,
                    observations_df           = infer_df.set_index("date"), # resample so assimilitation dates are weekly
                    parameters_range          = parameters_range,
                    state_space_range         = state_space_range,
                    model_settings            = model_settings,
                    if_settings               = if_settings,
                    perturbation              = σ_perturb)

path_to_results = os.path.join("results", "inference", "pf")
os.makedirs(path_to_results, exist_ok=True)

θpost = θpost.mean(-2)
np.savez_compressed(os.path.join(path_to_results, f"{idx_save_infer}".zfill(3)+"_infer.npz"),
                            θmle     = θmle,
                            θpost    = θpost,
                            δt       = δt,
                            tsim     = tsim,
                            xsim     = xsim,
                            id_infer = id_infer)