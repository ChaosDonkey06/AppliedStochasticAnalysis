from ifeakf import cooling, sample_uniform2, random_walk_perturbation, checkbound_params, check_state_space, sample_truncated_normal

from tqdm import tqdm
import pandas as pd
import numpy as np

def normal_loglikelihood(real_world_observations, model_observations, error_variance=None, A=0.1, num_times=100):
    if not error_variance:
        error_variance = 1 + (0.2*real_world_observations)**2

    nll =  A * np.exp(-0.5 * (real_world_observations - model_observations)**2 / error_variance) # Normal LL
    return - nll

def normalize_weights(w):
    """
    Weight normalization.
    w: Particle weights.
    """
    return w / np.sum(w)

def compute_effective_sample_size(w):
    """
    Effective sample size.
    """
    return 1/np.sum(w**2)

def importance_sampling(w, z, y, q):
    """ Importance sampling:
            Approximate likelihood P(z|θ) using the importance density q(z|y).
            Where y=g(x;θ) is the observation model used after using the state space model f(x;θ).
            Compute the relative weight of each particle respect the previous PF iteration and normalize the weights.
        w: Particle weights.
        z: World observations.
        y: Modelled observations.
        q: Proposal distribution.
    """
    loglik  = q(z , y)

    # Recompute weights and normalize them
    w = w * loglik
    w = normalize_weights(w)

    return w

def naive_weights(m):
    """
    Naive weights.
        Assume all particles have the same weight.
    """
    return np.ones(m)*1/m

def resample_particles(particles, x, w, m, p=None):
    """
    Particle resample.
    """
    if p:
        fixed_particles = np.sort(np.random.choice(np.arange(m), size=int(m*(1-p)), replace=False, p=w))
        particles_index = np.random.choice(np.arange(m), size=m, replace=True, p=w)
        particles_index[fixed_particles] = fixed_particles
    else:
        particles_index = np.sort(np.random.choice(np.arange(m), size=m, replace=True, p=w))

    w         = naive_weights(m)
    particles = particles[:, particles_index] # Replace particles.
    x_post    = x[:, particles_index]

    return particles, x_post, w

def pf(w, pprior, x, y, z, q):
    """
    Particle filter.
    """

    # IS(w, z, y, q)
    m               = pprior.shape[1]
    w               = importance_sampling(w, z, y, q)
    ppost, xpost, w = resample_particles(pprior, x, w, m, p=0.1)
    w               = importance_sampling(w, z, y, q)

    return ppost, xpost, w

def if_pf(process_model,
            observational_model,
            state_space_initial_guess,
            measure_density,
            observations_df,
            parameters_range,
            state_space_range,
            model_settings,
            if_settings,
            cooling_sequence = None,
            perturbation     = None,
            leave_progress   = False):

    if any('adjust_state_space' in key for key in if_settings.keys()):
        adjust_state_space = if_settings["adjust_state_space"]
    else:
        adjust_state_space = True

    if cooling_sequence is None:
        cooling_sequence = cooling(if_settings["Nif"], type_cool=if_settings["type_cooling"], cooling_factor=if_settings["shrinkage_factor"])

    k           = model_settings["k"] # Number of observations
    p           = model_settings["p"] # Number of parameters (to be estimated)
    n           = model_settings["n"] # Number of state variable
    m           = model_settings["m"] # Number of stochastic trajectories / particles / ensembles

    sim_dates   = model_settings["dates"]
    assim_dates = if_settings["assimilation_dates"]

    param_range = parameters_range.copy()
    std_param   = param_range[:, 1] - param_range[:,0]
    SIG         = std_param ** 2 / 4; #  Initial covariance of parameters

    if perturbation is None:
        perturbation = std_param / 10

    assimilation_times = len(assim_dates)

    param_post_all = np.full((p, m, assimilation_times, if_settings["Nif"]), np.nan)
    param_mean     = np.full((p, if_settings["Nif"]+1), np.nan)

    for n in tqdm(range(if_settings["Nif"]), leave=leave_progress):
        if n==0:
            p_prior          = sample_uniform2(param_range, m)
            x                = state_space_initial_guess(p_prior)
            param_mean[:, n] = np.mean(p_prior, -1)
            w                = naive_weights(m) # init particle filter weights

        else:
            pmean   = param_mean[:, n]
            pvar    = SIG * cooling_sequence[n]
            p_prior = p_post.copy() #sample_truncated_normal(pmean, pvar ** (0.5), param_range, m)
            x       = state_space_initial_guess(p_prior)
            w       = naive_weights(m) # init particle filter weights

        t_assim    = 0
        cum_obs    = np.zeros((k, m))
        param_time = np.full((p, m, assimilation_times), np.nan)

        for t, date in enumerate(sim_dates):
            x        = process_model(t, x, p_prior)
            y        = observational_model(t, x, p_prior)
            cum_obs += y

            if pd.to_datetime(date) == pd.to_datetime(assim_dates[t_assim]):
                pert_noise  = perturbation*cooling_sequence[n]
                p_prior     = random_walk_perturbation(p_prior, pert_noise)
                p_prior     = checkbound_params(p_prior, param_range)

                # Measured observations
                z = observations_df.loc[pd.to_datetime(date)][[f"y{i+1}" for i in range(k)]].values
                z = np.expand_dims(z, -1)

                x_post = x.copy()
                p_post = p_prior.copy()

                # Update parameter space
                p_post, x_post, w = pf(w, p_prior, x_post, z, cum_obs, measure_density)

                # check for a-physicalities in the state and parameter space.
                x_post = check_state_space(x_post, state_space_range)
                p_post = checkbound_params(p_post, param_range)

                p_prior = p_post.copy()
                x       = x_post.copy()

                # save posterior parameter
                param_time[:, :, t_assim] = p_post
                cum_obs                   = np.zeros((k, m))
                t_assim                   += 1

        param_post_all[:, :, :, n] = param_time
        param_mean[:, n+1]         = param_time.mean(-1).mean(-1) # average posterior over all assimilation times and them over all ensemble members
    return param_mean, param_post_all