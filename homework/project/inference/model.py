import pandas as pd
import numpy as np
import datetime

def brownian_noise(m):
    return np.random.randn(m)

def euler_maruyama_sde_01(x, t, δt, params):
    m = x.shape[1]

    α = params["α"]
    ρ = params["ρ"]
    β = params["β"]

    Xt = x[0, :]
    Yt = x[1, :]
    Zt = x[2, :]

    dUt = brownian_noise(m)
    dVt = brownian_noise(m)
    dWt = brownian_noise(m)

    dXt = δt * α * (Yt - Xt)          + np.sqrt(δt) * dUt
    dYt = δt * (Xt * (ρ - Zt) - Yt) + np.sqrt(δt) * dVt
    dZt = δt * (Xt * Yt - β * Zt)   + np.sqrt(δt) * dWt
    return np.array([Xt+dXt, Yt+dYt, Zt+dZt])

def simulate_em_sde01(tmax, δt, params,  m=3):
    t = np.arange(0, tmax, δt)

    x0            = np.array([[-5.91652, -5.52332, 24.57231]]).T * np.ones((1, m))
    xsim          = np.full((3, m, len(t)), np.nan)
    xsim[:, :, 0] = x0

    for i, ti in enumerate(t[1:]):
        xsim[:, :, i+1] = euler_maruyama_sde_01(xsim[:,:,i], ti, ti-t[i], params)

    return t, xsim

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