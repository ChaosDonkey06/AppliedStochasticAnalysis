from scipy.stats import truncnorm
import numpy as np

def sample_uniform2(xrange, m):
    p       = xrange.shape[0]
    samples = np.full((p, m), np.nan)
    for ip in range(p):
        samples[ip, :] = np.random.uniform(xrange[ip, 0], xrange[ip, 1], m)
    return samples

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd )

def sample_truncated_normal(mean, sd, xrange, m):
    p       = xrange.shape[0]
    samples = np.full((p, m), np.nan)
    for ip in range(p):
        samples[ip, :] = get_truncated_normal(mean=mean[ip], sd=sd[ip], low=xrange[ip, 0], upp=xrange[ip, 1]).rvs(m)
    return samples