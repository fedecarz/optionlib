import numpy as np

def simulate_gbm(S, r, sigma, T, steps, q=0.0, seed=None):
    """
    Generates a single GBM price path.
    Returns a numpy array of length steps+1 (including S at t=0).
    """

    if seed is not None:
        np.random.seed(seed)        # fix seed for reproducibility

    dt = T / steps                        # length of each time step
    Z = np.random.normal(0, 1, steps)     # standard normal shocks
    drift = (r - q - 0.5*sigma**2) * dt   # risk-neutral drift per step
    diffusion = sigma * np.sqrt(dt) * Z   # stochastic component per step
    # prepend S0 and compute cumulative log-returns to get full path
    path = np.concatenate([[S], S * np.exp(np.cumsum(drift + diffusion))])
    
    return path 
