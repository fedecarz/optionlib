import numpy as np

def simulate_gbm(S, r, sigma, T, steps, q=0.0, seed=None):
    """
    Generates a single GBM price path.
    Returns a numpy array of length steps+1 (including S at t=0).
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    Z = np.random.normal(0, 1, steps)
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    path = np.concatenate([[S], S * np.exp(np.cumsum(drift + diffusion))])
    
    return path 