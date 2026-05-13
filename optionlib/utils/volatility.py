import pandas as pd
import numpy as np
import copy
from optionlib.engines.black_scholes import Black_Scholes
from optionlib.utils.greeks import Greeks

def historical_volatility(prices: pd.DataFrame):
    """
    Computes historical volatility
    """
    log_rets = np.log(prices/prices.shift())
    ann_vol = log_rets.std() * np.sqrt(252)
    return ann_vol


def implied_vol(option, market_price, tol = 1e-6, max_iter = 200):
    """
    Calculates implied volatility -> backed out through Newton Raphson solver
    """
    sigma = 0.2                                # initial guess
    option_copy = copy.deepcopy(option)        # avoid mutating the original option

    for i in range(max_iter):
        option_copy.sigma = sigma

        price = Black_Scholes.price(option_copy)
        vega = Greeks.analytical(option_copy, "vega") * 100        # rescale vega back to raw units

        diff = price - market_price        # pricing error

        if abs(diff) < tol:                # converged
            return sigma

        if abs(vega) < 1e-10:              # vega too small — avoid division by zero
            break
    
        sigma = sigma - diff / vega            # Newton-Raphson step
        sigma = max(0.001, min(sigma, 10.0))   # keep sigma in valid bounds
        
    raise ValueError("Implied vol did not converge")
