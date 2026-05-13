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


def implied_vol(option, market_price, tol = 1e-6, max_iter = 100):
    """
    Calculates implied volatility -> backed out through Newton Raphson solver
    """
    sigma = 0.2
    option_copy = copy.deepcopy(option)

    for i in range(max_iter):
        option_copy.sigma = sigma

        price = Black_Scholes.price(option_copy)
        vega = Greeks.analytical(option_copy, "vega")

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

    raise ValueError("Implied vol did not converge")
   

