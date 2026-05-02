import numpy as np
from scipy.stats import norm
from optionlib.core.vanilla import European


class Black_Scholes:
    
    @staticmethod
    def price(option: European) -> float:
        """
        Computes the Black-Scholes price for a European Option.
        """
        if not isinstance(option, European):
            raise TypeError("Black-Scholes can only price European options. Use Monte Carlo or trees for other option types.")

        S = option.S
        K = option.K
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        d1 = (np.log(S/K) + (r - q * 0.5 * sigma * 2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return price
    

opt = European(S=100, K=100, T=1, r=0.05, sigma=0.2)
