import copy
import numpy as np
from scipy.stats import norm
from optionlib.core.vanilla import European


class Greeks:

    @staticmethod
    def analytical(option, greek):
        if not isinstance(option, European):
            raise TypeError("Analytical Greeks are only available for European options. Use numerical instead.")
        
        S = option.S
        K = option.K
        T = option.T
        sigma = option.sigma
        r = option.r
        q = option.q

        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if greek == "delta":
            if option.option_type == "call":
                return np.exp(-q * T) * norm.cdf(d1)
            else:
                return -np.exp(-q * T) * norm.cdf(-d1)
        elif greek == "gamma":
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        elif greek == "vega":
            return S * np.exp(-q * T)*norm.pdf(d1) * np.sqrt(T) / 100
        elif greek == "theta":
            if option.option_type == "call":
                return (-S*np.exp(-q*T)*norm.pdf(d1) * sigma / (2*np.sqrt(T)) 
                - r*K*np.exp(-r * T)*norm.cdf(d2) 
                + q*S*np.exp(-q * T)*norm.cdf(d1)) / 365
            else:
                return (-S*np.exp(-q * T)*norm.pdf(d1) * sigma/(2*np.sqrt(T)) 
                        + r*K*np.exp(-r * T)*norm.cdf(-d2) 
                        - q*S*np.exp(-q * T)*norm.cdf(-d1)) / 365
        elif greek == "rho":
            if option.option_type == "call":
                return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        else:
            raise ValueError(f"Unknown greek: {greek}")


    @staticmethod
    def numerical(option, engine, greek, epsilon = 0.01):
        
        option_up = copy.deepcopy(option)
        option_down = copy.deepcopy(option)

        if greek == "delta":
            option_up.S = option.S + epsilon
            option_down.S = option.S - epsilon
        elif greek == "gamma":
            option_up.S = option.S + epsilon
            option_down.S = option.S - epsilon
        elif greek == "vega":
            option_up.sigma = option.sigma + epsilon
            option_down.sigma = option.sigma - epsilon
        elif greek == "theta":
            option_up.T = option.T + epsilon
            option_down.T = option.T - epsilon
        elif greek == "rho":
            option_up.r = option.r + epsilon
            option_down.r = option.r - epsilon
        else:
            raise ValueError(f"Unknown greek: {greek}")
        

        V_up = engine.price(option_up)
        V_down = engine.price(option_down)

        if greek == "gamma":
            V_mid = engine.price(option)
            return (V_up - 2*V_mid + V_down) / (epsilon**2)

        return (V_up - V_down) / (2 * epsilon)