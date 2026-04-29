"""
The abstract class allows to enforce a defined structure to subclasses
"""

from abc import ABC, abstractmethod

class Option(ABC):
    def __init__(self, S, K, T, r, sigma, q: float = 0.0, option_type: str = "call"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self._validate()        # internal method (_)

    def _validate(self):
        if self.S <= 0:
            raise ValueError("Spot price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to expiry must be positive")
        if self.r < 0:
            raise ValueError("Risk-free rate cannot be negative")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.q < 0 or self.q > 1:
            raise ValueError("Dividend yield q must be between 0 and 1")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        

    @abstractmethod
    def payoff(self, S):
        pass
