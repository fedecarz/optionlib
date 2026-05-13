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

    def price(self, engine = None, **kwargs):
        """
        Price the option using the most appropriate engine by default.
        Override with engine='bs', 'mc', 'binomial', 'trinomial'.
        """
        
        # Default engine selection
        if engine is None:
            if isinstance(self, European):
                engine = "bs"
            elif isinstance(self, American):
                engine = "binomial"
            elif isinstance(self, Barrier):
                engine = "trinomial"
            else:
                engine = "mc"
        
        if engine == "bs":
            return Black_Scholes.price(self)
        elif engine == "binomial":
            return Binomial.price(self, **kwargs)
        elif engine == "trinomial":
            return Trinomial.price(self, **kwargs)
        elif engine == "mc":
            return MonteCarlo.price(self, **kwargs)
        else:
            raise ValueError(f"Unknown engine: '{engine}'. Use 'bs', 'mc', 'binomial', or 'trinomial'.")
        
    def greeks(self, method = "analytical", engine = None, epsilon = 0.01):
        """
        Returns all Greeks as a dictionary.
        Analytical only available for European options.
        """

        greek_names = ["delta", "gamma", "vega", "theta", "rho"]

        if method == "analytical":
            if not isinstance(self, European):
                raise TypeError("Analytical Greeks only available for European options. Use method='numerical'.")
            return {g: Greeks.analytical(self, g) for g in greek_names}

        elif method == "numerical":
            if engine is None:
                if isinstance(self, European):
                    _engine = Black_Scholes
                elif isinstance(self, American):
                    _engine = Binomial
                elif isinstance(self, Barrier):
                    _engine = Trinomial
                else:
                    _engine = MonteCarlo
            else:
                _engine = {"bs": Black_Scholes, "binomial": Binomial,
                        "trinomial": Trinomial, "mc": MonteCarlo}[engine]

            return {g: Greeks.numerical(self, _engine, g, epsilon=epsilon) for g in greek_names}

        else:
            raise ValueError("method must be 'analytical' or 'numerical'.")
        
# Deferred imports to avoid circular dependecies
from optionlib.utils.greeks import Greeks
from optionlib.core.vanilla import European, American
from optionlib.core.barrier import Barrier
from optionlib.engines.black_scholes import Black_Scholes
from optionlib.engines.monte_carlo import MonteCarlo
from optionlib.engines.binomial import Binomial
from optionlib.engines.trinomial import Trinomial
