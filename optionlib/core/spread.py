from optionlib.core.option import Option


class Spread(Option):
    """
    Spread option - payoff is based on the difference between two underlying assets.
    S1 is passed as S to the base class, sigma1 as sigma.
    K can be 0 for a pure spread with no strike.
    """
    def __init__(self, S1, S2, K, T, r, sigma1, sigma2, rho, q = 0, option_type = "call"):
        super().__init__(S1, K, T, r, sigma1, q, option_type)
        if S2 <= 0:
            raise ValueError("Spot price S2 must be positive")
        if sigma2 <= 0:
            raise ValueError("Volatility sigma2 must be positive")
        if rho < -1 or rho > 1:
            raise ValueError("Correlation must be between -1 and 1")
        
        self.S2 = S2
        self.sigma2 = sigma2
        self.rho = rho

    def payoff(self, S1_final, S2_final):
        if self.option_type == "call":
            return max((S1_final - S2_final) - self.K, 0)
        else:
            return max(self.K - (S1_final - S2_final), 0)