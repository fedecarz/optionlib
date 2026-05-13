from optionlib.core.option import Option


class Rainbow(Option):
    """
    Rainbow option - payoff depends on the best or worst performing asset among a basket of underlyings. Best-of pays based on the highest final price, worst-of pays based on the lowest final price.
    """
    def __init__(self, spots: list, K, T, r, sigmas: list, rho, q = 0, option_type = "call", rainbow_type = "best_of"):
        if type(spots) != list:
            raise ValueError("spots must be a list of spot prices")
        if type(sigmas) != list:
            raise ValueError("sigmas must be a list of volatilities")
        if len(spots) != len(sigmas):
            raise ValueError("spots and sigmas must have the same length")
        
        for i, s in enumerate(spots):
            if s <= 0:
                raise ValueError(f"spots[{i}] must be positive")
        
        for i, sig in enumerate(sigmas):
            if sig <= 0:
                raise ValueError(f"sigmas[{i}] must be positive")
        
        super().__init__(spots[0], K, T, r, sigmas[0], q, option_type)

        if rainbow_type not in ("best_of","worst_of"):
            raise ValueError("rainbow_type must be either 'best of' or 'worst of'")
        self.rainbow_type = rainbow_type
        self.rho = rho
        self.spots = spots
        self.sigmas = sigmas

    


    def payoff(self, S):
        if self.rainbow_type == "best_of":
            if self.option_type == "call":
                return max(max(S) - self.K, 0)
            else:
                return max(self.K - min(S), 0)
        else:
            if self.option_type == "call":
                return max(min(S) - self.K, 0)
            else:
                return max(self.K - max(S), 0)
                