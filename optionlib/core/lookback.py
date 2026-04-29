from optionlib.core.option import Option


class Lookback(Option):
    """
    Lookback option - payoff depends on the optimal (maximum or minimum) price of the underlying observed over the life of the option.
    Fixed lookback uses a fixed strike with the optimal path price as spot.
    Floating lookback replaces the strike with the optimal path price entirely.
    K is only used for fixed lookback and ignored for floating lookback.
    """
    def __init__(self, S, K, T, r, sigma, q = 0, option_type = "call", lookback_type = "fixed"):
        super().__init__(S, K, T, r, sigma, q, option_type)
        if lookback_type not in ("fixed", "floating"):
            raise ValueError("Lookback type can be either fixed or floating")
        self.lookback_type = lookback_type      # K is required by the base class but ignored for floating lookback

    
    def payoff(self, path):
        S_max = max(path)
        S_min = min(path)
        S_final = path[-1]

        if self.lookback_type == "fixed":
            if self.option_type == "call":
                return max(S_max - self.K, 0)
            else:
                return max(self.K - S_min, 0)
        else:
            if self.option_type == "call":
                return S_final - S_min
            else:
                return S_max - S_final