from optionlib.core.option import Option

class Asian(Option):
    """
    Asian option - payoff depends on the average price of the underlying over the life of the option.
    Supports arithmetic and geometric averaging.
    """
    def __init__(self, S, K, T, r, sigma, q = 0, option_type = "call", averaging="arithmetic"):
        super().__init__(S, K, T, r, sigma, q, option_type)
        if averaging not in ("arithmetic", "geometric"):
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")
        self.averaging = averaging

    def payoff(self, S_avg):
        if self.option_type == "call":
            return max(S_avg - self.K, 0)
        else:
            return max(self.K - S_avg, 0)
