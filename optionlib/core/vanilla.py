from optionlib.core.option import Option

class European(Option):
    """Can only be exercised at expiry."""
    def payoff(self, S):
        if self.option_type == "call":
            return max(S - self.K, 0)
        else:
            return max(self.K - S, 0)
        

class American(Option):
    """Can be exercised at any point before or at expiry"""
    def payoff(self, S):
        if self.option_type == "call":
            return max(S - self.K, 0)
        else:
            return max(self.K - S, 0)
