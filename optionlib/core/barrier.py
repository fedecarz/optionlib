from optionlib.core.option import Option

class Barrier(Option):
    """
    Barrier option - a path-dependent option that is either activated or extinguished if the underlying price crosses a barrier level.
    Supports knock-in (activated on breach) and knock-out (extinguished on breach) in both up and down directions.
    """
    def __init__(self, S, K, T, r, sigma, barrier, q = 0, option_type = "call", barrier_type = "up-and-out"):
        super().__init__(S, K, T, r, sigma, q, option_type)
        if barrier_type not in ("up-and-out", "down-and-out", "up-and-in", "down-and-in"):
            raise ValueError("Barrier type needs to be on of: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'")
        if barrier <= 0:
            raise ValueError("Barrier must be positive")
        self.barrier = barrier
        self.barrier_type = barrier_type
    

    def payoff(self, S_final, path):
        if "up" in self.barrier_type:
            breached = any(path >= self.barrier)
        else:
            breached = any(path <= self.barrier)

        if "out" in self.barrier_type:
            alive = not breached
        else:
            alive = breached
        
        if alive:
            if self.option_type == "call":
                return max(S_final - self.K, 0)
            else:
                return max(self.K - S_final, 0)
        else:
            return 0