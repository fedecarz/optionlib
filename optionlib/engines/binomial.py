import numpy as np
from optionlib.core.vanilla import European, American
from optionlib.core.barrier import Barrier


class Binomial:

    """
    Binomial tree engine for pricing European, American, and knock-out Barrier options.
    Knock-in barriers are not supported due to node slippage — use the trinomial engine instead.
    Accuracy increases with N (number of steps).
    """

    @staticmethod
    def price(option, N=100):
        """
        Computes the binomial price for European, American and Barrier options.
        """

        if isinstance(option, Barrier) and "in" in option.barrier_type:
            raise NotImplementedError("Knock-in barriers are not supported in the binomial engine")

        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        # Computing the parameters
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r-q) * dt) - d) / (u - d)

        # Creating a empty tree
        tree = np.zeros((N + 1, N + 1))

        for J in range(N + 1):
            S_T = S * u**J * d**(N - J)
            tree[N, J] = option.payoff(S_T)

        # Roll backwards
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                tree[i, j] = np.exp(-r * dt) * (p * tree[i+1, j+1] + (1-p) * tree[i+1, j])
                if isinstance(option, American):
                    S_ij = S * u**j * d**(i - j)
                    tree[i, j] = max(tree[i, j], option.payoff(S_ij))
                if isinstance(option, Barrier):
                    S_ij = S * u**j * d**(i - j)
                    if "out" in option.barrier_type:
                        if "up" in option.barrier_type and S_ij >= option.barrier:
                            tree[i, j] = 0
                        elif "down" in option.barrier_type and S_ij <= option.barrier:
                            tree[i, j] = 0

        return tree[0, 0]
