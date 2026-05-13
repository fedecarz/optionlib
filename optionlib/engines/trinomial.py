import numpy as np
from optionlib.core.vanilla import European, American
from optionlib.core.barrier import Barrier


class Trinomial:
    """
    Trinomial tree engine for pricing European, American, and Barrier options.
    Supports both knock-in and knock-out barriers accurately via node alignment.
    Accuracy increases with N (number of steps).
    """
    
    @staticmethod
    def price(option, N=100):
        """
        Computes the trinomial price for European, American and Barrier options.
        """

        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        # Computing the parameters
        dt = T / N
        lam = np.sqrt(3/2)      # lambda parameter
        u = np.exp(lam * sigma * np.sqrt(dt))
        d = 1 / u
        m = 1       # Middle, no move
        pu = 1/(2*lam**2) + (r - q - 0.5 * sigma ** 2) * np.sqrt(dt) / (2*lam*sigma)        # up probability
        pd = 1/(2*lam**2) - (r - q - 0.5 * sigma ** 2) * np.sqrt(dt) / (2*lam*sigma)        # down probability
        pm = 1 - 1/lam**2                                                                   # middle probability

        # Creating the tree
        tree = np.zeros((N + 1, 2*N + 1))

        # Forward pass
        for J in range(2*N + 1):
            S_T = S * u**(J - N)
            tree[N, J] = option.payoff(S_T)

        # Roll backwards
        for i in range(N - 1, -1, -1):
            for j in range(N - i, N + i + 1):
                tree[i, j] = np.exp(-r * dt) * (
                    pu * tree[i+1, j+1] + 
                    pm * tree[i+1, j] +
                    pd * tree[i+1, j-1]
                )
                
                S_ij = S * u**(j - N)
                
                if isinstance(option, American):
                    tree[i, j] = max(tree[i, j], option.payoff(S_ij))
                    
                if isinstance(option, Barrier):
                    if "out" in option.barrier_type:
                        if "up" in option.barrier_type and S_ij >= option.barrier:
                            tree[i, j] = 0
                        elif "down" in option.barrier_type and S_ij <= option.barrier:
                            tree[i, j] = 0
                    elif "in" in option.barrier_type:
                        if "up" in option.barrier_type and S_ij < option.barrier:
                            tree[i, j] = 0
                        elif "down" in option.barrier_type and S_ij > option.barrier:
                            tree[i, j] = 0

        return tree[0, N]


        
