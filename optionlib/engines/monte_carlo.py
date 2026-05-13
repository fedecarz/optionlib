import numpy as np
from optionlib.core.vanilla import European, American
from optionlib.core.asian import Asian
from optionlib.core.barrier import Barrier
from optionlib.core.lookback import Lookback
from optionlib.core.spread import Spread
from optionlib.core.rainbow import Rainbow


class MonteCarlo:
    """
    Monte Carlo simulation engine for pricing all option types.
    Accuracy increases with M (number of simulations).
    """

    @staticmethod
    def price(option, M=10000, steps=252):
        """
        General function that routes to the correct pricing method based on the option type
        """
        
        # Note: American options priced via MC ignore early exercise - use trees for accuracy
        if isinstance(option, European) or isinstance(option, American):
            return MonteCarlo._price_european(option, M, steps)
        elif isinstance(option, Asian):
            return MonteCarlo._price_asian(option, M, steps)
        elif isinstance(option, Barrier):
            return MonteCarlo._price_barrier(option, M, steps)
        elif isinstance(option, Lookback):
            return MonteCarlo._price_lookback(option, M, steps)
        elif isinstance(option, Spread):
            return MonteCarlo._price_spread(option, M, steps)
        elif isinstance(option, Rainbow):
            return MonteCarlo._price_rainbow(option, M, steps)
        else:
            raise TypeError(f"MonteCarlo does not support {type(option).__name__}")


    @staticmethod
    def _price_european(option, M, steps):
        
        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        # Simulate paths (vectorization)
        dt = T / steps
        Z = np.random.normal(0, 1, (M, steps))
        drift = (r - q - 0.5*sigma**2) * dt     # GBM decomposition
        diffusion = sigma * np.sqrt(dt) * Z
        S_paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))

        S_final = S_paths[:, -1]
        payoffs = np.array([option.payoff(s) for s in S_final])
        price = np.exp(-r * T) * np.mean(payoffs)

        return price

    @staticmethod
    def _price_asian(option, M, steps):
        
        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        dt = T / steps
        Z = np.random.normal(0, 1, (M, steps))
        drift = (r - q - 0.5*sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        S_paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))

        payoffs = []
        for path in S_paths:
            if option.averaging == "arithmetic":
                S_avg = np.mean(path)
            else:
                S_avg = np.exp(np.mean(np.log(path)))
            payoffs.append(option.payoff(S_avg))

        price = np.exp(-r * T) * np.mean(payoffs)
        return price


    @staticmethod
    def _price_barrier(option, M, steps):
                
        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        dt = T / steps
        Z = np.random.normal(0, 1, (M, steps))
        drift = (r - q - 0.5*sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        S_paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))

        payoffs = []
        for path in S_paths:
            S_final = path[-1]
            
            # check breach
            if "up" in option.barrier_type:
                breached = any(path >= option.barrier)
            else:
                breached = any(path <= option.barrier)

            # check if alive
            if "out" in option.barrier_type:
                alive = not breached
            else:
                alive = breached

            # compute payoff
            if alive:
                payoffs.append(option.payoff(S_final, path))
            else:
                payoffs.append(0)

        price = np.exp(-r * T) * np.mean(payoffs)
        return price

    @staticmethod
    def _price_lookback(option, M, steps):
        
        S = option.S
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q

        dt = T / steps
        Z = np.random.normal(0, 1, (M, steps))
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        S_paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))

        payoffs = []
        for path in S_paths:
            payoff = option.payoff(path)
            payoffs.append(payoff)

        price = np.exp(-r * T) * np.mean(payoffs)
        return price


        

    @staticmethod
    def _price_spread(option, M, steps):

        S = option.S
        S2 = option.S2
        T = option.T
        r = option.r
        sigma = option.sigma
        sigma2 = option.sigma2
        q = option.q

        dt = T / steps
        Z1 = np.random.normal(0,1,(M,steps))
        Z2 = np.random.normal(0,1,(M,steps))
        Z2 = option.rho * Z1 + np.sqrt(1 - option.rho**2) * Z2

        drift1 = (r - q - 0.5 * sigma**2) * dt
        diffusion1 = sigma * np.sqrt(dt) * Z1
        S_paths = S * np.exp(np.cumsum(drift1 + diffusion1, axis=1))

        drift2 = (r - q - 0.5*sigma2**2) * dt
        diffusion2 = sigma2 * np.sqrt(dt) * Z2
        S2_paths = S2 * np.exp(np.cumsum(drift2 + diffusion2, axis=1))


        payoffs = []
        for path1, path2 in zip(S_paths, S2_paths):
            S1_final = path1[-1]
            S2_final = path2[-1]
            payoffs.append(option.payoff(S1_final, S2_final))

        price = np.exp(-r * T) * np.mean(payoffs)
        return price



    @staticmethod
    def _price_rainbow(option, M, steps):
        
        spots = option.spots
        T = option.T
        r = option.r
        sigmas = option.sigmas
        q = option.q
        rho = option.rho

        N = len(spots)
        dt = T / steps
        corr_matrix = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        Z = np.random.normal(0,1,(M,steps,N))
        Z_correlated = Z @ L.T
        S_paths = np.zeros((N, M, steps))

        # loop over each asset i
        for i in range(N):
            drift_i = (r - q - 0.5 * sigmas[i]**2) * dt
            diffusion_i = sigmas[i] * np.sqrt(dt) * Z_correlated[:, :, i]
            S_paths[i] = spots[i] * np.exp(np.cumsum(drift_i + diffusion_i, axis=1))

        payoffs = []
        for m in range(M):
            finals = [S_paths[i][m, -1] for i in range(N)]
            payoffs.append(option.payoff(finals))
        
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
