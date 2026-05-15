import numpy as np
import pandas as pd
from optionlib.engines.black_scholes import Black_Scholes
from optionlib.core.vanilla import European

def generate_data(n_samples = 100000, seed = 42):
    """
    Generate synthetic option pricing data using Black-Scholes as ground truth.
    """
    rng = np.random.default_rng(seed=seed)

    S = rng.uniform(50,200,n_samples)
    K = rng.uniform(50,200,n_samples)
    T = rng.uniform(0.05,2.0,n_samples)
    r = rng.uniform(0.01,0.1,n_samples)
    sigma = rng.uniform(0.05,0.8,n_samples)
    q = np.zeros(n_samples)
    option_type = rng.choice(["call", "put"], n_samples)

    prices = []
    for i in range(n_samples):
        option = European(S[i], K[i], T[i], r[i], sigma[i], q[i], option_type[i])
        price = Black_Scholes.price(option)
        prices.append(price)

    df = pd.DataFrame({
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": q,
        "option_type": (option_type == "call").astype(int),  # encode: call=1, put=0
        "price": prices
    })

    return df