# optionlib

**An open-source Python library for pricing vanilla and exotic options — Black-Scholes, Monte Carlo, Binomial, Trinomial trees, and a Machine Learning engine for European options.**

*Personal project — built to implement and understand the core pricing methods used in quantitative finance, including a neural network pricer trained on synthetic data.*

---

## Overview

optionlib provides a complete options pricing framework in Python:

- price European and American options using closed-form Black-Scholes and tree methods
- price path-dependent exotics (Asian, Barrier, Lookback) using Monte Carlo simulation
- price multi-asset exotics (Spread, Rainbow) with correlated GBM paths
- price European options using a trained neural network (ML engine)
- compute Greeks analytically (Black-Scholes) or numerically via finite differences
- solve for implied volatility using Newton-Raphson inversion
- generate synthetic GBM price paths for testing and visualisation
- fetch live market data via yfinance (optional)

The main demonstration lives in:

- `notebooks/demo.ipynb` — end-to-end walkthrough of all option types, pricing engines, and ML sensitivity analysis

---

## Supported Options

| Option Type | Description |
|-------------|-------------|
| **European** | Exercise at expiry only |
| **American** | Exercise at any time before expiry |
| **Asian** | Payoff based on average price over the life of the option |
| **Barrier** | Activated or extinguished when price crosses a barrier level |
| **Lookback** | Payoff based on the optimal price observed over the life of the option |
| **Spread** | Payoff based on the difference between two underlying assets |
| **Rainbow** | Payoff based on the best or worst performing asset in a basket |

---

## Supported Engines

| Engine | European | American | Asian | Barrier | Lookback | Spread | Rainbow |
|--------|:--------:|:--------:|:-----:|:-------:|:--------:|:------:|:-------:|
| Black-Scholes | ✅ exact | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Binomial Tree | ✅ | ✅ | ❌ | ⚠️ knock-out only | ❌ | ❌ | ❌ |
| Trinomial Tree | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Monte Carlo | ✅ | ⚠️ approx | ✅ | ✅ | ✅ | ✅ | ✅ |
| ML (Neural Net) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

⚠️ approximate — works but with known limitations (see docs)

---

## Quickstart

```python
from optionlib import European, American, Asian, Barrier
from optionlib import Black_Scholes, MonteCarlo, Binomial, Trinomial
from optionlib import Greeks
from optionlib import MLPricer

# price a European call with Black-Scholes
opt = European(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
price = Black_Scholes.price(opt)
print(f"BS Price: {price:.4f}")

# price the same option with the ML engine
price_ml = MLPricer.price(opt)
print(f"ML Price: {price_ml:.4f}")

# price the same option with Monte Carlo
price_mc = MonteCarlo.price(opt, M=50000)
print(f"MC Price: {price_mc:.4f}")

# price an American put with the trinomial tree
am_put = American(S=100, K=105, T=1, r=0.05, sigma=0.2, option_type="put")
price_tree = Trinomial.price(am_put, N=200)
print(f"American Put: {price_tree:.4f}")

# compute Greeks analytically
delta = Greeks.analytical(opt, "delta")
vega  = Greeks.analytical(opt, "vega")
print(f"Delta: {delta:.4f}, Vega: {vega:.4f}")
```

---

## ML Engine

The ML engine is a feedforward neural network trained on 500,000 synthetic European options priced with Black-Scholes.

**Architecture:** 7 inputs → 64 → 64 → 64 → 1 output (price)  
**Training:** Adam optimizer, MSE loss, 80/20 train/val split, feature scaling (MinMaxScaler)  
**Accuracy:** ~$0.07 average error vs Black-Scholes on a $10 ATM option

You can retrain the model from scratch in trainer.py

Weights are saved automatically to `optionlib/ml/training/model_weights.pt`.

---

## Module Details

### core/

Defines all option types as Python classes. Every class inherits from the abstract `Option` base class and implements a `payoff` method.

```python
from optionlib import European, Asian, Barrier, Lookback, Spread, Rainbow

# vanilla
opt = European(S=100, K=100, T=1, r=0.05, sigma=0.2)

# asian — arithmetic or geometric averaging
asian = Asian(S=100, K=100, T=1, r=0.05, sigma=0.2, averaging="arithmetic")

# barrier — knock-in or knock-out, up or down
barrier = Barrier(S=100, K=100, T=1, r=0.05, sigma=0.2,
                  barrier=120, barrier_type="up-and-out")

# lookback — fixed or floating strike
lookback = Lookback(S=100, K=100, T=1, r=0.05, sigma=0.2, lookback_type="fixed")

# spread — two correlated underlyings
spread = Spread(S1=100, S2=95, K=5, T=1, r=0.05,
                sigma1=0.2, sigma2=0.18, rho=0.6)

# rainbow — best-of or worst-of basket
rainbow = Rainbow(spots=[100, 105, 98], K=100, T=1, r=0.05,
                  sigmas=[0.2, 0.22, 0.18], rho=rho_matrix)
```

### engines/

Four traditional pricing engines plus the ML engine. All expose a single `price(option)` static method.

```python
from optionlib import Black_Scholes, MonteCarlo, Binomial, Trinomial
from optionlib import MLPricer

Black_Scholes.price(opt)                    # European only, exact closed-form
MonteCarlo.price(opt, M=100000, steps=252)  # all option types
Binomial.price(opt, N=200)                  # European and American
Trinomial.price(opt, N=200)                 # European, American, all barriers
MLPricer.price(opt)                         # European only, neural network
```

### ml/

```
optionlib/ml/
├── data_gen.py     # generate synthetic training data using Black-Scholes
├── model.py        # neural network architecture (PyTorch)
├── trainer.py      # training loop with validation and feature scaling
├── pricer.py       # load weights and price any European option
└── training/
    ├── model_weights.pt   # saved model weights (not tracked by git)
    └── scaler.pkl         # saved feature scaler (not tracked by git)
```

### utils/

```python
from optionlib import Greeks, historical_volatility, implied_vol, simulate_gbm

Greeks.analytical(opt, "delta")
Greeks.numerical(asian, MonteCarlo, "delta", epsilon=0.01)
ann_vol = historical_volatility(prices)
iv = implied_vol(opt, market_price=10.50)
path = simulate_gbm(S=100, r=0.05, sigma=0.2, T=1, steps=252, seed=42)
```

---

## Pipeline

```
Define option → Choose engine → Price → Compute Greeks → Analyse
   core/           engines/      .price()   utils/greeks    notebook
```

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/fedecarz/optionlib.git
cd optionlib
pip install -e .
```

---

## Repository Structure

```
optionlib/
│
├── optionlib/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── option.py          # abstract base class
│   │   ├── vanilla.py         # European, American
│   │   ├── asian.py
│   │   ├── barrier.py
│   │   ├── lookback.py
│   │   ├── spread.py
│   │   └── rainbow.py
│   │
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── black_scholes.py
│   │   ├── monte_carlo.py
│   │   ├── binomial.py
│   │   └── trinomial.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── greeks.py
│   │   ├── volatility.py
│   │   ├── simulation.py      # GBM
│   │   └── market.py          # yfinance wrapper
│   │
│   └── ml/                    
│        ├── __init__.py
│        ├── data_gen.py     # generate training data
│        ├── model.py        # neural net (PyTorch)
│        ├── trainer.py
│        ├── pricer.py
│        └── training/
│           ├── model_weights.pt   # saved model weights (not tracked by git)
│           └── scaler.pkl         # saved feature scaler (not tracked by git)
│
├── tests/                         
│   ├── __init__.py
│   ├── test_european.py         # Planned
│   ├── test_american.py         # Planned
│   └── test_exotics.py          # Planned
│
├── notebooks/
│   └── demo.ipynb
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## Roadmap

| Feature | Status |
|---------|--------|
| European / American pricing | ✅ Done |
| Exotic options (Asian, Barrier, Lookback, Spread, Rainbow) | ✅ Done |
| Black-Scholes engine | ✅ Done |
| Monte Carlo engine | ✅ Done |
| Binomial tree engine | ✅ Done |
| Trinomial tree engine | ✅ Done |
| Analytical and numerical Greeks | ✅ Done |
| Implied vol solver | ✅ Done |
| GBM simulation utility | ✅ Done |
| yfinance market data wrapper | ✅ Done |
| ML pricing engine (neural network) | ✅ Done |
| Demo notebook | ✅ Done |
| Tests | 📋 Planned |

---

## Requirements

Python 3.8+ · numpy · scipy · pandas · torch · scikit-learn · joblib

Optional: yfinance · matplotlib · jupyter

---

## Limitations and Natural Extensions

| Area | Current | Natural Extension |
|------|---------|-------------------|
| American MC | Ignores early exercise | Longstaff-Schwartz algorithm |
| Binomial barriers | Knock-out only, slippage error | Node alignment, trinomial preferred |
| Volatility surface | Flat vol per option | Vol surface interpolation (SVI, SABR) |
| ML engine | European only | Train on all option types |
| Greeks for exotics | Numerical only | AAD (Adjoint Algorithmic Differentiation) |

---

## License

MIT
