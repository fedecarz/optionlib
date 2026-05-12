# optionlib

**An open-source Python library for pricing vanilla and exotic options — Black-Scholes, Monte Carlo, Binomial and Trinomial trees.**

*Personal project — built to implement and understand the core pricing methods used in quantitative finance.*

---

## Overview

optionlib provides a complete options pricing framework in pure Python:

- price European and American options using closed-form Black-Scholes and tree methods
- price path-dependent exotics (Asian, Barrier, Lookback) using Monte Carlo simulation
- price multi-asset exotics (Spread, Rainbow) with correlated GBM paths
- compute Greeks analytically (Black-Scholes) or numerically via finite differences
- solve for implied volatility using Newton-Raphson inversion
- generate synthetic GBM price paths for testing and visualisation
- fetch live market data via yfinance (optional)

The main demonstration lives in:

- `notebooks/demo.ipynb` — end-to-end walkthrough of all option types and pricing engines

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

⚠️ approximate — works but with known limitations (see docs)

---

## Quickstart

```python
from optionlib import European, American, Asian, Barrier
from optionlib import Black_Scholes, MonteCarlo, Binomial, Trinomial
from optionlib import Greeks

# price a European call with Black-Scholes
opt = European(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
price = Black_Scholes.price(opt)
print(f"BS Price: {price:.4f}")

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

# compute Greeks numerically for any option/engine
delta_num = Greeks.numerical(opt, MonteCarlo, "delta")
print(f"Numerical Delta: {delta_num:.4f}")
```

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

Four independent pricing engines. All expose a single `price(option)` static method.

```python
from optionlib import Black_Scholes, MonteCarlo, Binomial, Trinomial

# Black-Scholes — European only, exact closed-form
Black_Scholes.price(opt)

# Monte Carlo — all option types, accuracy increases with M
MonteCarlo.price(opt, M=100000, steps=252)

# Binomial tree — European and American, knock-out barriers
Binomial.price(opt, N=200)

# Trinomial tree — European, American, all barrier types
Trinomial.price(opt, N=200)
```

### utils/

#### Greeks

```python
from optionlib import Greeks

# analytical — Black-Scholes closed-form, European only, instant
Greeks.analytical(opt, "delta")   # → 0.5398
Greeks.analytical(opt, "gamma")   # → 0.0199
Greeks.analytical(opt, "vega")    # → 0.3752
Greeks.analytical(opt, "theta")   # → -0.0136
Greeks.analytical(opt, "rho")     # → 0.4623

# numerical — finite differences, any option, any engine
Greeks.numerical(asian, MonteCarlo, "delta", epsilon=0.01)
Greeks.numerical(barrier, Trinomial, "vega", epsilon=0.01)
```

#### Volatility

```python
from optionlib import historical_volatility, implied_vol

# historical vol from a price series
ann_vol = historical_volatility(prices)   # → 0.1923

# implied vol — Newton-Raphson inversion of Black-Scholes
iv = implied_vol(opt, market_price=10.50)  # → 0.2134
```

#### Simulation

```python
from optionlib import simulate_gbm

# generate a single GBM path
path = simulate_gbm(S=100, r=0.05, sigma=0.2, T=1, steps=252, seed=42)
# returns numpy array of length 253 (including S at t=0)
```

#### Market Data (optional — requires yfinance)

```python
from optionlib.utils.market import get_spot, get_historical_prices, get_risk_free_rate

spot   = get_spot("AAPL")
prices = get_historical_prices("AAPL", period="1y")
r      = get_risk_free_rate()
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

Core dependencies are installed automatically. For optional features:

```bash
pip install yfinance          # market data
pip install torch             # ML engine (Phase 2)
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
│   │   ├── option.py          # abstract base class
│   │   ├── vanilla.py         # European, American
│   │   ├── asian.py
│   │   ├── barrier.py
│   │   ├── lookback.py
│   │   ├── spread.py
│   │   └── rainbow.py
│   │
│   ├── engines/
│   │   ├── black_scholes.py
│   │   ├── monte_carlo.py
│   │   ├── binomial.py
│   │   └── trinomial.py
│   │
│   ├── utils/
│   │   ├── greeks.py
│   │   ├── volatility.py
│   │   ├── simulation.py
│   │   └── market.py          # yfinance wrapper (optional)
│   │
│   └── ml/                    # Phase 2
│       └── __init__.py
│
├── tests/
│   ├── test_european.py
│   ├── test_american.py
│   └── test_exotics.py
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
| Demo notebook | 🔄 In progress |
| ML pricing engine (neural network) | 📋 Phase 2 |
| Tests | 📋 Planned |

---

## Requirements

Python 3.8+ · numpy · scipy · pandas

Optional: yfinance · torch · scikit-learn · matplotlib · jupyter

---

## Limitations and Natural Extensions

| Area | Current | Natural Extension |
|------|---------|-------------------|
| American MC | Ignores early exercise | Longstaff-Schwartz algorithm |
| Binomial barriers | Knock-out only, slippage error | Node alignment, trinomial preferred |
| Volatility surface | Flat vol per option | Vol surface interpolation (SVI, SABR) |
| ML engine | Not yet built | Neural net trained on BS/MC prices |
| Greeks for exotics | Numerical only | AAD (Adjoint Algorithmic Differentiation) |

---

## License

MIT
