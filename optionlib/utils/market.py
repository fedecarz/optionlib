try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance to use market data features: pip install yfinance")

import pandas as pd


def get_spot(ticker: str):
    spot = yf.Ticker(ticker).fast_info["lastPrice"]
    return float(spot)
    
def get_historical_prices(ticker: str, period="1y"):
    prices = yf.Ticker(ticker).history(period = period)["Close"]
    return pd.Series(prices)

def get_risk_free_rate():
    rf = yf.Ticker("^IRX").fast_info["lastPrice"]
    return rf / 100        # convert from % to decimal

def from_market(ticker: str, K, expiry, option_type = "call", option_class = "european", period = "1y"):
    """
    helper to price options with '.price()'
    Params:
        ticker      : e.g. "AAPL"
        K           : strike price
        expiry      : expiry date as string "YYYY-MM-DD" or float (years)
        option_type : "call" or "put"
        option_class: "european" or "american"
        period      : historical period for vol estimation (default "1y")
    """

    from datetime import date

    # Fetch spot
    S = get_spot(ticker)

    # Fetch risk-free-rate
    r = get_risk_free_rate()

    # Compute T
    if isinstance(expiry, str):
        expiry_date = date.fromisoformat(expiry)
        T = (expiry_date - date.today()).days / 365        # convert calendar days to years
        if T <= 0:
            raise ValueError(f"Expiry date {expiry} is in the past.")
    else:
        T = float (expiry)        # already in years

    # Compute historical vol
    prices = get_historical_prices(ticker, period=period)
    sigma = float(historical_volatility(prices))

    # Build option
    if option_class == "european":
        return European(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    elif option_class == "american":
        return American(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    else:
        raise ValueError("option_class must be 'european' or 'american'")
