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
    return rf / 100