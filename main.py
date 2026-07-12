# ---------------------------LIB Imports---------------------------

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------DATA Imports---------------------------

# Portfolio definition: asset class, risk factor, ticker, position ($M)
portfolio = pd.DataFrame([
    {"asset_class": "Equities",    "risk_factor": "S&P 500",                 "ticker": "^GSPC",      "position_usd_mm": 15},
    {"asset_class": "Equities",    "risk_factor": "Euro Stoxx 50",           "ticker": "^STOXX50E",  "position_usd_mm": 10},
    {"asset_class": "Equities",    "risk_factor": "FTSE 100",                "ticker": "^FTSE",      "position_usd_mm": 10},
    {"asset_class": "FX",          "risk_factor": "EUR/USD",                 "ticker": "EURUSD=X",   "position_usd_mm": 10},
    {"asset_class": "FX",          "risk_factor": "GBP/USD",                 "ticker": "GBPUSD=X",   "position_usd_mm": 5},
    {"asset_class": "FX",          "risk_factor": "USD/JPY",                 "ticker": "USDJPY=X",   "position_usd_mm": 5},
    {"asset_class": "Rates/Bonds", "risk_factor": "US 10Y Treasury proxy",   "ticker": "IEF",         "position_usd_mm": 20},
    {"asset_class": "Rates/Bonds", "risk_factor": "US Long Bond proxy",      "ticker": "TLT",         "position_usd_mm": 15},
    {"asset_class": "Rates/Bonds", "risk_factor": "IG Corporate Bond proxy", "ticker": "LQD",         "position_usd_mm": 10},
])

tickers = portfolio["ticker"].tolist()

# Pull daily price history from Yahoo Finance
prices = yf.download(tickers, start="2015-01-01", auto_adjust=True)["Close"]
prices = prices[tickers]  # preserve portfolio order

print(prices.tail())
