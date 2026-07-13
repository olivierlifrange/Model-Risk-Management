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

# ---------------------------DATA PREPARATION---------------------------

# Pull daily price history from Yahoo Finance
prices = yf.download(tickers, start="2015-01-01", auto_adjust=True)["Close"]
prices = prices[tickers]  # preserve portfolio order

# Daily log returns per risk factor
log_returns = np.log(prices / prices.shift(1)).dropna()

positions = portfolio.set_index("ticker")["position_usd_mm"][tickers]


# --------------------------- EMPIRICAL DISTRIBUTION DIAGNOSTICS---------------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, ticker in zip(axes.flatten(), tickers):
    r = log_returns[ticker].dropna()
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{ticker}", fontsize=10)
fig.suptitle("Empirical Log-Return Distributions vs. Normal", fontsize=14)
plt.tight_layout()
plt.show()
# --------------------------- V1: PLAIN HISTORICAL SIMULATION VaR / ES---------------------------

LOOKBACK_DAYS = 252
CONFIDENCE = 0.99

# Most recent 252 days of returns, revalue today's positions under each historical scenario
scenario_returns = log_returns.tail(LOOKBACK_DAYS)
scenario_pnl = (np.exp(scenario_returns) - 1) * positions.values
portfolio_pnl = scenario_pnl.sum(axis=1)

v1var = -portfolio_pnl.quantile(1 - CONFIDENCE)
v1es = -portfolio_pnl[portfolio_pnl <= -v1var].mean()

v1_summary = pd.DataFrame({
    "1-day": [v1var, v1es],
    "10-day": [v1var * np.sqrt(10), v1es * np.sqrt(10)],
}, index=[f"VaR {CONFIDENCE:.0%}", f"ES {CONFIDENCE:.0%}"])

print(v1_summary.round(4).to_string())

# --------------------------- V2: PLAIN HISTORICAL SIMULATION VaR / ES---------------------------

