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

log_returns = np.log(prices / prices.shift(1)).dropna()
# --------------------------- EMPIRICAL DISTRIBUTION DIAGNOSTICS---------------------------------

# QQ PLOTS
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

# HISTOGRAMS
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, ticker in zip(axes.flatten(), tickers):
    r = log_returns[ticker].dropna()
    mu, sigma = r.mean(), r.std()
    sns.histplot(r, bins=60, stat="density", color="steelblue", ax=ax, alpha=0.6)
    x = np.linspace(r.min(), r.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="firebrick", lw=1.0, label="Fitted Normal")
    ax.set_title(f"{ticker}", fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle("Empirical Density vs. Fitted Normal", fontsize=14)
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

# --------------------------- V2: MODEL-BUILDING (VARIANCE-COVARIANCE) VaR / ES---------------------------

def portfolio_sd(Sigma, positions):
    """
    Portfolio standard deviation under the linear model (Hull, Ch.13, Sec 13.1).

    sigma_P^2 = positions^T @ Sigma @ positions, where Sigma is the covariance
    matrix of risk-factor (log-)returns and positions is the $ position vector,
    both ordered consistently with `tickers`.
    """
    positions = np.asarray(positions, dtype=float)
    variance = positions @ (Sigma @ positions)
    return np.sqrt(variance)

# --------------------------- Model 2a: Equal-Weighted (Sample) Covariance ---------------------------

# Sample covariance of log-returns over the same trailing window used by Model 1
Sigma_ew = log_returns.tail(LOOKBACK_DAYS).cov().values
sigma_p_ew = portfolio_sd(Sigma_ew, positions.values)

# --------------------------- Model 2b: EWMA (RiskMetrics) Covariance ---------------------------

LAMBDA = 0.94
BURN_IN = 100

# EWMA covariance matrix (Hull, Ch.13, Sec 13.3 / RiskMetrics, lambda = 0.94).
# Recursive update Sigma_t = lambda * Sigma_(t-1) + (1 - lambda) * outer(r_(t-1), r_(t-1)),
# run over the full available return history (unbounded, exponentially decaying weights,
# per Hull's own treatment rather than a fixed window). Seeded with the equal-weighted
# sample covariance over the first BURN_IN days.
returns_arr = log_returns.values
n_days, n_assets = returns_arr.shape

Sigma_ewma = np.zeros((n_days, n_assets, n_assets))
Sigma_ewma[BURN_IN - 1] = np.cov(returns_arr[:BURN_IN], rowvar=False)

for t in range(BURN_IN, n_days):
    r_prev = returns_arr[t - 1]
    Sigma_ewma[t] = LAMBDA * Sigma_ewma[t - 1] + (1 - LAMBDA) * np.outer(r_prev, r_prev)

# Full time series of sigma_P,t = sqrt(positions^T @ Sigma_t @ positions)
sigma_p_ewma = pd.Series(
    [portfolio_sd(Sigma_ewma[t], positions.values) for t in range(BURN_IN - 1, n_days)],
    index=log_returns.index[BURN_IN - 1:],
    name="sigma_p_ewma",
)
