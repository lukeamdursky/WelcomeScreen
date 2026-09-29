"""
Portfolio Risk Dashboard
-------------------------
Computes and visualizes risk metrics for a user-defined portfolio of equities:
volatility, VaR/CVaR (historical & parametric), beta, max drawdown,
rolling volatility, and correlation structure.

Run with:
    streamlit run risk_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

DEFAULT_TICKERS = "AAPL, JPM, JNJ, XOM, PG, CAT, NEE, AMZN, LIN, AMT"
DEFAULT_BENCHMARK = "SPY"

# Rough GICS sector tags for the default tickers (used for display/context only;
# doesn't affect calculations, and isn't looked up for custom tickers you enter).
SECTOR_MAP = {
    "AAPL": "Information Technology",
    "JPM": "Financials",
    "JNJ": "Health Care",
    "XOM": "Energy",
    "PG": "Consumer Staples",
    "CAT": "Industrials",
    "NEE": "Utilities",
    "AMZN": "Consumer Discretionary",
    "LIN": "Materials",
    "AMT": "Real Estate",
}

# ---------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: list[str], benchmark: str, period: str) -> pd.DataFrame:
    all_symbols = list(dict.fromkeys(tickers + [benchmark]))
    data = yf.download(all_symbols, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = all_symbols
    return close.dropna(how="all").ffill().dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------
# Risk metric functions
# ---------------------------------------------------------------------

def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return returns.std() * np.sqrt(trading_days)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss number (e.g. 0.03 = 3% loss)."""
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = -historical_var(returns, confidence)  # var as negative threshold
    tail = returns[returns <= var]
    return -tail.mean() if len(tail) > 0 else np.nan


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def max_drawdown(prices: pd.Series) -> tuple[float, pd.Series]:
    cum = prices / prices.iloc[0]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min(), drawdown


def beta_vs_benchmark(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    cov = np.cov(asset_returns, bench_returns)[0, 1]
    var = np.var(bench_returns)
    return cov / var if var != 0 else np.nan


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights)
    w = w / w.sum()
    aligned = returns[w.index]
    return (aligned * w).sum(axis=1)


def monte_carlo_portfolio_var(
    asset_returns: pd.DataFrame,
    weights: dict,
    confidence: float = 0.95,
    n_sims: int = 10000,
    horizon_days: int = 1,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Simulate portfolio returns over `horizon_days` using a correlated multivariate-normal
    model calibrated to the historical mean/covariance of daily asset returns, then derive
    VaR and CVaR from the simulated distribution.

    Approach:
      1. Estimate daily mean vector (mu) and covariance matrix (Sigma) from historical returns.
      2. Cholesky-decompose Sigma so independent draws can be correlated to match Sigma.
      3. Draw n_sims x horizon_days x n_assets standard normal shocks, correlate them,
         and add the drift (mu) back in.
      4. Sum daily asset returns across the horizon (additive approximation) to get a
         horizon-length asset return per simulation, then combine via portfolio weights.
      5. VaR = negative of the (1-confidence) percentile of simulated portfolio returns.
         CVaR = negative mean of the tail beyond that percentile (expected shortfall).

    Returns (var, cvar, simulated_portfolio_returns).
    """
    rng = np.random.default_rng(seed)
    w = pd.Series(weights)
    w = w / w.sum()
    tickers = list(w.index)

    mu = asset_returns[tickers].mean().values
    sigma = asset_returns[tickers].cov().values
    n_assets = len(tickers)

    # Cholesky decomposition to correlate independent normal draws.
    # Add a tiny jitter to the diagonal in case the covariance matrix is near-singular
    # (e.g. two tickers that are almost perfectly correlated).
    jitter = 1e-10 * np.eye(n_assets)
    L = np.linalg.cholesky(sigma + jitter)

    # Shape: (n_sims, horizon_days, n_assets)
    z = rng.standard_normal((n_sims, horizon_days, n_assets))
    correlated_shocks = z @ L.T
    simulated_daily_returns = mu + correlated_shocks

    # Sum across the horizon to get a cumulative (additive) return per simulation per asset.
    simulated_horizon_returns = simulated_daily_returns.sum(axis=1)  # (n_sims, n_assets)

    # Combine into portfolio returns using weights.
    simulated_portfolio_returns = simulated_horizon_returns @ w.values

    var = -np.percentile(simulated_portfolio_returns, (1 - confidence) * 100)
    tail = simulated_portfolio_returns[simulated_portfolio_returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else np.nan

    return var, cvar, simulated_portfolio_returns


def rolling_var_series(returns: pd.Series, window: int, confidence: float,
                        method: str = "historical") -> pd.Series:
    """
    Out-of-sample rolling VaR: at each day t, estimate VaR using only the `window` days
    strictly BEFORE t (via shift(1)), then that estimate is compared against the actual
    realized return on day t. This avoids look-ahead bias — the model never sees the
    day it's being tested against.

    Returns a Series of VaR estimates (positive numbers = loss magnitude), aligned to `returns`.
    """
    if method == "historical":
        var_series = returns.rolling(window).apply(
            lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
        )
    else:  # parametric (rolling mean/std, normal assumption)
        mu = returns.rolling(window).mean()
        sigma = returns.rolling(window).std()
        z = norm.ppf(1 - confidence)
        var_series = -(mu + z * sigma)
    return var_series.shift(1)


def kupiec_pof_test(n: int, x: int, confidence: float) -> tuple[float, float]:
    """
    Kupiec (1995) Proportion-of-Failures test: checks whether the observed breach rate
    is statistically consistent with the VaR model's stated confidence level.

    H0: the true breach probability equals p = 1 - confidence (the model is well-calibrated).
    Returns (LR statistic, p-value). LR is chi-square distributed with 1 degree of freedom
    under H0; a low p-value (e.g. < 0.05) means the model's breach rate is significantly
    off from what it claims, in either direction (too many OR too few breaches).
    """
    from scipy.stats import chi2

    p = 1 - confidence
    pi_hat = x / n if n > 0 else np.nan

    def log_lik(p_, x_, n_):
        # Log-likelihood of x breaches out of n trials under breach probability p_,
        # handling the x=0 / x=n edge cases where one term would be log(0)*0.
        term1 = (n_ - x_) * np.log(1 - p_) if p_ < 1 else 0.0
        term2 = x_ * np.log(p_) if p_ > 0 else 0.0
        return term1 + term2

    ll_null = log_lik(p, x, n)
    ll_alt = log_lik(pi_hat, x, n)
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    return lr_stat, p_value


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("⚠️ Portfolio Risk Dashboard")
st.caption("Volatility, VaR/CVaR, beta, drawdown, and correlation analysis")

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_area("Tickers (comma-separated)", DEFAULT_TICKERS, height=70)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    benchmark = st.text_input("Benchmark", DEFAULT_BENCHMARK).strip().upper()
    period = st.selectbox("History window", ["6mo", "1y", "2y", "5y"], index=2)
    confidence = st.slider("VaR/CVaR confidence level", 0.90, 0.99, 0.95, 0.01)

    st.markdown("---")
    st.markdown("**Monte Carlo simulation**")
    n_sims = st.select_slider("Number of simulations", [1000, 5000, 10000, 25000, 50000], value=10000)
    horizon_days = st.slider("Horizon (trading days)", 1, 20, 1)

    st.markdown("---")
    st.markdown("**VaR backtesting**")
    backtest_window = st.slider("Rolling estimation window (days)", 60, 500, 250, step=10)
    backtest_method = st.radio("Backtest method", ["Historical", "Parametric"], horizontal=True)

    st.markdown("---")
    st.markdown("**Portfolio weights**")
    weight_mode = st.radio("Weighting", ["Equal weight", "Custom"], horizontal=True)
    weights = {}
    if weight_mode == "Equal weight":
        for t in tickers:
            weights[t] = 1.0
    else:
        for t in tickers:
            weights[t] = st.number_input(f"{t} weight", min_value=0.0, value=1.0, step=0.1)

if len(tickers) < 1:
    st.warning("Enter at least one ticker.")
    st.stop()

with st.spinner("Pulling price history..."):
    prices = fetch_prices(tickers, benchmark, period)

returns = compute_returns(prices)
asset_returns = returns[tickers]
bench_returns = returns[benchmark]
port_ret = portfolio_returns(returns, weights)

# ---------------------------------------------------------------------
# Section 1: Summary risk table
# ---------------------------------------------------------------------

st.subheader("Risk Metrics by Position")

rows = []
for t in tickers:
    r = asset_returns[t]
    dd, _ = max_drawdown(prices[t])
    rows.append({
        "Ticker": t,
        "Sector": SECTOR_MAP.get(t, "—"),
        "Ann. Volatility": annualized_vol(r),
        f"Hist. VaR ({int(confidence*100)}%, daily)": historical_var(r, confidence),
        f"Hist. CVaR ({int(confidence*100)}%, daily)": historical_cvar(r, confidence),
        f"Parametric VaR ({int(confidence*100)}%, daily)": parametric_var(r, confidence),
        "Beta vs " + benchmark: beta_vs_benchmark(r, bench_returns),
        "Max Drawdown": dd,
    })

risk_df = pd.DataFrame(rows).set_index("Ticker")
pct_cols = [c for c in risk_df.columns if c not in (f"Beta vs {benchmark}", "Sector")]
st.dataframe(
    risk_df.style.format({c: "{:.2%}" for c in pct_cols} | {f"Beta vs {benchmark}": "{:.2f}"}),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Section 2: Portfolio-level risk
# ---------------------------------------------------------------------

st.subheader("Portfolio-Level Risk")

port_dd, port_dd_series = max_drawdown((1 + port_ret).cumprod())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ann. Volatility", f"{annualized_vol(port_ret):.2%}")
c2.metric(f"Hist. VaR ({int(confidence*100)}%)", f"{historical_var(port_ret, confidence):.2%}")
c3.metric(f"Hist. CVaR ({int(confidence*100)}%)", f"{historical_cvar(port_ret, confidence):.2%}")
c4.metric(f"Beta vs {benchmark}", f"{beta_vs_benchmark(port_ret, bench_returns):.2f}")
c5.metric("Max Drawdown", f"{port_dd:.2%}")

st.markdown("**Portfolio Drawdown Over Time**")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=port_dd_series.index, y=port_dd_series.values,
                             fill="tozeroy", line=dict(color="crimson"), name="Drawdown"))
fig_dd.update_layout(height=350, yaxis_title="Drawdown", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------
# Section 3: Rolling volatility
# ---------------------------------------------------------------------

st.subheader("Rolling Volatility (Annualized, 21-day window)")

roll_vol = asset_returns.rolling(21).std() * np.sqrt(252)
roll_vol[benchmark] = bench_returns.rolling(21).std() * np.sqrt(252)

fig_vol = go.Figure()
for col in roll_vol.columns:
    style = dict(dash="dash") if col == benchmark else {}
    fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol[col], mode="lines",
                                  name=col, line=style))
fig_vol.update_layout(height=400, yaxis_title="Annualized Volatility",
                       margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------------------
# Section 4: Correlation matrix
# ---------------------------------------------------------------------

st.subheader("Correlation Matrix (Daily Returns)")

corr = asset_returns.corr()
fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, aspect="auto")
fig_corr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------
# Section 5: Return distribution (VaR visualization)
# ---------------------------------------------------------------------

st.subheader("Portfolio Daily Return Distribution")

var_line = -historical_var(port_ret, confidence)
fig_hist = px.histogram(port_ret, nbins=60, labels={"value": "Daily Return"})
fig_hist.add_vline(x=var_line, line_color="red", line_dash="dash",
                    annotation_text=f"VaR ({int(confidence*100)}%)")
fig_hist.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------
# Section 6: Monte Carlo VaR
# ---------------------------------------------------------------------

st.subheader("Monte Carlo VaR / CVaR")
st.caption(
    f"Simulates {n_sims:,} correlated return paths over a {horizon_days}-day horizon using a "
    "multivariate-normal model calibrated to the portfolio's historical mean and covariance "
    "(Cholesky decomposition preserves cross-asset correlation). This relaxes the single-path "
    "assumption of historical VaR and lets you stress a longer horizon than one day."
)

with st.spinner("Running simulation..."):
    mc_var, mc_cvar, mc_sim_returns = monte_carlo_portfolio_var(
        asset_returns, weights, confidence=confidence, n_sims=n_sims, horizon_days=horizon_days
    )

# For a fair comparison, scale the 1-day historical/parametric estimates to the same horizon
# using the square-root-of-time rule (standard approximation for iid returns).
horizon_scale = np.sqrt(horizon_days)
hist_var_h = historical_var(port_ret, confidence) * horizon_scale
hist_cvar_h = historical_cvar(port_ret, confidence) * horizon_scale
param_var_h = parametric_var(port_ret, confidence) * horizon_scale

st.markdown(f"**VaR / CVaR comparison at {int(confidence*100)}% confidence, {horizon_days}-day horizon**")
compare_df = pd.DataFrame({
    "Method": ["Historical (scaled)", "Parametric (scaled)", "Monte Carlo"],
    "VaR": [hist_var_h, param_var_h, mc_var],
    "CVaR": [hist_cvar_h, hist_cvar_h, mc_cvar],
})
# Historical CVaR scaling reuses the same scaled figure since tail-average scaling is less standard;
# Monte Carlo's CVaR is simulated directly and is the most reliable of the three for the tail.
compare_df.loc[1, "CVaR"] = np.nan  # parametric normal CVaR omitted (redundant with parametric VaR under normality)

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Monte Carlo VaR", f"{mc_var:.2%}")
mc2.metric("Monte Carlo CVaR", f"{mc_cvar:.2%}")
mc3.metric("Historical VaR (scaled)", f"{hist_var_h:.2%}")

fig_compare = px.bar(
    compare_df, x="Method", y="VaR",
    labels={"VaR": f"VaR at {int(confidence*100)}% confidence"},
    text_auto=".2%",
)
fig_compare.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown(f"**Simulated Portfolio Return Distribution ({horizon_days}-day horizon)**")
fig_mc_hist = px.histogram(mc_sim_returns, nbins=80,
                            labels={"value": f"{horizon_days}-Day Portfolio Return"})
fig_mc_hist.add_vline(x=-mc_var, line_color="red", line_dash="dash",
                       annotation_text=f"VaR ({int(confidence*100)}%)")
fig_mc_hist.add_vline(x=-mc_cvar, line_color="darkred", line_dash="dot",
                       annotation_text="CVaR")
fig_mc_hist.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_mc_hist, use_container_width=True)

st.caption(
    "Monte Carlo VaR/CVaR assume returns are multivariate-normal and that the historical mean/covariance "
    "over the selected lookback window is a reasonable estimate of near-term risk. Real returns exhibit "
    "fatter tails and volatility clustering that a normal model understates — treat this as one lens on "
    "risk, not a guarantee. The square-root-of-time scaling used for historical/parametric comparison "
    "also assumes independent, identically distributed daily returns."
)

# ---------------------------------------------------------------------
# Section 7: VaR Backtesting (Kupiec POF Test)
# ---------------------------------------------------------------------

st.subheader("VaR Backtesting")
st.caption(
    f"Re-estimates {backtest_method.lower()} VaR each day using only the trailing "
    f"{backtest_window} trading days of data (never the day being tested — no look-ahead bias), "
    "then checks how often the actual next-day loss exceeded that VaR estimate. A well-calibrated "
    f"{int(confidence*100)}% VaR model should be breached on roughly {int((1-confidence)*100)}% of days — "
    "materially more suggests the model understates risk; materially fewer suggests it's overly conservative."
)

rolling_var = rolling_var_series(port_ret, backtest_window, confidence, method=backtest_method.lower())
backtest_df = pd.DataFrame({"Return": port_ret, "VaR": rolling_var}).dropna()

if len(backtest_df) < 30:
    st.warning(
        f"Only {len(backtest_df)} out-of-sample days available after the {backtest_window}-day "
        "estimation window — increase the history window or shrink the rolling window for a "
        "more reliable backtest."
    )
else:
    backtest_df["Breach"] = backtest_df["Return"] < -backtest_df["VaR"]
    n_obs = len(backtest_df)
    n_breaches = int(backtest_df["Breach"].sum())
    expected_breaches = n_obs * (1 - confidence)
    breach_rate = n_breaches / n_obs
    lr_stat, p_value = kupiec_pof_test(n_obs, n_breaches, confidence)
    model_ok = p_value >= 0.05

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Observations", f"{n_obs}")
    b2.metric("Breaches", f"{n_breaches}", f"expected ≈ {expected_breaches:.1f}")
    b3.metric("Breach Rate", f"{breach_rate:.2%}", f"target {int((1-confidence)*100)}%")
    b4.metric("Kupiec p-value", f"{p_value:.3f}", "Pass ✅" if model_ok else "Fail ⚠️")

    if model_ok:
        st.success(
            f"Kupiec test p-value = {p_value:.3f} (≥ 0.05): the breach rate is statistically "
            "consistent with a well-calibrated model at this confidence level."
        )
    else:
        direction = "more" if breach_rate > (1 - confidence) else "fewer"
        st.error(
            f"Kupiec test p-value = {p_value:.3f} (< 0.05): the model produced significantly {direction} "
            "breaches than expected — its VaR estimates may be miscalibrated for this asset mix and period."
        )

    st.markdown("**Actual Returns vs. Rolling VaR Threshold**")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df["Return"], mode="markers",
        marker=dict(size=5, color=np.where(backtest_df["Breach"], "red", "steelblue")),
        name="Daily Return",
    ))
    fig_bt.add_trace(go.Scatter(
        x=backtest_df.index, y=-backtest_df["VaR"], mode="lines",
        line=dict(color="orange", dash="dash"), name=f"-VaR ({int(confidence*100)}%)",
    ))
    fig_bt.update_layout(height=420, yaxis_title="Daily Return",
                          margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_bt, use_container_width=True)
    st.caption("Red points are breach days — the actual loss exceeded the VaR estimate made using only prior data.")

    if n_breaches > 0:
        st.markdown("**Breach Details**")
        breach_table = backtest_df[backtest_df["Breach"]][["Return", "VaR"]].copy()
        breach_table.columns = ["Actual Return", "VaR Estimate"]
        st.dataframe(
            breach_table.style.format({"Actual Return": "{:.2%}", "VaR Estimate": "{:.2%}"}),
            use_container_width=True,
        )

    st.caption(
        "The Kupiec Proportion-of-Failures test is the same style of check regulators require banks to run "
        "on internal VaR models (Basel traffic-light backtesting). It only tests whether the breach FREQUENCY "
        "is correct — it doesn't check whether breaches cluster together in time (for that, a test like "
        "Christoffersen's independence test would be the next step)."
    )

st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
           "VaR/CVaR are daily historical/parametric estimates based on the selected lookback window — "
           "not forward-looking guarantees. Beta computed vs. daily returns of the benchmark.")
