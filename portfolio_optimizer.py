"""
Portfolio Construction Dashboard
----------------------------------
Builds and compares several portfolio construction methodologies on a 20-stock,
11-sector universe: Equal-Weight, Minimum-Variance, Risk Parity, Max-Sharpe
(historical returns), and Max-Sharpe (Black-Litterman equilibrium returns, no views).

Covariance can be estimated via sample covariance or Ledoit-Wolf shrinkage.
Includes an efficient frontier visualization and an out-of-sample backtest that
computes weights on a training period only and evaluates them on a strictly later,
untouched holdout period (no look-ahead bias).

Methodology notes and known limitations are called out directly in the UI rather
than hidden in comments — see the "Methodology & Limitations" section at the bottom.

Run with:
    streamlit run portfolio_optimizer.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime

st.set_page_config(page_title="Portfolio Construction Dashboard", layout="wide")

# ---------------------------------------------------------------------
# Universe: 20 stocks across all 11 GICS sectors
# ---------------------------------------------------------------------

SECTOR_MAP = {
    "MSFT": "Information Technology", "AAPL": "Information Technology",
    "UNH": "Health Care", "JNJ": "Health Care",
    "JPM": "Financials", "BAC": "Financials",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "GOOGL": "Communication Services", "VZ": "Communication Services",
    "CAT": "Industrials", "HON": "Industrials",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "XOM": "Energy", "CVX": "Energy",
    "NEE": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate",
    "LIN": "Materials",
}
DEFAULT_TICKERS = list(SECTOR_MAP.keys())
DEFAULT_BENCHMARK = "SPY"

TRADING_DAYS = 252

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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_caps(tickers: list[str]) -> dict:
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("marketCap", None)
        except Exception:
            caps[t] = None
    return caps


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------
# Covariance & expected return estimation
# ---------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * TRADING_DAYS


def shrinkage_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance. Shrinks the noisy sample covariance toward a
    structured target, reducing estimation error — particularly valuable when the number
    of assets is not tiny relative to the number of return observations. Returns
    (annualized covariance matrix, shrinkage intensity used, 0=no shrinkage, 1=full shrinkage)."""
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_ * TRADING_DAYS, lw.shrinkage_


def historical_mean_returns(returns: pd.DataFrame) -> np.ndarray:
    return returns.mean().values * TRADING_DAYS


def implied_risk_aversion(bench_returns: pd.Series, rf_annual: float) -> float:
    """delta = (E[market return] - Rf) / Var(market return). This is the risk-aversion
    coefficient implied by how the market currently prices risk, using the benchmark
    as a proxy for 'the market portfolio'."""
    bench_annual_ret = bench_returns.mean() * TRADING_DAYS
    bench_annual_var = bench_returns.var() * TRADING_DAYS
    return (bench_annual_ret - rf_annual) / bench_annual_var if bench_annual_var > 0 else 2.5


def black_litterman_equilibrium_returns(cov: np.ndarray, market_weights: np.ndarray,
                                          delta: float) -> np.ndarray:
    """pi = delta * Sigma * w_mkt. These are the returns that would make the current
    market-cap-weighted portfolio the optimal (unconstrained mean-variance) portfolio —
    i.e. 'what the market must collectively believe, given how it's currently positioned.'
    No investor views are blended in here; this is the pure equilibrium prior."""
    return delta * (cov @ market_weights)


def estimate_shares_outstanding(current_market_caps: pd.Series, current_prices: pd.Series) -> pd.Series:
    """Approximates shares outstanding as (today's market cap) / (today's price). Shares
    outstanding drift over time due to buybacks and issuance, but far more slowly than market
    cap itself (which moves with price every day) — so this is a reasonable time-invariant
    proxy, not an exact figure. Used only to reconstruct a POINT-IN-TIME market-cap-weight
    estimate for periods other than today (see market_cap_weights_at), which avoids using
    today's market cap directly when estimating weights for a historical training window."""
    return current_market_caps / current_prices


def market_cap_weights_at(shares_outstanding: pd.Series, prices_row: pd.Series) -> np.ndarray:
    """Reconstructs an approximate market-cap-weight vector AS OF a specific point in time
    (prices_row), using the time-invariant shares-outstanding proxy. This is what should be
    used for Black-Litterman inside a backtest's training window — using TODAY's actual
    market-cap weights there would leak future information into the past (a company that's
    large today may not have been large at the end of the training period)."""
    tickers = shares_outstanding.index
    proxy_caps = shares_outstanding[tickers] * prices_row[tickers]
    return (proxy_caps / proxy_caps.sum()).values


# ---------------------------------------------------------------------
# Portfolio optimizers (long-only, fully invested: weights sum to 1, no shorting)
# ---------------------------------------------------------------------

def _base_bounds(n: int) -> list:
    return [(0.0, 1.0)] * n


def optimize_min_variance(cov: np.ndarray) -> np.ndarray:
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                    bounds=_base_bounds(n), constraints=cons, options={"maxiter": 1000})
    return res.x if res.success else x0


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 1e6

    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP",
                    bounds=_base_bounds(n), constraints=cons, options={"maxiter": 1000})
    return res.x if res.success else x0


def optimize_risk_parity(cov: np.ndarray, n_restarts: int = 6, seed: int = 0) -> np.ndarray:
    """Equal risk contribution portfolio. Uses a scale-invariant objective (percentage
    risk contribution, not absolute) since the absolute-contribution formulation can
    silently fail to converge when portfolio variance is small (a numerical-tolerance
    trap, not a modeling choice) — several random restarts guard against local minima."""
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-6, 1.0)] * n

    def rp_objective(w):
        port_var = w @ cov @ w
        risk_contrib_pct = (w * (cov @ w)) / port_var
        return np.sum((risk_contrib_pct - 1.0 / n) ** 2)

    rng = np.random.default_rng(seed)
    best_x, best_fun = np.ones(n) / n, np.inf
    for _ in range(n_restarts):
        x0 = rng.dirichlet(np.ones(n))
        res = minimize(rp_objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 1000, "ftol": 1e-12})
        if res.success and res.fun < best_fun:
            best_x, best_fun = res.x, res.fun
    return best_x


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 30) -> pd.DataFrame:
    """Traces the full risk-return parabola (both the efficient upper branch and the
    inefficient lower branch), then flags which points are actually on the efficient
    frontier (return >= the global minimum-variance portfolio's return)."""
    n = len(cov)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    rows = []
    for tr in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=tr: w @ mu - tr},
        ]
        x0 = np.ones(n) / n
        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                        bounds=_base_bounds(n), constraints=cons, options={"maxiter": 500})
        if res.success:
            rows.append({"Return": tr, "Volatility": np.sqrt(res.fun)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    min_var_return = df.loc[df["Volatility"].idxmin(), "Return"]
    df["Efficient"] = df["Return"] >= min_var_return
    return df.sort_values("Return").reset_index(drop=True)


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> dict:
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return {"Expected Return": ret, "Volatility": vol, "Sharpe": sharpe}


# ---------------------------------------------------------------------
# Backtesting helpers
# ---------------------------------------------------------------------

def train_test_split_returns(returns: pd.DataFrame, train_frac: float = 0.7):
    split_idx = int(len(returns) * train_frac)
    return returns.iloc[:split_idx], returns.iloc[split_idx:]


def performance_stats(port_returns: pd.Series, rf_annual: float = 0.0) -> dict:
    total_return = (1 + port_returns).prod() - 1
    n_years = len(port_returns) / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (port_returns.mean() * TRADING_DAYS - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + port_returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"CAGR": cagr, "Ann. Volatility": ann_vol, "Sharpe": sharpe, "Max Drawdown": max_dd,
            "Total Return": total_return}


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("🏗️ Portfolio Construction Dashboard")
st.caption(
    "Compares Equal-Weight, Minimum-Variance, Risk Parity, and Max-Sharpe portfolios "
    "(using both naive historical returns and Black-Litterman equilibrium returns), "
    "then backtests them out-of-sample. Read the Methodology & Limitations section at "
    "the bottom before drawing conclusions from this — it matters."
)

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    benchmark = st.text_input("Benchmark", DEFAULT_BENCHMARK).strip().upper()
    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    rf_annual = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.15,
                                 value=0.04, step=0.005, format="%.3f")
    use_shrinkage = st.checkbox("Use Ledoit-Wolf shrinkage covariance", value=True)

    st.markdown("---")
    st.markdown("**Backtest**")
    train_frac = st.slider("Training window (% of history)", 0.4, 0.85, 0.7, 0.05)

if len(tickers) < 3:
    st.warning("Enter at least 3 tickers.")
    st.stop()

with st.spinner("Pulling price history..."):
    prices = fetch_prices(tickers, benchmark, period)
with st.spinner("Pulling market cap data..."):
    market_caps_raw = fetch_market_caps(tickers)

returns = compute_returns(prices)
asset_returns = returns[tickers]
bench_returns = returns[benchmark]

# Market-cap weights for Black-Litterman; fall back to equal weight for any missing caps
caps = pd.Series({t: (market_caps_raw.get(t) or np.nan) for t in tickers})
if caps.isna().any():
    missing = caps[caps.isna()].index.tolist()
    st.info(f"Market cap unavailable for {missing} — using equal weight as a fallback for those in the BL prior only.")
    caps = caps.fillna(caps.mean())
w_mkt = (caps / caps.sum()).values

# Approximate shares outstanding (today's cap / today's price) so we can reconstruct
# point-in-time market-cap weights for the backtest's training window instead of reusing
# today's weights — see estimate_shares_outstanding / market_cap_weights_at docstrings.
shares_outstanding = estimate_shares_outstanding(caps, prices[tickers].iloc[-1])

# ---------------------------------------------------------------------
# Section 1: Return & covariance estimation
# ---------------------------------------------------------------------

st.subheader("1. Return & Covariance Estimation")

cov_sample = sample_covariance(asset_returns)
cov_shrink, shrinkage_intensity = shrinkage_covariance(asset_returns)
cov = cov_shrink if use_shrinkage else cov_sample

mu_hist = historical_mean_returns(asset_returns)
delta = implied_risk_aversion(bench_returns, rf_annual)
mu_bl = black_litterman_equilibrium_returns(cov, w_mkt, delta)

if use_shrinkage:
    st.caption(f"Ledoit-Wolf shrinkage intensity: **{shrinkage_intensity:.3f}** "
               "(0 = pure sample covariance, 1 = fully shrunk toward the structured target). "
               "A non-trivial intensity here means the raw sample covariance was judged noisy "
               "enough to warrant meaningfully adjusting it.")

est_df = pd.DataFrame({
    "Sector": [SECTOR_MAP.get(t, "—") for t in tickers],
    "Market Cap Weight": w_mkt,
    "Historical Return": mu_hist,
    "BL Equilibrium Return": mu_bl,
}, index=tickers)
st.dataframe(
    est_df.style.format({"Market Cap Weight": "{:.2%}", "Historical Return": "{:.2%}",
                          "BL Equilibrium Return": "{:.2%}"}),
    use_container_width=True,
)
st.caption(
    f"Implied market risk aversion (δ) from {benchmark}: **{delta:.2f}**. Notice how much more "
    "volatile the historical return column is than the BL equilibrium column — that volatility "
    "in the historical estimates is exactly the estimation-error problem the equilibrium prior "
    "is designed to avoid. No investor views are blended in here; these are pure equilibrium returns."
)

# ---------------------------------------------------------------------
# Section 2: Portfolio construction
# ---------------------------------------------------------------------

st.subheader("2. Portfolio Construction")

w_eq = np.ones(len(tickers)) / len(tickers)
w_minvar = optimize_min_variance(cov)
w_riskparity = optimize_risk_parity(cov)
w_maxsharpe_hist = optimize_max_sharpe(mu_hist, cov, rf_annual)
w_maxsharpe_bl = optimize_max_sharpe(mu_bl, cov, rf_annual)

methods = {
    "Equal Weight": w_eq,
    "Min-Variance": w_minvar,
    "Risk Parity": w_riskparity,
    "Max-Sharpe (Historical)": w_maxsharpe_hist,
    "Max-Sharpe (BL Equilibrium)": w_maxsharpe_bl,
}

weights_df = pd.DataFrame(methods, index=tickers)
st.markdown("**Portfolio Weights by Method**")
fig_weights = go.Figure()
for method in methods:
    fig_weights.add_trace(go.Bar(x=tickers, y=weights_df[method], name=method))
fig_weights.update_layout(barmode="group", height=450, yaxis_title="Weight",
                           margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_weights, use_container_width=True)

st.markdown("**Expected Risk/Return by Method** (using the estimation-period mu/cov shown above — "
            "these are in-sample by construction, not a forecast of future performance)")
stats_rows = []
for method, w in methods.items():
    mu_for_stats = mu_bl if "BL" in method else mu_hist
    s = portfolio_stats(w, mu_for_stats, cov, rf_annual)
    s["Method"] = method
    stats_rows.append(s)
stats_df = pd.DataFrame(stats_rows).set_index("Method")
st.dataframe(
    stats_df.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}),
    use_container_width=True,
)
st.caption(
    "Notice Max-Sharpe (Historical) likely concentrates heavily into whichever stock(s) had the best "
    "trailing Sharpe ratio in this sample — this is the 'estimation-error maximization' problem in "
    "action: mean-variance optimization aggressively bets on the noisiest part of its own input "
    "(expected returns). Min-Variance and Risk Parity don't use return forecasts at all, so they're "
    "far less sensitive to this — that's precisely why they're more widely trusted in practice."
)

# ---------------------------------------------------------------------
# Section 3: Efficient frontier
# ---------------------------------------------------------------------

st.subheader("3. Efficient Frontier")
st.caption("Traced using historical returns and the selected covariance estimator. "
           "Gray points are mathematically valid but inefficient (a portfolio with equal "
           "or lower risk exists for the same or higher return) — only the colored branch "
           "is the actual efficient frontier.")

with st.spinner("Tracing efficient frontier..."):
    frontier_df = efficient_frontier(mu_hist, cov, n_points=30)

fig_frontier = go.Figure()
if len(frontier_df) > 0:
    ineff = frontier_df[~frontier_df["Efficient"]]
    eff = frontier_df[frontier_df["Efficient"]]
    if len(ineff) > 0:
        fig_frontier.add_trace(go.Scatter(x=ineff["Volatility"], y=ineff["Return"],
                                           mode="lines", line=dict(color="lightgray", dash="dot"),
                                           name="Inefficient branch"))
    fig_frontier.add_trace(go.Scatter(x=eff["Volatility"], y=eff["Return"],
                                       mode="lines", line=dict(color="steelblue", width=3),
                                       name="Efficient frontier"))

# Individual assets
asset_vols = np.sqrt(np.diag(cov))
fig_frontier.add_trace(go.Scatter(x=asset_vols, y=mu_hist, mode="markers+text",
                                   text=tickers, textposition="top center",
                                   marker=dict(size=7, color="gray"), name="Individual assets"))

# Portfolio methods
method_colors = {"Equal Weight": "black", "Min-Variance": "green", "Risk Parity": "purple",
                  "Max-Sharpe (Historical)": "red", "Max-Sharpe (BL Equilibrium)": "orange"}
for method, w in methods.items():
    mu_for_stats = mu_bl if "BL" in method else mu_hist
    s = portfolio_stats(w, mu_for_stats, cov, rf_annual)
    fig_frontier.add_trace(go.Scatter(x=[s["Volatility"]], y=[s["Expected Return"]],
                                       mode="markers", marker=dict(size=14, symbol="star",
                                                                    color=method_colors.get(method)),
                                       name=method))

fig_frontier.update_layout(height=500, xaxis_title="Volatility (annualized)",
                            yaxis_title="Expected Return (annualized)",
                            margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_frontier, use_container_width=True)

# ---------------------------------------------------------------------
# Section 4: Out-of-sample backtest
# ---------------------------------------------------------------------

st.subheader("4. Out-of-Sample Backtest")
st.caption(
    "Weights for every method are computed using ONLY the training period below. Those fixed "
    "weights are then applied, unchanged, to the strictly later test period — the optimizer never "
    "sees the test period's data. This is the minimum bar for an honest backtest; skipping this "
    "split (i.e. testing on the same data used to build the portfolio) is a common and serious error."
)

train_returns, test_returns = train_test_split_returns(returns, train_frac)
st.caption(f"Training period: {train_returns.index[0].date()} to {train_returns.index[-1].date()} "
           f"({len(train_returns)} days) — Test period: {test_returns.index[0].date()} to "
           f"{test_returns.index[-1].date()} ({len(test_returns)} days)")

if len(test_returns) < 30:
    st.warning("Test period is very short — widen the history window or lower the training "
               "fraction for a more meaningful backtest.")
else:
    train_asset_returns = train_returns[tickers]
    train_bench_returns = train_returns[benchmark]
    test_asset_returns = test_returns[tickers]
    test_bench_returns = test_returns[benchmark]

    train_cov = (shrinkage_covariance(train_asset_returns)[0] if use_shrinkage
                 else sample_covariance(train_asset_returns))
    train_mu_hist = historical_mean_returns(train_asset_returns)
    train_delta = implied_risk_aversion(train_bench_returns, rf_annual)

    # Use market-cap weights AS OF the end of the training period, not today's — reusing
    # today's weights here would leak future information into the backtest (see the
    # market_cap_weights_at docstring). We approximate this with the shares-outstanding
    # proxy applied to the training period's last available price.
    train_end_prices = prices[tickers].loc[train_asset_returns.index[-1]]
    train_w_mkt = market_cap_weights_at(shares_outstanding, train_end_prices)
    train_mu_bl = black_litterman_equilibrium_returns(train_cov, train_w_mkt, train_delta)

    st.caption(
        "Note: the BL Equilibrium method's market-cap weights here are reconstructed as of "
        "the end of the training period (via a shares-outstanding proxy), not today's actual "
        "weights — otherwise this backtest would leak future information into the past."
    )

    bt_methods = {
        "Equal Weight": np.ones(len(tickers)) / len(tickers),
        "Min-Variance": optimize_min_variance(train_cov),
        "Risk Parity": optimize_risk_parity(train_cov),
        "Max-Sharpe (Historical)": optimize_max_sharpe(train_mu_hist, train_cov, rf_annual),
        "Max-Sharpe (BL Equilibrium)": optimize_max_sharpe(train_mu_bl, train_cov, rf_annual),
    }

    fig_growth = go.Figure()
    perf_rows = []
    for method, w in bt_methods.items():
        test_port_ret = pd.Series(test_asset_returns.values @ w, index=test_asset_returns.index)
        growth = (1 + test_port_ret).cumprod()
        fig_growth.add_trace(go.Scatter(x=growth.index, y=growth.values, mode="lines", name=method))
        stats = performance_stats(test_port_ret, rf_annual)
        stats["Method"] = method
        perf_rows.append(stats)

    bench_growth = (1 + test_bench_returns).cumprod()
    fig_growth.add_trace(go.Scatter(x=bench_growth.index, y=bench_growth.values, mode="lines",
                                     line=dict(color="black", dash="dash"), name=f"{benchmark} (buy & hold)"))
    bench_stats = performance_stats(test_bench_returns, rf_annual)
    bench_stats["Method"] = f"{benchmark} (benchmark)"
    perf_rows.append(bench_stats)

    fig_growth.update_layout(height=450, yaxis_title="Growth of $1", margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.markdown("**Out-of-Sample Growth of $1**")
    st.plotly_chart(fig_growth, use_container_width=True)

    perf_df = pd.DataFrame(perf_rows).set_index("Method")
    st.markdown("**Out-of-Sample Performance Statistics**")
    st.dataframe(
        perf_df.style.format({"CAGR": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.2f}",
                               "Max Drawdown": "{:.2%}", "Total Return": "{:.2%}"}),
        use_container_width=True,
    )
    st.caption(
        "No transaction costs, taxes, or rebalancing are modeled here — weights are set once at the "
        "start of the test period and held (buy-and-hold), not rebalanced. Real implementation costs "
        "would reduce every method's returns, and would especially penalize any method that "
        "concentrates into a small number of names (higher tracking error, harder to trade at scale)."
    )

# ---------------------------------------------------------------------
# Section 5: Methodology & Limitations
# ---------------------------------------------------------------------

st.subheader("5. Methodology & Limitations")
st.markdown("""
This dashboard is a methodology comparison exercise, not an investment recommendation. Specific
limitations worth understanding before drawing conclusions:

- **Estimation error in expected returns.** Historical average returns are a very noisy estimator
  of *future* expected returns — mean-variance optimization using them tends to aggressively
  overweight whichever asset got randomly lucky in the sample window (Michaud's "estimation-error
  maximization" problem). This is why Max-Sharpe (Historical) often looks concentrated, while
  Min-Variance and Risk Parity (which don't use return forecasts at all) tend to look more balanced.

- **Black-Litterman here uses no investor views.** The "BL Equilibrium" returns shown are the pure
  market-implied equilibrium (reverse-optimized from market-cap weights), not views-adjusted
  returns — this avoids fabricating a market call this project has no basis for making, while still
  providing a more disciplined return estimate than a raw historical average. In the out-of-sample
  backtest, market-cap weights are reconstructed as of the *end of the training period* (via an
  approximate shares-outstanding proxy: today's market cap ÷ today's price, applied to historical
  prices) rather than reusing today's actual weights — using today's weights for a past training
  window would leak future information into the backtest. This proxy still isn't exact, since real
  shares outstanding drift over time from buybacks and share issuance; it's a meaningfully better
  approximation than reusing today's market cap directly, not a perfect reconstruction.

- **Covariance matrix conditioning.** With ~20 assets and typically 250–1250 daily observations,
  the sample covariance matrix is usable but not especially stable — Ledoit-Wolf shrinkage (toggle
  in the sidebar) is a standard, well-cited correction, not a guarantee of accuracy.

- **Non-stationarity.** A single static covariance/return estimate assumes the recent past
  represents the near future. Correlations and volatilities shift meaningfully across market
  regimes (e.g. 2020, 2022) — a limitation shared by every method shown here.

- **Non-normality.** These optimizers use variance as the risk measure, which assumes
  symmetric, roughly normal returns. Real equity returns have fatter tails and negative skew that
  variance understates — see the companion Risk Dashboard's historical/Monte Carlo VaR for a more
  tail-aware view of risk on any of these portfolios.

- **Survivorship bias.** Tickers are pulled by their current identity — companies that were
  delisted, went bankrupt, or were acquired during the lookback window are silently excluded,
  which tends to inflate historical returns and understate historical risk.

- **No transaction costs, taxes, or capacity constraints.** The backtest assumes frictionless,
  instantaneous, infinitely-scalable trading. Real costs would reduce all returns shown, and would
  disproportionately penalize concentrated portfolios.

- **Hand-picked universe.** These 20 tickers were selected for sector coverage and liquidity, not
  through a rules-based, pre-specified universe construction process (e.g. "all S&P 500 members
  with 5+ years of history") — a real institutional process would define the universe by explicit
  rules rather than by a person choosing recognizable names, even reasonable ones.
""")

st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
