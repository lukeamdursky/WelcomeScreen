"""
Shared functions and constants used across all pages of this app: data loading,
covariance/return estimation, portfolio optimizers, walk-forward backtesting,
performance and risk metrics, and shared presentation helpers.

Kept in one place so Home.py and every file under pages/ import from here rather than
duplicating this logic.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import norm, chi2, t as t_dist
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Presentation constants
#   Blue = neutral controls/series. Red is reserved for risk, loss and breaches.
# ---------------------------------------------------------------------

NEUTRAL_BLUE = "#3B82F6"
LOSS_RED = "#E5484D"
WARN_AMBER = "#F59E0B"
BENCH_COLOR = "#C9CED6"                      # light gray: visible on dark and light backgrounds
BENCH_LINE = dict(color=BENCH_COLOR, dash="dash", width=2)

PCT0, PCT1, PCT2 = ".0%", ".1%", ".2%"       # Plotly/d3 tick formats for fractional values
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}   # hides the Plotly toolbar

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

# ---------------------------------------------------------------------
# Portfolio construction methods (names are used as keys across pages)
# ---------------------------------------------------------------------

M_EQ = "Equal Weight"
M_MV = "Min-Variance"
M_RP = "Risk Parity"
M_MS_HIST = "Max-Sharpe (Historical)"
M_MS_PRIOR = "Max-Sharpe (BL Equilibrium Prior)"
METHODS = [M_EQ, M_MV, M_RP, M_MS_HIST, M_MS_PRIOR]

# One fixed, well-separated hue per method, reused on every chart. No red: red is
# reserved for risk/loss/breach signals elsewhere in the app.
METHOD_COLORS = {
    M_EQ: "#F472B6",        # pink
    M_MV: "#3B82F6",        # blue
    M_RP: "#10B981",        # green
    M_MS_HIST: "#A78BFA",   # violet
    M_MS_PRIOR: "#F59E0B",  # amber
}

# 26 bright, non-red colors: readable on the dark theme and never confusable with the red risk signals.
_TICKER_PALETTE = [
    "#60A5FA", "#34D399", "#A78BFA", "#22D3EE", "#A3E635", "#818CF8", "#2DD4BF", "#E879F9",
    "#4ADE80", "#38BDF8", "#C084FC", "#84CC16", "#67E8F9", "#86EFAC", "#93C5FD", "#D8B4FE",
    "#5EEAD4", "#BEF264", "#7DD3FC", "#C4B5FD", "#6EE7B7", "#F0ABFC", "#A5B4FC", "#99F6E4",
    "#BAE6FD", "#DDD6FE",
]


def get_ticker_colors(tickers: list[str]) -> dict:
    """Stable ticker -> color map (assigned by sorted ticker order, so a ticker keeps its color)."""
    ordered = sorted(tickers)
    return {t: _TICKER_PALETTE[i % len(_TICKER_PALETTE)] for i, t in enumerate(ordered)}


# ---------------------------------------------------------------------
# Chart / table display wrappers
# ---------------------------------------------------------------------

def show_chart(fig):
    """Renders a Plotly figure full-width with the toolbar hidden. Falls back to the older
    `use_container_width` argument on Streamlit versions that don't accept `width`."""
    try:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def show_table(data, max_rows: int = 25):
    """Renders a DataFrame/Styler full-width, tall enough to show up to `max_rows` rows without an
    internal scrollbar (same version fallback as show_chart)."""
    df = data.data if hasattr(data, "export") else data      # Styler -> underlying frame
    height = (min(len(df), max_rows) + 1) * 35 + 3
    try:
        st.dataframe(data, width="stretch", height=height)
    except TypeError:
        st.dataframe(data, use_container_width=True, height=height)


def dash_style(df: pd.DataFrame, formats: dict) -> pd.DataFrame:
    """Formats each column with its format string and shows a dash for empty cells. Streamlit renders
    null cells as 'None' whatever the Styler says, so these small tables are converted to text up
    front (all-string columns also avoid Arrow mixed-type conversion warnings)."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        fmt = formats.get(col)
        out[col] = ["\u2014" if pd.isna(v) else (fmt.format(v) if fmt else str(v)) for v in df[col]]
    return out


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: list[str], benchmark: str, period: str) -> pd.DataFrame:
    """Daily adjusted closes from Yahoo Finance. auto_adjust=True returns split- and
    dividend-adjusted prices, so pct_change() gives total-return-style daily returns."""
    all_symbols = list(dict.fromkeys(tickers + [benchmark]))
    data = yf.download(all_symbols, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = all_symbols
    result = close.dropna(how="all").ffill().dropna()
    if result.empty:
        # Raising (instead of returning an empty frame) stops st.cache_data from caching a
        # transient failure, which would otherwise leave every visitor stuck for the full TTL.
        raise ValueError("Yahoo Finance returned no overlapping price history")
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_caps(tickers: list[str]) -> dict:
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("marketCap", None)
        except Exception:
            caps[t] = None
    return caps


def load_prices(tickers: list[str], benchmark: str, period: str) -> pd.DataFrame:
    """fetch_prices with friendly failure handling (bad ticker, rate limit, no overlap)."""
    msg = ("Could not load price history for these tickers. Check the ticker symbols in the sidebar; if "
           "they are correct, Yahoo Finance may be rate-limiting, so wait a minute and reload.")
    try:
        prices = fetch_prices(tickers, benchmark, period)
    except Exception:
        st.error(msg)
        st.stop()
    wanted = list(dict.fromkeys(tickers + [benchmark]))
    if prices is None or prices.empty or any(t not in prices.columns for t in wanted):
        st.error(msg)
        st.stop()
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_risk_free_series(period: str) -> pd.Series:
    """Historical short-term risk-free rate: the 13-week Treasury bill discount yield (Yahoo
    ticker ^IRX, quoted in percent -- e.g. 5.25 means 5.25% annualized), daily, over `period`.
    Returned as a decimal annual rate indexed by date (0.0525, not 5.25)."""
    data = yf.download("^IRX", period=period, auto_adjust=True, progress=False)
    close = data["Close"]["^IRX"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
    series = close.dropna() / 100.0
    series.name = "rf_annual"
    return series


def load_risk_free_series(period: str) -> pd.Series | None:
    """fetch_risk_free_series with silent failure (returns None) so a caller can fall back to a
    constant rate rather than stopping the app -- this data feed is a convenience, not something
    that should block using the rest of the tool if Yahoo Finance is unreachable."""
    try:
        series = fetch_risk_free_series(period)
    except Exception:
        return None
    return series if series is not None and not series.empty else None


def data_caption(prices: pd.DataFrame):
    st.caption(
        f"Data: Yahoo Finance (yfinance), split- and dividend-adjusted daily closes, "
        f"{prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} trading days, "
        f"common history of all tickers)."
    )


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------
# Covariance & expected return estimation
# ---------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * TRADING_DAYS


def shrinkage_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance (annualized) and the shrinkage intensity used."""
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_ * TRADING_DAYS, lw.shrinkage_


def historical_mean_returns(returns: pd.DataFrame) -> np.ndarray:
    return returns.mean().values * TRADING_DAYS


def implied_risk_aversion(bench_returns: pd.Series, rf_annual: float) -> float:
    """delta = (E[Rm] - Rf) / Var(Rm), estimated from the benchmark. Clipped to [1, 10]:
    in short or flat windows the raw estimate can be near zero or negative, which would
    flip the sign of every equilibrium return and make the prior meaningless."""
    bench_annual_ret = bench_returns.mean() * TRADING_DAYS
    bench_annual_var = bench_returns.var() * TRADING_DAYS
    raw = (bench_annual_ret - rf_annual) / bench_annual_var if bench_annual_var > 0 else 2.5
    return float(np.clip(raw, 1.0, 10.0))


def black_litterman_equilibrium_returns(cov: np.ndarray, market_weights: np.ndarray,
                                          delta: float) -> np.ndarray:
    """Equilibrium (implied) EXCESS returns pi = delta * Sigma * w_mkt, i.e. returns over the
    risk-free rate. Add the risk-free rate (see equilibrium_expected_returns) before comparing
    them with a Sharpe ratio that already subtracts it.

    This is ONLY the prior step of Black-Litterman. Full Black-Litterman blends this prior with
    investor views (P, Q, Omega) into a posterior:
        mu_BL = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 pi + P' Omega^-1 Q].
    This app supplies no views, so the posterior equals the prior. Consequence: the max-Sharpe
    portfolio on these returns is the market portfolio (Sigma^-1 pi is proportional to w_mkt),
    so without views this method reproduces cap weighting unless a constraint binds."""
    return delta * (cov @ market_weights)


def equilibrium_expected_returns(cov: np.ndarray, market_weights: np.ndarray, delta: float,
                                   rf_annual: float) -> np.ndarray:
    """Total expected returns implied by equilibrium: risk-free rate + excess returns (pi)."""
    return rf_annual + black_litterman_equilibrium_returns(cov, market_weights, delta)


def estimate_shares_outstanding(current_market_caps: pd.Series, current_prices: pd.Series) -> pd.Series:
    """Approximate shares outstanding as (today's market cap) / (today's price), so market-cap
    weights can be rebuilt point-in-time as (shares x historical price). Ignores buybacks/issuance."""
    return current_market_caps / current_prices


def market_cap_weights_at(shares_outstanding: pd.Series, prices_row: pd.Series) -> np.ndarray:
    tickers = shares_outstanding.index
    proxy_caps = shares_outstanding[tickers] * prices_row[tickers]
    return (proxy_caps / proxy_caps.sum()).values


# ---------------------------------------------------------------------
# Portfolio optimizers: long-only, fully invested, optional per-position cap
# ---------------------------------------------------------------------

def cap_is_feasible(n: int, max_weight: float) -> bool:
    """A long-only, fully-invested optimizer max weight of max_weight across n assets is only
    feasible if n * max_weight >= 1 (otherwise the weights cannot reach 100% while respecting
    every bound). _bounds() silently relaxes an infeasible cap to 1/n so the optimizer always has
    something to solve; callers that surface the cap to the user should check this first and say so."""
    return n * max_weight >= 1.0 - 1e-9


def _bounds(n: int, max_weight: float = 1.0) -> list:
    cap = min(1.0, max(float(max_weight), 1.0 / n + 1e-9))   # cap must allow sum(w) = 1
    return [(0.0, cap)] * n


def optimize_min_variance(cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                    bounds=_bounds(n, max_weight), constraints=cons, options={"maxiter": 1000})
    return res.x if res.success else x0


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, max_weight: float = 1.0) -> np.ndarray:
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 1e6

    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP",
                    bounds=_bounds(n, max_weight), constraints=cons, options={"maxiter": 1000})
    return res.x if res.success else x0


def optimize_risk_parity(cov: np.ndarray, n_restarts: int = 5, seed: int = 0) -> np.ndarray:
    """Equal-risk-contribution portfolio via the convex log-barrier formulation
    (Maillard, Roncalli & Teiletche 2010; Spinu 2013): minimize 0.5 w'Sigma w - sum(log w_i)
    over w > 0, then normalize to sum to 1. The first-order condition gives w_i (Sigma w)_i =
    constant for all i, i.e. equal risk contributions; the objective is convex, so the
    solution is unique. (A sum-of-squares objective with SLSQP silently failed at n = 20.)
    The position cap does not apply: risk parity is defined by equal risk, not by weight limits."""
    n = len(cov)

    def objective(w):
        return 0.5 * w @ cov @ w - np.sum(np.log(w))

    def gradient(w):
        return cov @ w - 1.0 / w

    bounds = [(1e-8, None)] * n
    rng = np.random.default_rng(seed)
    best_x, best_fun = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(0.5, 1.5, n) / n
        res = minimize(objective, x0, jac=gradient, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
        if res.success and res.fun < best_fun:
            best_x, best_fun = res.x, res.fun
    if best_x is None:
        return np.ones(n) / n
    return best_x / best_x.sum()


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 30,
                        max_weight: float = 1.0) -> pd.DataFrame:
    """Traces the risk-return parabola under the same constraints as the optimizers and flags
    the efficient branch (return >= the global minimum-variance portfolio's return)."""
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
                        bounds=_bounds(n, max_weight), constraints=cons, options={"maxiter": 500})
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


def portfolio_returns_series(returns: pd.DataFrame, tickers: list[str], weights: np.ndarray) -> pd.Series:
    return pd.Series(returns[tickers].values @ weights, index=returns.index)


# ---------------------------------------------------------------------
# Concentration and risk contribution
# ---------------------------------------------------------------------

def effective_n_weights(w: np.ndarray) -> float:
    """Effective number of positions = 1 / sum(w_i^2) (inverse Herfindahl). Equals N for equal weights."""
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of portfolio variance: w_i (Sigma w)_i / (w' Sigma w). Sums to 1."""
    w = np.asarray(w, dtype=float)
    port_var = w @ cov @ w
    return w * (cov @ w) / port_var


def effective_n_risk(rc: np.ndarray) -> float:
    """Effective number of independent risk bets = 1 / sum(rc_i^2), on risk shares clipped at
    zero (hedging positions with negative contribution are ignored) and renormalized."""
    rc = np.clip(np.asarray(rc, dtype=float), 0.0, None)
    rc = rc / rc.sum()
    return float(1.0 / np.sum(rc ** 2))


# ---------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------

def train_test_split_returns(returns: pd.DataFrame, train_frac: float = 0.7):
    split_idx = int(len(returns) * train_frac)
    return returns.iloc[:split_idx], returns.iloc[split_idx:]


def _as_daily_rf(rf, index: pd.Index):
    """Converts an annual risk-free input to a DAILY rate aligned to `index`. `rf` is either a
    constant annual rate (float) -- returned as a scalar (rf / 252), which broadcasts against any
    return series -- or a pd.Series of annualized rates indexed by date (e.g. a historical T-bill
    yield curve) -- reindexed to `index` with forward/back-fill (for dates the series doesn't cover,
    such as before the T-bill data starts) and converted to a daily rate. Both branches agree
    exactly when `rf` happens to be constant, so every function using this is a drop-in
    generalization of the constant-rate version, not a behavior change for existing callers."""
    if isinstance(rf, pd.Series):
        return rf.reindex(index).ffill().bfill() / TRADING_DAYS
    return float(rf) / TRADING_DAYS


def performance_stats(port_returns: pd.Series, rf=0.0,
                       bench_returns: pd.Series | None = None) -> dict:
    """Annualization: 252 trading days. CAGR is geometric; Sharpe/Sortino use the arithmetic mean
    of daily excess returns (return minus that day's risk-free rate), annualized. Sortino's
    downside deviation uses a minimum acceptable return equal to the risk-free rate. Tracking
    error is the annualized std of daily active returns vs the benchmark; information ratio =
    annualized mean active return / tracking error. `rf` may be a constant annual rate (float) or
    a pd.Series of annualized rates by date (see _as_daily_rf) -- e.g. a historical T-bill yield,
    so Sharpe/Sortino reflect the actual rate regime on each day rather than one constant guess."""
    r = port_returns.dropna()
    total_return = (1 + r).prod() - 1
    n_years = len(r) / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)

    rf_daily = _as_daily_rf(rf, r.index)
    excess = r - rf_daily
    mean_excess_ann = excess.mean() * TRADING_DAYS
    sharpe = mean_excess_ann / ann_vol if ann_vol > 0 else np.nan

    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = mean_excess_ann / downside_dev if downside_dev > 0 else np.nan

    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    out = {"CAGR": cagr, "Ann. Volatility": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
           "Max Drawdown": max_dd, "Total Return": total_return}
    if bench_returns is not None:
        active = (r - bench_returns.reindex(r.index)).dropna()
        te = active.std() * np.sqrt(TRADING_DAYS)
        out["Tracking Error"] = te
        out["Information Ratio"] = active.mean() * TRADING_DAYS / te if te > 0 else np.nan
    return out


def calendar_year_returns(port_returns: pd.Series) -> pd.Series:
    return port_returns.groupby(port_returns.index.year).apply(lambda r: (1 + r).prod() - 1)


def rolling_sharpe(port_returns: pd.Series, window: int, rf) -> pd.Series:
    """`rf` may be a constant annual rate or a pd.Series of annualized rates by date (see
    _as_daily_rf); volatility is still measured on raw (not excess) returns, the usual convention."""
    rf_daily = _as_daily_rf(rf, port_returns.index)
    roll_mean_excess = (port_returns - rf_daily).rolling(window).mean() * TRADING_DAYS
    roll_std = port_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return roll_mean_excess / roll_std


def rolling_alpha_beta(port_returns: pd.Series, bench_returns: pd.Series, window: int,
                        rf=0.0) -> pd.DataFrame:
    """Rolling single-factor OLS of portfolio excess return on benchmark excess return
    (closed form): beta = Cov/Var, alpha = mean(port) - beta*mean(bench), alpha annualized. `rf`
    may be a constant annual rate or a pd.Series of annualized rates by date (see _as_daily_rf)."""
    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    rf_daily = _as_daily_rf(rf, aligned.index)
    port_ex = aligned["port"] - rf_daily
    bench_ex = aligned["bench"] - rf_daily

    roll_cov = port_ex.rolling(window).cov(bench_ex)
    roll_var = bench_ex.rolling(window).var()
    beta = roll_cov / roll_var
    alpha_daily = port_ex.rolling(window).mean() - beta * bench_ex.rolling(window).mean()
    return pd.DataFrame({"Beta": beta, "Alpha (annualized)": alpha_daily * TRADING_DAYS})


def capm_regression(port_returns: pd.Series, bench_returns: pd.Series, rf=0.0) -> dict:
    """Full-period single-factor regression: port_excess = alpha + beta * bench_excess + residual,
    estimated by closed-form OLS (equivalent to a general least-squares fit for one regressor).
    `rf` may be a constant annual rate or a pd.Series of annualized rates by date (see
    _as_daily_rf). Returns annualized alpha and its standard error, beta, R-squared (share of
    portfolio excess-return variance explained by the benchmark), the alpha t-statistic, its
    two-sided p-value, and the number of observations -- so a large alpha can be read alongside
    how precisely it's estimated, rather than taken at face value."""
    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    rf_daily = _as_daily_rf(rf, aligned.index)
    port_ex = (aligned["port"] - rf_daily).values
    bench_ex = (aligned["bench"] - rf_daily).values
    n = len(aligned)

    bench_mean, port_mean = bench_ex.mean(), port_ex.mean()
    ss_x = float(((bench_ex - bench_mean) ** 2).sum())
    beta = float(((bench_ex - bench_mean) * (port_ex - port_mean)).sum() / ss_x) if ss_x > 0 else np.nan
    alpha_daily = port_mean - beta * bench_mean if ss_x > 0 else np.nan

    resid = port_ex - (alpha_daily + beta * bench_ex) if ss_x > 0 else np.full(n, np.nan)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((port_ex - port_mean) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if ss_x > 0 and n > 2:
        resid_var = ss_res / (n - 2)
        se_alpha_daily = np.sqrt(resid_var * (1.0 / n + bench_mean ** 2 / ss_x))
        t_stat = alpha_daily / se_alpha_daily if se_alpha_daily > 0 else np.nan
        p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))) if np.isfinite(t_stat) else np.nan
    else:
        se_alpha_daily, t_stat, p_value = np.nan, np.nan, np.nan

    return {"alpha_annual": alpha_daily * TRADING_DAYS if ss_x > 0 else np.nan, "beta": beta, "r2": r2,
            "alpha_se_annual": se_alpha_daily * TRADING_DAYS if se_alpha_daily == se_alpha_daily else np.nan,
            "t_stat": t_stat, "p_value": p_value, "n": n}


def max_drawdown(prices: pd.Series) -> tuple[float, pd.Series]:
    cum = prices / prices.iloc[0]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min(), drawdown


# ---------------------------------------------------------------------
# Walk-forward out-of-sample backtest with rebalancing, turnover and costs
# ---------------------------------------------------------------------

def _simulate_rebalanced(ret: np.ndarray, start: int, targets: dict, cost_rate: float):
    """Simulates one strategy from index `start`. At each index in `targets` the portfolio is
    rebalanced to the target weights; in between, holdings drift with returns (buy-and-hold).
    Cost = cost_rate x traded notional (sum |w_target - w_drifted|) charged on the rebalance day,
    including the initial purchase from cash. One-way turnover = 0.5 x traded notional.
    Returns (daily net returns array, {rebalance index: one-way turnover})."""
    T, n = ret.shape
    value = 1.0
    holdings = np.zeros(n)
    daily = np.full(T, np.nan)
    turnover = {}
    for t in range(start, T):
        value_start = value
        if t in targets:
            w_new = targets[t]
            w_cur = holdings / holdings.sum() if holdings.sum() > 0 else np.zeros(n)
            traded = float(np.abs(w_new - w_cur).sum())
            turnover[t] = 0.5 * traded
            value = value - cost_rate * traded * value
            holdings = value * w_new
        holdings = holdings * (1.0 + ret[t])
        value = float(holdings.sum())
        daily[t] = value / value_start - 1.0
    return daily, turnover


@st.cache_data(show_spinner=False)
def walk_forward_backtest(asset_returns: pd.DataFrame, bench_returns: pd.Series,
                           asset_prices: pd.DataFrame, shares_out: pd.Series, rf_annual: float,
                           lookback: int, rebalance_every: int, cost_bps: float,
                           use_shrinkage: bool, max_weight: float):
    """Walk-forward out-of-sample test of all five methods.

    At each rebalance date t, weights are estimated using ONLY the trailing `lookback` trading
    days ending the day before t (covariance, historical means, benchmark-implied risk aversion,
    and point-in-time market-cap weights for the equilibrium prior), then held until the next
    rebalance. Repeating this over the whole sample gives an out-of-sample return series that
    never uses future data, evaluated over many rebalances instead of a single split.
    Returns None if the sample is too short for at least two rebalances."""
    tickers = list(asset_returns.columns)
    ret = asset_returns.values
    T, n = ret.shape
    dates = asset_returns.index
    rebal_idx = list(range(lookback, T, rebalance_every))
    if len(rebal_idx) < 2:
        return None

    shares = shares_out.reindex(tickers)
    targets = {m: {} for m in METHODS}
    for i in rebal_idx:
        window = asset_returns.iloc[i - lookback:i]
        bench_window = bench_returns.iloc[i - lookback:i]
        cov = shrinkage_covariance(window)[0] if use_shrinkage else sample_covariance(window)
        mu_hist = historical_mean_returns(window)
        delta = implied_risk_aversion(bench_window, rf_annual)
        w_mkt = market_cap_weights_at(shares, asset_prices.loc[dates[i - 1]])
        mu_prior = equilibrium_expected_returns(cov, w_mkt, delta, rf_annual)

        targets[M_EQ][i] = np.ones(n) / n
        targets[M_MV][i] = optimize_min_variance(cov, max_weight)
        targets[M_RP][i] = optimize_risk_parity(cov)
        targets[M_MS_HIST][i] = optimize_max_sharpe(mu_hist, cov, rf_annual, max_weight)
        targets[M_MS_PRIOR][i] = optimize_max_sharpe(mu_prior, cov, rf_annual, max_weight)

    oos_index = dates[lookback:]
    out = {}
    for m in METHODS:
        net, turn = _simulate_rebalanced(ret, lookback, targets[m], cost_bps / 10000.0)
        gross, _ = _simulate_rebalanced(ret, lookback, targets[m], 0.0)
        out[m] = {
            "net": pd.Series(net[lookback:], index=oos_index),
            "gross": pd.Series(gross[lookback:], index=oos_index),
            "turnover": pd.Series({dates[i]: turn[i] for i in rebal_idx}),
            "weights": pd.DataFrame({dates[i]: targets[m][i] for i in rebal_idx}, index=tickers).T,
        }
    out["_meta"] = {"n_rebalances": len(rebal_idx), "oos_start": dates[lookback], "oos_end": dates[-1],
                    "oos_years": len(oos_index) / TRADING_DAYS}
    return out


# ---------------------------------------------------------------------
# Risk metrics: VaR / CVaR / beta / Monte Carlo / backtesting
# ---------------------------------------------------------------------

def annualized_vol(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    return returns.std() * np.sqrt(trading_days)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = -historical_var(returns, confidence)
    tail = returns[returns <= var]
    return -tail.mean() if len(tail) > 0 else np.nan


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def parametric_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Normal-distribution expected shortfall: -mu + sigma * phi(z_alpha) / alpha, alpha = 1 - confidence."""
    alpha = 1 - confidence
    mu, sigma = returns.mean(), returns.std()
    return -mu + sigma * norm.pdf(norm.ppf(alpha)) / alpha


def beta_vs_benchmark(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    cov = np.cov(asset_returns, bench_returns)[0, 1]
    var = np.var(bench_returns)
    return cov / var if var != 0 else np.nan


def monte_carlo_portfolio_var(asset_returns: pd.DataFrame, tickers: list[str], weights: np.ndarray,
                                confidence: float = 0.95, n_sims: int = 10000,
                                horizon_days: int = 1, seed: int = 42,
                                dist: str = "normal", df: float = 5.0):
    """Correlated Monte Carlo VaR/CVaR. Shocks are correlated through the Cholesky factor of the
    historical covariance. dist='normal' draws multivariate-normal shocks; dist='t' draws
    multivariate Student-t shocks with `df` degrees of freedom, rescaled so the covariance still
    matches the historical covariance but joint tails are fatter. Fully reproducible: results are
    identical for the same seed, inputs and settings."""
    rng = np.random.default_rng(seed)
    mu = asset_returns[tickers].mean().values
    sigma = asset_returns[tickers].cov().values
    n_assets = len(tickers)
    L = np.linalg.cholesky(sigma + 1e-10 * np.eye(n_assets))

    z = rng.standard_normal((n_sims, horizon_days, n_assets))
    shocks = z @ L.T
    if dist == "t":
        nu = max(float(df), 2.1)
        w = rng.chisquare(nu, size=(n_sims, horizon_days, 1)) / nu       # one mixing draw per path-day
        shocks = shocks * np.sqrt((nu - 2.0) / nu) / np.sqrt(w)          # keeps Cov(shocks) = Sigma
    simulated_horizon_returns = (mu + shocks).sum(axis=1)
    simulated_portfolio_returns = simulated_horizon_returns @ weights

    var = -np.percentile(simulated_portfolio_returns, (1 - confidence) * 100)
    tail = simulated_portfolio_returns[simulated_portfolio_returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else np.nan
    return var, cvar, simulated_portfolio_returns


def rolling_var_series(returns: pd.Series, window: int, confidence: float,
                        method: str = "historical") -> pd.Series:
    """Out-of-sample rolling VaR: each day's VaR uses only the `window` days before it (shift(1))."""
    if method == "historical":
        var_series = returns.rolling(window).apply(
            lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
        )
    else:
        mu = returns.rolling(window).mean()
        sigma = returns.rolling(window).std()
        z = norm.ppf(1 - confidence)
        var_series = -(mu + z * sigma)
    return var_series.shift(1)


def _binom_loglik(x: float, n: float, p: float) -> float:
    """Log-likelihood of x events in n Bernoulli(p) trials (0*log(0) treated as 0)."""
    return float(xlogy(n - x, 1 - p) + xlogy(x, p))


def kupiec_pof_test(n: int, x: int, confidence: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test of unconditional coverage:
    H0: breach probability = 1 - confidence. LR ~ chi-square(1). Returns (LR, p-value)."""
    p = 1 - confidence
    pi_hat = x / n if n > 0 else 0.0
    lr = max(-2 * (_binom_loglik(x, n, p) - _binom_loglik(x, n, pi_hat)), 0.0)
    return lr, float(1 - chi2.cdf(lr, df=1))


def christoffersen_tests(breaches, confidence: float) -> dict:
    """Christoffersen (1998) tests on a 0/1 breach series.

    * POF (Kupiec): are breaches the right FREQUENCY? chi-square(1)
    * Independence: does a breach today make one tomorrow more likely (clustering)? chi-square(1),
      from a first-order Markov transition matrix of breach states.
    * Conditional coverage = POF + Independence: right frequency AND no clustering. chi-square(2)."""
    b = np.asarray(breaches, dtype=int)
    n, x = len(b), int(b.sum())
    lr_pof, p_pof = kupiec_pof_test(n, x, confidence)

    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / total if total > 0 else 0.0

    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr_ind = max(float(-2 * (ll_null - ll_alt)), 0.0)
    p_ind = float(1 - chi2.cdf(lr_ind, df=1))

    lr_cc = lr_pof + lr_ind
    p_cc = float(1 - chi2.cdf(lr_cc, df=2))
    return {"n": n, "breaches": x, "lr_pof": lr_pof, "p_pof": p_pof, "lr_ind": lr_ind, "p_ind": p_ind,
            "lr_cc": lr_cc, "p_cc": p_cc, "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": pi01, "pi11": pi11}


# ---------------------------------------------------------------------
# Shared sidebar settings (matching widget keys keep values in sync across pages)
# ---------------------------------------------------------------------

def render_shared_settings(show_backtest: bool = False, show_confidence: bool = False,
                             show_shrinkage: bool = True, show_constraints: bool = False,
                             show_rf: bool = True, show_rf_mode: bool = False) -> dict:
    st.sidebar.header("Settings")
    st.sidebar.subheader("Model inputs")
    ticker_input = st.sidebar.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS),
                                          height=100, key="shared_ticker_input")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    benchmark = st.sidebar.text_input("Benchmark", DEFAULT_BENCHMARK, key="shared_benchmark").strip().upper()
    period = st.sidebar.selectbox("History window", ["2y", "5y", "10y"], index=1, key="shared_period",
                                   help="Longer windows give the walk-forward test more rebalances "
                                        "and the covariance estimates more observations.")

    settings = {"tickers": tickers, "benchmark": benchmark, "period": period}

    if show_rf and show_rf_mode:
        # Historical mode belongs here only: Performance measures REALIZED excess returns
        # (Sharpe, Sortino, alpha), where the risk-free rate someone actually earned on cash
        # changed a lot from 2021 to today, so a day-by-day rate is the more honest comparison.
        rf_mode = st.sidebar.radio("Risk-free rate", ["Constant", "Historical (13-week T-bill)"],
                                    key="shared_rf_mode", horizontal=True,
                                    help="Historical uses the actual ^IRX 13-week Treasury bill yield, "
                                         "day by day, instead of one constant rate across the whole "
                                         "window. Portfolio Construction and Risk Report always use a "
                                         "single scalar rate instead — see their own inputs.")
        rf_series = load_risk_free_series(period) if rf_mode == "Historical (13-week T-bill)" else None
        if rf_mode == "Historical (13-week T-bill)" and rf_series is None:
            st.sidebar.warning("Couldn't load the historical T-bill series; using a constant rate instead.")

        if rf_series is not None:
            rf_annual = float(rf_series.mean())          # period-average, for legacy scalar consumers
            rf = rf_series                                 # full daily series, for precise per-day use
            st.sidebar.caption(f"Using ^IRX (13-week T-bill), period average {rf_annual:.2%}.")
        else:
            rf_annual = st.sidebar.number_input("Constant annual risk-free rate", min_value=0.0,
                                                  max_value=0.15, value=0.04, step=0.005, format="%.3f",
                                                  key="shared_rf")
            rf = rf_annual
        settings["rf_annual"], settings["rf"] = rf_annual, rf
        settings["rf_mode"] = "Historical" if rf_series is not None else "Constant"
    elif show_rf:
        # Portfolio Construction: Max-Sharpe and the Black-Litterman risk-aversion estimate need one
        # scalar hurdle rate for the optimization horizon, not a day-by-day series threaded into the
        # objective — so only a plain constant is offered here, deliberately without the Historical
        # toggle Performance has. (Min-Variance and Risk Parity don't use this input at all.)
        rf_annual = st.sidebar.number_input(
            "Optimization risk-free rate (annual)", min_value=0.0, max_value=0.15, value=0.04,
            step=0.005, format="%.3f", key="shared_rf",
            help="A single scalar hurdle rate used by Max-Sharpe and the Black-Litterman equilibrium "
                 "prior. Always constant here by design — a mean-variance optimization needs one rate "
                 "for its horizon, not a day-by-day series; Performance's Historical mode is for "
                 "measuring realized excess returns after the fact, a different use of the rate.")
        settings["rf_annual"], settings["rf"], settings["rf_mode"] = rf_annual, rf_annual, "Constant"
    # else (Risk Report): no risk-free input at all -- VaR/CVaR model the distribution of portfolio
    # losses, and beta is a covariance sensitivity, so neither uses a risk-free rate.

    if show_shrinkage:
        settings["use_shrinkage"] = st.sidebar.checkbox("Use Ledoit-Wolf shrinkage covariance",
                                                          value=True, key="shared_shrinkage")
    if show_constraints:
        enforce_cap = st.sidebar.checkbox(
            "Enforce position cap", value=False, key="shared_enforce_cap",
            help="Off: optimizers are unconstrained by weight (equivalent to a 100% cap). Risk Parity "
                 "is never capped either way, since it is defined by equal risk contribution.")
        if enforce_cap:
            cap_pct = st.sidebar.slider("Optimizer max weight (%)", 10, 100, 20, 5,
                                         key="shared_max_weight_pct",
                                         help="Applies to Min-Variance, both Max-Sharpe portfolios and "
                                              "the efficient frontier — not Risk Parity.")
            settings["max_weight"] = cap_pct / 100.0
        else:
            settings["max_weight"] = 1.0
    if show_confidence:
        settings["confidence"] = st.sidebar.slider("VaR/CVaR confidence level", 0.90, 0.99, 0.95, 0.01,
                                                     key="shared_confidence")
    if show_backtest:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Backtest assumptions")
        settings["train_frac"] = st.sidebar.slider("Training window (% of history)", 0.4, 0.85, 0.7, 0.05,
                                                      key="shared_train_frac")
    return settings


# ---------------------------------------------------------------------
# Carry-forward state and navigation helpers
# ---------------------------------------------------------------------

def _current_construction_fingerprint() -> tuple:
    """A snapshot of the sidebar settings that determine how a portfolio's weights are built:
    universe, benchmark, history window, covariance estimator, optimizer cap, and risk-free rate
    (both Max-Sharpe methods use it directly, and the equilibrium prior adds it to the return, so
    it changes the optimized weights, not just downstream performance figures). Portfolio
    Construction's own risk-free input is always a plain scalar (see render_shared_settings), so
    this only ever needs to track that one number, not a mode flag. Read directly from
    session_state (not passed in) so it means the same thing whether it's called from the page that
    carries a portfolio forward or a later page that only checks it: the widgets that set these
    values live only on Portfolio Construction, but their session_state entries persist across page
    navigation once that page has been visited in this session."""
    ticker_input = st.session_state.get("shared_ticker_input", ", ".join(DEFAULT_TICKERS))
    tickers = tuple(sorted({t.strip().upper() for t in ticker_input.split(",") if t.strip()}))
    benchmark = st.session_state.get("shared_benchmark", DEFAULT_BENCHMARK).strip().upper()
    period = st.session_state.get("shared_period", "5y")
    use_shrinkage = bool(st.session_state.get("shared_shrinkage", True))
    enforce_cap = bool(st.session_state.get("shared_enforce_cap", False))
    max_weight_pct = int(st.session_state.get("shared_max_weight_pct", 20)) if enforce_cap else 100
    rf_annual = round(float(st.session_state.get("shared_rf", 0.04)), 4)
    return (tickers, benchmark, period, use_shrinkage, max_weight_pct, rf_annual)


def get_carried_portfolio():
    """Returns the carried-forward portfolio, if any, with a `stale` flag: True when the sidebar's
    current construction settings (tickers, benchmark, history window, covariance estimator, cap)
    no longer match what was in effect when it was carried forward, so a caller can warn rather
    than silently treat a portfolio built under different assumptions as current."""
    if "carried_tickers" in st.session_state and "carried_weights" in st.session_state:
        stored_fp = st.session_state.get("carried_fingerprint")
        return {
            "tickers": st.session_state["carried_tickers"],
            "weights": st.session_state["carried_weights"],
            "method_name": st.session_state.get("carried_method_name", "Custom"),
            "stale": stored_fp is not None and stored_fp != _current_construction_fingerprint(),
        }
    return None


def set_carried_portfolio(tickers: list[str], weights: np.ndarray, method_name: str):
    st.session_state["carried_tickers"] = tickers
    st.session_state["carried_weights"] = weights
    st.session_state["carried_method_name"] = method_name
    st.session_state["carried_fingerprint"] = _current_construction_fingerprint()


def anchor(slug: str):
    st.markdown(f'<a name="{slug}"></a>', unsafe_allow_html=True)


def toc(entries: list[tuple[str, str]]):
    lines = [f"- [{text}](#{slug})" for text, slug in entries]
    st.markdown("**Contents**\n" + "\n".join(lines))


def in_sample_notice():
    st.info(
        "In-sample view: the carried-forward weights were estimated on this same history, so the "
        "results below flatter the strategy by construction. The out-of-sample evidence is the "
        "walk-forward backtest on the Portfolio Construction page."
    )


def methodology_panel(max_weight: float | None = None, cost_bps: float | None = None,
                       rebalance_label: str | None = None, lookback: int | None = None):
    """Concise methodology and assumptions panel (collapsed by default)."""
    cap = ("no cap enforced (100%)" if (max_weight is None or max_weight >= 0.999)
           else f"optimizer max weight {max_weight:.0%} (enforced)")
    wf = ""
    if cost_bps is not None and rebalance_label and lookback:
        wf = (f" Current walk-forward settings: {lookback}-day estimation window, {rebalance_label} "
              f"rebalance, {cost_bps:g} bps per side.")
    with st.expander("Methodology and assumptions"):
        st.markdown(
            "- **Data:** Yahoo Finance via `yfinance`, daily closes with `auto_adjust=True` "
            "(split- and dividend-adjusted), so returns are total-return style. Simple daily returns; "
            "only dates where every ticker trades are used.\n"
            "- **Universe:** 20 large caps chosen by hand, up to 2 per GICS sector. The list is not "
            "rules-based and contains only companies that survive today (survivorship bias).\n"
            f"- **Constraints:** long-only, fully invested (weights sum to 100%), no leverage, no shorting, "
            f"{cap}, no sector or turnover constraints. The cap (when enforced) applies to Min-Variance, "
            "both Max-Sharpe portfolios and the frontier; Risk Parity is defined by equal risk "
            "contribution and is never capped.\n"
            "- **Estimation:** covariance is Ledoit-Wolf shrinkage (optional) or sample covariance. "
            "'Historical' expected returns are trailing mean returns, which are very noisy.\n"
            "- **BL equilibrium prior vs full Black-Litterman:** the equilibrium method reverse-engineers "
            "excess returns from market-cap weights (pi = delta x Sigma x w_mkt), adds the risk-free rate, "
            "and maximizes Sharpe. No investor views are blended in, so it is only the prior step of "
            "Black-Litterman, not the full posterior. With equilibrium returns the max-Sharpe portfolio "
            "is the market portfolio by construction, so without views (and without a binding cap) this "
            "method reproduces cap weighting. **Market-cap weights are approximate**: they are rebuilt "
            "point-in-time as (shares outstanding today) x (historical price), holding shares outstanding "
            "fixed over the whole sample, so buybacks and issuance are not reflected. Treat the "
            "equilibrium-prior method's out-of-sample result as approximate for this reason. Risk aversion "
            "comes from the benchmark, clipped to [1, 10]; it scales the returns but does not change the "
            "weights.\n"
            "- **Walk-forward and rebalancing:** each rebalance uses only the trailing estimation window "
            "(data strictly before the rebalance date), sets new weights, and holds them (drifting) until "
            "the next rebalance. One-way turnover at rebalance t is T_t = 0.5 x sum_i |w_i,t - w_i,t-|, "
            "where w_i,t- is the pre-rebalance (drifted) weight; the cost charged is (cost rate) x "
            "(2 x T_t), i.e. the per-side rate applied to total traded notional, including the initial "
            "purchase from cash. Costs ignore bid-ask spread, market impact and taxes." + wf + "\n"
            "- **Conventions:** 252 trading days per year; volatility = daily std x sqrt(252); Sharpe and "
            "Sortino use arithmetic mean excess return over the risk-free rate (Sortino's minimum "
            "acceptable return = risk-free rate) — a single constant rate everywhere except "
            "Performance's Historical mode, which uses that day's actual rate instead (see below); "
            "CAGR is geometric; tracking error and information ratio use daily active returns vs the "
            "benchmark; VaR and CVaR are 1-day unless a horizon is set, with multi-day historical "
            "figures scaled by the square root of time.\n"
            "- **Risk-free rate by page:** Portfolio Construction and its walk-forward backtest always "
            "use one scalar rate (a mean-variance optimization needs a single hurdle rate for its "
            "horizon, not a day-by-day series) — it sets the Max-Sharpe hurdle and is added to the "
            "Black-Litterman equilibrium prior. Performance offers a Historical mode (the ^IRX 13-week "
            "T-bill yield, day by day) for Sharpe, Sortino and the alpha/beta regression, since those "
            "measure realized excess returns after the fact and short rates moved a great deal over a "
            "multi-year window. Risk Report uses no risk-free rate at all: VaR, CVaR and Monte Carlo "
            "model the distribution of portfolio losses rather than excess return over cash, and beta "
            "is a covariance sensitivity that doesn't involve it either.\n"
            "- **Monte Carlo:** fixed random seed, so results reproduce exactly. Normal shocks understate "
            "tail risk; the Student-t option adds fatter joint tails while matching the same covariance.\n"
            "- **Known biases and limits:** estimation error in expected returns, non-stationary "
            "correlations, non-normal returns, survivorship bias, approximate point-in-time market-cap "
            "weights (see above), and short out-of-sample windows that give little statistical power. "
            "Educational tool, not investment advice."
        )
