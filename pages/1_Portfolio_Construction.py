"""
Portfolio Construction
------------------------
Builds and compares five portfolio construction methods on a user-defined universe,
then tests them out-of-sample with a walk-forward backtest that includes rebalancing,
turnover and transaction costs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import common as c

st.set_page_config(page_title="Portfolio Construction", layout="wide")

st.title("Portfolio Construction")
st.caption(
    "Compares Equal-Weight, Minimum-Variance, Risk Parity, and two Max-Sharpe portfolios "
    "(one using historical returns, one using a Black-Litterman equilibrium prior), then tests "
    "them out-of-sample with a walk-forward backtest."
)

c.toc([
    ("1. Return & Covariance Estimation", "returns-covariance"),
    ("2. Portfolio Construction", "construction"),
    ("3. Efficient Frontier", "frontier"),
    ("4. Walk-Forward Out-of-Sample Backtest", "backtest"),
    ("5. Carry a Portfolio Forward", "carry-forward"),
])

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
settings = c.render_shared_settings(show_backtest=False, show_shrinkage=True, show_constraints=True)
tickers, benchmark, period, rf_annual = (settings["tickers"], settings["benchmark"],
                                          settings["period"], settings["rf_annual"])
use_shrinkage, max_weight = settings["use_shrinkage"], settings["max_weight"]

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest assumptions")
st.sidebar.caption("Walk-forward")
lookback = st.sidebar.slider("Estimation window (trading days)", 126, 756, 252, 21, key="wf_lookback")
rebal_label = st.sidebar.radio("Rebalance frequency", ["Monthly", "Quarterly"], horizontal=True, key="wf_rebal")
rebalance_every = 21 if rebal_label == "Monthly" else 63
cost_bps = st.sidebar.number_input("Transaction cost (bps per side)", 0.0, 100.0, 10.0, 1.0, key="wf_cost")

st.sidebar.caption("Single train/test split (comparison)")
train_frac = st.sidebar.slider("Training window (% of history)", 0.4, 0.85, 0.7, 0.05, key="shared_train_frac")

if len(tickers) < 3:
    st.warning("Enter at least 3 tickers in the sidebar.")
    st.stop()

if max_weight < 0.999 and not c.cap_is_feasible(len(tickers), max_weight):
    st.warning(
        f"A {max_weight:.0%} optimizer max weight across {len(tickers)} tickers is infeasible — the "
        f"weights can't reach 100% while every position stays under that cap ({len(tickers)} x "
        f"{max_weight:.0%} = {len(tickers) * max_weight:.0%}). The optimizers below automatically relax "
        f"it to {1 / len(tickers):.1%} (equal weight) instead; raise the cap or add tickers to use a "
        "binding one."
    )

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
with st.spinner("Pulling price history..."):
    prices = c.load_prices(tickers, benchmark, period)
with st.spinner("Pulling market cap data..."):
    market_caps_raw = c.fetch_market_caps(tickers)

returns = c.compute_returns(prices)
asset_returns = returns[tickers]
bench_returns = returns[benchmark]

caps = pd.Series({t: (market_caps_raw.get(t) or np.nan) for t in tickers})
if caps.isna().any():
    missing = caps[caps.isna()].index.tolist()
    st.info(f"Market cap unavailable for {missing}; the mean cap of the others is used as a fallback "
            "for those names in the equilibrium prior only.")
    caps = caps.fillna(caps.mean())
w_mkt = (caps / caps.sum()).values
shares_outstanding = c.estimate_shares_outstanding(caps, prices[tickers].iloc[-1])

c.data_caption(prices)
c.methodology_panel(max_weight=max_weight, cost_bps=cost_bps,
                    rebalance_label=rebal_label.lower(), lookback=lookback)

# ---------------------------------------------------------------------
c.anchor("returns-covariance")
st.subheader("1. Return & Covariance Estimation")

cov_sample = c.sample_covariance(asset_returns)
cov_shrink, shrinkage_intensity = c.shrinkage_covariance(asset_returns)
cov = cov_shrink if use_shrinkage else cov_sample

mu_hist = c.historical_mean_returns(asset_returns)
delta = c.implied_risk_aversion(bench_returns, rf_annual)
mu_prior = c.equilibrium_expected_returns(cov, w_mkt, delta, rf_annual)

if use_shrinkage:
    st.caption(f"Ledoit-Wolf shrinkage intensity: **{shrinkage_intensity:.3f}** "
               "(0 = pure sample covariance, 1 = fully shrunk toward the structured target).")

est_df = pd.DataFrame({
    "Sector": [c.SECTOR_MAP.get(t, "—") for t in tickers],
    "Market Cap Weight": w_mkt,
    "Historical Return": mu_hist,
    "Equilibrium Expected Return": mu_prior,
}, index=tickers)
c.show_table(est_df.style.format({"Market Cap Weight": "{:.2%}", "Historical Return": "{:.2%}",
                                   "Equilibrium Expected Return": "{:.2%}"}))
st.caption(
    f"Implied market risk aversion (δ) from {benchmark}: **{delta:.2f}** (clipped to [1, 10]). "
    "**The equilibrium returns are only the prior step of Black-Litterman, not the full model.** "
    "Excess returns are reverse-engineered from market-cap weights (π = δ · Σ · w_mkt) and the "
    "risk-free rate is added to get the expected return shown. Full Black-Litterman would blend this "
    "prior with investor views into a posterior; no views are supplied here, so the posterior equals "
    "the prior. The historical column is far noisier, which is the estimation-error problem the prior "
    "is meant to soften."
)

# ---------------------------------------------------------------------
c.anchor("construction")
st.subheader("2. Portfolio Construction")

cap_text = "no per-position cap" if max_weight >= 0.999 else f"a {max_weight:.0%} cap per position"
st.caption(
    f"**Constraints (all methods):** long-only, fully invested (weights sum to 100%), no leverage or "
    f"shorting, {cap_text}, no sector or turnover constraints. The cap applies to Min-Variance, both "
    "Max-Sharpe portfolios and the frontier; Risk Parity is defined by equal risk contribution and is not capped."
)

methods = {
    c.M_EQ: np.ones(len(tickers)) / len(tickers),
    c.M_MV: c.optimize_min_variance(cov, max_weight),
    c.M_RP: c.optimize_risk_parity(cov),
    c.M_MS_HIST: c.optimize_max_sharpe(mu_hist, cov, rf_annual, max_weight),
    c.M_MS_PRIOR: c.optimize_max_sharpe(mu_prior, cov, rf_annual, max_weight),
}
if max_weight < 0.999 and methods[c.M_RP].max() > max_weight + 1e-6:
    st.caption(f"Note: Risk Parity's largest weight ({methods[c.M_RP].max():.1%}) exceeds the cap because "
               "that method is not capped.")

prior_gap = float(np.abs(methods[c.M_MS_PRIOR] - w_mkt).max())
if w_mkt.max() <= max_weight + 1e-9:
    st.caption(
        f"**Check on the equilibrium method:** with equilibrium returns the max-Sharpe portfolio is the market "
        f"portfolio, and here it matches the market-cap weights to within {prior_gap * 100:.2f} percentage "
        "points. So without investor views this method is cap weighting; it adds nothing until views are "
        "supplied."
    )
else:
    st.caption(
        f"**Check on the equilibrium method:** without a cap this portfolio would equal the market-cap weights. "
        f"The {max_weight:.0%} cap binds on the largest names, so it departs from them (largest gap "
        f"{prior_gap * 100:.1f} percentage points)."
    )

weights_df = pd.DataFrame(methods, index=tickers)
st.markdown("**Portfolio Weights by Method**")
st.caption("Heatmap rather than grouped bars: 20 tickers × 5 methods is 100 bars, too thin to compare. "
           "Darker cells are larger weights; hover for exact values.")
fig_weights = px.imshow(weights_df.T, aspect="auto", color_continuous_scale="Blues", text_auto=".0%",
                         labels=dict(x="Ticker", y="Method", color="Weight"))
fig_weights.update_traces(hovertemplate="Method: %{y}<br>Ticker: %{x}<br>Weight: %{z:.2%}<extra></extra>")
fig_weights.update_coloraxes(colorbar_tickformat=c.PCT0, colorbar_title="Weight")
fig_weights.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
c.show_chart(fig_weights)

st.markdown("**Risk/Return and Concentration by Method** (in-sample: estimated and measured on the same "
            "history, so these are not forecasts)")
stats_rows = []
for method, w in methods.items():
    mu_for_stats = mu_prior if method == c.M_MS_PRIOR else mu_hist
    row = c.portfolio_stats(w, mu_for_stats, cov, rf_annual)
    row["Effective N (weights)"] = c.effective_n_weights(w)
    row["Largest Position"] = float(w.max())
    row["Top 5 Weight"] = float(np.sort(w)[::-1][:5].sum())
    row["Positions > 1%"] = int((w > 0.01).sum())
    row["Method"] = method
    stats_rows.append(row)
stats_df = pd.DataFrame(stats_rows).set_index("Method")
c.show_table(stats_df.style.format({
    "Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
    "Effective N (weights)": "{:.1f}", "Largest Position": "{:.1%}", "Top 5 Weight": "{:.1%}",
    "Positions > 1%": "{:d}"}))
st.caption(
    "Effective N = 1 / Σw², the number of equal-sized positions with the same concentration "
    f"(equal weight over {len(tickers)} names gives {len(tickers)}). Max-Sharpe on historical returns "
    "typically concentrates into whichever names had the best trailing Sharpe ratio: estimation-error "
    "maximization. Min-Variance and Risk Parity use no return forecasts, so they are less exposed to it."
)

# ---------------------------------------------------------------------
c.anchor("frontier")
st.subheader("3. Efficient Frontier")
st.caption("Traced from historical returns and the selected covariance estimator under the same "
           "constraints as the optimizers. The dotted branch is mathematically valid but inefficient; "
           "only the solid branch is the efficient frontier. Asset tickers appear on hover, since 20 "
           "fixed labels would overlap. Every marker below is plotted using this same historical return "
           "vector and covariance matrix, including Max-Sharpe (BL Equilibrium Prior), which is optimized "
           "against a different (equilibrium) return vector — see section 2 for that method's return "
           "against its own optimization target instead.")

with st.spinner("Tracing efficient frontier..."):
    frontier_df = c.efficient_frontier(mu_hist, cov, n_points=30, max_weight=max_weight)

fig_frontier = go.Figure()
if len(frontier_df) > 0:
    ineff = frontier_df[~frontier_df["Efficient"]]
    eff = frontier_df[frontier_df["Efficient"]]
    if len(ineff) > 0:
        fig_frontier.add_trace(go.Scatter(x=ineff["Volatility"], y=ineff["Return"], mode="lines",
                                           line=dict(color="#64748B", dash="dot"), name="Inefficient branch"))
    fig_frontier.add_trace(go.Scatter(x=eff["Volatility"], y=eff["Return"], mode="lines",
                                       line=dict(color="#CBD5E1", width=3), name="Efficient frontier"))

asset_vols = np.sqrt(np.diag(cov))
fig_frontier.add_trace(go.Scatter(
    x=asset_vols, y=mu_hist, mode="markers", marker=dict(size=7, color="#8892A6"),
    name="Individual assets", text=tickers,
    hovertemplate="%{text}<br>Volatility: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"))
for method, w in methods.items():
    # Every marker is evaluated on the historical mu/cov that trace this frontier — not each
    # method's own optimization target — so a marker's position is always consistent with the
    # curve it sits on. (The BL-equilibrium method optimizes against a different return vector;
    # see section 2 for its return measured against that target instead.)
    frontier_stats = c.portfolio_stats(w, mu_hist, cov, rf_annual)
    fig_frontier.add_trace(go.Scatter(
        x=[frontier_stats["Volatility"]], y=[frontier_stats["Expected Return"]], mode="markers", name=method,
        marker=dict(size=15, symbol="star", color=c.METHOD_COLORS[method], line=dict(width=1, color="#0E1117")),
        hovertemplate=f"{method}<br>Volatility: %{{x:.1%}}<br>Return: %{{y:.1%}}<extra></extra>"))
fig_frontier.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(title="Volatility (annualized)", tickformat=c.PCT0),
                            yaxis=dict(title="Expected return (annualized)", tickformat=c.PCT0))
c.show_chart(fig_frontier)

# ---------------------------------------------------------------------
c.anchor("backtest")
st.subheader("4. Walk-Forward Out-of-Sample Backtest")
st.caption(
    f"At each {rebal_label.lower()} rebalance, every method re-estimates its inputs from only the trailing "
    f"{lookback} trading days, sets new weights, and holds them (drifting) until the next rebalance. "
    f"Costs are {cost_bps:g} bps per side on traded notional, including the initial purchase. Repeating this "
    "across the whole sample tests the methods on many out-of-sample decisions instead of a single split."
)

with st.spinner("Running walk-forward backtest..."):
    wf = c.walk_forward_backtest(asset_returns, bench_returns, prices[tickers], shares_outstanding,
                                  rf_annual, lookback, rebalance_every, cost_bps, use_shrinkage, max_weight)

if wf is None:
    st.warning(f"The sample is too short for a walk-forward test with a {lookback}-day estimation window "
               f"and {rebal_label.lower()} rebalancing (needs the window plus at least two rebalance "
               "periods). Increase the History window or shorten the estimation window.")
else:
    meta = wf["_meta"]
    st.caption(f"Out-of-sample period: {meta['oos_start'].date()} to {meta['oos_end'].date()} "
               f"({meta['oos_years']:.1f} years, {meta['n_rebalances']} rebalances). "
               "Growth is net of transaction costs.")
    oos_index = wf[c.M_EQ]["net"].index
    bench_oos = bench_returns.loc[oos_index]

    fig_growth = go.Figure()
    for method in c.METHODS:
        growth = (1 + wf[method]["net"]).cumprod()
        fig_growth.add_trace(go.Scatter(x=growth.index, y=growth.values, mode="lines", name=method,
                                         line=dict(color=c.METHOD_COLORS[method])))
    bench_growth = (1 + bench_oos).cumprod()
    fig_growth.add_trace(go.Scatter(x=bench_growth.index, y=bench_growth.values, mode="lines",
                                     line=c.BENCH_LINE, name=f"{benchmark} (buy & hold)"))
    fig_growth.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                              yaxis=dict(title="Growth of $1", tickprefix="$", tickformat=".2f"))
    st.markdown("**Out-of-Sample Growth of $1 (net of costs)**")
    c.show_chart(fig_growth)

    rows = {}
    for method in c.METHODS:
        net, gross = wf[method]["net"], wf[method]["gross"]
        s = c.performance_stats(net, rf_annual, bench_oos)
        turn = wf[method]["turnover"].iloc[1:]            # exclude the initial purchase from cash
        s["Annual Turnover"] = turn.sum() / meta["oos_years"]
        s["Cost Drag (CAGR)"] = c.performance_stats(gross, rf_annual)["CAGR"] - s["CAGR"]
        rows[method] = s
    rows[f"{benchmark} (benchmark)"] = c.performance_stats(bench_oos, rf_annual)
    perf_df = pd.DataFrame(rows).T[["CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown",
                                     "Tracking Error", "Information Ratio", "Annual Turnover", "Cost Drag (CAGR)"]]
    st.markdown("**Out-of-Sample Performance (net of costs)**")
    c.show_table(c.dash_style(perf_df, {
        "CAGR": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
        "Max Drawdown": "{:.2%}", "Tracking Error": "{:.2%}", "Information Ratio": "{:.2f}",
        "Annual Turnover": "{:.0%}", "Cost Drag (CAGR)": "{:.2%}"}))
    st.caption(
        f"{meta['n_rebalances']} rebalances over {meta['oos_years']:.1f} years is a small sample, so "
        "differences between methods are usually within statistical noise; treat the ranking as "
        "descriptive. Annual turnover is one-way (half of traded notional) and excludes the initial "
        "purchase. Cost drag is gross minus net CAGR. Costs are a flat per-side rate: no spread, "
        "market impact or taxes, so real frictions would be higher, especially for concentrated portfolios."
    )

    with st.expander("Rebalance detail: turnover per rebalance by method"):
        turn_df = pd.DataFrame({m: wf[m]["turnover"] for m in c.METHODS})
        turn_df.index = turn_df.index.strftime("%Y-%m-%d")
        c.show_table(turn_df.style.format("{:.1%}"))

    with st.expander("Gross vs. net performance (before transaction costs)"):
        st.caption("Same walk-forward weights and rebalances, simulated with zero transaction cost, so "
                   "the difference from the net table above is entirely the cost drag shown there.")
        gross_rows = {m: c.performance_stats(wf[m]["gross"], rf_annual, bench_oos) for m in c.METHODS}
        gross_df = pd.DataFrame(gross_rows).T[["CAGR", "Ann. Volatility", "Sharpe", "Sortino",
                                                "Max Drawdown", "Tracking Error", "Information Ratio"]]
        c.show_table(c.dash_style(gross_df, {
            "CAGR": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max Drawdown": "{:.2%}", "Tracking Error": "{:.2%}", "Information Ratio": "{:.2f}"}))

# Single split, kept as a simpler comparison
with st.expander("Single train/test split (simpler comparison)"):
    train_returns, test_returns = c.train_test_split_returns(returns, train_frac)
    st.caption(
        f"Weights are estimated once on the training period, then held (no rebalancing, no costs) through "
        f"the later test period. Training: {train_returns.index[0].date()} to {train_returns.index[-1].date()} "
        f"({len(train_returns)} days). Test: {test_returns.index[0].date()} to {test_returns.index[-1].date()} "
        f"({len(test_returns)} days). One split is a single draw, which is why the walk-forward test above "
        "is the main evidence."
    )
    if len(test_returns) < 30:
        st.warning("The test period is very short. Widen the History window or lower the training fraction.")
    else:
        tr_assets, tr_bench = train_returns[tickers], train_returns[benchmark]
        te_assets, te_bench = test_returns[tickers], test_returns[benchmark]
        tr_cov = c.shrinkage_covariance(tr_assets)[0] if use_shrinkage else c.sample_covariance(tr_assets)
        tr_mu = c.historical_mean_returns(tr_assets)
        tr_delta = c.implied_risk_aversion(tr_bench, rf_annual)
        tr_w_mkt = c.market_cap_weights_at(shares_outstanding, prices[tickers].loc[tr_assets.index[-1]])
        tr_prior = c.equilibrium_expected_returns(tr_cov, tr_w_mkt, tr_delta, rf_annual)
        split_methods = {
            c.M_EQ: np.ones(len(tickers)) / len(tickers),
            c.M_MV: c.optimize_min_variance(tr_cov, max_weight),
            c.M_RP: c.optimize_risk_parity(tr_cov),
            c.M_MS_HIST: c.optimize_max_sharpe(tr_mu, tr_cov, rf_annual, max_weight),
            c.M_MS_PRIOR: c.optimize_max_sharpe(tr_prior, tr_cov, rf_annual, max_weight),
        }
        fig_split = go.Figure()
        rows = {}
        for method, w in split_methods.items():
            r = pd.Series(te_assets.values @ w, index=te_assets.index)
            g = (1 + r).cumprod()
            fig_split.add_trace(go.Scatter(x=g.index, y=g.values, mode="lines", name=method,
                                            line=dict(color=c.METHOD_COLORS[method])))
            rows[method] = c.performance_stats(r, rf_annual, te_bench)
        bg = (1 + te_bench).cumprod()
        fig_split.add_trace(go.Scatter(x=bg.index, y=bg.values, mode="lines", line=c.BENCH_LINE,
                                        name=f"{benchmark} (buy & hold)"))
        rows[f"{benchmark} (benchmark)"] = c.performance_stats(te_bench, rf_annual)
        fig_split.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                                 yaxis=dict(title="Growth of $1", tickprefix="$", tickformat=".2f"))
        c.show_chart(fig_split)
        split_df = pd.DataFrame(rows).T[["CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown",
                                          "Tracking Error", "Information Ratio"]]
        c.show_table(c.dash_style(split_df, {
            "CAGR": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max Drawdown": "{:.2%}", "Tracking Error": "{:.2%}", "Information Ratio": "{:.2f}"}))

# ---------------------------------------------------------------------
c.anchor("carry-forward")
st.subheader("5. Carry a Portfolio Forward")
st.caption(
    "Pick one portfolio to analyze on the Performance and Risk Report pages. Those pages use the "
    "full-history weights shown in section 2, which were estimated on the same data they are "
    "measured on (in-sample); the walk-forward results above are the out-of-sample evidence."
)
carry_choice = st.selectbox("Portfolio to carry forward", list(methods.keys()))
if st.button("Carry this portfolio forward", type="primary"):
    c.set_carried_portfolio(tickers, methods[carry_choice], carry_choice)
    st.success(f"'{carry_choice}' carried forward with {len(tickers)} tickers. "
               "Open the Performance or Risk Report page to see it analyzed.")
carried = c.get_carried_portfolio()
if carried is not None:
    if carried.get("stale"):
        st.warning(
            f"Currently carried forward: **{carried['method_name']}** ({len(carried['tickers'])} tickers) — "
            "but the sidebar settings have changed since it was built (tickers, benchmark, history window, "
            "covariance estimator, cap, or risk-free rate), so its weights no longer reflect the current settings. Carry it "
            "forward again to refresh it."
        )
    else:
        st.info(f"Currently carried forward: **{carried['method_name']}** ({len(carried['tickers'])} tickers)")
