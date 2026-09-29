"""
Risk Report
------------
Volatility, contribution to risk, VaR/CVaR (historical, parametric, Monte Carlo with a
Student-t option), and VaR backtesting (Kupiec and Christoffersen tests) for the portfolio
carried forward from Portfolio Construction (or a manually specified one).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import common as c

st.set_page_config(page_title="Risk Report", layout="wide")

st.title("Risk Report")
st.caption(
    "Volatility, contribution to risk, VaR/CVaR (historical, parametric, Monte Carlo), and VaR "
    "backtesting for the portfolio carried forward from Portfolio Construction."
)

c.toc([
    ("1. Risk Metrics by Position", "by-position"),
    ("2. Portfolio-Level Risk", "portfolio-risk"),
    ("3. Contribution to Risk and Diversification", "contribution"),
    ("4. Rolling Volatility", "rolling-vol"),
    ("5. Correlation Matrix", "correlation"),
    ("6. Return Distribution", "distribution"),
    ("7. Monte Carlo VaR / CVaR", "monte-carlo"),
    ("8. VaR Backtesting (Kupiec and Christoffersen)", "backtesting"),
])

settings = c.render_shared_settings(show_backtest=False, show_confidence=True, show_shrinkage=False,
                                      show_rf=False)
benchmark, period, confidence = settings["benchmark"], settings["period"], settings["confidence"]
conf_pct = int(round(confidence * 100))

with st.sidebar:
    st.markdown("---")
    st.markdown("**Monte Carlo simulation**")
    n_sims = st.select_slider("Number of simulations", [1000, 5000, 10000, 25000, 50000], value=10000)
    horizon_days = st.slider("Horizon (trading days)", 1, 20, 1)
    seed = int(st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1,
                                help="Fixed seed: the same inputs and seed always give identical results."))
    mc_dist = st.radio("Simulated shocks", ["Normal", "Student-t"], horizontal=True)
    t_df = st.slider("Student-t degrees of freedom", 3, 30, 5,
                      help="Lower = fatter tails. About 4 to 6 is typical for daily equity returns.",
                      disabled=(mc_dist != "Student-t"))

    st.markdown("---")
    st.markdown("**VaR backtesting**")
    backtest_window = st.slider("Rolling estimation window (days)", 60, 500, 250, step=10)
    backtest_method = st.radio("Backtest method", ["Historical", "Parametric"], horizontal=True)

    st.markdown("---")
    st.markdown("**Portfolio weights**")
    use_custom = st.checkbox("Override with custom weights instead of the carried portfolio")

carried = c.get_carried_portfolio()

if use_custom or carried is None:
    if carried is None and not use_custom:
        st.warning(
            "No portfolio has been carried forward yet, so this page shows an equal-weight portfolio of "
            "the sidebar tickers. Go to Portfolio Construction and click 'Carry this portfolio forward' "
            "to analyze a specific method."
        )
    tickers = settings["tickers"]
    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers in the sidebar.")
        st.stop()
    if use_custom:
        weight_mode = st.sidebar.radio("Weighting", ["Equal weight", "Custom"], horizontal=True)
        weights_dict = {}
        for t in tickers:
            weights_dict[t] = 1.0 if weight_mode == "Equal weight" else st.sidebar.number_input(
                f"{t} weight", min_value=0.0, value=1.0, step=0.1)
        w_series = pd.Series(weights_dict)
        weights = (w_series / w_series.sum()).values
        method_name = "Custom weights"
    else:
        weights = np.ones(len(tickers)) / len(tickers)
        method_name = "Equal Weight"
else:
    tickers = carried["tickers"]
    weights = np.asarray(carried["weights"], dtype=float)
    method_name = carried["method_name"]
    if carried.get("stale"):
        st.warning(
            f"Analyzing **{method_name}** ({len(tickers)} tickers), but the sidebar settings have changed "
            "on Portfolio Construction since this portfolio was carried forward (tickers, benchmark, "
            "history window, covariance estimator, cap, or risk-free rate), so these weights may no longer reflect the "
            "current settings. Go back and carry it forward again to refresh it."
        )
    else:
        st.info(f"Analyzing carried-forward portfolio: **{method_name}** ({len(tickers)} tickers). Tick "
                "'Override with custom weights' in the sidebar to analyze something else.")

with st.spinner("Pulling price history..."):
    prices = c.load_prices(tickers, benchmark, period)

returns = c.compute_returns(prices)
asset_returns = returns[tickers]
bench_returns = returns[benchmark]
port_ret = c.portfolio_returns_series(returns, tickers, weights)
port_color = c.METHOD_COLORS.get(method_name, c.NEUTRAL_BLUE)

cov_ann = c.sample_covariance(asset_returns)
rc = c.risk_contributions(weights, cov_ann)
asset_vols = np.sqrt(np.diag(cov_ann))

c.data_caption(prices)
c.methodology_panel()
c.in_sample_notice()

# ---------------------------------------------------------------------
c.anchor("by-position")
st.subheader("1. Risk Metrics by Position")

rows = []
for i, t in enumerate(tickers):
    r = asset_returns[t]
    dd, _ = c.max_drawdown(prices[t])
    rows.append({
        "Ticker": t,
        "Sector": c.SECTOR_MAP.get(t, "—"),
        "Weight": weights[i],
        "Risk Contribution": rc[i],
        "Ann. Volatility": c.annualized_vol(r),
        f"Hist. VaR ({conf_pct}%, 1d)": c.historical_var(r, confidence),
        f"Hist. CVaR ({conf_pct}%, 1d)": c.historical_cvar(r, confidence),
        f"Parametric VaR ({conf_pct}%, 1d)": c.parametric_var(r, confidence),
        f"Beta vs {benchmark}": c.beta_vs_benchmark(r, bench_returns),
        "Max Drawdown": dd,
    })
risk_df = pd.DataFrame(rows).set_index("Ticker")
fmt = {col: "{:.2%}" for col in risk_df.columns if col not in ("Sector", f"Beta vs {benchmark}")}
fmt[f"Beta vs {benchmark}"] = "{:.2f}"
c.show_table(risk_df.style.format(fmt))

# ---------------------------------------------------------------------
c.anchor("portfolio-risk")
st.subheader("2. Portfolio-Level Risk")

port_dd, port_dd_series = c.max_drawdown((1 + port_ret).cumprod())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ann. Volatility", f"{c.annualized_vol(port_ret):.2%}")
k2.metric(f"Hist. VaR ({conf_pct}%, 1d)", f"{c.historical_var(port_ret, confidence):.2%}")
k3.metric(f"Hist. CVaR ({conf_pct}%, 1d)", f"{c.historical_cvar(port_ret, confidence):.2%}")
k4.metric(f"Beta vs {benchmark}", f"{c.beta_vs_benchmark(port_ret, bench_returns):.2f}")
k5.metric("Max Drawdown", f"{port_dd:.2%}")

st.markdown("**Portfolio Drawdown Over Time**")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=port_dd_series.index, y=port_dd_series.values, mode="lines", name="Drawdown",
                             fill="tozeroy", line=dict(color=c.LOSS_RED), fillcolor="rgba(229,72,77,0.30)"))
fig_dd.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      yaxis=dict(title="Drawdown", tickformat=c.PCT0))
c.show_chart(fig_dd)

# ---------------------------------------------------------------------
c.anchor("contribution")
st.subheader("3. Contribution to Risk and Diversification")
st.caption(
    "Each position's contribution to risk is its share of total portfolio variance, w_i(Σw)_i / w'Σw, using "
    "the sample covariance of the same history. Contributions sum to 100%. A position whose bar of risk "
    "exceeds its bar of weight is adding more risk than capital."
)

eff_w = c.effective_n_weights(weights)
eff_r = c.effective_n_risk(rc)
port_vol_ann = float(np.sqrt(weights @ cov_ann @ weights))
div_ratio = float(weights @ asset_vols / port_vol_ann)
top3_share = float(np.sort(rc)[::-1][:3].sum())
e1, e2, e3, e4 = st.columns(4)
e1.metric("Effective N (weights)", f"{eff_w:.1f}", help="1 / Σw²: equal-sized positions with the same concentration.")
e2.metric("Effective N (risk)", f"{eff_r:.1f}", help="1 / Σ(risk share²): how many equal risk bets the portfolio is really made of.")
e3.metric("Top 3 share of risk", f"{top3_share:.0%}")
e4.metric("Diversification ratio", f"{div_ratio:.2f}", help="Weighted-average asset volatility / portfolio volatility. Higher means more diversification benefit.")

order = np.argsort(-rc)
fig_rc = go.Figure()
fig_rc.add_trace(go.Bar(y=[tickers[i] for i in order], x=weights[order], orientation="h", name="Weight",
                         marker_color=c.NEUTRAL_BLUE, hovertemplate="%{y} weight: %{x:.2%}<extra></extra>"))
fig_rc.add_trace(go.Bar(y=[tickers[i] for i in order], x=rc[order], orientation="h", name="Share of risk",
                         marker_color=c.LOSS_RED, hovertemplate="%{y} risk share: %{x:.2%}<extra></extra>"))
fig_rc.update_layout(barmode="group", height=max(360, 26 * len(tickers) + 100),
                      margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                      xaxis=dict(title="Share of portfolio", tickformat=c.PCT0),
                      yaxis=dict(autorange="reversed"))
c.show_chart(fig_rc)

# ---------------------------------------------------------------------
c.anchor("rolling-vol")
st.subheader("4. Rolling Volatility (annualized, 21-day window)")
st.caption("The portfolio against the benchmark, with optional individual positions. The default adds the "
           "three largest risk contributors; twenty overlapping lines are unreadable.")

roll = lambda s: s.rolling(21).std() * np.sqrt(c.TRADING_DAYS)
default_extras = [tickers[i] for i in order[:3]]
extras = st.multiselect("Add individual positions to the chart", tickers, default=default_extras)
tk_colors = c.get_ticker_colors(tickers)

fig_vol = go.Figure()
for t in extras:
    rv = roll(asset_returns[t])
    fig_vol.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name=t,
                                  line=dict(color=tk_colors[t], width=1), opacity=0.75))
rv_p, rv_b = roll(port_ret), roll(bench_returns)
fig_vol.add_trace(go.Scatter(x=rv_b.index, y=rv_b.values, mode="lines", name=f"{benchmark} (benchmark)",
                              line=c.BENCH_LINE))
fig_vol.add_trace(go.Scatter(x=rv_p.index, y=rv_p.values, mode="lines", name=f"Portfolio ({method_name})",
                              line=dict(color=port_color, width=3)))
fig_vol.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                       yaxis=dict(title="Annualized volatility", tickformat=c.PCT0))
c.show_chart(fig_vol)

# ---------------------------------------------------------------------
c.anchor("correlation")
st.subheader("5. Correlation Matrix (daily returns)")
st.caption("Tickers are grouped by sector. The color scale is diverging and centered at zero — blue for "
           "negative correlation, red for positive — scaled to the largest absolute off-diagonal "
           "correlation, so a mildly negative cell reads as visually distinct from a near-zero one instead "
           "of both looking the same; the diagonal is always 1.00 by definition and shows at the reddest end.")
sector_order = sorted(tickers, key=lambda t: (c.SECTOR_MAP.get(t, "~"), t))
corr = asset_returns[sector_order].corr()
off_diag = corr.values.copy()
np.fill_diagonal(off_diag, np.nan)
corr_bound = max(float(np.nanmax(np.abs(off_diag))), 0.05)
fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-corr_bound, zmax=corr_bound, aspect="auto")
fig_corr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
c.show_chart(fig_corr)

# ---------------------------------------------------------------------
c.anchor("distribution")
st.subheader("6. Portfolio Daily Return Distribution")
hist_var, hist_cvar = c.historical_var(port_ret, confidence), c.historical_cvar(port_ret, confidence)
fig_hist = go.Figure(go.Histogram(x=port_ret, nbinsx=60, marker_color=c.NEUTRAL_BLUE,
                                   hovertemplate="Return %{x}<br>Days %{y}<extra></extra>"))
fig_hist.add_vline(x=-hist_var, line_color=c.LOSS_RED, line_dash="dash",
                    annotation_text=f"VaR ({conf_pct}%)", annotation_position="top left")
fig_hist.add_vline(x=-hist_cvar, line_color=c.LOSS_RED, line_dash="dot",
                    annotation_text="CVaR", annotation_position="top left")
fig_hist.update_layout(height=340, margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
                        xaxis=dict(title="Daily return", tickformat=c.PCT1), yaxis=dict(title="Days"))
c.show_chart(fig_hist)

# ---------------------------------------------------------------------
c.anchor("monte-carlo")
st.subheader("7. Monte Carlo VaR / CVaR")
st.caption(
    f"Simulates {n_sims:,} correlated {horizon_days}-day return paths from the portfolio's historical mean "
    "and covariance (Cholesky decomposition keeps cross-asset correlation). Normal shocks understate joint "
    "tail risk; the Student-t option draws fatter-tailed shocks with the same covariance. "
    f"The random seed is fixed at {seed}, so results reproduce exactly on every run."
)

mc_n = c.monte_carlo_portfolio_var(asset_returns, tickers, weights, confidence, n_sims, horizon_days, seed,
                                    "normal")
mc_t = c.monte_carlo_portfolio_var(asset_returns, tickers, weights, confidence, n_sims, horizon_days, seed,
                                    "t", float(t_df))
mc_var, mc_cvar, mc_sims = mc_t if mc_dist == "Student-t" else mc_n

scale = np.sqrt(horizon_days)
comp = pd.DataFrame({
    "Method": ["Historical", "Parametric (normal)", "Monte Carlo (normal)", f"Monte Carlo (Student-t, df={t_df})"],
    "VaR": [hist_var * scale, c.parametric_var(port_ret, confidence) * scale, mc_n[0], mc_t[0]],
    "CVaR": [hist_cvar * scale, c.parametric_cvar(port_ret, confidence) * scale, mc_n[1], mc_t[1]],
})
q1, q2, q3 = st.columns(3)
q1.metric(f"Monte Carlo VaR ({mc_dist})", f"{mc_var:.2%}")
q2.metric(f"Monte Carlo CVaR ({mc_dist})", f"{mc_cvar:.2%}")
q3.metric("Historical VaR (scaled)", f"{hist_var * scale:.2%}")

st.markdown(f"**VaR and CVaR at {conf_pct}% confidence, {horizon_days}-day horizon**")
fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(x=comp["Method"], y=comp["VaR"], name="VaR", marker_color=c.NEUTRAL_BLUE,
                          text=[f"{v:.2%}" for v in comp["VaR"]], textposition="outside"))
fig_cmp.add_trace(go.Bar(x=comp["Method"], y=comp["CVaR"], name="CVaR (expected shortfall)",
                          marker_color=c.LOSS_RED, text=[f"{v:.2%}" for v in comp["CVaR"]],
                          textposition="outside"))
fig_cmp.update_layout(barmode="group", height=360, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                       yaxis=dict(title="Loss (fraction of portfolio)", tickformat=c.PCT1,
                                   rangemode="tozero"))
c.show_chart(fig_cmp)
st.caption(f"Historical and parametric figures are 1-day estimates scaled by the square root of {horizon_days} "
           "(assumes independent, identically distributed daily returns). Where Student-t CVaR exceeds normal "
           "CVaR, the normal model is understating tail risk. At 95% the Student-t VaR can come out lower than "
           "the normal VaR: with the same variance, the t is more peaked in the middle and puts its extra "
           "weight further out in the tail, so the gap shows up in CVaR and at higher confidence levels "
           "(try 99%).")

st.markdown(f"**Simulated {horizon_days}-day portfolio returns ({mc_dist} shocks)**")
fig_mc = go.Figure(go.Histogram(x=mc_sims, nbinsx=80, marker_color=c.NEUTRAL_BLUE,
                                 hovertemplate="Return %{x}<br>Paths %{y}<extra></extra>"))
fig_mc.add_vline(x=-mc_var, line_color=c.LOSS_RED, line_dash="dash", annotation_text=f"VaR ({conf_pct}%)",
                  annotation_position="top left")
fig_mc.add_vline(x=-mc_cvar, line_color=c.LOSS_RED, line_dash="dot", annotation_text="CVaR",
                  annotation_position="top left")
fig_mc.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
                      xaxis=dict(title=f"{horizon_days}-day portfolio return", tickformat=c.PCT1),
                      yaxis=dict(title="Simulated paths"))
c.show_chart(fig_mc)

# ---------------------------------------------------------------------
c.anchor("backtesting")
st.subheader("8. VaR Backtesting (Kupiec and Christoffersen)")
st.caption(
    f"Re-estimates {backtest_method.lower()} VaR each day from only the trailing {backtest_window} days "
    f"(never the day being tested), then counts how often the actual loss exceeded it. A well-calibrated "
    f"{conf_pct}% VaR is breached on about {100 - conf_pct}% of days. Kupiec tests the breach frequency; "
    "Christoffersen adds tests for clustering (independence) and for both together (conditional coverage)."
)

rolling_var = c.rolling_var_series(port_ret, backtest_window, confidence, method=backtest_method.lower())
bt = pd.DataFrame({"Return": port_ret, "VaR": rolling_var}).dropna()

if len(bt) < 30:
    st.warning(f"Only {len(bt)} out-of-sample days remain after the {backtest_window}-day window. "
               "Increase the History window or shrink the rolling window.")
else:
    bt["Breach"] = bt["Return"] < -bt["VaR"]
    tests = c.christoffersen_tests(bt["Breach"].astype(int).values, confidence)
    n_obs, n_br = tests["n"], tests["breaches"]
    expected = n_obs * (1 - confidence)

    r1a, r1b, r1c = st.columns(3)
    r1a.metric("Observations", f"{n_obs}")
    r1b.metric("Breaches", f"{n_br}", f"expected about {expected:.1f}", delta_color="off", delta_arrow="off")
    r1c.metric("Breach rate", f"{n_br / n_obs:.2%}", f"target {100 - conf_pct}%", delta_color="off", delta_arrow="off")

    def verdict(p):
        return ("No rejection at 5%", "normal") if p >= 0.05 else ("Rejected at 5%", "inverse")
    v_pof, v_ind, v_cc = verdict(tests["p_pof"]), verdict(tests["p_ind"]), verdict(tests["p_cc"])
    r2a, r2b, r2c = st.columns(3)
    r2a.metric("Kupiec p-value (frequency)", f"{tests['p_pof']:.3f}", v_pof[0], delta_color=v_pof[1], delta_arrow="off")
    r2b.metric("Christoffersen p-value (independence)", f"{tests['p_ind']:.3f}", v_ind[0], delta_color=v_ind[1], delta_arrow="off")
    r2c.metric("Conditional coverage p-value (both)", f"{tests['p_cc']:.3f}", v_cc[0], delta_color=v_cc[1], delta_arrow="off")

    if tests["p_cc"] >= 0.05:
        st.success(
            "At the 5% significance level, the conditional-coverage null is not rejected. The observed "
            "breach frequency is consistent with the stated confidence level, with no statistically "
            "significant evidence of breach clustering."
        )
    else:
        problems = []
        if tests["p_pof"] < 0.05:
            problems.append("the breach frequency is " + ("too high" if n_br > expected else "too low"))
        if tests["p_ind"] < 0.05:
            problems.append("breaches cluster in time")
        if not problems:
            problems.append("frequency and clustering are jointly off even though neither test is rejected alone")
        st.error("At the 5% significance level, the conditional-coverage null is rejected: " +
                  "; ".join(problems) + ".")
    if tests["p_pof"] >= 0.05 and tests["p_ind"] < 0.05:
        st.warning("Kupiec alone would not be rejected, but the independence test is: the breach frequency "
                   "is consistent with the stated confidence level while the losses arrive clustered in "
                   "time. That is the gap the Christoffersen test exists to catch.")

    st.markdown("**Actual Returns vs. Rolling VaR Threshold**")
    fig_bt = go.Figure()
    ok_days, br_days = bt[~bt["Breach"]], bt[bt["Breach"]]
    fig_bt.add_trace(go.Scatter(x=ok_days.index, y=ok_days["Return"], mode="markers", name="Daily return",
                                 marker=dict(size=5, color=c.NEUTRAL_BLUE)))
    fig_bt.add_trace(go.Scatter(x=br_days.index, y=br_days["Return"], mode="markers", name="Breach",
                                 marker=dict(size=8, color=c.LOSS_RED)))
    fig_bt.add_trace(go.Scatter(x=bt.index, y=-bt["VaR"], mode="lines", name=f"-VaR ({conf_pct}%)",
                                 line=dict(color=c.WARN_AMBER, dash="dash")))
    fig_bt.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                          yaxis=dict(title="Daily return", tickformat=c.PCT1))
    c.show_chart(fig_bt)

    with st.expander("Breach transition counts (used by the independence test)"):
        st.markdown(
            f"- Day after a non-breach: {tests['n00']} non-breaches, {tests['n01']} breaches "
            f"(P(breach | no breach yesterday) = {tests['pi01']:.1%})\n"
            f"- Day after a breach: {tests['n10']} non-breaches, {tests['n11']} breaches "
            f"(P(breach | breach yesterday) = {tests['pi11']:.1%})\n\n"
            "If breaches were independent, these two probabilities would be about equal."
        )

    if n_br > 0:
        st.markdown("**Breach Details**")
        breach_table = br_days[["Return", "VaR"]].copy()
        breach_table.index = breach_table.index.strftime("%Y-%m-%d")
        breach_table.index.name = "Date"
        breach_table.columns = ["Actual Return", "VaR Estimate"]
        c.show_table(breach_table.style.format({"Actual Return": "{:.2%}", "VaR Estimate": "{:.2%}"}), max_rows=12)

st.caption(
    "VaR and CVaR are historical estimates from the selected window, not forward-looking guarantees. "
    "Beta is computed from daily returns against the benchmark."
)
