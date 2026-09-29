"""
Performance
------------
Performance analytics for the portfolio carried forward from Portfolio Construction
(or an equal-weight fallback): growth, relative and risk-adjusted metrics, calendar-year
returns, rolling Sharpe, rolling alpha/beta, and drawdown.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import common as c

st.set_page_config(page_title="Performance", layout="wide")

st.title("Performance")
st.caption(
    "Growth, risk-adjusted and benchmark-relative metrics, rolling Sharpe, rolling alpha and beta, "
    "and drawdown for the portfolio carried forward from Portfolio Construction."
)

c.toc([
    ("1. Portfolio Overview", "overview"),
    ("2. Static Historical Backcast & Summary Metrics", "growth"),
    ("3. Calendar-Year Returns", "calendar"),
    ("4. Rolling Sharpe Ratio", "rolling-sharpe"),
    ("5. Rolling Alpha & Beta", "rolling-alpha-beta"),
    ("6. Drawdown", "drawdown"),
])

settings = c.render_shared_settings(show_backtest=False, show_confidence=False, show_shrinkage=False,
                                      show_rf_mode=True)
benchmark, period, rf_annual = settings["benchmark"], settings["period"], settings["rf_annual"]

carried = c.get_carried_portfolio()
if carried is None:
    st.warning(
        "No portfolio has been carried forward yet, so this page shows an equal-weight portfolio of the "
        "sidebar tickers. Go to Portfolio Construction and click 'Carry this portfolio forward' to "
        "analyze a specific method."
    )
    tickers = settings["tickers"]
    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers in the sidebar.")
        st.stop()
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

with st.spinner("Pulling price history..."):
    prices = c.load_prices(tickers, benchmark, period)

returns = c.compute_returns(prices)
bench_returns = returns[benchmark]
port_returns = c.portfolio_returns_series(returns, tickers, weights)
rf_obj = settings["rf"]                      # constant float, or a historical daily-rate Series
port_color = c.METHOD_COLORS.get(method_name, c.NEUTRAL_BLUE)

c.data_caption(prices)
if settings["rf_mode"] == "Historical":
    st.caption(f"Risk-free rate: historical 13-week T-bill yield, day by day (period average "
               f"{rf_annual:.2%}) — used in Sharpe, Sortino and the alpha/beta regression below.")
else:
    st.caption(f"Risk-free rate: constant annual rate of {rf_annual:.2%}, applied uniformly across "
               "the whole history window.")
c.methodology_panel()
c.in_sample_notice()

# ---------------------------------------------------------------------
c.anchor("overview")
st.subheader("1. Portfolio Overview")
st.markdown(f"Analyzing **{method_name}** across **{len(tickers)}** tickers, benchmarked against **{benchmark}**.")

order = np.argsort(-weights)
top_ticker = tickers[int(order[0])]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Positions above 1%", f"{int((weights > 0.01).sum())}")
m2.metric("Effective N (weights)", f"{c.effective_n_weights(weights):.1f}")
m3.metric("Largest position", f"{weights.max():.1%}", top_ticker, delta_color="off", delta_arrow="off")
m4.metric("Top 5 weight", f"{np.sort(weights)[::-1][:5].sum():.0%}")

fig_w = go.Figure(go.Bar(x=[tickers[i] for i in order], y=weights[order], marker_color=port_color,
                          hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
fig_w.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                     yaxis=dict(title="Weight", tickformat=c.PCT0))
c.show_chart(fig_w)
st.caption("Effective N = 1 / Σw²: the number of equal-sized positions with the same concentration.")

# ---------------------------------------------------------------------
c.anchor("growth")
st.subheader(f"2. Static Historical Backcast & Summary Metrics — {method_name}")
st.caption("This is a static historical backcast, not a forecast or a live track record: weights are "
           "fixed at today's values and applied to the full history window, buy-and-hold (no "
           "rebalancing, no transaction costs). It shows how the current portfolio WOULD HAVE performed, "
           "not evidence that it will. The walk-forward backtest on Portfolio Construction is the "
           "out-of-sample evidence.")

port_growth = (1 + port_returns).cumprod()
bench_growth = (1 + bench_returns).cumprod()
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=port_growth.index, y=port_growth.values, mode="lines", name=method_name,
                                 line=dict(color=port_color, width=2)))
fig_growth.add_trace(go.Scatter(x=bench_growth.index, y=bench_growth.values, mode="lines",
                                 name=f"{benchmark} (buy & hold)", line=c.BENCH_LINE))
fig_growth.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                          yaxis=dict(title="Growth of $1", tickprefix="$", tickformat=".2f"))
c.show_chart(fig_growth)

port_stats = c.performance_stats(port_returns, rf_obj, bench_returns)
bench_stats = c.performance_stats(bench_returns, rf_obj)
reg = c.capm_regression(port_returns, bench_returns, rf_obj)
port_stats["Beta"], port_stats["Alpha (ann.)"], port_stats["R²"] = reg["beta"], reg["alpha_annual"], reg["r2"]
summary_df = pd.DataFrame([port_stats, bench_stats], index=[method_name, f"{benchmark} (benchmark)"])
summary_df = summary_df[["CAGR", "Total Return", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown",
                          "Tracking Error", "Information Ratio", "Beta", "Alpha (ann.)", "R²"]]
c.show_table(c.dash_style(summary_df, {
    "CAGR": "{:.2%}", "Total Return": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.2f}",
    "Sortino": "{:.2f}", "Max Drawdown": "{:.2%}", "Tracking Error": "{:.2%}",
    "Information Ratio": "{:.2f}", "Beta": "{:.2f}", "Alpha (ann.)": "{:.2%}", "R²": "{:.2f}"}))
st.caption(
    "Sortino penalizes only returns below the risk-free rate (downside deviation). Tracking error is the "
    "annualized standard deviation of daily returns relative to the benchmark; the information ratio is "
    "annualized active return divided by tracking error. Alpha and beta come from a single-factor "
    "regression on the benchmark; R² is the share of the portfolio's excess-return variance that "
    "regression explains — low R² means alpha and beta describe only a small part of the story."
)
with st.expander("Regression diagnostics (alpha's standard error and significance)"):
    st.caption(
        "A large estimated alpha is not automatically meaningful: it also has an estimation error, and "
        "for a short or volatile history that error can be large relative to the estimate itself."
    )
    diag_df = pd.DataFrame([{
        "Alpha (ann.)": reg["alpha_annual"], "Alpha std. error (ann.)": reg["alpha_se_annual"],
        "t-statistic": reg["t_stat"], "p-value": reg["p_value"], "R²": reg["r2"], "Observations": reg["n"],
    }])
    c.show_table(c.dash_style(diag_df, {
        "Alpha (ann.)": "{:.2%}", "Alpha std. error (ann.)": "{:.2%}", "t-statistic": "{:.2f}",
        "p-value": "{:.3f}", "R²": "{:.2f}", "Observations": "{:.0f}"}))
    st.caption("A p-value below 0.05 is the conventional (if debated) threshold for calling alpha "
               "statistically distinguishable from zero at this sample size.")

# ---------------------------------------------------------------------
c.anchor("calendar")
st.subheader("3. Calendar-Year Returns")
st.caption("Daily returns compounded within each calendar year. Years marked * start partway through "
           "(limited by the History window); YTD is in progress and reflects only the days available so far.")

cal_df = pd.DataFrame({method_name: c.calendar_year_returns(port_returns),
                       benchmark: c.calendar_year_returns(bench_returns)})
data_start, data_end = port_returns.index.min(), port_returns.index.max()
cal_labels = []
for y in cal_df.index:
    lbl = str(y)
    if y == data_end.year and not (data_end.month == 12 and data_end.day >= 28):
        lbl += " YTD"
    elif y == data_start.year and not (data_start.month == 1 and data_start.day <= 3):
        lbl += "*"
    cal_labels.append(lbl)
fig_cal = go.Figure()
fig_cal.add_trace(go.Bar(x=cal_labels, y=cal_df[method_name], name=method_name,
                          marker_color=port_color))
fig_cal.add_trace(go.Bar(x=cal_labels, y=cal_df[benchmark], name=benchmark,
                          marker_color=c.BENCH_COLOR))
fig_cal.update_layout(barmode="group", height=380, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                       yaxis=dict(title="Calendar-year return", tickformat=c.PCT0))
c.show_chart(fig_cal)

# ---------------------------------------------------------------------
c.anchor("rolling-sharpe")
st.subheader("4. Rolling Sharpe Ratio")
st.caption("A single full-period Sharpe ratio hides how risk-adjusted performance varied over time. "
           "The rolling version shows strong and weak stretches instead of averaging them away.")

window = st.select_slider("Rolling window (trading days)", options=[63, 126, 252], value=126,
                            key="perf_window",
                            help="63 trading days is about one quarter, 126 about six months, 252 about "
                                 "one year. 126 is the default: a 63-day annualized Sharpe is quite "
                                 "noisy, since its mean is estimated from only about three months of data.")
rs_port = c.rolling_sharpe(port_returns, window, rf_obj)
rs_bench = c.rolling_sharpe(bench_returns, window, rf_obj)
fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(x=rs_port.index, y=rs_port.values, mode="lines", name=method_name,
                             line=dict(color=port_color)))
fig_rs.add_trace(go.Scatter(x=rs_bench.index, y=rs_bench.values, mode="lines",
                             name=f"{benchmark} (benchmark)", line=c.BENCH_LINE))
fig_rs.add_hline(y=0, line_color=c.BENCH_COLOR, line_dash="dot", line_width=1)
fig_rs.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                      yaxis=dict(title=f"Rolling {window}-day Sharpe (annualized)"))
c.show_chart(fig_rs)

# ---------------------------------------------------------------------
c.anchor("rolling-alpha-beta")
st.subheader("5. Rolling Alpha & Beta")
st.caption(
    "Rolling single-factor regression of portfolio excess return on benchmark excess return. Beta is "
    "market sensitivity; annualized alpha is the return left over after accounting for that exposure. "
    f"Both use the same {window}-day window as above and are backward-looking descriptions, not forecasts."
)
rab = c.rolling_alpha_beta(port_returns, bench_returns, window, rf_obj)

fig_beta = go.Figure()
fig_beta.add_trace(go.Scatter(x=rab.index, y=rab["Beta"], mode="lines", line=dict(color=c.NEUTRAL_BLUE)))
fig_beta.add_hline(y=1, line_color=c.BENCH_COLOR, line_dash="dot", line_width=1,
                    annotation_text="Beta = 1", annotation_position="bottom right")
fig_beta.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                        yaxis=dict(title="Rolling beta"))
st.markdown(f"**Rolling {window}-day beta vs. {benchmark}**")
c.show_chart(fig_beta)

fig_alpha = go.Figure()
fig_alpha.add_trace(go.Scatter(x=rab.index, y=rab["Alpha (annualized)"], mode="lines",
                                line=dict(color="#10B981")))
fig_alpha.add_hline(y=0, line_color=c.BENCH_COLOR, line_dash="dot", line_width=1)
fig_alpha.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                         yaxis=dict(title="Rolling annualized alpha", tickformat=c.PCT0))
st.markdown(f"**Rolling {window}-day annualized alpha vs. {benchmark}**")
c.show_chart(fig_alpha)

# ---------------------------------------------------------------------
c.anchor("drawdown")
st.subheader("6. Drawdown")
st.caption("Peak-to-trough decline in cumulative value over the full history window.")
_, dd_port = c.max_drawdown(port_growth)
_, dd_bench = c.max_drawdown(bench_growth)
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=dd_port.index, y=dd_port.values, mode="lines", name=method_name,
                             fill="tozeroy", line=dict(color=c.LOSS_RED), fillcolor="rgba(229,72,77,0.30)"))
fig_dd.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench.values, mode="lines",
                             name=f"{benchmark} (benchmark)", line=c.BENCH_LINE))
fig_dd.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), legend=c.LEGEND_TOP,
                      yaxis=dict(title="Drawdown", tickformat=c.PCT0))
c.show_chart(fig_dd)

st.caption("Historical, backward-looking figures with no transaction costs, taxes, or rebalancing.")
