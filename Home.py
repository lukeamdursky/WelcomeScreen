"""
Equity Portfolio Toolkit — Home
--------------------------------
Entry point for a three-stage workflow: Portfolio Construction -> Performance -> Risk Report.

Run with:
    streamlit run Home.py
"""

import streamlit as st

import common as c

st.set_page_config(page_title="Equity Portfolio Toolkit", layout="wide")

st.title("Equity Portfolio Toolkit")
st.caption(
    "Construct equity portfolios with several methods, test them out-of-sample with costs, analyze "
    "their performance, and stress-test their risk. The portfolio you choose on page one carries "
    "through to the later pages."
)

st.markdown("## Contents")
st.page_link("pages/1_Portfolio_Construction.py", label="1. Portfolio Construction",
              help="Equal-Weight, Min-Variance, Risk Parity and two Max-Sharpe portfolios, plus a "
                   "walk-forward backtest with rebalancing, turnover and transaction costs")
st.page_link("pages/2_Performance.py", label="2. Performance",
              help="Sharpe, Sortino, tracking error, information ratio, rolling alpha and beta, drawdown")
st.page_link("pages/3_Risk_Report.py", label="3. Risk Report",
              help="Contribution to risk, VaR/CVaR (historical, parametric, Monte Carlo), Kupiec and "
                   "Christoffersen backtests")

st.markdown("---")
c.methodology_panel()

st.markdown("""
### How the pages fit together

**Settings are shared.** Tickers, benchmark, history window and risk-free rate are set once and carry
across all three pages.

**The portfolio you build carries forward.** On Portfolio Construction, choose a method and click
"Carry this portfolio forward". Performance and Risk Report then analyze that portfolio. Without one,
they fall back to an equal-weight portfolio of your tickers.

**In-sample versus out-of-sample.** The carried-forward weights are estimated on the full history, so the
Performance and Risk pages show in-sample results. The walk-forward backtest on Portfolio Construction,
which re-estimates using only past data at each rebalance and charges transaction costs, is the
out-of-sample evidence.

*Educational tool, not investment advice.*
""")
