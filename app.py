import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
import ta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Giants Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #7c3aed;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Dashboard Controls")
    st.markdown("---")

    all_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    selected_stocks = st.multiselect(
        "Select Stocks",
        options=all_stocks,
        default=["AAPL", "MSFT", "GOOGL", "AMZN"],
        help="Choose 2–6 stocks to analyse",
    )
    if len(selected_stocks) < 2:
        st.error("Please select at least 2 stocks.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2000, max_value=2023, value=2004)
    with col2:
        end_year = st.number_input("End Year", min_value=2001, max_value=2025, value=2024)

    risk_free_rate = st.slider(
        "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.3, step=0.1
    ) / 100

    benchmark = st.selectbox("Benchmark", ["^GSPC (S&P 500)", "^IXIC (NASDAQ)", "^DJI (Dow Jones)"])
    benchmark_ticker = benchmark.split(" ")[0]

    st.markdown("---")
    st.caption("Data sourced from Yahoo Finance via yfinance.")

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def load_data(tickers, benchmark_ticker, start_year, end_year):
    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-31"
    all_tickers = tickers + [benchmark_ticker]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw.columns = [c if c != benchmark_ticker else "BENCHMARK" for c in raw.columns]
    raw = raw.dropna(how="all")
    returns = raw.pct_change().dropna()
    return raw, returns

prices, returns = load_data(selected_stocks, benchmark_ticker, start_year, end_year)
stock_returns = returns[selected_stocks]
bench_returns = returns["BENCHMARK"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Tech Giants Stock Market Analysis")
st.markdown(
    f"Analysing **{', '.join(selected_stocks)}** from **{start_year}** to **{end_year}** "
    f"against **{benchmark}**."
)
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
cum_returns = (1 + stock_returns).prod() - 1
best_stock  = cum_returns.idxmax()
annualised  = (1 + cum_returns) ** (252 / len(stock_returns)) - 1

kpi_cols = st.columns(len(selected_stocks))
for i, ticker in enumerate(selected_stocks):
    cr  = cum_returns[ticker] * 100
    ann = annualised[ticker] * 100
    with kpi_cols[i]:
        st.metric(
            label=ticker,
            value=f"{cr:,.1f}%",
            delta=f"CAGR {ann:.1f}%",
        )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Stock Performance",
    "📉 Returns Analysis",
    "⚖️ Risk Metrics",
    "🏆 Portfolio Optimizer",
    "🕯️ Technical Analysis",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Stock Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Normalized Price Performance")
    st.caption("All prices rebased to 100 at the start of the selected period.")

    normalized = prices[selected_stocks].div(prices[selected_stocks].iloc[0]) * 100

    fig_price = go.Figure()
    for ticker in selected_stocks:
        fig_price.add_trace(go.Scatter(
            x=normalized.index, y=normalized[ticker],
            name=ticker, mode="lines", line=dict(width=2),
        ))
    fig_price.update_layout(
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date", yaxis_title="Indexed Price (Base = 100)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")

    def build_stats(returns_df):
        rows = []
        for col in returns_df.columns:
            r = returns_df[col].dropna()
            rows.append({
                "Ticker": col,
                "Mean Daily Return": f"{r.mean()*100:.4f}%",
                "Geo Mean Daily Return": f"{((1+r).prod()**(1/len(r))-1)*100:.4f}%",
                "Cumulative Return": f"{((1+r).prod()-1)*100:.2f}%",
                "Median Daily Return": f"{r.median()*100:.4f}%",
                "Std Dev (Daily)": f"{r.std()*100:.4f}%",
                "Annualised Volatility": f"{r.std()*np.sqrt(252)*100:.2f}%",
                "Skewness": f"{stats.skew(r):.4f}",
                "Kurtosis": f"{stats.kurtosis(r):.4f}",
                "Coeff of Variation": f"{(r.std()/r.mean()):.2f}",
            })
        return pd.DataFrame(rows).set_index("Ticker")

    stats_df = build_stats(stock_returns)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr = stock_returns.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "A correlation close to **1** means the stocks move together; "
        "close to **-1** means they move inversely — ideal for diversification."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Returns Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Cumulative Returns Over Time")
    cum_ret_ts = (1 + stock_returns).cumprod() - 1

    fig_cum = go.Figure()
    for ticker in selected_stocks:
        fig_cum.add_trace(go.Scatter(
            x=cum_ret_ts.index, y=cum_ret_ts[ticker] * 100,
            name=ticker, mode="lines", line=dict(width=2),
        ))
    fig_cum.update_layout(
        hovermode="x unified", height=420,
        xaxis_title="Date", yaxis_title="Cumulative Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Daily Returns Distribution")
        selected_for_dist = st.selectbox("Stock", selected_stocks, key="dist_stock")
        r_series = stock_returns[selected_for_dist]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=r_series * 100, nbinsx=80,
            name=selected_for_dist,
            marker_color="#7c3aed", opacity=0.75,
        ))
        fig_dist.update_layout(
            height=380, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.subheader("Rolling 30-Day Volatility")
        fig_vol = go.Figure()
        for ticker in selected_stocks:
            rolling_vol = stock_returns[ticker].rolling(30).std() * np.sqrt(252) * 100
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol,
                name=ticker, mode="lines", line=dict(width=1.5),
            ))
        fig_vol.update_layout(
            height=380, xaxis_title="Date", yaxis_title="Annualised Volatility (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("Annual Returns Heatmap")
    annual = stock_returns.copy()
    annual.index = pd.to_datetime(annual.index)
    annual_ret = annual.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    annual_ret.index = annual_ret.index.year

    fig_heat = px.imshow(
        annual_ret.T,
        text_auto=".1f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(x="Year", y="Stock", color="Return (%)"),
    )
    fig_heat.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Green = positive annual return · Red = negative annual return.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Beta & Alpha vs. Benchmark")
    st.caption(
        "Beta measures sensitivity to market moves. "
        "Alpha is excess return after accounting for market risk."
    )

    def compute_beta_alpha(stock_ret, bench_ret, rf):
        excess_stock = stock_ret - rf / 252
        excess_bench = bench_ret - rf / 252
        X = excess_bench.values.reshape(-1, 1)
        y = excess_stock.values
        model = LinearRegression().fit(X, y)
        beta  = model.coef_[0]
        alpha = model.intercept_ * 252  # annualise
        return round(beta, 4), round(alpha, 4)

    risk_rows = []
    for ticker in selected_stocks:
        aligned = pd.concat([stock_returns[ticker], bench_returns], axis=1).dropna()
        aligned.columns = ["stock", "bench"]
        beta, alpha = compute_beta_alpha(aligned["stock"], aligned["bench"], risk_free_rate)

        r      = aligned["stock"]
        vol    = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252 - risk_free_rate) / vol
        downside = r[r < 0].std() * np.sqrt(252)
        sortino  = (r.mean() * 252 - risk_free_rate) / downside if downside > 0 else np.nan
        cum      = (1 + r).cumprod()
        rolling_max = cum.cummax()
        max_dd   = ((cum - rolling_max) / rolling_max).min()

        risk_rows.append({
            "Ticker": ticker,
            "Beta": beta,
            "Alpha (Ann.)": f"{alpha*100:.2f}%",
            "Annualised Vol": f"{vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
        })

    risk_df = pd.DataFrame(risk_rows).set_index("Ticker")
    st.dataframe(risk_df, use_container_width=True)

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("Sharpe vs. Annualised Return")
        scatter_data = []
        for ticker in selected_stocks:
            r   = stock_returns[ticker].dropna()
            vol = r.std() * np.sqrt(252)
            ann_ret = r.mean() * 252
            sharpe  = (ann_ret - risk_free_rate) / vol
            scatter_data.append({"Ticker": ticker, "Ann. Return (%)": ann_ret*100, "Sharpe": round(sharpe,3), "Volatility (%)": vol*100})
        scat_df = pd.DataFrame(scatter_data)
        fig_scat = px.scatter(
            scat_df, x="Volatility (%)", y="Ann. Return (%)",
            text="Ticker", size="Sharpe", color="Sharpe",
            color_continuous_scale="Viridis",
        )
        fig_scat.update_traces(textposition="top center")
        fig_scat.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_r2:
        st.subheader("Rolling 6-Month Sharpe Ratio")
        fig_sharpe = go.Figure()
        window = 126
        for ticker in selected_stocks:
            r = stock_returns[ticker]
            roll_sharpe = (
                r.rolling(window).mean() * 252 - risk_free_rate
            ) / (r.rolling(window).std() * np.sqrt(252))
            fig_sharpe.add_trace(go.Scatter(
                x=roll_sharpe.index, y=roll_sharpe,
                name=ticker, mode="lines", line=dict(width=1.5),
            ))
        fig_sharpe.add_hline(y=1, line_dash="dash", line_color="white", opacity=0.4)
        fig_sharpe.update_layout(
            height=380, hovermode="x unified",
            xaxis_title="Date", yaxis_title="Sharpe Ratio",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    st.markdown("---")
    st.subheader("Drawdown Chart")
    fig_dd = go.Figure()
    for ticker in selected_stocks:
        cum = (1 + stock_returns[ticker]).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax() * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd, name=ticker,
            mode="lines", line=dict(width=1.5), fill="tozeroy", opacity=0.4,
        ))
    fig_dd.update_layout(
        hovermode="x unified", height=380,
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    st.caption("Drawdown shows the percentage decline from the rolling peak — a key measure of downside risk.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Portfolio Optimizer (Markowitz)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Markowitz Mean-Variance Portfolio Optimization")
    st.markdown(
        "Using the **Efficient Frontier**, we find the portfolio weights that "
        "**maximise the Sharpe ratio** given the historical return & covariance structure."
    )

    price_data = prices[selected_stocks].dropna()

    try:
        mu  = expected_returns.mean_historical_return(price_data)
        cov = risk_models.sample_cov(price_data)
        ef  = EfficientFrontier(mu, cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights_raw = ef.clean_weights()
        weights = {k: v for k, v in weights_raw.items() if v > 0.001}

        perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        opt_ret, opt_vol, opt_sharpe = perf

        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        col_w1.metric("Expected Annual Return", f"{opt_ret*100:.2f}%")
        col_w2.metric("Annual Volatility",      f"{opt_vol*100:.2f}%")
        col_w3.metric("Sharpe Ratio",           f"{opt_sharpe:.3f}")
        col_w4.metric("Risk-Free Rate",         f"{risk_free_rate*100:.1f}%")

        st.markdown("---")
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("Optimal Portfolio Weights")
            fig_pie = go.Figure(go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.45,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.subheader("All Weights (incl. zero)")
            bar_data = pd.DataFrame({
                "Ticker": list(weights_raw.keys()),
                "Weight (%)": [v * 100 for v in weights_raw.values()],
            })
            fig_bar = px.bar(
                bar_data, x="Ticker", y="Weight (%)",
                color="Weight (%)", color_continuous_scale="Viridis",
            )
            fig_bar.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("Efficient Frontier")
        st.caption("Each dot represents a random portfolio; the gold star is the optimal (max Sharpe) portfolio.")

        n_portfolios = 3000
        results = np.zeros((3, n_portfolios))
        np.random.seed(42)
        n = len(selected_stocks)
        for i in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            p_ret = np.dot(w, mu.values)
            p_vol = np.sqrt(w @ cov.values @ w)
            results[0, i] = p_vol * 100
            results[1, i] = p_ret * 100
            results[2, i] = (p_ret - risk_free_rate) / p_vol

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=results[0], y=results[1],
            mode="markers",
            marker=dict(color=results[2], colorscale="Viridis", size=4, opacity=0.6,
                        colorbar=dict(title="Sharpe")),
            name="Random Portfolios",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol * 100], y=[opt_ret * 100],
            mode="markers", marker=dict(symbol="star", size=18, color="gold"),
            name="Max Sharpe Portfolio",
        ))
        fig_ef.update_layout(
            height=460, xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        st.markdown("---")
        st.subheader("Optimized vs. Equal-Weight Portfolio")

        eq_weights = np.array([1 / len(selected_stocks)] * len(selected_stocks))
        eq_returns = (stock_returns[selected_stocks] @ eq_weights)
        opt_w_arr  = np.array([weights_raw.get(t, 0) for t in selected_stocks])
        opt_returns_ts = (stock_returns[selected_stocks] @ opt_w_arr)

        cum_eq  = (1 + eq_returns).cumprod() - 1
        cum_opt = (1 + opt_returns_ts).cumprod() - 1

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=cum_eq.index, y=cum_eq * 100,
            name="Equal-Weight", line=dict(width=2, dash="dash"),
        ))
        fig_comp.add_trace(go.Scatter(
            x=cum_opt.index, y=cum_opt * 100,
            name="Optimized (Max Sharpe)", line=dict(width=2.5),
        ))
        fig_comp.update_layout(
            hovermode="x unified", height=400,
            xaxis_title="Date", yaxis_title="Cumulative Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        def portfolio_metrics(ret_series, rf):
            ann_ret = ret_series.mean() * 252
            ann_vol = ret_series.std() * np.sqrt(252)
            sharpe  = (ann_ret - rf) / ann_vol
            down    = ret_series[ret_series < 0].std() * np.sqrt(252)
            sortino = (ann_ret - rf) / down if down > 0 else np.nan
            cum     = (1 + ret_series).cumprod()
            max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
            cagr    = (1 + cum.iloc[-1] - 1) ** (252 / len(ret_series)) - 1
            return {
                "CAGR": f"{cagr*100:.2f}%",
                "Cumulative Return": f"{(cum.iloc[-1]-1)*100:.2f}%",
                "Annual Volatility": f"{ann_vol*100:.2f}%",
                "Sharpe Ratio": f"{sharpe:.3f}",
                "Sortino Ratio": f"{sortino:.3f}",
                "Max Drawdown": f"{max_dd*100:.2f}%",
            }

        m_eq  = portfolio_metrics(eq_returns, risk_free_rate)
        m_opt = portfolio_metrics(opt_returns_ts, risk_free_rate)
        comp_df = pd.DataFrame({"Equal-Weight": m_eq, "Optimized": m_opt})
        st.dataframe(comp_df, use_container_width=True)

    except Exception as e:
        st.error(f"Optimisation failed: {e}. Try a wider date range or different stocks.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Technical Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Technical Analysis")

    tech_ticker = st.selectbox("Select Stock for Technical Analysis", selected_stocks, key="tech_stock")
    ma_short = st.slider("Short MA (days)", 10, 50, 20, key="ma_short")
    ma_long  = st.slider("Long MA (days)", 50, 200, 50, key="ma_long")

    @st.cache_data(ttl=3600)
    def load_ohlcv(ticker, start_year, end_year):
        return yf.download(ticker, start=f"{start_year}-01-01", end=f"{end_year}-12-31", auto_adjust=True)

    ohlcv = load_ohlcv(tech_ticker, start_year, end_year)
    if ohlcv.empty:
        st.warning("No OHLCV data available for this selection.")
    else:
        close = ohlcv["Close"].squeeze()
        volume = ohlcv["Volume"].squeeze()

        ma_s = close.rolling(ma_short).mean()
        ma_l = close.rolling(ma_long).mean()
        rsi  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd_ind = ta.trend.MACD(close=close)
        macd_line = macd_ind.macd()
        signal_line = macd_ind.macd_signal()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

        fig_tech = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.2, 0.2, 0.15],
            vertical_spacing=0.03,
            subplot_titles=[
                f"{tech_ticker} Price + Bollinger Bands + MAs",
                "RSI (14)",
                "MACD",
                "Volume",
            ],
        )

        # Candlestick
        fig_tech.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"].squeeze(), high=ohlcv["High"].squeeze(),
            low=ohlcv["Low"].squeeze(), close=close,
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            showlegend=False,
        ), row=1, col=1)

        fig_tech.add_trace(go.Scatter(x=close.index, y=ma_s, name=f"MA{ma_short}",
                                      line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=close.index, y=ma_l, name=f"MA{ma_long}",
                                      line=dict(color="#60a5fa", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=close.index, y=bb.bollinger_hband(),
                                      name="BB Upper", line=dict(color="gray", dash="dot", width=1),
                                      showlegend=True), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=close.index, y=bb.bollinger_lband(),
                                      name="BB Lower", line=dict(color="gray", dash="dot", width=1),
                                      fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
                                      showlegend=True), row=1, col=1)

        # RSI
        fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI",
                                      line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=2, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # MACD
        fig_tech.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD",
                                      line=dict(color="#34d399", width=1.5)), row=3, col=1)
        fig_tech.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal",
                                      line=dict(color="#f87171", width=1.5)), row=3, col=1)
        macd_hist = macd_ind.macd_diff()
        fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Histogram",
                                  marker_color="#7c3aed", opacity=0.6), row=3, col=1)

        # Volume
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(ohlcv["Close"].squeeze(), ohlcv["Open"].squeeze())]
        fig_tech.add_trace(go.Bar(x=ohlcv.index, y=volume, name="Volume",
                                  marker_color=colors, showlegend=False), row=4, col=1)

        fig_tech.update_layout(
            height=900, hovermode="x unified",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_tech.update_yaxes(title_text="RSI",       row=2, col=1)
        fig_tech.update_yaxes(title_text="MACD",      row=3, col=1)
        fig_tech.update_yaxes(title_text="Volume",    row=4, col=1)

        st.plotly_chart(fig_tech, use_container_width=True)

        st.markdown("---")
        st.markdown("""
**Reading the chart:**
- **Candlesticks** — green = price closed higher; red = closed lower.
- **Bollinger Bands** — when price touches the upper band, it may be overbought; lower band, oversold.
- **RSI > 70** = overbought · **RSI < 30** = oversold.
- **MACD** — when the MACD line crosses above the signal line, it is a bullish signal.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · Data from Yahoo Finance via yfinance · "
    "Portfolio optimisation via PyPortfolioOpt · Technical indicators via TA-Lib (ta)."
)
