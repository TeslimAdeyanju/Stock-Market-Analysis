# Stock Market Analysis of Tech Giants Using Python

A quantitative finance project analysing 20 years of stock market data for four major technology companies — Apple (AAPL), Microsoft (MSFT), Google (GOOGL), and Amazon (AMZN) — using Python. The project covers statistical performance analysis, risk-adjusted return evaluation, Markowitz portfolio optimisation, and technical analysis, delivered through both a research-grade Jupyter notebook and an interactive Streamlit dashboard.

---

## Overview

The stock market presents three persistent challenges for investors: identifying consistently profitable assets, managing portfolio-level risk, and allocating capital efficiently across multiple positions. This project addresses all three using rigorous quantitative methods grounded in modern portfolio theory.

The four companies were selected for their market dominance, sector leadership, and the richness of their 20-year historical data. Together they represent the defining technology companies of the 21st century, making them an ideal subject for long-term equity analysis.

The analysis spans January 2004 to December 2024, deliberately chosen to capture major market cycles: the 2008 Global Financial Crisis, the 2020 COVID-19 crash and recovery, and the 2022 Federal Reserve rate-hike driven tech sell-off.

---

## Project Objectives

1. Analyse 20 years of historical stock data to uncover return patterns, trends, and statistical properties.
2. Evaluate individual stock performance using a full suite of risk and return metrics.
3. Benchmark each stock against the S&P 500 using the CAPM framework (Beta and Alpha).
4. Measure risk-adjusted performance via the Sharpe Ratio, Sortino Ratio, and maximum drawdown.
5. Construct and optimise a multi-asset portfolio using the Markowitz Mean-Variance model.
6. Interpret price action and momentum using technical indicators as a complement to statistical analysis.

---

## Key Results

### Individual Stock Performance (2004 - 2024)

| Stock | Cumulative Return | CAGR | Daily Std Dev | Sharpe Ratio | Coeff of Variation |
| ----- | ---------------- | ---- | ------------- | ------------ | ------------------ |
| AAPL | 51,204% | ~36% | 2.04% | ~1.05 | 14.3 (best) |
| AMZN | 10,663% | ~30% | 2.39% | ~0.75 | 19.9 |
| GOOGL | 6,645% | ~25% | 1.93% | ~0.77 | 19.1 |
| MSFT | 2,415% | ~20% | 1.70% | ~0.65 | 21.9 |

Apple (AAPL) leads on virtually every metric — highest cumulative return, highest Sharpe Ratio, and lowest Coefficient of Variation (risk per unit of return). A $1,000 investment in Apple in January 2004 would have grown to approximately $513,000 by December 2024.

### Portfolio Optimisation Results

| Metric | Equal-Weight Baseline | Markowitz Optimised | S&P 500 Benchmark |
| ------ | -------------------- | ------------------- | ----------------- |
| Cumulative Return | 1,735.27% | 2,468.60% | 327.42% |
| CAGR | 18.23% | 20.54% | 8.12% |
| Annual Volatility | 23.8% | 24.2% | 18.1% |
| Sharpe Ratio | 1.20 | 1.24 | 0.79 |
| Sortino Ratio | 1.76 | 1.83 | 1.14 |
| Maximum Drawdown | -44.7% | -32.86% | -56.8% |
| Best Year | +76.2% | +76.75% | +32.4% |

Optimal portfolio weights: AAPL 51.31% -- AMZN 37.57% -- MSFT 11.12% -- GOOGL 0.00%

The Markowitz-optimised portfolio delivered 733 additional percentage points of cumulative return and a 12% reduction in maximum drawdown relative to the equal-weight baseline, while improving both the Sharpe and Sortino Ratios.

---

## Methodology

### 1. Data Collection and Preprocessing

Historical adjusted closing prices were retrieved from Yahoo Finance using the `yfinance` library. Adjusted close prices account for stock splits and dividend reinvestment, providing a total-return perspective. Daily percentage returns were computed as:

```text
r_t = (P_t - P_{t-1}) / P_{t-1}
```

This transforms the non-stationary price series into a stationary return series suitable for statistical analysis.

### 2. Statistical Performance Analysis

Nine metrics were computed for each stock to build a comprehensive return distribution profile:

| Metric | Purpose |
| ------ | ------- |
| Mean Daily Return | Average daily gain or loss |
| Geometric Mean | Compound daily growth rate, accounting for volatility drag |
| Harmonic Mean | Rate-of-return average, penalising extreme values |
| Cumulative Return | Total wealth multiplier over the full period |
| Median Daily Return | Robust central tendency, unaffected by outliers |
| Standard Deviation | Daily volatility — spread of returns around the mean |
| Skewness | Asymmetry of the return distribution |
| Kurtosis | Fat-tailedness — probability of extreme return events |
| Coefficient of Variation | Risk per unit of return, enabling cross-stock comparison |

All four stocks exhibit leptokurtic return distributions (kurtosis > 3), meaning extreme returns occur more frequently than normal distribution models predict. This is a critical finding for risk management, as standard Value-at-Risk models based on normality underestimate tail exposure.

### 3. Beta and Alpha Analysis

Beta and Alpha were estimated using the Capital Asset Pricing Model (CAPM) Security Characteristic Line, implemented via linear regression with `scikit-learn`:

```text
r_i - r_f = alpha_i + beta_i * (r_m - r_f) + epsilon_i
```

Where r_i is the stock return, r_f is the risk-free rate, and r_m is the S&P 500 return. Beta measures market sensitivity; Alpha measures excess return above the CAPM prediction.

Results (2004-2024):

| Stock | Beta | Daily Alpha | Annualised Alpha |
| ----- | ---- | ----------- | ---------------- |
| AAPL | 0.0035 | 0.00143 | ~36% |
| AMZN | 0.0218 | 0.00120 | ~30% |
| GOOGL | -0.0191 | 0.00101 | ~25% |
| MSFT | -0.0129 | 0.00078 | ~20% |

All four stocks generated significant positive Alpha, indicating they consistently outperformed CAPM expectations over the 20-year period.

### 4. Sharpe and Sortino Ratio Analysis

The Sharpe Ratio measures return per unit of total risk:

```text
S = (mean_portfolio_return - risk_free_rate) / portfolio_std_dev
```

The Sortino Ratio uses only downside deviation in the denominator, making it more appropriate for asymmetric return distributions. Both metrics were computed using `quantstats`, which also produced full tearsheet reports comparing each stock and portfolio against the S&P 500 benchmark.

### 5. Portfolio Construction

An equal-weight baseline portfolio was constructed first, allocating 25% to each stock. Portfolio daily returns are the weighted average of individual returns:

```text
r_portfolio = 0.25 * r_AAPL + 0.25 * r_MSFT + 0.25 * r_GOOGL + 0.25 * r_AMZN
```

This naïve baseline delivered 3,429.9% cumulative return vs 296.86% for the S&P 500 over the same period — demonstrating the sector's extraordinary long-term performance.

### 6. Markowitz Mean-Variance Optimisation

The Markowitz (1952) framework finds portfolio weights that maximise the Sharpe Ratio subject to the constraint that weights sum to one and are non-negative (long-only):

```text
maximise: (w^T * mu - r_f) / sqrt(w^T * Sigma * w)
subject to: sum(w_i) = 1, w_i >= 0 for all i
```

Where mu is the vector of expected returns and Sigma is the covariance matrix of returns. Expected returns were estimated using mean historical returns (`pypfopt.expected_returns.mean_historical_return`) and the covariance matrix using the sample covariance (`pypfopt.risk_models.sample_cov`). The optimisation was solved using the `EfficientFrontier` class from `PyPortfolioOpt`, which wraps the `cvxpy` convex optimisation solver.

The Efficient Frontier was visualised by simulating 3,000 random portfolios and plotting their risk-return coordinates, with the maximum Sharpe portfolio marked explicitly.

### 7. Technical Analysis

Technical indicators were applied to Apple stock as a case study, using the `ta` (Technical Analysis) library:

| Indicator | Type | Signal |
| --------- | ---- | ------ |
| SMA-20 and SMA-50 | Trend | Golden Cross (SMA-20 crosses above SMA-50) = bullish signal |
| Bollinger Bands (20-day, 2 std dev) | Volatility | Price at upper band = overbought; lower band = oversold |
| RSI (14-day) | Momentum | Above 70 = overbought; below 30 = oversold |
| MACD (12/26/9) | Momentum | MACD crossing above signal line = bullish |

Charts were built using `plotly` with full interactivity — candlestick with volume, overlaid moving averages, and separate RSI and MACD panels.

---

## Interactive Dashboard

An interactive Streamlit dashboard (`app.py`) extends the notebook analysis into a live, exploratory tool. The dashboard allows users to select any combination of stocks, adjust the date range and risk-free rate, and view all analyses update dynamically.

Dashboard tabs:

| Tab | Content |
| --- | ------- |
| Stock Performance | Normalised price chart rebased to 100, summary statistics table, correlation heatmap |
| Returns Analysis | Cumulative returns time series, daily return distribution histogram, rolling 30-day volatility, annual returns heatmap |
| Risk Metrics | Beta and Alpha vs benchmark, Sharpe vs annualised return scatter, rolling 6-month Sharpe, drawdown chart |
| Portfolio Optimiser | Efficient Frontier with 3,000 simulated portfolios, max-Sharpe weights (pie and bar charts), optimised vs equal-weight cumulative return comparison, performance metrics table |
| Technical Analysis | Candlestick chart with Bollinger Bands, RSI panel, MACD panel, volume — configurable moving average windows |

To run the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard can be deployed to Streamlit Cloud by connecting the GitHub repository, producing a shareable public URL with no server configuration required.

---

## Project Structure

```text
portfolio-03-stock-market-analysis/
├── app.py                                                          # Streamlit interactive dashboard
├── requirements.txt                                                # Python dependencies
├── Portfolio-Stock-Market-Analysis-of-Tech-Giants-Using-Python.ipynb  # Research notebook
└── README.md
```

---

## Technology Stack

| Category | Libraries and Tools |
| -------- | ------------------- |
| Data retrieval | `yfinance`, `pandas_datareader` |
| Data manipulation | `pandas`, `numpy` |
| Statistical analysis | `scipy.stats`, `skimpy` |
| Visualisation | `plotly`, `matplotlib`, `seaborn`, `plotnine` |
| Portfolio analysis | `quantstats`, `PyPortfolioOpt` |
| Technical analysis | `ta` |
| Machine learning | `scikit-learn`, `optuna` |
| Dashboard | `streamlit` |

Python version: 3.10+

---

## Limitations and Caveats

**Survivorship bias.** The four companies analysed are among the most successful in history. A randomly selected portfolio of 2004 technology stocks would have included many companies that no longer exist. The results should be interpreted as retrospective analysis of known winners, not as a forward-looking performance guarantee.

**In-sample optimisation.** The Markowitz optimisation was performed using the same data used for evaluation. True out-of-sample performance should be validated through rolling-window or walk-forward backtesting, which would show lower but more realistic improvements over equal-weight.

**Parameter estimation error.** Markowitz optimisation is sensitive to errors in the expected return vector and covariance matrix. Small estimation errors can produce large changes in optimal weights. More robust approaches include Ledoit-Wolf shrinkage estimators, the Black-Litterman model, or minimum weight constraints to prevent extreme concentration.

**Normal distribution assumption.** Sharpe Ratio calculations assume normally distributed returns. All four stocks exhibit fat-tailed, leptokurtic distributions, which means the Sharpe Ratio underestimates true tail risk. Complementary measures such as Conditional Value-at-Risk (CVaR) should be used for a complete risk picture.

**No transaction costs or taxes.** All return calculations assume costless rebalancing. Real-world implementation would incur brokerage fees, bid-ask spreads, and capital gains taxes that would reduce net returns.

---

## Proposed Extensions

1. **Walk-forward optimisation** — Re-optimise weights annually on a rolling basis and evaluate on the subsequent out-of-sample period to produce realistic performance estimates.
2. **Expanded asset universe** — Include bonds, commodities, REITs, and international equities to introduce genuine diversification across asset classes rather than within a single sector.
3. **Black-Litterman model** — Blend market-equilibrium implied returns with analyst or model-derived views to produce more robust expected return estimates for optimisation.
4. **Machine learning return forecasting** — Train LSTM or transformer models on price and macro data to generate return forecasts that feed into the optimiser as forward-looking expected returns.
5. **Factor attribution** — Decompose returns into Fama-French factors (market, size, value, momentum, profitability) to understand what is driving Alpha and whether it is persistent.
6. **Monte Carlo simulation** — Project portfolio value distributions under stochastic return assumptions to quantify the range of outcomes over different investment horizons.

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business*, 39(1), 119-138.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915-1953.
