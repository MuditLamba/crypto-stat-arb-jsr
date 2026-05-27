# crypto-stat-arb-jsr

Systematic crypto stat arb research platform focused on cross-sectional alpha generation, market-neutral portfolio construction, and intraday factor research across Binance.US crypto assets.

Built and evaluated short-horizon quantitative strategies on 193 USDT pairs (2019–2026), combining:

* volume-gated cross-sectional reversal,
* top-decile momentum,
* IVOL anomaly signals on BTC-residualized returns,
* and intraday seasonality effects.

The framework includes:

* data ingestion/cleaning pipelines,
* factor regression tooling,
* residual return modeling,
* rank-demean-normalize transforms,
* volatility scaling,
* turnover-aware weighting optimization,
* and backtesting infrastructure with transaction cost/slippage modeling.

The primary strategy achieved:

* 1.43 net backtested Sharpe ratio,
* BTC beta ≈ 0,
* ~19% annualized alpha,
* using 4h rebalancing with out-of-sample validation from 2024 onward.
