#Shared utilities for the crypto stat arb project.

from __future__ import annotations

import numpy as np
import pandas as pd


BARS_PER_YEAR_DAILY = 365
BARS_PER_YEAR_4H = 6 * 365

DEFAULT_COST_BPS_MARKET = 20
DEFAULT_COST_BPS_LIMIT = 7


def compute_returns(px):
    return px / px.shift(1) - 1


def rank_demean_normalize(signal, universe=None):
    """Cross-sectional signal to dollar-neutral weights. Each row sums to 0
    and abs-sums to 1."""
    sig = signal.copy()
    if universe is not None:
        sig = sig.where(universe)
    ranked = sig.rank(axis=1)
    demeaned = ranked.subtract(ranked.mean(axis=1), axis=0)
    denom = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return demeaned.divide(denom, axis=0).fillna(0)


def compute_turnover(weights):
    """Sum of delta weights per period."""
    return (weights.fillna(0) - weights.shift().fillna(0)).abs().sum(axis=1)


def apply_costs(gross_returns, turnover, cost_bps=DEFAULT_COST_BPS_MARKET):
    return gross_returns.subtract(turnover * cost_bps * 1e-4, fill_value=0)


def drawdown(px):
    """Drawdown series."""
    return px / px.expanding(min_periods=1).max() - 1


def max_drawdown(returns):
    """Max drawdown of a return series, as a negative number."""
    equity = (1 + returns.fillna(0)).cumprod()
    return drawdown(equity).min()


def get_stats(returns, bars_per_year=BARS_PER_YEAR_4H, name=None):
    """Annualized return, vol, Sharpe, max DD, n_obs."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return pd.Series(
            {'ret_ann': np.nan, 'vol_ann': np.nan, 'sharpe': np.nan,
             'max_dd': np.nan, 'n_obs': len(r)},
            name=name,
        )
    return pd.Series({
        'ret_ann': r.mean() * bars_per_year,
        'vol_ann': r.std() * np.sqrt(bars_per_year),
        'sharpe': r.mean() / r.std() * np.sqrt(bars_per_year),
        'max_dd': max_drawdown(r),
        'n_obs': len(r),
    }, name=name)


def rolling_beta(returns, benchmark, window=60):
    """Rolling beta of each column to benchmark. Vectorized via corr * vol_asset / vol_bench."""
    corr = returns.rolling(window).corr(benchmark)
    vol_asset = returns.rolling(window).std()
    vol_bench = benchmark.rolling(window).std()
    return (corr * vol_asset).divide(vol_bench, axis=0)


def residual_returns(returns, benchmark, window=60):
    """Returns minus rolling beta times benchmark. Strips market factor."""
    beta = rolling_beta(returns, benchmark, window=window)
    return returns.subtract(beta.multiply(benchmark, axis=0), fill_value=0)


def residual_returns_no_fill(returns, benchmark, window=60):
    """Like residual_returns but adds NaNs through the warmup window."""
    beta = rolling_beta(returns, benchmark, window=window)
    return returns.subtract(beta.multiply(benchmark, axis=0))


def factor_regression(strat_returns, factor_returns, bars_per_year=BARS_PER_YEAR_4H):
    """Full-sample regression. Returns beta, alpha_ann, corr, info_ratio, n_obs."""
    df = pd.concat([strat_returns, factor_returns], axis=1).dropna()
    nan_row = pd.Series({'beta': np.nan, 'alpha_ann': np.nan,
                         'corr': np.nan, 'info_ratio': np.nan, 'n_obs': len(df)})
    if len(df) < 10:
        return nan_row
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    if x.var() == 0:
        return nan_row
    beta = np.cov(y, x, ddof=1)[0, 1] / x.var()
    alpha_per_bar = y.mean() - beta * x.mean()
    resid_vol = (y - beta * x).std()
    info_ratio = (alpha_per_bar * np.sqrt(bars_per_year)) / resid_vol if resid_vol > 0 else np.nan
    return pd.Series({
        'beta': beta,
        'alpha_ann': alpha_per_bar * bars_per_year,
        'corr': y.corr(x),
        'info_ratio': info_ratio,
        'n_obs': len(df),
    })


def optimal_weights(sigma, mu):
    """Mean-variance: inv(sigma) @ mu, abs-normalized."""
    w = np.linalg.inv(np.asarray(sigma)) @ np.asarray(mu)
    w = w / np.abs(w).sum()
    return pd.Series(w, index=mu.index) if isinstance(mu, pd.Series) else w


def eqvol_weights(sigma):
    """Inverse-vol, abs-normalized."""
    sigma_arr = np.asarray(sigma)
    w = 1 / np.sqrt(np.diag(sigma_arr))
    w = w / np.abs(w).sum()
    return pd.Series(w, index=sigma.index) if isinstance(sigma, pd.DataFrame) else w


def sr_weights(sigma, mu):
    """mu / diag(sigma), abs-normalized."""
    w = np.asarray(mu) / np.diag(np.asarray(sigma))
    w = w / np.abs(w).sum()
    return pd.Series(w, index=mu.index) if isinstance(mu, pd.Series) else w


def gmv_weights(sigma):
    """Global minimum variance: inv(sigma) @ 1, abs-normalized."""
    sigma_arr = np.asarray(sigma)
    w = np.linalg.inv(sigma_arr) @ np.ones(sigma_arr.shape[0])
    w = w / np.abs(w).sum()
    return pd.Series(w, index=sigma.index) if isinstance(sigma, pd.DataFrame) else w
