"""Module 3 — Realized volatility and realized correlation.

Inputs: equity_daily_2021_jan2022 (15 months of daily total-return for 4 singles).
Outputs (all per-date scalars or per-(ticker,date) Series):
    - realized_vol_rolling   : annualised rolling σ per name
    - realized_corr_rolling  : average pairwise rolling ρ across basket
    - realized_basket_variance: vega-weighted basket variance scalar
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ANNUALISATION = 252.0   # trading days per year


def _log_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from total_return column.

    equity must have columns: ticker, date, total_return.
    Returns wide DataFrame indexed by date, columns = tickers.
    """
    equity = equity.sort_values(["ticker", "date"])
    # total_return is a simple return (not log); compute log
    wide = (
        equity
        .pivot(index="date", columns="ticker", values="total_return")
        .sort_index()
    )
    log_ret = np.log1p(wide)
    return log_ret


def realized_vol_rolling(
    equity: pd.DataFrame,
    windows: list[int] = (5, 10, 21),
) -> pd.DataFrame:
    """Per-(ticker, date) annualised realized volatility.

    Returns a DataFrame indexed by date, with MultiIndex columns
    (window, ticker).  E.g. df[21]["AAPL"] gives the 21d rolling σ for AAPL.
    """
    log_ret = _log_returns(equity)
    frames = {}
    for w in windows:
        # Rolling std then annualise
        rv = log_ret.rolling(w, min_periods=w).std() * np.sqrt(ANNUALISATION)
        frames[w] = rv
    return pd.concat(frames, axis=1)


def realized_corr_rolling(
    equity: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Average pairwise realized correlation across the basket, rolling.

    Returns a Series indexed by date.
    """
    log_ret = _log_returns(equity)
    tickers = log_ret.columns.tolist()
    n = len(tickers)
    if n < 2:
        return pd.Series(np.nan, index=log_ret.index, name="realized_corr")

    # Rolling pairwise correlations
    pairs = [(tickers[i], tickers[j]) for i in range(n) for j in range(i + 1, n)]
    pair_corrs = []
    for t1, t2 in pairs:
        r = (
            log_ret[[t1, t2]]
            .rolling(window, min_periods=window)
            .corr()
            .unstack()[(t1, t2)]
        )
        pair_corrs.append(r)

    corr_df = pd.concat(pair_corrs, axis=1)
    avg_corr = corr_df.mean(axis=1)
    avg_corr.name = "realized_corr"
    return avg_corr


def realized_basket_variance(
    equity: pd.DataFrame,
    weights: dict[str, float],
    window: int = 21,
) -> pd.Series:
    """Vega-weighted basket realized variance (scalar per date).

    σ_basket² = Σ_i w_i² σ_i² + 2 Σ_{i<j} w_i w_j ρ_{ij} σ_i σ_j

    `weights` maps ticker → weight (normalised externally or here).
    Returns a Series indexed by date (variance, not vol).
    """
    log_ret = _log_returns(equity)
    tickers = list(weights.keys())

    # Normalise weights
    total_w = sum(weights.values())
    w = {t: weights[t] / total_w for t in tickers}

    # Rolling variance per ticker
    rv_dict = {}
    for t in tickers:
        if t not in log_ret.columns:
            continue
        rv_dict[t] = (
            log_ret[t].rolling(window, min_periods=window).var() * ANNUALISATION
        )

    # Rolling pairwise cov
    result = pd.Series(np.nan, index=log_ret.index, name="basket_variance")
    # Diagonal
    diag = sum(w[t] ** 2 * rv_dict[t] for t in tickers if t in rv_dict)
    # Off-diagonal
    n = len(tickers)
    cross = pd.Series(0.0, index=log_ret.index)
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = tickers[i], tickers[j]
            if t1 not in rv_dict or t2 not in rv_dict:
                continue
            cov_ij = (
                log_ret[[t1, t2]]
                .rolling(window, min_periods=window)
                .cov()
                .unstack()[(t1, t2)]
                * ANNUALISATION
            )
            cross = cross.add(2 * w[t1] * w[t2] * cov_ij, fill_value=0)

    result = diag.add(cross, fill_value=np.nan)
    result.name = "basket_variance"
    return result


def build_realized_summary(
    equity: pd.DataFrame,
    weights: dict[str, float],
    windows: list[int] = (5, 10, 21),
    default_window: int = 21,
) -> pd.DataFrame:
    """Combine rolling vol, corr, and basket variance into one summary DataFrame.

    Returns a DataFrame indexed by date with columns:
        realized_corr, basket_variance, and <ticker>_vol_<window> for each window.
    """
    vol_df = realized_vol_rolling(equity, windows=windows)
    corr_s = realized_corr_rolling(equity, window=default_window)
    bv_s   = realized_basket_variance(equity, weights=weights, window=default_window)

    # Flatten vol multi-index → ticker_vol_W columns
    vol_flat = vol_df.copy()
    vol_flat.columns = [f"{t}_vol_{w}" for w, t in vol_flat.columns]

    summary = pd.concat([vol_flat, corr_s, bv_s], axis=1)
    return summary


__all__ = [
    "realized_vol_rolling",
    "realized_corr_rolling",
    "realized_basket_variance",
    "build_realized_summary",
    "ANNUALISATION",
]
