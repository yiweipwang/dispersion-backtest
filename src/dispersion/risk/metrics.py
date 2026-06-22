"""Module 12 — Risk evaluation.

Implements per the Risk Track deliverable:
    - Vega exposure per variant, per day
    - Tail-correlation stress (set ρ = 1, recompute MTM)
    - Historical VaR and Expected Shortfall (1 %, 1-day)
    - Rolling max-drawdown
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISATION = 252.0


# ── Vega exposure ──────────────────────────────────────────────────────────────

def daily_vega_exposure(positions_df: pd.DataFrame) -> pd.Series:
    """Compute net vega exposure per day.

    `positions_df` must have columns: date, vega_notional (signed).
    Returns a Series indexed by date.
    """
    return positions_df.groupby("date")["vega_notional"].sum().rename("vega_exposure")


# ── Tail-correlation stress ────────────────────────────────────────────────────

def tail_correlation_stress(
    basket_vs_var: pd.Series,         # realized basket variance per date
    index_vs_var: pd.Series,          # index variance per date
    single_vols: pd.DataFrame,        # (date, ticker) → vol
    weights: dict[str, float],
    position_vega: pd.Series,         # signed vega per date
    shock_corr: float = 1.0,
) -> pd.Series:
    """Recompute basket variance under tail correlation (ρ → shock_corr).

    Basket variance under stress:
        σ_basket²_stress = Σ_i w_i² σ_i² + 2 * shock_corr * Σ_{i<j} w_i w_j σ_i σ_j

    MTM shock = (σ_basket²_stress - σ_basket²_normal) * vega_per_var_unit

    Returns a Series of daily stress P&L.
    """
    tickers = list(weights.keys())
    total_w = sum(weights.values())
    w = {t: weights[t] / total_w for t in tickers}

    stress_pnl = []
    for date in basket_vs_var.index:
        row_vols = {
            t: float(single_vols.loc[date, t])
            for t in tickers
            if date in single_vols.index and t in single_vols.columns
            and not np.isnan(single_vols.loc[date, t])
        }
        if len(row_vols) < 2:
            stress_pnl.append(0.0)
            continue

        diag = sum(w[t]**2 * row_vols[t]**2 for t in row_vols)
        off_diag_normal = sum(
            2 * w[t1] * w[t2] * row_vols[t1] * row_vols[t2]
            for i, t1 in enumerate(tickers)
            for t2 in list(tickers)[i+1:]
            if t1 in row_vols and t2 in row_vols
        )
        var_normal = float(basket_vs_var.get(date, diag + off_diag_normal))
        var_stress = diag + shock_corr * off_diag_normal

        delta_var = var_stress - var_normal
        vega = float(position_vega.get(date, 0.0))
        # Rough conversion: P&L ≈ vega * d(vol) ≈ vega * delta_var / (2*vol)
        avg_vol = np.sqrt(var_normal) if var_normal > 0 else 0.15
        pnl_shock = vega * (delta_var / (2 * avg_vol)) if avg_vol > 0 else 0.0
        stress_pnl.append(pnl_shock)

    return pd.Series(stress_pnl, index=basket_vs_var.index, name="stress_pnl")


# ── VaR and Expected Shortfall ────────────────────────────────────────────────

def historical_var(pnl: pd.Series, confidence: float = 0.99) -> float:
    """Historical 1-day VaR at `confidence` level (returns a negative number = loss)."""
    return float(np.percentile(pnl.dropna(), (1 - confidence) * 100))


def expected_shortfall(pnl: pd.Series, confidence: float = 0.99) -> float:
    """Historical expected shortfall (CVaR) at `confidence` level."""
    var = historical_var(pnl, confidence)
    tail = pnl[pnl <= var]
    return float(tail.mean()) if not tail.empty else var


# ── Max drawdown ───────────────────────────────────────────────────────────────

def rolling_max_drawdown(pnl: pd.Series, window: int | None = None) -> pd.Series:
    """Rolling max drawdown on cumulative P&L."""
    cum = pnl.cumsum()
    if window:
        roll_max = cum.rolling(window, min_periods=1).max()
    else:
        roll_max = cum.cummax()
    drawdown = cum - roll_max
    return drawdown.rename("drawdown")


# ── Risk summary ───────────────────────────────────────────────────────────────

def risk_summary(
    daily_pnl: pd.DataFrame,
    var_confidence: float = 0.99,
) -> dict:
    """Compute all risk metrics; return as a dict for tearsheet."""
    pnl = daily_pnl["net_pnl"].dropna()
    dd  = rolling_max_drawdown(pnl)
    var = historical_var(pnl, confidence=var_confidence)
    es  = expected_shortfall(pnl, confidence=var_confidence)

    return {
        "var_1pct_1d":       round(var, 0),
        "es_1pct_1d":        round(es, 0),
        "max_drawdown":      round(float(dd.min()), 0),
        "worst_day":         round(float(pnl.min()), 0),
        "best_day":          round(float(pnl.max()), 0),
        "pnl_skew":          round(float(pnl.skew()), 3),
        "pnl_kurt":          round(float(pnl.kurt()), 3),
    }


__all__ = [
    "daily_vega_exposure",
    "tail_correlation_stress",
    "historical_var",
    "expected_shortfall",
    "rolling_max_drawdown",
    "risk_summary",
]
