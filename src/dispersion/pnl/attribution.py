"""Module 11 — P&L + attribution.

Per-trade and per-day P&L; per-variant cost attribution waterfall.

Attribution table:
    gross_alpha         = signal P&L pre-costs
    opt_half_spread     = option half-spread cost
    opt_commission      = option commission
    opt_sec_occ         = option SEC/OCC fees
    stk_half_spread     = stock-leg half-spread
    stk_impact          = stock-leg market impact
    stk_commission      = stock-leg commission/fees
    financing           = vega financing / margin cost
    net_alpha           = gross_alpha - all costs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISATION = 252.0


def build_daily_pnl(
    daily_rows: list[dict],
) -> pd.DataFrame:
    """Build a daily P&L DataFrame from a list of per-day dicts.

    Each dict must have: date, gross_pnl, and cost sub-keys matching
    the attribution waterfall.
    """
    df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Fill missing cost columns with 0
    cost_cols = [
        "opt_half_spread", "opt_commission", "opt_sec_occ",
        "stk_half_spread", "stk_impact", "stk_commission",
        "stk_finra", "financing",
    ]
    for col in cost_cols:
        if col not in df.columns:
            df[col] = 0.0

    df["total_cost"] = df[cost_cols].sum(axis=1)
    df["net_pnl"]    = df["gross_pnl"] - df["total_cost"]
    df["cum_gross"]  = df["gross_pnl"].cumsum()
    df["cum_net"]    = df["net_pnl"].cumsum()
    return df


def attribution_waterfall(daily_pnl: pd.DataFrame, basis_notional: float = 1_000_000) -> pd.DataFrame:
    """Produce the attribution waterfall table (bps annualised + total $).

    `basis_notional` is used to convert $ P&L to bps.
    bps = ($ / notional) * 10_000 * (ANNUALISATION / n_days)
    """
    n_days = len(daily_pnl)
    scale = 10_000 * ANNUALISATION / max(n_days, 1) / basis_notional

    lines = [
        ("Gross alpha (pre-costs)",    "gross_pnl"),
        ("Less: option half-spread",   "opt_half_spread"),
        ("Less: option commission",    "opt_commission"),
        ("Less: option SEC/OCC fees",  "opt_sec_occ"),
        ("Less: stock half-spread",    "stk_half_spread"),
        ("Less: stock market impact",  "stk_impact"),
        ("Less: stock commission/fees","stk_commission"),
        ("Less: financing/margin",     "financing"),
    ]

    rows = []
    for label, col in lines:
        usd = float(daily_pnl[col].sum()) if col in daily_pnl.columns else 0.0
        bps = usd * scale
        rows.append({"Line": label, "bps_annualised": round(bps, 1), "usd": round(usd, 0)})

    # Net alpha
    net_usd = float(daily_pnl["net_pnl"].sum()) if "net_pnl" in daily_pnl.columns else 0.0
    rows.append({"Line": "= Net alpha", "bps_annualised": round(net_usd * scale, 1), "usd": round(net_usd, 0)})
    return pd.DataFrame(rows)


def compute_headline_stats(daily_pnl: pd.DataFrame) -> dict:
    """Compute Sharpe, max drawdown, hit rate from net P&L series."""
    pnl = daily_pnl["net_pnl"].dropna()
    if pnl.empty:
        return {}

    mu    = pnl.mean() * ANNUALISATION
    sigma = pnl.std() * np.sqrt(ANNUALISATION)
    sharpe = mu / sigma if sigma > 0 else np.nan

    cum = pnl.cumsum()
    roll_max = cum.cummax()
    drawdown = cum - roll_max
    max_dd = float(drawdown.min())

    hit_rate = float((pnl > 0).mean())
    n_trades = int(daily_pnl.get("trade_id", pd.Series([0])).nunique()) - 1  # subtract 0

    return {
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(max_dd, 0),
        "hit_rate":  round(hit_rate, 4),
        "n_days":    len(pnl),
        "n_trades":  max(n_trades, 0),
        "total_net": round(float(pnl.sum()), 0),
    }


__all__ = [
    "build_daily_pnl",
    "attribution_waterfall",
    "compute_headline_stats",
]
