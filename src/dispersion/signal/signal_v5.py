"""Module 7 — V5 Signal: Realized-vs-implied correlation.

Entry: ρ_implied − ρ_realized  >  entry_threshold
Trade: short index variance (implied), long basket variance (implied).
Exit:  premium compresses below exit_threshold OR holding_max_days exceeded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals_v5(
    surface: pd.DataFrame,          # from build_implied_surface(): date, corr_premium, ...
    earnings_dates: pd.DataFrame,
    entry_threshold: float = 0.10,  # 10 pp correlation premium
    exit_threshold:  float = 0.03,
    corr_window:     int   = 21,
    holding_max_days: int  = 21,
    target_vega_notional: float = 1_000_000,
) -> pd.DataFrame:
    """Return per-date signal DataFrame.

    Columns: date, signal, position_open, corr_premium, trade_id.
    """
    df = surface.copy().sort_values("date").reset_index(drop=True)
    blackout_dates = set(pd.to_datetime(earnings_dates["date"].values))

    signals = []
    position_open = False
    entry_date = None
    trade_id = 0

    for _, row in df.iterrows():
        date = pd.Timestamp(row["date"])
        premium = row.get("corr_premium", np.nan)
        is_blackout = date in blackout_dates

        if not position_open:
            if (not np.isnan(premium)
                    and premium > entry_threshold
                    and not is_blackout):
                position_open = True
                entry_date = date
                trade_id += 1
                sig = 1
            else:
                sig = 0
        else:
            days_held = (date - entry_date).days if entry_date else 0
            exit_cond = (
                (not np.isnan(premium) and premium < exit_threshold)
                or days_held >= holding_max_days
                or is_blackout
            )
            if exit_cond:
                position_open = False
                entry_date = None
                sig = -1
            else:
                sig = 0

        signals.append({
            "date":          date,
            "signal":        sig,
            "position_open": position_open,
            "corr_premium":  premium,
            "implied_corr":  row.get("implied_corr", np.nan),
            "realized_corr": row.get("realized_corr", np.nan),
            "trade_id":      trade_id if position_open or sig == -1 else 0,
        })

    return pd.DataFrame(signals)
