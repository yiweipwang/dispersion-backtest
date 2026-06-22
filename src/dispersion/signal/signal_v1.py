"""Module 7 — V1 Signal: Classic variance dispersion.

Entry: K_VS_index − Σ w_i K_VS_i  >  entry_threshold  (vol points)
Trade: short index VS, long basket of single-name VS (vega-neutralised)
Exit:  spread compresses below exit_threshold OR holding_max_days exceeded
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dispersion.basket.weights import get_weights


def generate_signals_v1(
    vs_basket: pd.DataFrame,            # from basket_vs_strike(): date, vs_spread_vol, ...
    earnings_dates: pd.DataFrame,       # from load_earnings_calendar()
    entry_threshold: float = 2.0,       # vol points
    exit_threshold:  float = 0.5,
    holding_max_days: int  = 21,
    target_vega_notional: float = 1_000_000,
) -> pd.DataFrame:
    """Return a per-date signal DataFrame.

    Columns:
        date, signal (1=enter, 0=flat, -1=exit), position_open (bool),
        vs_spread_vol, trade_id
    """
    df = vs_basket.copy().sort_values("date").reset_index(drop=True)

    # Earnings blackout: exclude dates where any name reports
    blackout_dates = set(pd.to_datetime(earnings_dates["date"].values))

    signals = []
    position_open = False
    entry_date = None
    trade_id = 0

    for _, row in df.iterrows():
        date = pd.Timestamp(row["date"])
        spread = row.get("vs_spread_vol", np.nan)
        is_blackout = date in blackout_dates

        if not position_open:
            # Entry condition
            if (not np.isnan(spread)
                    and spread > entry_threshold
                    and not is_blackout):
                position_open = True
                entry_date = date
                trade_id += 1
                sig = 1       # enter
            else:
                sig = 0
        else:
            # Check exit conditions
            days_held = (date - entry_date).days if entry_date else 0
            exit_signal = (
                (not np.isnan(spread) and spread < exit_threshold)
                or days_held >= holding_max_days
                or is_blackout
            )
            if exit_signal:
                position_open = False
                entry_date = None
                sig = -1      # exit
            else:
                sig = 0       # hold

        signals.append({
            "date":            date,
            "signal":          sig,
            "position_open":   position_open,
            "vs_spread_vol":   spread,
            "trade_id":        trade_id if position_open or sig == -1 else 0,
            "entry_threshold": entry_threshold,
        })

    return pd.DataFrame(signals)
