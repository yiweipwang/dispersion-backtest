"""Module 7 — V2 Signal: Replicated correlation dispersion.

Same entry condition as V1 (VS-spread threshold), but the index leg is
built from a strip of OTM SPX options (log-contract replication portfolio)
rather than a direct variance swap.  Daily delta rebalancing applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals_v2(
    vs_basket: pd.DataFrame,
    earnings_dates: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold:  float = 0.5,
    holding_max_days: int  = 21,
    target_vega_notional: float = 1_000_000,
) -> pd.DataFrame:
    """Signal generation is identical to V1; the structural difference is in
    Module 9 (execution) where the index leg fills via the OTM replication strip.

    Returns the same schema as signal_v1.
    """
    from dispersion.signal.signal_v1 import generate_signals_v1
    signals = generate_signals_v1(
        vs_basket=vs_basket,
        earnings_dates=earnings_dates,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        holding_max_days=holding_max_days,
        target_vega_notional=target_vega_notional,
    )
    # Tag as V2 — execution layer reads this to use the replication portfolio
    signals["variant"] = "v2"
    signals["delta_hedge_daily"] = True
    return signals
