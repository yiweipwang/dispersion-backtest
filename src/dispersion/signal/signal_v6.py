"""Module 7 — V6 Signal: 0DTE-driven dispersion.

At 15:30, compute dealer GEX on SPX/SPXW 0DTE.
When dealer GEX is strongly negative (< gex_threshold), dispersion pays —
enter at MOC, close at MOO next morning.

GEX (dealer Gamma EXposure):
    GEX = Σ_contracts (sign × gamma × OI × multiplier × spot²) / 1e8
    sign: +1 if dealer short the option (net buyer flow), -1 if long

Lee-Ready sign inference assigns trade direction from NBBO quote:
    price >= ask midpoint → buyer-initiated → dealer sold → dealer short → positive GEX
    price <= bid midpoint → seller-initiated → dealer bought → dealer long → negative GEX
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _lee_ready_sign(
    trade_price: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
) -> np.ndarray:
    """Lee-Ready sign: +1 buyer-initiated, -1 seller-initiated, 0 ambiguous."""
    mid = (bid + ask) / 2.0
    sign = np.where(trade_price >= mid, 1, np.where(trade_price <= mid, -1, 0))
    return sign.astype(float)


def compute_dealer_gex(
    trades_nbbo: pd.DataFrame,
    chain_eod: pd.DataFrame,
    date: pd.Timestamp,
    spot: float,
    multiplier: float = 100.0,
) -> float:
    """Compute net dealer GEX for SPX/SPXW 0DTE options at 15:30 ET.

    trades_nbbo: intraday option trades + NBBO for `date`
    chain_eod: EOD chain for `date` (used for OI and gamma)
    Returns total dealer GEX in USD (notional).
    """
    # Filter to 0DTE contracts: expiry == date
    date_norm = pd.Timestamp(date).normalize()
    zero_dte = chain_eod[
        (chain_eod["date"] == date_norm)
        & (chain_eod["expiry"] == date_norm)
    ].copy()
    if zero_dte.empty:
        return 0.0

    # Build a strike → gamma map from EOD chain
    if "gamma_recomp" in zero_dte.columns:
        gamma_col = "gamma_recomp"
    elif "gamma" in zero_dte.columns:
        gamma_col = "gamma"
    else:
        return 0.0

    gamma_map = (
        zero_dte.groupby(["strike", "right"])[gamma_col].mean().to_dict()
    )
    oi_map = (
        zero_dte.groupby(["strike", "right"])["open_interest"].sum().to_dict()
    )

    # Filter trades to 15:00–15:30 window (last 30 min before signal time)
    if "timestamp" in trades_nbbo.columns:
        t_open  = pd.Timestamp(f"{date_norm.date()} 15:00:00")
        t_close = pd.Timestamp(f"{date_norm.date()} 15:30:00")
        trades = trades_nbbo[
            (trades_nbbo["timestamp"] >= t_open)
            & (trades_nbbo["timestamp"] <= t_close)
        ].copy()
    else:
        trades = trades_nbbo.copy()

    if trades.empty:
        # Fall back to OI-based GEX
        gex = 0.0
        for (k, r), g in gamma_map.items():
            oi = oi_map.get((k, r), 0)
            gex += g * oi * multiplier * spot ** 2
        return gex

    # Lee-Ready signs on intraday trades
    bid_col = next((c for c in ("bid", "nbbo_bid") if c in trades.columns), None)
    ask_col = next((c for c in ("ask", "nbbo_ask") if c in trades.columns), None)
    price_col = next((c for c in ("price", "trade_price") if c in trades.columns), None)

    if bid_col and ask_col and price_col:
        signs = _lee_ready_sign(
            trades[price_col].values,
            trades[bid_col].values,
            trades[ask_col].values,
        )
        # Dealer is on the opposite side of the trade
        dealer_sign = -signs   # buyer-initiated → dealer sold → dealer short → +GEX
    else:
        dealer_sign = np.zeros(len(trades))

    # Merge gamma from chain
    right_col = next((c for c in ("right",) if c in trades.columns), None)
    strike_col = next((c for c in ("strike",) if c in trades.columns), None)
    size_col   = next((c for c in ("size", "quantity") if c in trades.columns), None)

    gex_total = 0.0
    for i, row in trades.iterrows():
        k = row.get(strike_col, np.nan) if strike_col else np.nan
        r = row.get(right_col, "C")    if right_col   else "C"
        sz = row.get(size_col, 1)      if size_col    else 1
        g = gamma_map.get((k, r), 0.0)
        d = dealer_sign[i] if isinstance(dealer_sign, np.ndarray) else 0.0
        gex_total += d * g * sz * multiplier * spot ** 2

    return float(gex_total) / 1e8   # normalise to billions


def generate_signals_v6(
    chain_eod: pd.DataFrame,
    trades_nbbo_by_date: dict,      # date → DataFrame (lazy-loaded)
    equity_daily: pd.DataFrame,     # for spot price
    earnings_dates: pd.DataFrame,
    gex_threshold: float = -500_000_000,
    target_vega_notional: float = 500_000,
) -> pd.DataFrame:
    """Return per-date signal DataFrame.

    Columns: date, signal, position_open, dealer_gex, trade_id.
    Trade cycle: enter at close today (signal==1), exit at open next day (signal==-1).
    """
    backtest_dates = sorted(chain_eod["date"].dropna().unique())
    blackout_dates = set(pd.to_datetime(earnings_dates["date"].values))

    # Build spot price lookup
    equity_daily = equity_daily.copy()
    spx_close = (
        equity_daily.groupby("date")["close_auction_price"]
        .mean()
        .to_dict()
        if "close_auction_price" in equity_daily.columns
        else {}
    )

    signals = []
    position_open = False
    pending_exit = False
    trade_id = 0

    for date in backtest_dates:
        ts = pd.Timestamp(date)
        is_blackout = ts in blackout_dates

        # If we entered yesterday, exit today at MOO
        if pending_exit:
            pending_exit = False
            position_open = False
            sig = -1
        elif position_open:
            sig = 0
        else:
            sig = 0

        if not position_open and not is_blackout:
            # Compute dealer GEX at 15:30
            day_chain = chain_eod[chain_eod["date"] == ts]
            spot = float(spx_close.get(ts, np.nan))
            if np.isnan(spot):
                # Fallback: use median underlying price from chain
                spot_col = next(
                    (c for c in ("spot", "underlier_price") if c in day_chain.columns),
                    None,
                )
                spot = float(day_chain[spot_col].median()) if spot_col else 4500.0

            nbbo = trades_nbbo_by_date.get(ts, pd.DataFrame())
            gex = compute_dealer_gex(day_chain, day_chain, ts, spot)

            # Normalise sign: strongly negative GEX → enter dispersion
            if gex < gex_threshold / 1e8:   # threshold already normalised above
                trade_id += 1
                position_open = True
                pending_exit = True   # close tomorrow at MOO
                sig = 1
            else:
                gex_val = gex
        else:
            gex = np.nan

        signals.append({
            "date":          ts,
            "signal":        sig,
            "position_open": position_open,
            "dealer_gex":    gex if "gex" in dir() else np.nan,
            "trade_id":      trade_id if sig != 0 else 0,
        })

    return pd.DataFrame(signals)
