"""Module 5 — Variance-swap pricer via log-contract replication.

Standard discrete approximation:

    K_VS = (2/T) * [ Σ_{K < F} P(K)/K² * dK  +  Σ_{K >= F} C(K)/K² * dK ]

where F is the forward price, T is time to expiry (years), and the sum runs
over available OTM strikes.

Validation gate: K_VS should sit 0.5–5 vol points above ATM IV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_price(chain_slice: pd.DataFrame) -> float:
    """Estimate forward from put-call parity at the strike nearest the money.

    Uses: C - P = F * exp(-rT) - K * exp(-rT) ≈ S - K  (r ≈ 0 short-circuit).
    Falls back to spot price if insufficient data.
    """
    spot_col = next(
        (c for c in ("spot", "underlier_price", "close") if c in chain_slice.columns),
        None,
    )
    if spot_col:
        return float(chain_slice[spot_col].median())

    # Put-call parity fallback: find ATM strike pair
    calls = chain_slice[chain_slice["right"] == "C"][["strike", "mid"]].set_index("strike")
    puts  = chain_slice[chain_slice["right"] == "P"][["strike", "mid"]].set_index("strike")
    common = calls.index.intersection(puts.index)
    if common.empty:
        return np.nan
    diffs = (calls.loc[common, "mid"] - puts.loc[common, "mid"])
    atm_k = common[np.argmin(np.abs(diffs.values))]
    # C - P = S - K → F ≈ S ≈ K + (C - P)
    F = float(atm_k + diffs.loc[atm_k])
    return F


def variance_swap_strike(
    chain_slice: pd.DataFrame,
    T_years: float,
    use_mid: bool = True,
) -> float:
    """Compute VS strike (annualised vol² × 100 scale, i.e. 'vol points squared').

    `chain_slice` is the option chain for a *single* (ticker, date, expiry).
    Required columns: strike, right (C/P), mid (or bid+ask), and ideally spot.

    Returns the VS strike in **variance** units (σ²).  Multiply by 100 to convert
    to 'vol-point²'; take sqrt for the equivalent vol.
    """
    if T_years <= 0:
        return np.nan

    price_col = "mid" if (use_mid and "mid" in chain_slice.columns) else "bid"
    if price_col not in chain_slice.columns:
        return np.nan

    F = _forward_price(chain_slice)
    if np.isnan(F) or F <= 0:
        return np.nan

    df = chain_slice.copy().dropna(subset=["strike", price_col])
    df = df[df[price_col] > 0]          # drop zero-priced options
    df = df.sort_values("strike")

    # OTM puts: K < F
    puts  = df[(df["right"] == "P") & (df["strike"] < F)].copy()
    # OTM calls: K >= F
    calls = df[(df["right"] == "C") & (df["strike"] >= F)].copy()

    # dK approximation — midpoint rule between adjacent strikes
    def _integrate(sub: pd.DataFrame) -> float:
        if sub.empty:
            return 0.0
        strikes = sub["strike"].values
        prices  = sub[price_col].values
        # dK: average spacing between adjacent strikes
        if len(strikes) == 1:
            # Single strike: use a nominal dK of 5 (SPX spacing typical)
            dK = np.array([5.0])
        else:
            dK = np.diff(strikes)
            # Pad first/last
            dK = np.concatenate([[dK[0]], (dK[:-1] + dK[1:]) / 2, [dK[-1]]])
        weights = prices / (strikes ** 2) * dK
        return float(np.sum(weights))

    integral = _integrate(puts) + _integrate(calls)
    K_VS_var = (2.0 / T_years) * integral       # variance units
    return K_VS_var


def build_vs_strikes(
    chain: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Compute VS strike per (ticker, date, expiry).

    Returns a DataFrame: ticker, date, expiry, dte, vs_strike_var, vs_strike_vol.
    """
    rows = []
    group_keys = ["ticker", "date", "expiry"]
    for keys, grp in chain.groupby(group_keys):
        ticker, date, expiry = keys
        if ticker not in tickers:
            continue
        dte = grp["dte"].iloc[0] if "dte" in grp.columns else np.nan
        T = max(float(dte), 0.0) / 365.0 if not np.isnan(dte) else 0.0
        vs_var = variance_swap_strike(grp, T_years=T)
        vs_vol = float(np.sqrt(vs_var)) if (vs_var is not None and vs_var > 0) else np.nan
        rows.append({
            "ticker":        ticker,
            "date":          date,
            "expiry":        expiry,
            "dte":           dte,
            "vs_strike_var": vs_var,   # annualised variance
            "vs_strike_vol": vs_vol,   # equivalent vol (σ units, 0–1 scale)
        })
    return pd.DataFrame(rows).sort_values(["ticker", "date", "expiry"]).reset_index(drop=True)


def near_month_vs_strikes(vs_df: pd.DataFrame, min_dte: int = 7) -> pd.DataFrame:
    """Select the near-month (smallest DTE > min_dte) VS strike per (ticker, date)."""
    valid = vs_df[vs_df["dte"] > min_dte].copy()
    idx = valid.groupby(["ticker", "date"])["dte"].idxmin()
    return valid.loc[idx].reset_index(drop=True)


def basket_vs_strike(
    nm_vs: pd.DataFrame,
    weights: dict[str, float],
    index_ticker: str,
) -> pd.DataFrame:
    """Vega-weighted basket VS strike per date.

    Returns a DataFrame: date, index_vs_vol, basket_vs_vol, vs_spread_vol.
    (vs_spread_vol = index_vs_vol - basket_vs_vol — the V1/V2 entry signal.)
    """
    # Normalise weights
    total_w = sum(weights.values())
    w = {t: v / total_w for t, v in weights.items()}

    singles = [t for t in weights if t != index_ticker]

    # Pivot to wide
    wide = nm_vs.pivot(index="date", columns="ticker", values="vs_strike_vol")

    rows = []
    for date, row in wide.iterrows():
        idx_vol = row.get(index_ticker, np.nan)
        basket_vol = sum(w[t] * float(row.get(t, np.nan) or np.nan)
                         for t in singles) if singles else np.nan
        spread = (float(idx_vol) - basket_vol
                  if not (np.isnan(idx_vol) or np.isnan(basket_vol))
                  else np.nan)
        rows.append({
            "date":           date,
            "index_vs_vol":   idx_vol,
            "basket_vs_vol":  basket_vol,
            "vs_spread_vol":  spread,   # entry signal for V1/V2
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


__all__ = [
    "variance_swap_strike",
    "build_vs_strikes",
    "near_month_vs_strikes",
    "basket_vs_strike",
]
