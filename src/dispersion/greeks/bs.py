"""Module 2 — Black-Scholes Greeks recompute (gamma + vega).

Validation gate: EOD recomputed gamma & vega vs vendor columns.
Median |relative error| must be < 1 % for both (Module 13 gate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Core BS formulas ──────────────────────────────────────────────────────────

_SQRT2PI = np.sqrt(2 * np.pi)


def _d1(S: np.ndarray, K: np.ndarray, T: np.ndarray,
        r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """d1 from Black-Scholes formula.  All inputs must be broadcastable arrays."""
    # Guard against zero or negative T / sigma
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(d1: np.ndarray, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    return d1 - sigma * np.sqrt(T)


def _nprime(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / _SQRT2PI


def bs_gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Gamma = N'(d1) / (S * sigma * sqrt(T)).

    Units: per dollar squared (raw BS gamma).
    Multiply by S^2/100 to convert to 'dollar gamma' if needed.
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    d1 = _d1(S, K, T, r, sigma)
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    return _nprime(d1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Vega = S * sqrt(T) * N'(d1).

    Units: change in option price per 1-unit change in sigma.
    Divide by 100 to get vega per 1 vol-point (1 %).
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    d1 = _d1(S, K, T, r, sigma)
    T = np.maximum(T, 1e-8)
    return S * np.sqrt(T) * _nprime(d1)


def bs_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    right: str | np.ndarray = "C",
) -> np.ndarray:
    """Delta.  right: 'C' or 'P'."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    d1 = _d1(S, K, T, r, sigma)
    d = norm.cdf(d1)
    if isinstance(right, str):
        return d if right == "C" else d - 1.0
    right = np.asarray(right)
    return np.where(right == "C", d, d - 1.0)


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    right: str | np.ndarray = "C",
) -> np.ndarray:
    """Black-Scholes option price."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if isinstance(right, str):
        if right == "C":
            return call
        else:
            return call - S + K * np.exp(-r * T)   # put-call parity
    right = np.asarray(right)
    put = call - S + K * np.exp(-r * T)
    return np.where(right == "C", call, put)


# ── DataFrame-level helpers ────────────────────────────────────────────────────

def recompute_greeks(
    chain: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Add/overwrite `gamma_recomp` and `vega_recomp` columns on an option chain.

    Required columns: spot (or underlier_price), strike, iv, dte (calendar days).
    `risk_free_rate` defaults to 0; caller can pass SOFR/365 if desired.

    Returns a copy of `chain` with two new columns.
    """
    df = chain.copy()

    # Time to expiry in years
    T = df["dte"].clip(lower=0).values / 365.0

    # Spot — column name varies by data source
    spot_col = next(
        (c for c in ("spot", "underlier_price", "close", "last") if c in df.columns),
        None,
    )
    if spot_col is None:
        raise KeyError(
            "Cannot find spot price column in chain DataFrame. "
            "Expected one of: 'spot', 'underlier_price', 'close', 'last'."
        )
    S = df[spot_col].values

    iv_col = next((c for c in ("iv", "implied_vol", "impliedvol") if c in df.columns), None)
    if iv_col is None:
        raise KeyError("Cannot find IV column.  Expected 'iv', 'implied_vol', or 'impliedvol'.")
    sigma = df[iv_col].values

    K = df["strike"].values
    r = risk_free_rate

    df["gamma_recomp"] = bs_gamma(S, K, T, r, sigma)
    df["vega_recomp"]  = bs_vega(S, K, T, r, sigma)
    return df


def recompute_greeks_trades(
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Same as `recompute_greeks` but for the trades-with-greeks file.

    Requires columns: underlier_price (or spot), strike, iv (or implied_vol), dte.
    """
    return recompute_greeks(trades, risk_free_rate=risk_free_rate)


__all__ = [
    "bs_gamma",
    "bs_vega",
    "bs_delta",
    "bs_price",
    "recompute_greeks",
    "recompute_greeks_trades",
]
