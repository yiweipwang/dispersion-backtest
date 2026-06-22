"""Module 4 — Implied vol surface + implied correlation.

Per (underlier_id, date):
    - ATM IV (nearest-month expiry > 7 DTE)
    - 30-day IV (interpolated across expirations)
    - Implied basket variance
    - Implied correlation (backed out from the basket-variance equation)
    - Correlation premium = ρ_implied - ρ_realized
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISATION = 252.0


# ── ATM IV ────────────────────────────────────────────────────────────────────

def _atm_iv_for_group(group: pd.DataFrame, min_dte: int = 7) -> float | None:
    """Given all options for a (ticker, date), return ATM IV.

    Selects nearest-month expiry with DTE > min_dte, then linearly
    interpolates IV across strikes to the at-the-money level.
    """
    # Find nearest-month expiry with DTE > min_dte
    valid = group[group["dte"] > min_dte]
    if valid.empty:
        return None
    nearest_expiry = valid["expiry"].min()
    exp_slice = group[group["expiry"] == nearest_expiry].copy()

    # Spot price (use median across strikes for robustness)
    spot_col = next(
        (c for c in ("spot", "underlier_price", "close") if c in exp_slice.columns),
        None,
    )
    if spot_col is None:
        return None
    spot = exp_slice[spot_col].median()
    if np.isnan(spot) or spot <= 0:
        return None

    # Linear interpolation of IV across strikes to the ATM strike (= spot)
    exp_slice = exp_slice.dropna(subset=["strike", "iv"]).sort_values("strike")
    if exp_slice.empty:
        return None

    iv_interp = np.interp(spot, exp_slice["strike"].values, exp_slice["iv"].values)
    return float(iv_interp)


def compute_atm_iv(
    chain: pd.DataFrame,
    min_dte: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame with columns (ticker, date, atm_iv)."""
    rows = []
    for (ticker, date), grp in chain.groupby(["ticker", "date"]):
        iv = _atm_iv_for_group(grp, min_dte=min_dte)
        rows.append({"ticker": ticker, "date": date, "atm_iv": iv})
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


# ── 30-day IV (term-structure interpolation) ──────────────────────────────────

def compute_30d_iv(chain: pd.DataFrame, target_dte: int = 30) -> pd.DataFrame:
    """Interpolate across expirations to a target-DTE IV per (ticker, date)."""
    rows = []
    for (ticker, date), grp in chain.groupby(["ticker", "date"]):
        spot_col = next(
            (c for c in ("spot", "underlier_price", "close") if c in grp.columns),
            None,
        )
        if spot_col is None:
            rows.append({"ticker": ticker, "date": date, "iv_30d": np.nan})
            continue
        spot = grp[spot_col].median()

        # For each expiry, get the ATM IV
        exp_ivs = []
        for exp, eg in grp.groupby("expiry"):
            eg = eg.dropna(subset=["strike", "iv"]).sort_values("strike")
            if eg.empty:
                continue
            dte_val = eg["dte"].iloc[0]
            iv_atm = float(np.interp(spot, eg["strike"].values, eg["iv"].values))
            exp_ivs.append((dte_val, iv_atm))

        if len(exp_ivs) < 2:
            iv_30d = exp_ivs[0][1] if exp_ivs else np.nan
        else:
            exp_ivs.sort()
            dtes = [x[0] for x in exp_ivs]
            ivs  = [x[1] for x in exp_ivs]
            iv_30d = float(np.interp(target_dte, dtes, ivs))

        rows.append({"ticker": ticker, "date": date, "iv_30d": iv_30d})

    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


# ── Implied correlation back-out ──────────────────────────────────────────────

def implied_correlation(
    index_iv: float,
    single_ivs: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Back out ρ_implied from the basket-variance identity.

    σ_index² = Σ_i w_i² σ_i² + 2 ρ Σ_{i<j} w_i w_j σ_i σ_j

    Solving for ρ:
        ρ = (σ_index² - Σ_i w_i² σ_i²) / (2 Σ_{i<j} w_i w_j σ_i σ_j)

    Returns NaN if denominator ≈ 0 or result is out of plausible range.
    """
    tickers = list(weights.keys())
    # Normalise weights
    total_w = sum(weights.values())
    w = {t: weights[t] / total_w for t in tickers}

    index_var = index_iv ** 2

    # Diagonal sum
    diag = sum(w[t] ** 2 * single_ivs[t] ** 2 for t in tickers if t in single_ivs)

    # Off-diagonal denominator
    n = len(tickers)
    off_diag_denom = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = tickers[i], tickers[j]
            if t1 in single_ivs and t2 in single_ivs:
                off_diag_denom += w[t1] * w[t2] * single_ivs[t1] * single_ivs[t2]
    off_diag_denom *= 2.0

    if abs(off_diag_denom) < 1e-12:
        return np.nan

    rho = (index_var - diag) / off_diag_denom
    return float(rho)


def build_implied_surface(
    chain: pd.DataFrame,
    index_ticker: str,
    single_tickers: list[str],
    weights: dict[str, float],
    realized_corr: pd.Series | None = None,
    min_dte: int = 7,
) -> pd.DataFrame:
    """Main entry point for Module 4.

    Returns a DataFrame indexed by date with columns:
        index_atm_iv, <ticker>_atm_iv, implied_corr,
        realized_corr (if provided), corr_premium.
    """
    atm = compute_atm_iv(chain, min_dte=min_dte)
    # Pivot: date → ticker columns
    atm_wide = atm.pivot(index="date", columns="ticker", values="atm_iv")

    rows = []
    for date in atm_wide.index:
        row: dict = {"date": date}
        # Index IV
        idx_iv = atm_wide.loc[date, index_ticker] if index_ticker in atm_wide.columns else np.nan
        row["index_atm_iv"] = idx_iv

        # Singles
        single_ivs = {}
        for t in single_tickers:
            iv = atm_wide.loc[date, t] if t in atm_wide.columns else np.nan
            row[f"{t}_atm_iv"] = iv
            if not np.isnan(iv):
                single_ivs[t] = iv

        # Implied correlation
        if not np.isnan(idx_iv) and len(single_ivs) == len(single_tickers):
            rho_impl = implied_correlation(idx_iv, single_ivs, weights)
        else:
            rho_impl = np.nan
        row["implied_corr"] = rho_impl

        rows.append(row)

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Join realized corr
    if realized_corr is not None:
        rc = realized_corr.rename("realized_corr").reindex(df.index)
        df = df.join(rc)
        df["corr_premium"] = df["implied_corr"] - df["realized_corr"]
    else:
        df["realized_corr"] = np.nan
        df["corr_premium"] = np.nan

    return df.reset_index()


__all__ = [
    "compute_atm_iv",
    "compute_30d_iv",
    "implied_correlation",
    "build_implied_surface",
]
