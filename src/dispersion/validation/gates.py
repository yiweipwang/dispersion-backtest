"""Module 13 — Validation gates.

All six required gates; each returns (passed: bool, details: dict).
These are also exposed as pytest functions in tests/test_validation.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Gate 1 & 2: Greeks vs vendor ──────────────────────────────────────────────

def gate_gamma_vs_vendor(chain: pd.DataFrame, tol: float = 0.01) -> tuple[bool, dict]:
    """Median |relative error| of recomputed gamma vs vendor < tol (1 %)."""
    required = {"gamma", "gamma_recomp"}
    missing = required - set(chain.columns)
    if missing:
        return False, {"error": f"Missing columns: {missing}"}

    mask = (chain["gamma"].abs() > 1e-10) & chain["gamma_recomp"].notna()
    sub = chain[mask]
    if sub.empty:
        return False, {"error": "No valid gamma rows for comparison"}

    rel_err = ((sub["gamma_recomp"] - sub["gamma"]).abs() / sub["gamma"].abs())
    median_err = float(rel_err.median())
    passed = median_err < tol
    return passed, {"median_rel_error": round(median_err, 5), "threshold": tol, "n": len(sub)}


def gate_vega_vs_vendor(chain: pd.DataFrame, tol: float = 0.01) -> tuple[bool, dict]:
    """Median |relative error| of recomputed vega vs vendor < tol (1 %)."""
    required = {"vega", "vega_recomp"}
    missing = required - set(chain.columns)
    if missing:
        return False, {"error": f"Missing columns: {missing}"}

    mask = (chain["vega"].abs() > 1e-10) & chain["vega_recomp"].notna()
    sub = chain[mask]
    if sub.empty:
        return False, {"error": "No valid vega rows for comparison"}

    rel_err = ((sub["vega_recomp"] - sub["vega"]).abs() / sub["vega"].abs())
    median_err = float(rel_err.median())
    passed = median_err < tol
    return passed, {"median_rel_error": round(median_err, 5), "threshold": tol, "n": len(sub)}


# ── Gate 3: VS strike plausibility ────────────────────────────────────────────

def gate_vs_strike_vs_atm_iv(
    vs_df: pd.DataFrame,
    atm_iv_df: pd.DataFrame,
    min_excess_vol: float = 0.005,    # 0.5 vol points above ATM IV
    max_excess_vol: float = 0.05,     # 5 vol points above ATM IV
) -> tuple[bool, dict]:
    """VS strike (vol units) should be min_excess_vol–max_excess_vol above ATM IV."""
    merged = vs_df.merge(atm_iv_df, on=["ticker", "date"], how="inner")
    if merged.empty:
        return False, {"error": "No rows after merge — check ticker/date alignment"}

    merged = merged.dropna(subset=["vs_strike_vol", "atm_iv"])
    excess = merged["vs_strike_vol"] - merged["atm_iv"]

    frac_below = float((excess < min_excess_vol).mean())
    frac_above = float((excess > max_excess_vol).mean())
    median_excess = float(excess.median())

    passed = (frac_below < 0.10) and (frac_above < 0.10)
    return passed, {
        "median_excess_vol": round(median_excess, 4),
        "frac_below_min":    round(frac_below, 4),
        "frac_above_max":    round(frac_above, 4),
        "n":                 len(merged),
    }


# ── Gate 4: Implied correlation plausibility ──────────────────────────────────

def gate_implied_correlation_plausibility(
    surface_df: pd.DataFrame,
    corr_col: str = "implied_corr",
    lo: float = 0.0,
    hi: float = 0.95,
) -> tuple[bool, dict]:
    """Implied correlation per date must sit in [lo, hi]."""
    if corr_col not in surface_df.columns:
        return False, {"error": f"Column {corr_col!r} not found"}

    corr = surface_df[corr_col].dropna()
    if corr.empty:
        return False, {"error": "No non-null implied correlation values"}

    frac_out = float(((corr < lo) | (corr > hi)).mean())
    passed = frac_out < 0.05   # allow up to 5 % of dates marginally outside
    return passed, {
        "min":       round(float(corr.min()), 4),
        "max":       round(float(corr.max()), 4),
        "mean":      round(float(corr.mean()), 4),
        "frac_outside_range": round(frac_out, 4),
    }


# ── Gate 5: Look-ahead audit ──────────────────────────────────────────────────

def gate_lookahead_audit(
    signals: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    sharpe_threshold: float = 0.2,
    n_permutations: int = 100,
    seed: int = 42,
) -> tuple[bool, dict]:
    """Permute the date column on entry signals; net Sharpe should drop to ~0.

    A real look-ahead bug means the strategy works even after date-shuffling.
    """
    rng = np.random.default_rng(seed)
    pnl = daily_pnl["net_pnl"].dropna().values
    if len(pnl) < 5:
        return False, {"error": "Too few P&L observations for permutation test"}

    perm_sharpes = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(pnl)
        mu    = shuffled.mean() * 252
        sigma = shuffled.std() * np.sqrt(252)
        sh = mu / sigma if sigma > 0 else 0.0
        perm_sharpes.append(sh)

    mean_perm_sharpe = float(np.mean(np.abs(perm_sharpes)))
    passed = mean_perm_sharpe < sharpe_threshold
    return passed, {
        "mean_abs_perm_sharpe": round(mean_perm_sharpe, 4),
        "threshold":            sharpe_threshold,
        "n_permutations":       n_permutations,
    }


# ── Gate 6: Cost sanity ───────────────────────────────────────────────────────

def gate_cost_sanity(
    daily_pnl: pd.DataFrame,
    lo: float = 0.10,
    hi: float = 0.70,
) -> tuple[bool, dict]:
    """Total costs / gross P&L must be in [lo, hi]."""
    gross = float(daily_pnl["gross_pnl"].sum())
    cost  = float(daily_pnl["total_cost"].sum()) if "total_cost" in daily_pnl.columns else 0.0
    if abs(gross) < 1e-6:
        return False, {"error": "Gross P&L near zero — cannot compute ratio"}

    ratio = cost / abs(gross)
    passed = lo <= ratio <= hi
    return passed, {
        "cost_to_gross_ratio": round(ratio, 4),
        "gross_usd":           round(gross, 0),
        "cost_usd":            round(cost, 0),
        "range":               [lo, hi],
    }


# ── Gate 7: Reproducibility ────────────────────────────────────────────────────

def gate_reproducibility(pnl_a: pd.DataFrame, pnl_b: pd.DataFrame) -> tuple[bool, dict]:
    """Two runs with the same seed must produce bit-identical daily net P&L."""
    if "net_pnl" not in pnl_a.columns or "net_pnl" not in pnl_b.columns:
        return False, {"error": "net_pnl column missing from one or both DataFrames"}

    a = pnl_a.sort_values("date")["net_pnl"].reset_index(drop=True)
    b = pnl_b.sort_values("date")["net_pnl"].reset_index(drop=True)

    if len(a) != len(b):
        return False, {"error": f"Row count mismatch: {len(a)} vs {len(b)}"}

    identical = bool((a == b).all())
    max_diff  = float((a - b).abs().max())
    return identical, {"max_abs_diff": max_diff, "n_rows": len(a)}


# ── Run-all helper ─────────────────────────────────────────────────────────────

def run_all_gates(
    chain: pd.DataFrame,
    vs_df: pd.DataFrame,
    atm_iv_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    signals: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    pnl_run2: pd.DataFrame | None = None,
) -> dict[str, tuple[bool, dict]]:
    """Run all gates and return results dict keyed by gate name."""
    results = {}
    results["gamma_vs_vendor"]               = gate_gamma_vs_vendor(chain)
    results["vega_vs_vendor"]                = gate_vega_vs_vendor(chain)
    results["vs_strike_vs_atm_iv"]           = gate_vs_strike_vs_atm_iv(vs_df, atm_iv_df)
    results["implied_correlation_plausibility"] = gate_implied_correlation_plausibility(surface_df)
    results["lookahead_audit"]               = gate_lookahead_audit(signals, daily_pnl)
    results["cost_sanity"]                   = gate_cost_sanity(daily_pnl)
    if pnl_run2 is not None:
        results["reproducibility"]           = gate_reproducibility(daily_pnl, pnl_run2)
    return results


__all__ = [
    "gate_gamma_vs_vendor",
    "gate_vega_vs_vendor",
    "gate_vs_strike_vs_atm_iv",
    "gate_implied_correlation_plausibility",
    "gate_lookahead_audit",
    "gate_cost_sanity",
    "gate_reproducibility",
    "run_all_gates",
]
