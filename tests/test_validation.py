"""All six required validation gates as pytest cases.

These tests run against synthetic data so they can pass on a fresh checkout
without the staged data bundle.  The tests validate the *logic* of each gate.

When real data is available, run:
    pytest tests/ -v --data-integration
and the integration fixtures will load from Sample Data/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.validation.gates import (
    gate_gamma_vs_vendor,
    gate_vega_vs_vendor,
    gate_vs_strike_vs_atm_iv,
    gate_implied_correlation_plausibility,
    gate_lookahead_audit,
    gate_cost_sanity,
    gate_reproducibility,
)
from dispersion.greeks.bs import bs_gamma, bs_vega


# ── Synthetic fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_chain() -> pd.DataFrame:
    """A small synthetic option chain with known-correct BS greeks."""
    rng = np.random.default_rng(42)
    n = 200
    S = 4500.0
    Ks = np.linspace(4000, 5000, n)
    T = 30 / 365
    sigma = rng.uniform(0.15, 0.35, n)
    r = 0.005
    gamma_true = bs_gamma(S, Ks, T, r, sigma)
    vega_true  = bs_vega(S, Ks, T, r, sigma)

    return pd.DataFrame({
        "strike":       Ks,
        "dte":          np.full(n, 30),
        "iv":           sigma,
        "spot":         np.full(n, S),
        "right":        ["C" if k >= S else "P" for k in Ks],
        "gamma":        gamma_true,
        "vega":         vega_true,
        "gamma_recomp": gamma_true * (1 + rng.normal(0, 0.003, n)),   # <1% error
        "vega_recomp":  vega_true  * (1 + rng.normal(0, 0.003, n)),
    })


@pytest.fixture
def synthetic_chain_bad_greeks(synthetic_chain) -> pd.DataFrame:
    """Chain where recomputed greeks are systematically off."""
    df = synthetic_chain.copy()
    df["gamma_recomp"] = df["gamma"] * 1.5   # 50% error → should FAIL
    df["vega_recomp"]  = df["vega"]  * 1.5
    return df


@pytest.fixture
def synthetic_vs_df() -> pd.DataFrame:
    """VS strikes slightly above ATM IV (as expected)."""
    dates = pd.date_range("2022-01-03", "2022-01-31", freq="B")
    tickers = ["AAPL", "MSFT"]
    rows = []
    for d in dates:
        for t in tickers:
            atm = 0.30
            rows.append({
                "ticker": t, "date": d,
                "vs_strike_vol": atm + 0.02,   # 2 vol pts above ATM → should PASS
                "dte": 20,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_atm_iv_df() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", "2022-01-31", freq="B")
    tickers = ["AAPL", "MSFT"]
    rows = [{"ticker": t, "date": d, "atm_iv": 0.30}
            for d in dates for t in tickers]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_surface_df() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", "2022-01-31", freq="B")
    return pd.DataFrame({
        "date":         dates,
        "implied_corr": np.linspace(0.55, 0.70, len(dates)),
        "realized_corr":np.linspace(0.45, 0.60, len(dates)),
        "corr_premium": np.linspace(0.05, 0.15, len(dates)),
    })


@pytest.fixture
def synthetic_daily_pnl() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", "2022-01-31", freq="B")
    n = len(dates)
    gross = rng.normal(5000, 3000, n)
    cost  = gross * 0.30     # 30% cost ratio → inside [10%, 70%]
    return pd.DataFrame({
        "date":          dates,
        "gross_pnl":     gross,
        "net_pnl":       gross - cost,
        "total_cost":    cost,
        "opt_half_spread": cost * 0.5,
        "opt_commission":  cost * 0.2,
        "opt_sec_occ":     cost * 0.1,
        "stk_half_spread": cost * 0.1,
        "stk_impact":      cost * 0.05,
        "stk_commission":  cost * 0.03,
        "stk_finra":       0.0,
        "financing":       cost * 0.02,
        "trade_id":        [i // 5 for i in range(n)],
    })


@pytest.fixture
def synthetic_signals() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", "2022-01-31", freq="B")
    signals = [0] * len(dates)
    signals[0] = 1
    signals[4] = -1
    signals[8] = 1
    signals[12] = -1
    return pd.DataFrame({"date": dates, "signal": signals, "trade_id": [0]*len(dates)})


# ── Gate 1: gamma_vs_vendor ────────────────────────────────────────────────────

def test_gamma_vs_vendor_pass(synthetic_chain):
    passed, details = gate_gamma_vs_vendor(synthetic_chain)
    assert passed, f"Expected PASS but got FAIL: {details}"
    assert details["median_rel_error"] < 0.01


def test_gamma_vs_vendor_fail(synthetic_chain_bad_greeks):
    passed, details = gate_gamma_vs_vendor(synthetic_chain_bad_greeks)
    assert not passed, f"Expected FAIL but got PASS: {details}"


# ── Gate 2: vega_vs_vendor ────────────────────────────────────────────────────

def test_vega_vs_vendor_pass(synthetic_chain):
    passed, details = gate_vega_vs_vendor(synthetic_chain)
    assert passed, f"Expected PASS but got FAIL: {details}"
    assert details["median_rel_error"] < 0.01


def test_vega_vs_vendor_fail(synthetic_chain_bad_greeks):
    passed, details = gate_vega_vs_vendor(synthetic_chain_bad_greeks)
    assert not passed, f"Expected FAIL but got PASS: {details}"


# ── Gate 3: vs_strike_vs_atm_iv ───────────────────────────────────────────────

def test_vs_strike_vs_atm_iv_pass(synthetic_vs_df, synthetic_atm_iv_df):
    passed, details = gate_vs_strike_vs_atm_iv(synthetic_vs_df, synthetic_atm_iv_df)
    assert passed, f"Expected PASS: {details}"


def test_vs_strike_vs_atm_iv_fail_below():
    """VS strike systematically below ATM IV → should FAIL."""
    dates = pd.date_range("2022-01-03", "2022-01-10", freq="B")
    vs_df   = pd.DataFrame({"ticker": "AAPL", "date": dates, "vs_strike_vol": 0.28, "dte": 20})
    atm_df  = pd.DataFrame({"ticker": "AAPL", "date": dates, "atm_iv": 0.30})
    passed, details = gate_vs_strike_vs_atm_iv(vs_df, atm_df, min_excess_vol=0.005)
    assert not passed, f"Expected FAIL: {details}"


# ── Gate 4: implied_correlation_plausibility ──────────────────────────────────

def test_implied_corr_plausibility_pass(synthetic_surface_df):
    passed, details = gate_implied_correlation_plausibility(synthetic_surface_df)
    assert passed, f"Expected PASS: {details}"


def test_implied_corr_plausibility_fail():
    df = pd.DataFrame({"implied_corr": [1.5, -0.2, 0.6, 0.7, 1.2] * 4})
    passed, details = gate_implied_correlation_plausibility(df)
    assert not passed, f"Expected FAIL: {details}"


# ── Gate 5: lookahead_audit ───────────────────────────────────────────────────

def test_lookahead_audit_pass(synthetic_signals, synthetic_daily_pnl):
    """Randomly generated P&L should have near-zero permuted Sharpe."""
    passed, details = gate_lookahead_audit(
        synthetic_signals, synthetic_daily_pnl, sharpe_threshold=0.5, n_permutations=50
    )
    # Permuted random P&L should easily pass (Sharpe ≈ 0)
    assert passed, f"Expected PASS: {details}"


# ── Gate 6: cost_sanity ───────────────────────────────────────────────────────

def test_cost_sanity_pass(synthetic_daily_pnl):
    passed, details = gate_cost_sanity(synthetic_daily_pnl)
    assert passed, f"Expected PASS: {details}"
    assert 0.10 <= details["cost_to_gross_ratio"] <= 0.70


def test_cost_sanity_fail_high_cost(synthetic_daily_pnl):
    df = synthetic_daily_pnl.copy()
    df["total_cost"] = df["gross_pnl"] * 0.95   # 95% cost ratio → FAIL
    passed, details = gate_cost_sanity(df)
    assert not passed, f"Expected FAIL: {details}"


# ── Gate 7: reproducibility ───────────────────────────────────────────────────

def test_reproducibility_pass(synthetic_daily_pnl):
    passed, details = gate_reproducibility(synthetic_daily_pnl, synthetic_daily_pnl.copy())
    assert passed, f"Expected PASS: {details}"
    assert details["max_abs_diff"] == 0.0


def test_reproducibility_fail(synthetic_daily_pnl):
    df2 = synthetic_daily_pnl.copy()
    df2["net_pnl"] = df2["net_pnl"] + 1.0   # tiny difference → should FAIL
    passed, details = gate_reproducibility(synthetic_daily_pnl, df2)
    assert not passed, f"Expected FAIL: {details}"


# ── Unit tests for underlying math ────────────────────────────────────────────

def test_bs_gamma_positive():
    """Gamma must be positive for all standard inputs."""
    g = bs_gamma(S=4500, K=4500, T=30/365, r=0.005, sigma=0.25)
    assert g > 0, f"Gamma should be positive, got {g}"


def test_bs_vega_positive():
    """Vega must be positive."""
    v = bs_vega(S=4500, K=4500, T=30/365, r=0.005, sigma=0.25)
    assert v > 0, f"Vega should be positive, got {v}"


def test_bs_vega_zero_at_expiry():
    """Vega → 0 as T → 0."""
    v = bs_vega(S=4500, K=4500, T=1e-8, r=0.005, sigma=0.25)
    assert v < 1e-3, f"Vega near expiry should be ~0, got {v}"


def test_implied_correlation_backout():
    """Verify the implied-corr back-out is self-consistent."""
    from dispersion.implied.surface import implied_correlation
    single_ivs = {"AAPL": 0.30, "MSFT": 0.25, "NVDA": 0.45, "TSLA": 0.55}
    weights    = {"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.20, "TSLA": 0.20}
    rho_target = 0.65

    # Forward: compute index variance from known rho
    tickers = list(weights.keys())
    total_w = sum(weights.values())
    w = {t: weights[t] / total_w for t in tickers}
    diag = sum(w[t]**2 * single_ivs[t]**2 for t in tickers)
    off  = sum(2 * w[t1] * w[t2] * single_ivs[t1] * single_ivs[t2]
               for i, t1 in enumerate(tickers)
               for t2 in list(tickers)[i+1:])
    index_var  = diag + rho_target * off
    index_iv   = np.sqrt(index_var)

    rho_back = implied_correlation(index_iv, single_ivs, weights)
    assert abs(rho_back - rho_target) < 1e-6, (
        f"Back-out failed: expected {rho_target}, got {rho_back}"
    )
