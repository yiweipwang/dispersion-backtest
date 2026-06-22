"""Module 8 — Basket construction / vega-weighted weights.

Three weighting methods:
    vega_neutral   (default): w_i = 1/(4*σ_i), normalised
    spx_weight:               actual SPX weights, renormalised to our 4 names
    equal_vega:               w_i = 1/4 (uniform)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate SPX constituent weights for our 4 names as of Jan 2022
# Source: S&P 500 index methodology (approx.)
_SPX_APPROX_WEIGHTS = {
    "AAPL": 0.0694,
    "MSFT": 0.0621,
    "NVDA": 0.0198,
    "TSLA": 0.0213,
}


def vega_neutral_weights(single_ivs: dict[str, float]) -> dict[str, float]:
    """w_i = 1/(4 * σ_i), normalised so that Σ w_i = 1.

    The idea: positions are sized so that each name contributes equal vega
    to the basket, regardless of its IV level.
    """
    raw = {t: 1.0 / (4.0 * sigma) for t, sigma in single_ivs.items() if sigma > 0}
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return {t: 1.0 / n for t in raw}
    return {t: v / total for t, v in raw.items()}


def spx_weight_matched(tickers: list[str]) -> dict[str, float]:
    """Scale by approximate SPX constituent weights, renormalise to our 4 names."""
    raw = {t: _SPX_APPROX_WEIGHTS.get(t, 0.0) for t in tickers}
    total = sum(raw.values())
    if total == 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    return {t: v / total for t, v in raw.items()}


def equal_vega_weights(tickers: list[str]) -> dict[str, float]:
    """Uniform: w_i = 1/N."""
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def get_weights(
    method: str,
    tickers: list[str],
    single_ivs: dict[str, float] | None = None,
) -> dict[str, float]:
    """Dispatch to the right weighting scheme.

    method: 'vega_neutral' | 'spx_weight' | 'equal_vega'
    single_ivs required for 'vega_neutral'.
    """
    if method == "vega_neutral":
        if single_ivs is None:
            raise ValueError("single_ivs required for vega_neutral weighting")
        return vega_neutral_weights(single_ivs)
    elif method == "spx_weight":
        return spx_weight_matched(tickers)
    elif method == "equal_vega":
        return equal_vega_weights(tickers)
    else:
        raise ValueError(f"Unknown basket weight method: {method!r}")


def vega_notional_to_contracts(
    target_vega_notional: float,
    single_ivs: dict[str, float],
    weights: dict[str, float],
    vega_per_contract: dict[str, float],
) -> dict[str, int]:
    """Convert a target vega notional (USD) to contract counts per name.

    The short-index leg has notional vega = target_vega_notional.
    Each long-basket leg is sized so that Σ_i w_i * n_i * vega_i = target_vega_notional.

    vega_per_contract: dict mapping ticker → vega per contract (from Module 2).
    Returns: dict ticker → number of contracts (floored to int).
    """
    contracts = {}
    for t, w in weights.items():
        leg_vega_notional = w * target_vega_notional
        vpc = vega_per_contract.get(t, np.nan)
        if np.isnan(vpc) or vpc <= 0:
            contracts[t] = 0
        else:
            contracts[t] = max(1, int(leg_vega_notional / vpc))
    return contracts


__all__ = [
    "vega_neutral_weights",
    "spx_weight_matched",
    "equal_vega_weights",
    "get_weights",
    "vega_notional_to_contracts",
]
