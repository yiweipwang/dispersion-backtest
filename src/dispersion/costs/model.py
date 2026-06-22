"""Module 10 — Cost model.

Per-leg costs in USD per fill.

Option leg:
    half_spread        = (ask - bid) / 2 * n_contracts * multiplier
    commission         = $0.65 per contract per side
    sec_occ_fee        = $0.04 per contract on sells
    vega_financing     = margin posting * SOFR per overnight

Stock leg (delta hedge):
    half_spread        = from intraday NBBO last print in auction window
    auction_impact     = alpha * sqrt(participation) * notional
    commission/fees    = $0.005 per share + SEC/FINRA constant

Target total round-trip: 40–80 bps on the option leg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

COMMISSION_PER_CONTRACT = 0.65      # USD per side
SEC_OCC_FEE_SELL        = 0.04      # USD per contract on sells
STOCK_COMMISSION        = 0.005     # USD per share
STOCK_FINRA_FEE         = 0.000119  # per share (FINRA TAF)
MARKET_IMPACT_ALPHA     = 0.10      # sqrt-participation coefficient


# ── Option-leg costs ───────────────────────────────────────────────────────────

def option_leg_cost(
    bid: float,
    ask: float,
    n_contracts: int,
    multiplier: int = 100,
    is_sell: bool = False,
) -> dict[str, float]:
    """Cost breakdown for one option-leg fill."""
    half_spread  = (ask - bid) / 2.0 * n_contracts * multiplier
    commission   = COMMISSION_PER_CONTRACT * n_contracts
    sec_occ      = SEC_OCC_FEE_SELL * n_contracts if is_sell else 0.0
    total        = half_spread + commission + sec_occ
    return {
        "half_spread": half_spread,
        "commission":  commission,
        "sec_occ":     sec_occ,
        "total":       total,
    }


def vega_financing_cost(
    vega_notional: float,
    sofr: float,           # annual rate, e.g. 0.005
    n_days: int = 1,
) -> float:
    """Financing cost on vega margin posting (SOFR * notional * days/365)."""
    return vega_notional * sofr * n_days / 365.0


# ── Stock-leg costs ────────────────────────────────────────────────────────────

def nbbo_half_spread(
    intraday_nbbo: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
) -> float:
    """Last NBBO half-spread in the auction window for `ticker` on `date`."""
    mask = (intraday_nbbo["date"] == date) & (intraday_nbbo["ticker"] == ticker)
    df = intraday_nbbo[mask]
    if df.empty:
        return 0.005   # fallback: 0.5c default
    last = df.sort_values("timestamp").iloc[-1]
    bid = last.get("bid", last.get("nbbo_bid", np.nan))
    ask = last.get("ask", last.get("nbbo_ask", np.nan))
    if np.isnan(bid) or np.isnan(ask):
        return 0.005
    return float((ask - bid) / 2.0)


def auction_market_impact(
    shares: float,
    auction_volume: float,
    price: float,
    alpha: float = MARKET_IMPACT_ALPHA,
) -> float:
    """sqrt-participation impact: alpha * sqrt(|shares| / auction_volume) * notional."""
    if auction_volume <= 0 or shares == 0:
        return 0.0
    participation = abs(shares) / auction_volume
    impact_bps = alpha * np.sqrt(participation)
    notional = abs(shares) * price
    return impact_bps * notional


def stock_leg_cost(
    shares: float,
    price: float,
    half_spread: float,
    auction_volume: float,
    alpha: float = MARKET_IMPACT_ALPHA,
    is_sell: bool = False,
) -> dict[str, float]:
    """Cost breakdown for one stock-leg fill."""
    spread_cost   = half_spread * abs(shares)
    impact_cost   = auction_market_impact(shares, auction_volume, price, alpha)
    commission    = STOCK_COMMISSION * abs(shares)
    finra_fee     = STOCK_FINRA_FEE * abs(shares) if is_sell else 0.0
    total         = spread_cost + impact_cost + commission + finra_fee
    return {
        "half_spread": spread_cost,
        "impact":      impact_cost,
        "commission":  commission,
        "finra_fee":   finra_fee,
        "total":       total,
    }


# ── Aggregated cost per trade ──────────────────────────────────────────────────

def total_trade_cost(
    option_legs: list[dict],          # [{bid, ask, n_contracts, multiplier, is_sell}]
    stock_legs:  list[dict],          # [{shares, price, half_spread, auction_volume, is_sell}]
    vega_notional: float = 0.0,
    sofr: float = 0.005,
    n_days: int = 1,
) -> dict[str, float]:
    """Aggregate all cost components for a full trade."""
    opt_half_spread = 0.0
    opt_commission  = 0.0
    opt_sec_occ     = 0.0
    for leg in option_legs:
        c = option_leg_cost(**leg)
        opt_half_spread += c["half_spread"]
        opt_commission  += c["commission"]
        opt_sec_occ     += c["sec_occ"]

    stk_half_spread = 0.0
    stk_impact      = 0.0
    stk_commission  = 0.0
    stk_finra       = 0.0
    for leg in stock_legs:
        c = stock_leg_cost(**leg)
        stk_half_spread += c["half_spread"]
        stk_impact      += c["impact"]
        stk_commission  += c["commission"]
        stk_finra       += c["finra_fee"]

    financing = vega_financing_cost(vega_notional, sofr, n_days)
    total = (opt_half_spread + opt_commission + opt_sec_occ
             + stk_half_spread + stk_impact + stk_commission + stk_finra
             + financing)

    return {
        "opt_half_spread": opt_half_spread,
        "opt_commission":  opt_commission,
        "opt_sec_occ":     opt_sec_occ,
        "stk_half_spread": stk_half_spread,
        "stk_impact":      stk_impact,
        "stk_commission":  stk_commission,
        "stk_finra":       stk_finra,
        "financing":       financing,
        "total":           total,
    }


__all__ = [
    "option_leg_cost",
    "stock_leg_cost",
    "total_trade_cost",
    "vega_financing_cost",
    "nbbo_half_spread",
    "auction_market_impact",
]
