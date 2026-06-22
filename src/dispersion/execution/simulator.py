"""Module 9 — Option fill simulator + position evolution.

Fills at chain mid-price (or bid/ask as configured).
Marks-to-market daily at chain mid.
Delta-hedges in the underlying at the close auction price.
Handles two trade cadences:
    - Monthly: V1, V2, V5 (roll when DTE < roll_dte_threshold)
    - Overnight: V6 (enter MOC, exit MOO next day)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from dispersion.greeks.bs import bs_delta


# ── Position dataclass ─────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    """Represents a single option position."""
    ticker: str
    expiry: pd.Timestamp
    strike: float
    right: str          # 'C' or 'P'
    n_contracts: int
    direction: int      # +1 long, -1 short
    entry_price: float  # mid at fill
    current_price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    multiplier: int = 100


@dataclass
class EquityLeg:
    """Stock position used for delta hedge."""
    ticker: str
    shares: float
    direction: int
    entry_price: float
    current_price: float = 0.0


@dataclass
class Position:
    """Full position for one variant on one date."""
    trade_id: int
    variant: str
    entry_date: pd.Timestamp
    option_legs: list[OptionLeg] = field(default_factory=list)
    equity_legs: list[EquityLeg] = field(default_factory=list)
    closed: bool = False
    exit_date: pd.Timestamp | None = None


# ── Fill helper ────────────────────────────────────────────────────────────────

def _fill_price(
    chain_slice: pd.DataFrame,
    ticker: str,
    expiry: pd.Timestamp,
    strike: float,
    right: str,
    price_mode: Literal["mid", "bid", "ask"] = "mid",
) -> tuple[float, float, float, float]:
    """Return (fill_price, delta, gamma, vega) for the contract.

    Returns (nan, nan, nan, nan) if the contract is not found.
    """
    mask = (
        (chain_slice["ticker"] == ticker)
        & (chain_slice["expiry"] == expiry)
        & (chain_slice["strike"] == strike)
        & (chain_slice["right"] == right)
    )
    row = chain_slice[mask]
    if row.empty:
        return np.nan, np.nan, np.nan, np.nan

    r = row.iloc[0]
    price = r.get(price_mode, r.get("mid", np.nan))
    delta = r.get("delta", r.get("delta_recomp", np.nan))
    gamma = r.get("gamma_recomp", r.get("gamma", np.nan))
    vega  = r.get("vega_recomp",  r.get("vega",  np.nan))
    return float(price), float(delta), float(gamma), float(vega)


# ── Execution simulator ────────────────────────────────────────────────────────

class ExecutionSimulator:
    """Simulates option fills, daily MTM, and delta hedging."""

    def __init__(
        self,
        fill_mode: Literal["mid", "bid", "ask"] = "mid",
        multiplier: int = 100,
        participation_cap: float = 0.10,
    ):
        self.fill_mode = fill_mode
        self.multiplier = multiplier
        self.participation_cap = participation_cap

    def open_position(
        self,
        trade_id: int,
        variant: str,
        date: pd.Timestamp,
        legs: list[dict],          # [{ticker, expiry, strike, right, n_contracts, direction}]
        chain: pd.DataFrame,
    ) -> Position:
        """Create a Position from leg specs, filling at chain mid."""
        pos = Position(trade_id=trade_id, variant=variant, entry_date=date)
        for leg in legs:
            price, delta, gamma, vega = _fill_price(
                chain, leg["ticker"], leg["expiry"],
                leg["strike"], leg["right"], self.fill_mode,
            )
            ol = OptionLeg(
                ticker=leg["ticker"],
                expiry=leg["expiry"],
                strike=leg["strike"],
                right=leg["right"],
                n_contracts=leg["n_contracts"],
                direction=leg["direction"],
                entry_price=price,
                current_price=price,
                delta=delta if not np.isnan(delta) else 0.0,
                gamma=gamma if not np.isnan(gamma) else 0.0,
                vega=vega  if not np.isnan(vega)  else 0.0,
                multiplier=self.multiplier,
            )
            pos.option_legs.append(ol)
        return pos

    def mark_to_market(
        self,
        position: Position,
        date: pd.Timestamp,
        chain: pd.DataFrame,
    ) -> float:
        """Mark all option legs to mid; return day's option MTM P&L (USD)."""
        total_pnl = 0.0
        for leg in position.option_legs:
            price, delta, gamma, vega = _fill_price(
                chain, leg.ticker, leg.expiry,
                leg.strike, leg.right, "mid",
            )
            if np.isnan(price):
                continue
            prev_price = leg.current_price
            pnl = (price - prev_price) * leg.direction * leg.n_contracts * leg.multiplier
            leg.current_price = price
            leg.delta = delta if not np.isnan(delta) else leg.delta
            leg.gamma = gamma if not np.isnan(gamma) else leg.gamma
            leg.vega  = vega  if not np.isnan(vega)  else leg.vega
            total_pnl += pnl
        return total_pnl

    def delta_hedge(
        self,
        position: Position,
        date: pd.Timestamp,
        equity_prices: dict[str, float],   # ticker → close auction price
        auction_volume: dict[str, float],  # ticker → total auction volume (shares)
    ) -> tuple[list[EquityLeg], float]:
        """Compute delta-hedge trades.  Returns (new equity legs, hedge cost USD)."""
        # Net delta per ticker across all option legs
        net_delta: dict[str, float] = {}
        for leg in position.option_legs:
            t = leg.ticker
            d = leg.delta * leg.direction * leg.n_contracts * leg.multiplier
            net_delta[t] = net_delta.get(t, 0.0) + d

        new_legs = []
        hedge_cost = 0.0
        for t, target_delta in net_delta.items():
            # We want to hedge to delta-zero → trade -target_delta shares
            shares_needed = -target_delta

            # Participation cap
            avail = auction_volume.get(t, np.inf)
            capped_shares = np.clip(
                abs(shares_needed),
                0,
                avail * self.participation_cap,
            ) * np.sign(shares_needed)

            px = equity_prices.get(t, np.nan)
            if np.isnan(px) or px <= 0:
                continue

            eq = EquityLeg(
                ticker=t,
                shares=capped_shares,
                direction=int(np.sign(capped_shares)),
                entry_price=px,
                current_price=px,
            )
            new_legs.append(eq)
            hedge_cost += abs(capped_shares) * px  # notional (cost accounted in costs module)

        return new_legs, hedge_cost

    def close_position(
        self,
        position: Position,
        date: pd.Timestamp,
        chain: pd.DataFrame,
    ) -> float:
        """Close all legs at mid; return exit P&L."""
        pnl = self.mark_to_market(position, date, chain)
        position.closed = True
        position.exit_date = date
        return pnl


__all__ = [
    "OptionLeg",
    "EquityLeg",
    "Position",
    "ExecutionSimulator",
]
