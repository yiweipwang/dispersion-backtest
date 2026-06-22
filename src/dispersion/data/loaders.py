"""Module 1 — Data layer.

All loaders return clean, typed, point-in-time DataFrames.
Types are normalised at the boundary; no magic downstream.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

# ── Path resolution ────────────────────────────────────────────────────────────

_DEFAULT_ROOT = Path("Sample Data") / "jan2022_5ticker"


def _data_root() -> Path:
    env = os.environ.get("DISPERSION_DATA_DIR")
    return Path(env) if env else _DEFAULT_ROOT


def _path(filename: str) -> Path:
    return _data_root() / filename


# ── File name registry (matches base.yaml) ─────────────────────────────────────

_FILES = {
    "option_chain_eod":      "option_chain_eod_oct2021_jan2022.parquet",
    "option_trades_nbbo":    "option_trades_with_nbbo_jan2022.parquet",
    "option_trades_greeks":  "option_trades_with_greeks_jan2022.parquet",
    "underlier_mapping":     "option_underlier_ticker_mapping.parquet",
    "equity_daily":          "equity_daily_2021_jan2022.parquet",
    "stock_daily_close":     "stock_daily_close_2022.parquet",
    "auction_window_trades": "auction_window_trades_jan2022.parquet",
    "auction_volume_summary":"auction_volume_summary_jan2022.parquet",
    "intraday_nbbo":         "intraday_nbbo_close_open_windows_jan2022.parquet",
    "earnings_calendar":     "earnings_calendar_jan2022.parquet",
    "macro_regime":          "macro_regime_jan2022.parquet",
    "vix9d":                 "vix9d_jan2022.parquet",
    "sp500_constituents":    "index_constituents_sp500_jan2022.parquet",
}


def _read(key: str, **kwargs) -> pd.DataFrame:
    p = _path(_FILES[key])
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found: {p}\n"
            f"Set DISPERSION_DATA_DIR env var or place files at {_data_root()}"
        )
    return pd.read_parquet(p, **kwargs)


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _normalise_right(df: pd.DataFrame, col: str = "right") -> pd.DataFrame:
    """Standardise option right to 'C' / 'P'."""
    if col not in df.columns:
        return df
    mapping = {
        "CALL": "C", "call": "C", "Call": "C",
        "PUT":  "P", "put":  "P", "Put":  "P",
        "C": "C", "P": "P",
    }
    df = df.copy()
    df[col] = df[col].map(mapping).fillna(df[col])
    return df


def _ensure_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
    return df


def _ensure_strike_dollars(df: pd.DataFrame, col: str = "strike") -> pd.DataFrame:
    """Strike should be in dollars.  If values look like they're in cents (> 50k),
    divide by 100.  This is a heuristic guard — verify against your data dict."""
    if col not in df.columns:
        return df
    df = df.copy()
    median_strike = df[col].median()
    if median_strike > 50_000:
        df[col] = df[col] / 100.0
    return df


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_option_chain_eod(date_filter: str | None = None) -> pd.DataFrame:
    """Daily EOD option chain (Oct 2021 → Jan 2022).

    Columns (after normalisation):
        date, underlier_id, ticker, expiry, strike, right,
        bid, ask, mid, iv, open_interest, volume,
        gamma (vendor), vega (vendor), delta (vendor), ...
    """
    df = _read("option_chain_eod")
    df = _ensure_date(df, "date")
    df = _ensure_date(df, "expiry")
    df = _normalise_right(df)
    df = _ensure_strike_dollars(df)

    # Compute mid if not present
    if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # DTE (calendar days)
    if "dte" not in df.columns:
        df["dte"] = (df["expiry"] - df["date"]).dt.days

    # SPX vs SPXW flag
    if "ticker" in df.columns:
        df["is_spxw"] = df["ticker"].str.upper().str.contains("SPXW")

    if date_filter is not None:
        cutoff = pd.Timestamp(date_filter)
        df = df[df["date"] >= cutoff].copy()

    return df.reset_index(drop=True)


def load_option_trades_with_nbbo(date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Intraday OPRA option trades + NBBO (Jan 2022).

    Used for Lee-Ready trade-sign inference (V6 dealer-flow modelling).
    """
    df = _read("option_trades_nbbo")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.normalize()
    df = _normalise_right(df)

    if date is not None:
        d = pd.Timestamp(date).normalize()
        df = df[df["date"] == d].copy()

    return df.reset_index(drop=True)


def load_option_trades_with_greeks(date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-trade option Greeks (Jan 2022).  Gamma column absent — recompute via BS."""
    df = _read("option_trades_greeks")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.normalize()
    df = _normalise_right(df)
    df = _ensure_strike_dollars(df)

    if date is not None:
        d = pd.Timestamp(date).normalize()
        df = df[df["date"] == d].copy()

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_underlier_mapping() -> pd.DataFrame:
    """Map underlier_id ↔ ticker (5 underliers)."""
    df = _read("underlier_mapping")
    df.columns = df.columns.str.lower()
    return df


def load_equity_daily(date_filter: str | None = None) -> pd.DataFrame:
    """15 months of daily OHLC + total_return for the 4 single names.

    date_filter: if supplied, return rows with date >= date_filter.
    """
    df = _read("equity_daily")
    df = _ensure_date(df)
    df.columns = df.columns.str.lower()

    # Compute log return from total_return if log_ret absent
    if "log_ret" not in df.columns and "total_return" in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["log_ret"] = df.groupby("ticker")["total_return"].transform(
            lambda s: (1 + s).apply(pd.np.log) if hasattr(pd, "np") else _log1p_series(s)
        )

    if date_filter is not None:
        df = df[df["date"] >= pd.Timestamp(date_filter)].copy()

    return df.reset_index(drop=True)


def _log1p_series(s: pd.Series) -> pd.Series:
    import numpy as np
    return np.log1p(s)


def load_stock_daily_close() -> pd.DataFrame:
    """Independent full-year-2022 close-price feed (cross-check)."""
    df = _read("stock_daily_close")
    df = _ensure_date(df)
    df.columns = df.columns.str.lower()
    return df.reset_index(drop=True)


def load_auction_volume_summary() -> pd.DataFrame:
    """Per-(ticker, date, side) auction volume + VWAP."""
    df = _read("auction_volume_summary")
    df = _ensure_date(df)
    df.columns = df.columns.str.lower()
    return df.reset_index(drop=True)


def load_intraday_nbbo(
    date: str | pd.Timestamp | None = None,
    side: Literal["close", "open"] = "close",
) -> pd.DataFrame:
    """Millisecond NBBO snapshots around open/close windows (single names)."""
    df = _read("intraday_nbbo")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.normalize()
    df = _ensure_date(df)
    df.columns = df.columns.str.lower()

    if "window" in df.columns:
        df = df[df["window"] == side].copy()

    if date is not None:
        d = pd.Timestamp(date).normalize()
        df = df[df["date"] == d].copy()

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_earnings_calendar() -> pd.DataFrame:
    """Earnings dates: MSFT 1/25 AMC, TSLA 1/26 AMC, AAPL 1/27 AMC."""
    try:
        df = _read("earnings_calendar")
        df = _ensure_date(df)
        df.columns = df.columns.str.lower()
    except FileNotFoundError:
        # Hard-code the known dates from the brief
        df = pd.DataFrame({
            "ticker": ["MSFT", "TSLA", "AAPL"],
            "date":   pd.to_datetime(["2022-01-25", "2022-01-26", "2022-01-27"]),
            "time":   ["AMC", "AMC", "AMC"],
        })
    return df


@lru_cache(maxsize=1)
def load_macro_regime() -> pd.DataFrame:
    """One row per trading day: VIX, VIX3M, VIX9D, SOFR (joined view)."""
    macro = _read("macro_regime")
    macro = _ensure_date(macro)
    macro.columns = macro.columns.str.lower()

    try:
        vix9d = _read("vix9d")
        vix9d = _ensure_date(vix9d)
        vix9d.columns = vix9d.columns.str.lower()
        # keep only the close column; rename to vix9d_close
        close_col = [c for c in vix9d.columns if "close" in c]
        if close_col:
            vix9d = vix9d[["date", close_col[0]]].rename(
                columns={close_col[0]: "vix9d"}
            )
            macro = macro.merge(vix9d, on="date", how="left")
    except FileNotFoundError:
        pass

    return macro.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_sp500_constituents() -> pd.DataFrame:
    """SP500 membership table (universe-extension sanity check)."""
    df = _read("sp500_constituents")
    df.columns = df.columns.str.lower()
    return df


# ── Convenience re-exports ─────────────────────────────────────────────────────

__all__ = [
    "load_option_chain_eod",
    "load_option_trades_with_nbbo",
    "load_option_trades_with_greeks",
    "load_equity_daily",
    "load_stock_daily_close",
    "load_auction_volume_summary",
    "load_intraday_nbbo",
    "load_earnings_calendar",
    "load_macro_regime",
    "load_underlier_mapping",
    "load_sp500_constituents",
]
