"""Module 14 — Per-variant tearsheet generator.

Produces a multi-page PDF per variant with:
    1. Headline stats
    2. Cumulative net P&L + gross overlay
    3. Drawdown profile
    4. Daily P&L distribution
    5. Cost-attribution waterfall
    6. Variant-specific diagnostic (spread / correlation / GEX time series)
    7. Risk panel (vega, stress, VaR/ES)
    8. Validation gate summary
    9. Caveats page

Also produces a runs/comparison.pdf comparing all variants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from dispersion.pnl.attribution import compute_headline_stats, attribution_waterfall
from dispersion.risk.metrics import rolling_max_drawdown, risk_summary


VARIANT_CAVEATS = {
    "v1": (
        "V1 — Classic Variance Dispersion caveats:\n\n"
        "• 4-name basket covers <3% of SPX by weight; realized correlation "
        "will be noisier than a full-index dispersion book.\n"
        "• VS pricer uses discrete-strike OTM integration; accuracy degrades "
        "at wide bid-ask spreads (typical in Jan 2022 stressed market).\n"
        "• Jan 2022 sample is a single risk-off month — tail-correlation stress "
        "materialised repeatedly; Sharpe will understate steady-state capacity.\n"
        "• 4 months of EOD IV history (Oct 2021–Jan 2022) means rolling estimates "
        "at the start of January have limited pre-sample data."
    ),
    "v2": (
        "V2 — Replicated Correlation Dispersion caveats:\n\n"
        "• The OTM strip replication uses listed SPX/SPXW options at EOD mid; "
        "intraday execution would face additional bid-ask impact on each leg.\n"
        "• Delta rebalancing is done daily at close auction — real execution "
        "would use intraday triggers.\n"
        "• Same 4-name basket limitation and IV-history constraint as V1."
    ),
    "v5": (
        "V5 — Realized-vs-Implied Correlation caveats:\n\n"
        "• The 21-day rolling realized correlation with only 4 names has high "
        "estimation noise; a 500-name basket would have a much tighter estimate.\n"
        "• The correlation premium threshold of 10pp is calibrated on this sample "
        "only — requires walk-forward validation on a longer history.\n"
        "• V5 and V1 share the same trade mechanic; their P&L will be correlated."
    ),
    "v6": (
        "V6 — 0DTE-Driven Dispersion caveats:\n\n"
        "• Dealer GEX computed from EOD chain OI; intraday OI shifts are not "
        "captured between prints.\n"
        "• Lee-Ready sign inference on OPRA tapes is approximate; true dealer "
        "inventory requires prime-broker data.\n"
        "• 0DTE-flagged days in Jan 2022 are limited (VIX spike events); the "
        "strategy fires infrequently, producing a small-N result.\n"
        "• Overnight hold means exposure to gap risk — not modelled explicitly."
    ),
}

UNIVERSAL_CAVEATS = (
    "\nUniversal caveats (all variants):\n"
    "• Jan 2022 backtest: 21 trading days. No statistical significance.\n"
    "• 4-name basket (~3% SPX weight) is a crude approximation of full dispersion.\n"
    "• No market-impact model on option legs (assume mid fills throughout).\n"
    "• No short-gamma/jump risk modelling; all P&L is daily close-to-close.\n"
    "• V4 (sector dispersion) is deferred — sector ETF + constituent data not staged."
)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)


def _add_entry_exit_markers(ax: plt.Axes, signals: pd.DataFrame) -> None:
    entries = signals[signals["signal"] == 1]["date"]
    exits   = signals[signals["signal"] == -1]["date"]
    for d in entries:
        ax.axvline(x=d, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    for d in exits:
        ax.axvline(x=d, color="red", linestyle="--", linewidth=0.8, alpha=0.6)


# ── Page builders ──────────────────────────────────────────────────────────────

def _page_headline(
    fig: plt.Figure,
    variant: str,
    title: str,
    stats: dict,
    waterfall: pd.DataFrame,
) -> None:
    """Page 1: Headline stats + attribution waterfall."""
    fig.clf()
    ax_stats = fig.add_axes([0.05, 0.55, 0.9, 0.35])
    ax_stats.axis("off")
    ax_stats.set_title(f"{title}\nHeadline Statistics", fontsize=14, fontweight="bold", pad=10)

    lines = [
        f"Net Sharpe: {stats.get('sharpe', 'N/A')}",
        f"Max Drawdown: ${stats.get('max_dd', 'N/A'):,.0f}",
        f"Hit Rate: {stats.get('hit_rate', 'N/A'):.1%}",
        f"Sample: {stats.get('n_days', 0)} days, {stats.get('n_trades', 0)} trades",
        f"Total Net P&L: ${stats.get('total_net', 0):,.0f}",
    ]
    ax_stats.text(0.5, 0.5, "\n".join(lines), transform=ax_stats.transAxes,
                  ha="center", va="center", fontsize=12, fontfamily="monospace")

    # Waterfall table
    ax_table = fig.add_axes([0.05, 0.05, 0.9, 0.45])
    ax_table.axis("off")
    if not waterfall.empty:
        cols  = ["Line", "bps_annualised", "usd"]
        table = ax_table.table(
            cellText=waterfall[cols].values,
            colLabels=["P&L Line", "bps (ann.)", "USD"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)


def _page_pnl_curves(
    fig: plt.Figure,
    daily_pnl: pd.DataFrame,
    signals: pd.DataFrame,
) -> None:
    """Pages 2 & 3: Cumulative P&L + drawdown."""
    fig.clf()
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    pnl = daily_pnl.sort_values("date")
    ax1.plot(pnl["date"], pnl["gross_pnl"].cumsum(), label="Gross alpha", color="#1f77b4", linewidth=1.2)
    ax1.plot(pnl["date"], pnl["net_pnl"].cumsum(),   label="Net alpha",   color="#2ca02c", linewidth=1.5)
    _add_entry_exit_markers(ax1, signals)
    ax1.legend(fontsize=8)
    _style_ax(ax1, "Cumulative P&L", ylabel="USD")

    ax2 = fig.add_subplot(gs[1])
    dd = rolling_max_drawdown(pnl["net_pnl"])
    ax2.fill_between(pnl["date"], dd.values, 0, color="#d62728", alpha=0.5)
    _style_ax(ax2, "Drawdown", ylabel="USD")


def _page_distribution(fig: plt.Figure, daily_pnl: pd.DataFrame) -> None:
    """Page 4: Daily P&L distribution."""
    fig.clf()
    ax = fig.add_subplot(111)
    pnl = daily_pnl["net_pnl"].dropna()
    ax.hist(pnl, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(pnl.mean(), color="green", linestyle="--", label=f"Mean: ${pnl.mean():,.0f}")
    ax.axvline(pnl.quantile(0.01), color="red", linestyle="--",
               label=f"1% VaR: ${pnl.quantile(0.01):,.0f}")
    ax.legend(fontsize=8)
    _style_ax(ax, "Daily Net P&L Distribution", xlabel="USD", ylabel="Frequency")


def _page_diagnostic(
    fig: plt.Figure,
    variant: str,
    signals: pd.DataFrame,
    surface_df: pd.DataFrame | None = None,
    vs_basket: pd.DataFrame | None = None,
) -> None:
    """Page 6: Variant-specific diagnostic."""
    fig.clf()
    ax = fig.add_subplot(111)

    if variant in ("v1", "v2") and vs_basket is not None and "vs_spread_vol" in vs_basket.columns:
        vs = vs_basket.sort_values("date")
        ax.plot(vs["date"], vs["vs_spread_vol"], color="#9467bd", linewidth=1.2)
        ax.axhline(signals.get("entry_threshold", 2.0) if isinstance(signals, dict) else 2.0,
                   color="green", linestyle="--", linewidth=0.8, label="Entry threshold")
        _add_entry_exit_markers(ax, signals if isinstance(signals, pd.DataFrame) else pd.DataFrame())
        _style_ax(ax, f"{variant.upper()} VS Spread (vol points)", ylabel="Vol pts")

    elif variant == "v5" and surface_df is not None:
        surf = surface_df.sort_values("date")
        if "implied_corr" in surf.columns:
            ax.plot(surf["date"], surf["implied_corr"],   label="ρ implied",  color="#1f77b4")
        if "realized_corr" in surf.columns:
            ax.plot(surf["date"], surf["realized_corr"],  label="ρ realized", color="#ff7f0e")
        if "corr_premium" in surf.columns:
            ax2 = ax.twinx()
            ax2.bar(surf["date"], surf["corr_premium"], alpha=0.3, color="#2ca02c",
                    label="Premium")
            ax2.set_ylabel("Corr premium", fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        _style_ax(ax, "V5 Implied vs Realized Correlation", ylabel="Correlation")

    elif variant == "v6" and isinstance(signals, pd.DataFrame) and "dealer_gex" in signals.columns:
        sig = signals.sort_values("date")
        ax.bar(sig["date"], sig["dealer_gex"], color="#8c564b", alpha=0.7)
        _add_entry_exit_markers(ax, signals)
        _style_ax(ax, "V6 Dealer GEX + Signal Markers", ylabel="GEX (normalised)")

    else:
        ax.text(0.5, 0.5, "Diagnostic data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)


def _page_risk(fig: plt.Figure, daily_pnl: pd.DataFrame, risk: dict) -> None:
    """Page 7: Risk panel."""
    fig.clf()
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # VaR/ES stats
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    lines = [
        f"1% 1-day VaR:  ${risk.get('var_1pct_1d', 0):,.0f}",
        f"1% 1-day ES:   ${risk.get('es_1pct_1d', 0):,.0f}",
        f"Max Drawdown:  ${risk.get('max_drawdown', 0):,.0f}",
        f"Worst Day:     ${risk.get('worst_day', 0):,.0f}",
        f"Skew:  {risk.get('pnl_skew', 0):.2f}    Kurt: {risk.get('pnl_kurt', 0):.2f}",
    ]
    ax0.text(0.1, 0.5, "\n".join(lines), transform=ax0.transAxes,
             va="center", fontsize=10, fontfamily="monospace")
    ax0.set_title("Risk Summary", fontsize=10, fontweight="bold")

    # Cumulative P&L rolling std (proxy for regime)
    ax1 = fig.add_subplot(gs[0, 1])
    pnl = daily_pnl["net_pnl"].dropna()
    rolling_vol = pnl.rolling(5).std() * np.sqrt(252)
    ax1.plot(daily_pnl["date"].iloc[-len(rolling_vol):], rolling_vol.values,
             color="#d62728")
    _style_ax(ax1, "Rolling 5d Annualised Vol of P&L", ylabel="Vol")

    # Daily P&L bars
    ax2 = fig.add_subplot(gs[1, :])
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in daily_pnl["net_pnl"].fillna(0)]
    ax2.bar(daily_pnl["date"], daily_pnl["net_pnl"].fillna(0), color=colours, alpha=0.8)
    _style_ax(ax2, "Daily Net P&L", ylabel="USD")


def _page_validation(fig: plt.Figure, gate_results: dict) -> None:
    """Page 8: Validation gate summary."""
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Validation Gate Results", fontsize=14, fontweight="bold")

    rows = []
    for gate, (passed, details) in gate_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        detail_str = "  |  ".join(f"{k}: {v}" for k, v in details.items()
                                   if k not in ("error",))
        rows.append([gate, status, detail_str[:80]])

    table = ax.table(
        cellText=rows,
        colLabels=["Gate", "Status", "Details"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    # Colour pass/fail cells
    for i, (_, (passed, _)) in enumerate(gate_results.items()):
        colour = "#c8e6c9" if passed else "#ffcdd2"
        table[(i + 1, 1)].set_facecolor(colour)


def _page_caveats(fig: plt.Figure, variant: str) -> None:
    """Page 9: Caveats."""
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis("off")
    text = VARIANT_CAVEATS.get(variant, "") + UNIVERSAL_CAVEATS
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            va="top", ha="left", fontsize=9, wrap=True,
            fontfamily="monospace")
    ax.set_title("Caveats & Limitations", fontsize=12, fontweight="bold")


# ── Main tearsheet builder ─────────────────────────────────────────────────────

def generate_tearsheet(
    variant: str,
    title: str,
    daily_pnl: pd.DataFrame,
    signals: pd.DataFrame,
    gate_results: dict,
    output_path: Path,
    surface_df: pd.DataFrame | None = None,
    vs_basket: pd.DataFrame | None = None,
) -> None:
    """Generate a multi-page PDF tearsheet for one variant."""
    from matplotlib.backends.backend_pdf import PdfPages

    stats     = compute_headline_stats(daily_pnl)
    waterfall = attribution_waterfall(daily_pnl)
    risk      = risk_summary(daily_pnl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8.5))

    with PdfPages(str(output_path)) as pdf:
        _page_headline(fig, variant, title, stats, waterfall)
        pdf.savefig(fig, bbox_inches="tight")

        _page_pnl_curves(fig, daily_pnl, signals)
        pdf.savefig(fig, bbox_inches="tight")

        _page_distribution(fig, daily_pnl)
        pdf.savefig(fig, bbox_inches="tight")

        _page_diagnostic(fig, variant, signals, surface_df, vs_basket)
        pdf.savefig(fig, bbox_inches="tight")

        _page_risk(fig, daily_pnl, risk)
        pdf.savefig(fig, bbox_inches="tight")

        _page_validation(fig, gate_results)
        pdf.savefig(fig, bbox_inches="tight")

        _page_caveats(fig, variant)
        pdf.savefig(fig, bbox_inches="tight")

    plt.close(fig)
    print(f"Tearsheet written to {output_path}")


# ── Comparison page ────────────────────────────────────────────────────────────

def generate_comparison(
    variant_results: dict[str, dict],   # variant → {stats, daily_pnl}
    output_path: Path,
) -> None:
    """Single-page comparison PDF across all variants."""
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(14, 8.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    variants = list(variant_results.keys())
    colours  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Cumulative net P&L
    ax_cum = fig.add_subplot(gs[0, :])
    for i, v in enumerate(variants):
        pnl = variant_results[v]["daily_pnl"]["net_pnl"].cumsum()
        ax_cum.plot(variant_results[v]["daily_pnl"]["date"], pnl,
                    label=v.upper(), color=colours[i % len(colours)], linewidth=1.5)
    ax_cum.legend(fontsize=9)
    _style_ax(ax_cum, "Cumulative Net P&L — All Variants", ylabel="USD")

    # Summary table
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    rows = []
    for v in variants:
        s = variant_results[v]["stats"]
        rows.append([
            v.upper(),
            str(s.get("sharpe", "N/A")),
            f"${s.get('max_dd', 0):,.0f}",
            f"{s.get('hit_rate', 0):.1%}",
            f"${s.get('total_net', 0):,.0f}",
            str(s.get("n_trades", 0)),
        ])
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Variant", "Sharpe", "Max DD", "Hit Rate", "Total Net P&L", "N Trades"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison page written to {output_path}")


__all__ = [
    "generate_tearsheet",
    "generate_comparison",
]
