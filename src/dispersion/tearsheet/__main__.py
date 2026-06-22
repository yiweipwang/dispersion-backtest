"""Tearsheet CLI: python -m dispersion.tearsheet --variant v1  |  --all"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dispersion.tearsheet.generator import generate_tearsheet, generate_comparison


def _load_variant(variant: str) -> dict | None:
    out = Path("runs") / variant
    pnl_path = out / "daily_pnl.parquet"
    sig_path  = out / "signals.parquet"
    if not pnl_path.exists() or not sig_path.exists():
        print(f"  Skipping {variant}: run outputs not found at {out}/")
        return None

    daily_pnl  = pd.read_parquet(pnl_path)
    signals    = pd.read_parquet(sig_path)
    stats_path = out / "stats.json"
    stats      = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    gate_path  = out / "stats.json"
    gate_results = {}
    if gate_path.exists():
        raw = json.loads(gate_path.read_text()).get("gate_results", {})
        gate_results = {k: (v[0], v[1]) for k, v in raw.items()}

    surface_df = None
    vs_basket  = None
    surf_path = out / "surface.parquet"
    vs_path   = out / "vs_basket.parquet"
    if surf_path.exists():
        surface_df = pd.read_parquet(surf_path)
    if vs_path.exists():
        vs_basket = pd.read_parquet(vs_path)

    return {
        "daily_pnl":   daily_pnl,
        "signals":     signals,
        "stats":       stats,
        "gate_results": gate_results,
        "surface_df":  surface_df,
        "vs_basket":   vs_basket,
    }


TITLES = {
    "v1": "V1 — Classic Variance Dispersion",
    "v2": "V2 — Replicated Correlation Dispersion",
    "v5": "V5 — Realized vs Implied Correlation",
    "v6": "V6 — 0DTE-Driven Dispersion",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispersion tearsheet generator")
    parser.add_argument("--variant", choices=["v1", "v2", "v5", "v6"])
    parser.add_argument("--all", action="store_true", help="Generate tearsheets + comparison for all variants")
    args = parser.parse_args()

    if args.all:
        variant_results = {}
        for v in ["v1", "v2", "v5", "v6"]:
            data = _load_variant(v)
            if data is None:
                continue
            generate_tearsheet(
                variant=v,
                title=TITLES.get(v, v.upper()),
                daily_pnl=data["daily_pnl"],
                signals=data["signals"],
                gate_results=data["gate_results"],
                output_path=Path("runs") / v / "tearsheet.pdf",
                surface_df=data.get("surface_df"),
                vs_basket=data.get("vs_basket"),
            )
            variant_results[v] = {"stats": data["stats"], "daily_pnl": data["daily_pnl"]}

        if variant_results:
            generate_comparison(
                variant_results=variant_results,
                output_path=Path("runs") / "comparison.pdf",
            )
    elif args.variant:
        v = args.variant
        data = _load_variant(v)
        if data:
            generate_tearsheet(
                variant=v,
                title=TITLES.get(v, v.upper()),
                daily_pnl=data["daily_pnl"],
                signals=data["signals"],
                gate_results=data["gate_results"],
                output_path=Path("runs") / v / "tearsheet.pdf",
                surface_df=data.get("surface_df"),
                vs_basket=data.get("vs_basket"),
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
