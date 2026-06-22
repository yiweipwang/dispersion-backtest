# Dispersion Backtest — Implementation Sprint (Week 2)

End-to-end multi-variant volatility dispersion backtest against the Jan 2022 5-ticker sample.
Implements V1 (classic variance dispersion), V2 (replicated correlation dispersion),
V5 (realized-vs-implied correlation), and V6 (0DTE-driven dispersion).

## Quick start

```bash
pip install -e .          # or: make install
make all                  # run all four required variants + comparison tearsheet
make v1                   # run V1 only
python -m dispersion.run --variant v1   # equivalent
python -m dispersion.tearsheet --all    # comparison PDF only
```

## Data

Place the staged parquet bundle at `Sample Data/jan2022_5ticker/` (default).
Override with `--data-dir <path>` or set `DATA_DIR` in the environment.

## Structure

```
config/          YAML configs — base.yaml + per-variant overrides
src/dispersion/  Source (14 modules — see §6 of the brief)
tests/           pytest validation gates (all 6 required)
runs/            Output per variant (gitignored)
```

## Reproducibility

All randomness is seeded via `base.yaml:seed`. `make all` completes in < 60 min on a laptop.
Each run writes `runs/<variant>/run_metadata.json` with git SHA, config hash, data hashes, timestamp.

## Variants

| Key | Name | Status |
|-----|------|--------|
| v1  | Classic variance dispersion | Required |
| v2  | Replicated correlation dispersion | Required |
| v5  | Realized-vs-implied correlation | Required |
| v6  | 0DTE-driven dispersion | Required |
| v3  | Vol-of-vol dispersion | Deferred (optional stretch — scaffolded) |
| v4  | Sector dispersion | Deferred — sector ETF data not in bundle |
