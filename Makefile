.PHONY: all v1 v2 v5 v6 data backtest tearsheet install test clean

DATA_DIR ?= Sample\ Data/jan2022_5ticker
PYTHON    ?= python

install:
	pip install -e ".[dev]" --break-system-packages

data:
	$(PYTHON) -m dispersion.run --variant v1 --stage data

v1:
	$(PYTHON) -m dispersion.run --variant v1

v2:
	$(PYTHON) -m dispersion.run --variant v2

v5:
	$(PYTHON) -m dispersion.run --variant v5

v6:
	$(PYTHON) -m dispersion.run --variant v6

all: v1 v2 v5 v6
	$(PYTHON) -m dispersion.tearsheet --all

tearsheet:
	$(PYTHON) -m dispersion.tearsheet --variant $(VARIANT)

test:
	pytest tests/ -v

clean:
	rm -rf runs/*/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
