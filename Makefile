PYTHON ?= python3.11
VENV   ?= .venv
BIN     = $(VENV)/bin

.PHONY: install app test test-all icon clean run

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -e ".[dev]"

app: icon
	./scripts/make_app.sh

icon:
	$(BIN)/python scripts/make_icon.py scripts/StemForge.icns

run:
	$(BIN)/stemforge serve

test:
	$(BIN)/python -m pytest tests -q

test-all:
	$(BIN)/python -m pytest tests -q -m "slow or not slow"

clean:
	rm -rf dist build *.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
