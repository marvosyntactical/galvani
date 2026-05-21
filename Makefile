.PHONY: help install lint format typecheck test test-live refresh-fixtures clean

help:
	@echo "make install            install uv deps including extras + dev group"
	@echo "make lint               ruff check"
	@echo "make format             ruff format"
	@echo "make typecheck          mypy strict"
	@echo "make test               pytest, skips 'live' tests"
	@echo "make test-live          pytest including live tests (needs NEUPRINT_TOKEN)"
	@echo "make refresh-fixtures   re-pull HD-ring fixtures from neuPrint"

install:
	uv sync --all-extras --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not live" --cov=galvani --cov-report=term-missing

test-live:
	uv run pytest --cov=galvani --cov-report=term-missing

refresh-fixtures:
	@test -n "$$NEUPRINT_TOKEN" || (echo "NEUPRINT_TOKEN is not set" && exit 1)
	uv run python scripts/refresh_fixtures.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build dist *.egg-info
