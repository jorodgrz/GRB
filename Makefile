# Test and validation targets for the GRB classification project.
#
# Quality gates (every PR):
#   make lint     -- ruff format check + ruff lint (zero warnings)
#   make typecheck -- mypy on grb_*.py
#   make smoke    -- fast pytest subset (no data, no compas pin), under ~5 s
#   make coverage -- smoke subset + 70 percent coverage floor on grb_*.py
#   make ci       -- the bundle CI runs on every push and pull request
#
# Heavier targets (run locally; manual dispatch in CI):
#   make test     -- full pytest suite (data-bound tests skip if Data/ empty)
#
# Conda is the canonical dependency manager (see environment.yml); the
# Makefile assumes the active environment already has pytest, ruff, mypy,
# and pytest-cov installed.

.PHONY: lint typecheck smoke coverage ci test clean

lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy src/grb_classify.py src/grb_io.py src/grb_offsets.py src/grb_physics.py src/grb_plot_style.py src/grb_rates.py

smoke:
	python -m pytest -m "not slow and not requires_data and not requires_compas" --no-header

coverage:
	python -m pytest -m "unit or anchors" --cov-fail-under=70 --tb=short

ci: lint typecheck smoke

test:
	python -m pytest

clean:
	find tests -type d -name __pycache__ -exec rm -rf {} +
	find . -maxdepth 2 -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .coverage htmlcov
