.PHONY: install lint format-check typecheck test check

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format-check:
	ruff format --check .

typecheck:
	mypy

test:
	pytest

check: lint format-check typecheck test
