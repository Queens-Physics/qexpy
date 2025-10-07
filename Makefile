.PHONY: format test

format:
	ruff format
	ruff check --fix

test:
	uv run pytest --cov=qexpy --cov-report=html --doctest-modules
