.PHONY: install train test run fmt lint

install:
poetry install

train:
poetry run python -m src.model.train

test:
poetry run pytest -q

run:
poetry run python -m src.app.cli --help

fmt:
poetry run black .

lint:
poetry run ruff check .
