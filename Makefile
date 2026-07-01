PYTHON ?= python

.PHONY: lint test run run-api mlflow

lint:
	ruff check .
	ruff format .

test:
	pytest

run:
	$(PYTHON) -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

run-api:
	$(PYTHON) -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

mlflow:
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000