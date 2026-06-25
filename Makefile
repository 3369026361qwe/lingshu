install:
	pip install -e .
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

test-shujuku:
	python -m pytest tests/test_shujuku/ -v

test-coverage:
	python -m pytest tests/ -v --cov=. --cov-report=html

run:
	uvicorn jiekou.server:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check .

format:
	ruff format .
