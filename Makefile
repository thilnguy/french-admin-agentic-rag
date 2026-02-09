.PHONY: install test lint format run docker-build

install:
	uv sync

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run ruff format .

run:
	uv run uvicorn src.main:app --reload

docker-build:
	docker build -t french-admin-agent .

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
