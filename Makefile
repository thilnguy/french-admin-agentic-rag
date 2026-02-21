.PHONY: install test lint format run docker-build clean clean-logs eval

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

eval:
	uv run python evals/llm_judge.py

clean-logs:
	rm -f *.log *.txt evals/*.log evals/*.txt
	rm -f finetuning/data/train.jsonl finetuning/data/valid.jsonl

clean: clean-logs
	rm -rf .venv .pytest_cache .ruff_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
