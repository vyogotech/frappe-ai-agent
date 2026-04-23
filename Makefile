.PHONY: install test lint format typecheck serve clean

install:
	uv sync --all-extras

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright src/

serve:
	uv run uvicorn ai_agent.app:create_app --factory --host 0.0.0.0 --port 8484 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
