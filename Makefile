.PHONY: install lint format test api ui mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

api:
	uv run uvicorn timegrid.api.main:app --reload --port 8450

ui:
	TIMEGRID_API_URL=http://localhost:8450 uv run streamlit run src/timegrid/ui/app.py --server.port 8951

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5046

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
