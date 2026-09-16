.PHONY: infra infra-down migrate api worker test lint fmt
UV := uv run

infra:        ## Postgres+pgvector :5433, queue Redis :6380
	docker compose up -d db redis
infra-down:
	docker compose down
migrate:
	$(UV) alembic upgrade head
api:
	$(UV) uvicorn app.main:app --reload --port 8000
worker:       ## reloads on code change
	$(UV) watchfiles --filter python "arq app.workers.settings.WorkerSettings" app
test:
	$(UV) pytest -q
lint:
	$(UV) ruff check . && $(UV) ruff format --check .
fmt:
	$(UV) ruff format . && $(UV) ruff check --fix .
