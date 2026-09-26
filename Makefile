.PHONY: infra infra-down migrate api worker worker-dev test lint fmt
UV := uv run

infra:        ## Postgres+pgvector :5433, queue Redis :6380, cache Redis :6381
	docker compose up -d db redis cache
infra-down:
	docker compose down
migrate:
	$(UV) alembic upgrade head
api:
	$(UV) uvicorn app.main:app --reload --port 8000
worker:
	$(UV) arq app.workers.settings.WorkerSettings
worker-dev:   ## reloads on code change; cancels any in-flight job, so not for real ingests
	$(UV) watchfiles --filter python "arq app.workers.settings.WorkerSettings" app
test:
	$(UV) pytest -q
lint:
	$(UV) ruff check . && $(UV) ruff format --check .
fmt:
	$(UV) ruff format . && $(UV) ruff check --fix .
