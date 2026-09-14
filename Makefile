.PHONY: infra infra-down migrate test lint fmt
UV := uv run

infra:        ## Postgres+pgvector :5433
	docker compose up -d db
infra-down:
	docker compose down
migrate:
	$(UV) alembic upgrade head
test:
	$(UV) pytest -q
lint:
	$(UV) ruff check . && $(UV) ruff format --check .
fmt:
	$(UV) ruff format . && $(UV) ruff check --fix .
