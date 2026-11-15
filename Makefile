.PHONY: infra infra-down stack migrate api worker worker-dev test alerts-test lint fmt
UV := uv run
PROMTOOL := docker run --rm -v "$(CURDIR)/deploy/observability:/cfg:ro" --entrypoint promtool prom/prometheus:v3.15.0

infra:        ## Postgres+pgvector :5433, queue Redis :6380, cache Redis :6381, S3 :9005
	docker compose --profile s3 up -d db redis cache s3
infra-down:   ## stop whatever `infra` or `stack` started; data volumes are kept
	docker compose --profile '*' down
stack:        ## everything in containers, plus Prometheus :9090 and Grafana :3000
	docker compose --profile app --profile observability up -d --build
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
alerts-test:  ## validate the alert rules and run their unit tests
	$(PROMTOOL) check rules /cfg/alerts.yml && $(PROMTOOL) test rules /cfg/alerts_test.yml
lint:
	$(UV) ruff check . && $(UV) ruff format --check .
fmt:
	$(UV) ruff format . && $(UV) ruff check --fix .
