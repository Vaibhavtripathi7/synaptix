"""Prometheus metrics.

One process per container, scaled by adding replicas: the default single-process registry is
therefore correct and there is no multiprocess mode to misconfigure. Labels are bounded —
route templates rather than raw paths, model names from config — because a label taking
unbounded values (a UUID, a scanner's probe path) creates a time series per value and takes
the metrics backend down with it."""

from prometheus_client import Counter, Gauge, Histogram

_SLOW = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)

LLM_CALLS = Counter(
    "synaptix_llm_calls_total", "Model calls by outcome", ["model", "op", "outcome"]
)
LLM_LATENCY = Histogram(
    "synaptix_llm_call_duration_seconds", "Model call latency", ["model", "op"], buckets=_SLOW
)
LLM_TOKENS = Counter("synaptix_llm_tokens_total", "Tokens by model and kind", ["model", "kind"])

JOBS = Counter("synaptix_jobs_total", "Background jobs by outcome", ["job", "outcome"])
JOB_LATENCY = Histogram(
    "synaptix_job_duration_seconds", "Background job duration", ["job"], buckets=_SLOW
)
INGEST_STAGE = Histogram(
    "synaptix_ingest_stage_duration_seconds", "Time per ingestion stage", ["stage"], buckets=_SLOW
)
QUEUE_DEPTH = Gauge("synaptix_queue_depth", "Jobs waiting to be picked up")
