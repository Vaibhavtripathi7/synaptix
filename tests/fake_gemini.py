"""A fake Gemini REST API, served through `httpx.MockTransport`.

The real google-genai SDK builds its real requests and parses real-shaped responses, so the code
under test - failover, rate-limit waits, token accounting, structured output, streaming - runs
exactly as in production, with no network and no quota. Wire formats were taken from the requests
the SDK actually sends (generateContent, streamGenerateContent?alt=sse, batchEmbedContents).

Structured responses are generated from the `responseSchema` in each request, so every agent's
schema is answered without per-schema code; three shapes get realistic content because tests
assert on it: OCR batches, reranking scores, and query expansion."""

import hashlib
import json
import random
import re
from dataclasses import dataclass, field

import httpx

USAGE = {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}
ANSWER = "The quality factor is Q = (1/R) sqrt(L/C) [1]."
FIGURE = "A bar chart comparing costs across departments."
OCR_TEXT = "impedance and resonance."


def quota_error(per: str = "Day", retry_s: int = 20) -> httpx.Response:
    """A 429 exactly as Google shapes it: QuotaFailure naming the quota, plus RetryInfo."""
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": f"GenerateRequestsPer{per}PerProjectPerModel-FreeTier"}
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": f"{retry_s}s",
                    },
                ],
            }
        },
    )


def overloaded() -> httpx.Response:
    return httpx.Response(
        503,
        json={
            "error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}
        },
    )


def vector(text: str, dim: int) -> list[float]:
    """Deterministic unit vector per text: identical text always embeds identically."""
    rng = random.Random(hashlib.sha256(text.encode()).digest())
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


@dataclass
class Call:
    model: str
    method: str
    body: dict

    @property
    def text(self) -> str:
        parts = [
            p.get("text", "") for c in self.body.get("contents", []) for p in c.get("parts", [])
        ]
        return "\n".join(parts)

    @property
    def images(self) -> int:
        return sum(
            "inlineData" in p for c in self.body.get("contents", []) for p in c.get("parts", [])
        )

    @property
    def schema(self) -> dict | None:
        return self.body.get("generationConfig", {}).get("responseSchema")


@dataclass
class FakeGemini:
    dim: int
    calls: list[Call] = field(default_factory=list)
    # model -> responses to serve before behaving normally (errors, for failover tests)
    script: dict[str, list[httpx.Response]] = field(default_factory=dict)

    def handle(self, request: httpx.Request) -> httpx.Response:
        m = re.search(r"/models/([^:/]+):(\w+)", request.url.path)
        assert m, f"unexpected request {request.url}"
        model, method = m.groups()
        call = Call(model, method, json.loads(request.content or b"{}"))
        self.calls.append(call)
        if self.script.get(model):
            return self.script[model].pop(0)
        if method == "batchEmbedContents":
            texts = [p["text"] for r in call.body["requests"] for p in r["content"]["parts"]]
            return httpx.Response(
                200, json={"embeddings": [{"values": vector(t, self.dim)} for t in texts]}
            )
        if method == "generateContent":
            return httpx.Response(200, json=_response(self._text(call), model, final=True))
        if method == "streamGenerateContent":
            words = self._text(call).split(" ")
            pieces = [" ".join(words[: len(words) // 2]) + " ", " ".join(words[len(words) // 2 :])]
            events = [_response(p, model, final=i == len(pieces) - 1) for i, p in enumerate(pieces)]
            body = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        raise AssertionError(f"unhandled method {method}")

    def _text(self, call: Call) -> str:
        if call.schema:
            return json.dumps(_structured(call))
        if "Describe this figure" in call.text:
            return FIGURE
        if "say what this table reports" in call.text:
            return "The table reports measured values per configuration."
        return ANSWER

    def methods(self) -> list[str]:
        return [c.method for c in self.calls]


def _response(text: str, model: str, *, final: bool) -> dict:
    response = {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}],
        "modelVersion": model,
    }
    if final:
        response["candidates"][0]["finishReason"] = "STOP"
        response["usageMetadata"] = USAGE | {"totalTokenCount": sum(USAGE.values())}
    return response


def _structured(call: Call) -> dict:
    props = call.schema.get("properties", {})
    if "pages" in props:  # OCR batch: one transcription per "Page N:" label, in order
        labels = [int(n) for n in re.findall(r"^Page (\d+):", call.text, re.M)]
        return {
            "pages": [
                {
                    "page_index": n,
                    "markdown": f"## Scanned heading {n}\nTranscribed text of scan {n}: {OCR_TEXT}",
                }
                for n in labels
            ]
        }
    if "scores" in props:  # reranker: grade every listed passage, preserving their order
        passages = [int(n) for n in re.findall(r"passage \[(\d+)\]", call.text)]
        return {
            "scores": [{"index": i, "relevance": max(0, 10 - k)} for k, i in enumerate(passages)]
        }
    if "queries" in props:
        return {
            "queries": ["alternative wording of the question", "another phrasing with synonyms"]
        }
    return _instance(call.schema)


def _instance(schema: dict):
    """A minimal valid instance of an OpenAPI-style schema, as the SDK sends it."""
    if "anyOf" in schema:
        return _instance(next(s for s in schema["anyOf"] if s.get("type", "").upper() != "NULL"))
    kind = schema.get("type", "OBJECT").upper()
    if kind == "OBJECT":
        return {k: _instance(v) for k, v in schema.get("properties", {}).items()}
    if kind == "ARRAY":
        return [_instance(schema["items"])]
    if kind == "STRING":
        return schema.get("enum", ["fake text"])[0]
    if kind == "INTEGER":
        return max(int(schema.get("minimum", 1)), 1)
    if kind == "NUMBER":
        return 1.0
    if kind == "BOOLEAN":
        return True
    raise AssertionError(f"unhandled schema type {kind}")
