"""Token accounting per request or job.

A `Usage` lives in a context variable for the life of one request or job, and the LLM client
adds to it after every call. asyncio tasks inherit the context, and the variable holds a
reference, so model calls fanned out with `gather` inside one request all add to one ledger."""

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from app.core.settings import get_settings


@dataclass(slots=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    embedded_texts: int = 0
    cost_usd: float | None = None  # only when the operator configured prices
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, *, input: int, output: int, thinking: int, cached: int) -> None:
        self.calls += 1
        self.input_tokens += input
        self.output_tokens += output
        self.thinking_tokens += thinking
        self.cached_tokens += cached
        self.by_model[model] = self.by_model.get(model, 0) + input + output + thinking
        price = get_settings().llm_prices_per_mtok.get(model)
        if price is not None:
            # Thinking tokens bill at the output rate.
            spent = ((input * price[0]) + ((output + thinking) * price[1])) / 1_000_000
            self.cost_usd = round((self.cost_usd or 0.0) + spent, 6)

    def as_dict(self) -> dict:
        return asdict(self)


_current: ContextVar[Usage | None] = ContextVar("usage", default=None)


def begin() -> Usage:
    usage = Usage()
    _current.set(usage)
    return usage


def current() -> Usage | None:
    return _current.get()
