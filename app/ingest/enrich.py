"""Make non-text content retrievable: figure descriptions, table summaries, OCR for scans."""

import asyncio

import structlog

from app.db.models import ElementKind
from app.ingest.parse import Block, ParsedDoc
from app.llm.client import Gemini

log = structlog.get_logger()

_FIGURE = (
    "Describe this figure from a document so it can be found by search and used to answer "
    "questions. State what it shows, axes/labels, key values or trends, and the takeaway. "
    "Plain prose, no preamble."
)
_TABLE = (
    "In two or three sentences, say what this table reports and its most important figures "
    "or comparisons. Preserve exact numbers and units.\n\n"
)
_SCAN = (
    "Transcribe this scanned page to Markdown. Keep headings, lists and reading order; render "
    "tables as Markdown tables and all mathematics as LaTeX between $ delimiters. Describe any "
    "diagram or plot in square brackets where it appears. Output only the transcription."
)


async def enrich(parsed: ParsedDoc, llm: Gemini) -> int:
    """Fills `content` (figures, scans) or `meta['summary']` (tables) in place, concurrently;
    concurrency is bounded by the client's semaphore. Work that still fails after the client's
    retries leaves the block with its fallback (caption, raw table) and records the error in its
    meta, so one flaky request degrades one element instead of failing the document.
    Returns the number of degraded blocks."""
    blocks = parsed.blocks
    targets = [i for i, b in enumerate(blocks) if _needs(b)]
    results = await asyncio.gather(
        *(_enrich_one(blocks, i, llm) for i in targets), return_exceptions=True
    )

    failed = 0
    for i, r in zip(targets, results, strict=True):
        if not isinstance(r, BaseException):
            continue
        failed += 1
        b = blocks[i]
        b.meta["enrich_error"] = f"{type(r).__name__}: {r}"[:300]
        if not b.content:
            b.content = _adjacent_caption(blocks, i) or ""
        log.warning("enrich.degraded", page=b.page, role=b.role, error=b.meta["enrich_error"])
    return failed


def _needs(b: Block) -> bool:
    return b.image_png is not None or b.kind is ElementKind.table


async def _enrich_one(blocks: list[Block], i: int, llm: Gemini) -> None:
    b = blocks[i]
    if b.role == "scan":
        b.content = await llm.generate([_SCAN, Gemini.image(b.image_png)])
    elif b.kind is ElementKind.figure:
        hint = _adjacent_caption(blocks, i)
        prompt = _FIGURE + (f"\nThe document captions it: {hint}" if hint else "")
        b.content = await llm.generate([prompt, Gemini.image(b.image_png)])
        if hint:
            b.meta["caption"] = hint
    else:
        b.meta["summary"] = await llm.generate(_TABLE + b.content)


def _adjacent_caption(blocks: list[Block], i: int) -> str | None:
    for j in (i + 1, i - 1):
        if (
            0 <= j < len(blocks)
            and blocks[j].role == "caption"
            and blocks[j].page == blocks[i].page
        ):
            return blocks[j].content
    return None
