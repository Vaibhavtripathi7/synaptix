"""Retrieval and answer quality against a golden set.

    uv run python -m eval.run              # retrieval metrics only: no generation calls
    uv run python -m eval.run --answers    # also generate answers and grade them with an LLM judge
    uv run python -m eval.run --ablation   # dense vs BM25 vs hybrid vs reranked; eval/ablation.md

Retrieval is scored with expansion off (embeddings plus the configured reranker), so it is cheap to
run after every change to chunking, fusion or reranking. `hit@k` = an expected page appears in
the returned passages; `mrr` = 1/rank of the first passage on an expected page; `terms` =
fraction of expected terms present somewhere in the context."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.settings import get_settings
from app.db.models import DocStatus, Document
from app.db.session import SessionLocal, engine
from app.llm.client import Gemini
from app.llm.embeddings import build_embedder
from app.llm.untrusted import Fence
from app.rag.answer import Answerer
from app.retrieval.context import render
from app.retrieval.rerank import build_reranker

GOLDEN = Path(__file__).with_name("golden.jsonl")


class Grade(BaseModel):
    faithfulness: int = Field(
        ge=0, le=5, description="5 = every claim is supported by the passages"
    )
    correctness: int = Field(ge=0, le=5, description="5 = fully answers the question")
    unsupported_claims: list[str]


JUDGE = (
    "You are grading a question-answering system. Given the passages it retrieved, the question "
    "and its answer, score faithfulness (are all claims supported by the passages?) and "
    "correctness (does it answer the question well, judged only from the passages?) from 0 to 5, "
    "and list any claims not supported by the passages."
)


async def main(answers: bool) -> None:
    settings = get_settings().model_copy(update={"query_expansion": False})
    llm = Gemini(settings)
    answerer = Answerer(llm, build_embedder(settings, llm), build_reranker(settings, llm), settings)
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

    hits, rr, terms, grades = [], [], [], []
    async with SessionLocal() as session:
        docs = {
            d.filename: d.id
            for d in await session.scalars(
                select(Document).where(Document.status == DocStatus.ready)
            )
        }
        for c in cases:
            doc_id = docs.get(c["document"])
            if doc_id is None:
                print(f"skip (not ingested): {c['document']}")
                continue
            r = await answerer.retrieve(session, c["question"], [doc_id])
            ranks = [
                p.n
                for p in r.passages
                if any(p.page_start <= pg <= p.page_end for pg in c["pages"])
            ]
            context = " ".join(p.text for p in r.passages).lower()
            hit, mrr = bool(ranks), (1 / ranks[0] if ranks else 0.0)
            term_frac = sum(t.lower() in context for t in c["terms"]) / len(c["terms"])
            hits.append(hit)
            rr.append(mrr)
            terms.append(term_frac)
            line = (
                f"{'hit ' if hit else 'MISS'} mrr={mrr:.2f} terms={term_frac:.2f}  {c['question']}"
            )
            if answers:
                answer = await answerer.answer(c["question"], r.passages)
                fence = Fence()
                grade = await llm.generate_structured(
                    f"Passages:\n\n{render(r.passages, fence)}\n\nQuestion: {c['question']}\n\n"
                    f"Answer:\n{answer}",
                    Grade,
                    system=JUDGE,
                    think=True,
                )
                grades.append(grade)
                line += f"  | faithful={grade.faithfulness} correct={grade.correctness}"
                if grade.unsupported_claims:
                    line += f"  unsupported: {grade.unsupported_claims}"
            print(line)

    n = len(hits)
    print(
        f"\n{n} cases  hit@{settings.rerank_top_n}={sum(hits) / n:.2f}"
        f"  mrr={statistics.mean(rr):.2f}  terms={statistics.mean(terms):.2f}"
    )
    if grades:
        print(
            f"faithfulness={statistics.mean(g.faithfulness for g in grades):.2f}/5"
            f"  correctness={statistics.mean(g.correctness for g in grades):.2f}/5"
        )
    await engine.dispose()


VARIANTS = [
    # name, dense, sparse, rerank as (candidates, characters per passage) or None
    ("dense only", True, False, None),
    ("BM25 only", False, True, None),
    ("hybrid (RRF)", True, True, None),
    ("hybrid + rerank, 40 x 1200 chars (production)", True, True, (40, 1200)),
    ("hybrid + rerank, 20 x 600 chars", True, True, (20, 600)),
]


async def ablation(skip_rerank: bool = False) -> None:
    """Scores every retrieval variant on the same golden set. Each question is embedded once
    and the vector reused, so variants differ only in strategy. Expansion is off throughout:
    this isolates fusion and reranking."""
    from app.core import usage
    from app.rag.answer import answerer_for
    from app.retrieval.context import build_context
    from app.retrieval.hybrid import hybrid_search
    from app.retrieval.rerank import LLMReranker

    settings = get_settings()
    stack = answerer_for(None)
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    rows = []
    async with SessionLocal() as session:
        docs = {
            d.filename: d.id
            for d in await session.scalars(
                select(Document).where(Document.status == DocStatus.ready)
            )
        }
        cases = [c for c in cases if c["document"] in docs]
        vectors = await stack.embedder.embed_documents([c["question"] for c in cases])
        for name, dense, sparse, rerank in VARIANTS:
            if rerank and skip_rerank:
                continue
            hit1 = hit = rr = tokens = 0.0
            elapsed: list[float] = []
            for case, vector in zip(cases, vectors, strict=True):
                ledger = usage.begin()
                started = time.perf_counter()
                hits = await hybrid_search(
                    session,
                    [(case["question"], vector)],
                    document_ids=[docs[case["document"]]],
                    k=settings.retrieval_k,
                    dense=dense,
                    sparse=sparse,
                )
                await session.commit()
                if rerank:
                    candidates, chars = rerank
                    hits = await LLMReranker(stack.llm, chars).rerank(
                        case["question"], hits[:candidates], settings.rerank_top_n
                    )
                passages = build_context(hits[: settings.rerank_top_n], settings.context_tokens)
                elapsed.append(time.perf_counter() - started)
                ranks = [
                    p.n
                    for p in passages
                    if any(p.page_start <= pg <= p.page_end for pg in case["pages"])
                ]
                hit += bool(ranks)
                hit1 += bool(ranks) and ranks[0] == 1
                rr += 1 / ranks[0] if ranks else 0.0
                tokens += ledger.input_tokens
            n = len(cases)
            rows.append((name, hit1 / n, hit / n, rr / n, tokens / n, statistics.median(elapsed)))
            print(
                f"{name:48} hit@1 {hit1 / n:.2f}  hit@{settings.rerank_top_n} {hit / n:.2f}  "
                f"MRR {rr / n:.2f}  tokens/q {tokens / n:7.0f}  p50 {rows[-1][5]:.2f}s",
                flush=True,
            )

    k = settings.rerank_top_n
    table = [
        f"| variant | hit@1 | hit@{k} | MRR | input tokens / question | p50 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    table += [
        f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:,.0f} | {r[5]:.2f} s |"
        for r in rows
    ]
    header = (
        f"{len(cases)} questions over {len({c['document'] for c in cases})} documents "
        f"(`eval/golden.jsonl`); query expansion off; `{settings.embedding_model}`.\n\n"
    )
    Path(__file__).with_name("ablation.md").write_text(header + "\n".join(table) + "\n")
    await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="generate and grade answers too")
    ap.add_argument("--ablation", action="store_true", help="compare retrieval strategies")
    ap.add_argument("--skip-rerank", action="store_true", help="ablation without model calls")
    args = ap.parse_args()
    asyncio.run(ablation(args.skip_rerank) if args.ablation else main(args.answers))
