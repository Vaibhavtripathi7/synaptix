"""Retrieval and answer quality against a golden set.

    uv run python -m eval.run              # retrieval metrics only: no generation calls
    uv run python -m eval.run --answers    # also generate answers and grade them with an LLM judge

Retrieval is scored with expansion off (embeddings plus the configured reranker), so it is cheap to
run after every change to chunking, fusion or reranking. `hit@k` = an expected page appears in
the returned passages; `mrr` = 1/rank of the first passage on an expected page; `terms` =
fraction of expected terms present somewhere in the context."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.settings import get_settings
from app.db.models import DocStatus, Document
from app.db.session import SessionLocal, engine
from app.llm.client import Gemini
from app.llm.embeddings import build_embedder
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
                grade = await llm.generate_structured(
                    f"Passages:\n\n{render(r.passages)}\n\nQuestion: {c['question']}\n\n"
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="generate and grade answers too")
    asyncio.run(main(ap.parse_args().answers))
