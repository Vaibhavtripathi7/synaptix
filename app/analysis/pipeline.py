"""Executes an AnalysisRun.

competitor:  profile ─┐
             swot    ─┼─►  key_analysis(profile, swot, strategy, facets)
             strategy ┘
comparison:  facets(doc A) + facets(doc B)  →  comparison

Results are stored as JSON with the passages each stage used, so a client can resolve every
`evidence` number to a page range of the source document."""

import uuid
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from app.analysis import agents
from app.db.models import AnalysisKind, AnalysisRun, RunStatus
from app.db.session import SessionLocal
from app.rag.answer import Answerer
from app.retrieval.context import Passage

log = structlog.get_logger()


def _stage(output: BaseModel, passages: list[Passage]) -> dict:
    return {
        "output": output.model_dump(),
        "passages": [
            {
                "n": p.n,
                "document_id": str(p.document_id),
                "title": p.title,
                "heading_path": p.heading_path,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "kind": p.kind.value,
                "chunk_ids": [str(c) for c in p.chunk_ids],
            }
            for p in passages
        ],
    }


async def run_analysis(run_id: uuid.UUID, answerer: Answerer) -> None:
    async with SessionLocal() as session:
        run = await session.get(AnalysisRun, run_id)
        if run is None:
            log.warning("analysis.missing", run_id=str(run_id))
            return
        run.status = RunStatus.running
        await session.commit()
        try:
            if run.kind is AnalysisKind.competitor:
                run.result = await _competitor(session, answerer, run.document_ids)
            else:
                cmp, passages = await agents.compare(
                    session=session,
                    answerer=answerer,
                    first=run.document_ids[0],
                    second=run.document_ids[1],
                )
                run.result = {"comparison": _stage(cmp, passages)}
            run.status = RunStatus.done
        except Exception as exc:
            log.exception("analysis.failed", run_id=str(run_id))
            run.status, run.error = RunStatus.failed, f"{type(exc).__name__}: {exc}"[:2000]
        run.finished_at = datetime.now(UTC)
        await session.commit()


async def _competitor(session, answerer: Answerer, document_ids: list[uuid.UUID]) -> dict:
    """Three independent stages, then a fourth that reads all three. The three run one after
    another, not concurrently: they share the run's database session."""
    result, outputs = {}, {}
    for agent in (agents.PROFILE, agents.SWOT, agents.STRATEGY):
        output, passages = await agents.run_agent(agent, answerer, session, document_ids)
        result[agent.name], outputs[agent.name] = _stage(output, passages), output
    key, passages = await agents.run_agent(
        agents.KEY_ANALYSIS, answerer, session, document_ids, inputs=outputs
    )
    result[agents.KEY_ANALYSIS.name] = _stage(key, passages)
    return result
