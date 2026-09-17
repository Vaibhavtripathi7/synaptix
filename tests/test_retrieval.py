"""Hybrid search + context building against the live database (skipped when it is not up).
Vectors are synthetic, so this exercises the SQL, fusion and expansion logic, not the models."""

from app.db.session import SessionLocal
from app.retrieval.context import build_context
from app.retrieval.hybrid import hybrid_search
from tests.conftest import vec as _vec


async def test_hybrid_fuses_dense_and_sparse_and_expands_to_section(seeded):
    async with SessionLocal() as s:
        # dense query = chunk 2's vector; sparse query matches chunk 0's words
        hits = await hybrid_search(
            s, [("isochrone drawbacks", _vec(2))], document_ids=[seeded], k=10
        )
        by_text = {h.chunk.content[:8]: h for h in hits}
        assert by_text["Dijkstra"].dense_rank == 1
        assert by_text["The isoc"].sparse_rank == 1
        assert by_text["The isoc"].score > by_text["Genetic "].score  # in both lists vs one

        passages = build_context(hits, budget_tokens=500)
        # all three chunks share one short section -> one expanded passage covering them
        assert len(passages) == 1
        assert passages[0].text.startswith("Short section body")
        assert len(passages[0].chunk_ids) == 3
        assert passages[0].heading_path == ["Test Doc", "Methods"]
