"""Hybrid search against the live database (skipped when it is not up). Vectors are
synthetic, so this exercises the SQL and the fusion logic, not the models."""

from app.db.session import SessionLocal
from app.retrieval.hybrid import hybrid_search
from tests.conftest import vec as _vec


async def test_hybrid_fuses_dense_and_sparse(seeded):
    async with SessionLocal() as s:
        # dense query = chunk 2's vector; sparse query matches chunk 0's words
        hits = await hybrid_search(
            s, [("isochrone drawbacks", _vec(2))], document_ids=[seeded], k=10
        )
        by_text = {h.chunk.content[:8]: h for h in hits}
        assert by_text["Dijkstra"].dense_rank == 1
        assert by_text["The isoc"].sparse_rank == 1
        assert by_text["The isoc"].score > by_text["Genetic "].score  # in both lists vs one
