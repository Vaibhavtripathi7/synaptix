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


async def test_answerer_retrieve_with_stub_models(seeded):
    """Covers the retrieve() -> search() -> context path without any model calls."""
    from app.core.settings import get_settings
    from app.rag.answer import Answerer
    from app.retrieval.rerank import NoReranker

    class StubEmbedder:
        async def embed_documents(self, texts):
            return [_vec(2) for _ in texts]

        async def embed_query(self, text):
            return _vec(2)

    settings = get_settings().model_copy(update={"query_expansion": False})
    answerer = Answerer(llm=None, embedder=StubEmbedder(), reranker=NoReranker(), settings=settings)
    async with SessionLocal() as s:
        r = await answerer.retrieve(s, "isochrone drawbacks", [seeded])
    assert r.queries == ["isochrone drawbacks"]
    assert r.candidates == 3 and len(r.passages) == 1
    assert r.passages[0].n == 1


async def test_keyword_search_matches_questions_not_just_exact_term_sets(seeded):
    """A question's words are rarely all in one chunk. AND semantics made sparse retrieval
    return nothing for natural-language questions (measured hit rate 0.19); it must rank chunks
    that contain *some* of the terms."""
    async with SessionLocal() as s:
        question = "What are the drawbacks of the isochrone approach for weather routing?"
        hits = await hybrid_search(
            s, [(question, _vec(99))], document_ids=[seeded], k=10, dense=False
        )
    ranked = sorted((h for h in hits if h.sparse_rank), key=lambda h: h.sparse_rank)
    assert ranked, "no keyword hits for a question no single chunk fully contains"
    assert ranked[0].chunk.content.startswith("The isochrone method has drawbacks")


async def test_search_with_several_query_variants(seeded):
    """Query expansion sends three variants. A local once shadowed the `sparse` flag, so every
    variant after the first crashed - only multi-variant searches exposed it."""
    async with SessionLocal() as s:
        variants = [
            ("isochrone drawbacks", _vec(2)),
            ("isochrone limits", _vec(1)),
            ("grid", _vec(0)),
        ]
        hits = await hybrid_search(s, variants, document_ids=[seeded], k=10)
    assert hits and all(h.queries for h in hits)
    assert max(len(h.queries) for h in hits) > 1, "variants must fuse, not just run"
