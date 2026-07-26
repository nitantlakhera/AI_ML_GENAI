"""Chunking, vector store, hybrid retrieval, fusion, reranking, graph RAG, pipeline."""

from __future__ import annotations

from agentic_studio.core.types import Chunk, Document, Message, Retrieved
from agentic_studio.rag.chunking import chunk_documents, markdown_split, recursive_split
from agentic_studio.rag.fusion import deduplicate, reciprocal_rank_fusion, weighted_score_fusion
from agentic_studio.rag.graph_rag import KnowledgeGraph, extract_entities
from agentic_studio.rag.ingest import prepare_chunks, stable_chunk_id
from agentic_studio.rag.lexical import BM25Index
from agentic_studio.rag.pipeline import RagConfig, RagPipeline
from agentic_studio.rag.query_transform import QueryTransformer, rule_based_variants
from agentic_studio.rag.rerank import LexicalReranker, NoOpReranker, build_reranker
from agentic_studio.rag.vector_store import NumpyVectorStore, matches

# -- chunking ---------------------------------------------------------------


def test_recursive_split_respects_the_size_budget():
    text = ". ".join(f"sentence number {i} about retrieval" for i in range(60))

    pieces = recursive_split(text, chunk_size=200, chunk_overlap=20)

    assert len(pieces) > 1
    assert all(len(piece) <= 260 for piece, _ in pieces)


def test_markdown_split_keeps_the_heading_trail():
    text = "# Guide\n\nintro text\n\n## Hybrid\n\ndense plus sparse\n\n## Rerank\n\ncross encoder"

    pieces = markdown_split(text, chunk_size=500, overlap=0)
    headings = [meta["heading"] for _, meta in pieces]

    assert "Guide > Hybrid" in headings
    assert "Guide > Rerank" in headings


def test_chunking_attaches_parent_context():
    body = " ".join(f"paragraph {i} discusses reranking and fusion in depth." for i in range(40))
    document = Document(text=body, metadata={"source": "big.md"})

    chunks = chunk_documents([document], strategy="recursive", chunk_size=150,
                             chunk_overlap=0, parent_chunk_size=1200)

    assert any(chunk.parent_text for chunk in chunks)
    parented = next(chunk for chunk in chunks if chunk.parent_text)
    assert len(parented.parent_text) > len(parented.text)
    assert parented.context_text if isinstance(parented, Retrieved) else True


def test_stable_chunk_ids_are_deterministic():
    document = Document(text="hybrid retrieval merges dense and sparse", metadata={"source": "a.md"})

    first = prepare_chunks([document])
    second = prepare_chunks([Document(text=document.text, metadata=dict(document.metadata))])

    assert [c.id for c in first] == [c.id for c in second]
    assert all(chunk.id.startswith("chunk_") for chunk in first)
    assert stable_chunk_id(first[0]) == first[0].id


# -- vector store -----------------------------------------------------------


def test_upsert_replaces_instead_of_duplicating(tmp_path):
    store = NumpyVectorStore(path=tmp_path / "index", autoload=False)
    store.upsert([Chunk(text="original text", id="x", metadata={"source": "a.md"})])
    store.upsert([Chunk(text="revised text about fusion", id="x", metadata={"source": "a.md"})])

    assert len(store.all_chunks()) == 1
    assert "revised" in store.get("x").text


def test_delete_by_metadata_filter(store):
    removed = store.delete(where={"source": "doc1.md"})

    assert removed == 1
    assert all(chunk.source != "doc1.md" for chunk in store.all_chunks())


def test_index_round_trips_through_disk(tmp_path):
    original = NumpyVectorStore(path=tmp_path / "index", autoload=False)
    original.upsert([Chunk(text="graph rag traverses entity co-occurrence", id="g1",
                           metadata={"source": "g.md"})])
    original.save()

    reloaded = NumpyVectorStore(path=tmp_path / "index")

    assert len(reloaded.all_chunks()) == 1
    assert reloaded.search("entity graph", k=1)[0].chunk.id == "g1"


def test_metadata_predicates():
    metadata = {"filetype": "pdf", "page": 7, "title": "Retrieval Handbook"}

    assert matches(metadata, {"filetype": "pdf"})
    assert matches(metadata, {"page": {"$lte": 10, "$gte": 5}})
    assert matches(metadata, {"title": {"$contains": "handbook"}})
    assert matches(metadata, {"filetype": {"$in": ["pdf", "md"]}})
    assert not matches(metadata, {"filetype": "md"})
    assert not matches(metadata, {"page": {"$gt": 10}})


def test_dense_search_honours_a_metadata_filter(store):
    results = store.search("BM25 exact identifiers", k=4, where={"source": "doc1.md"})

    assert results
    assert {result.chunk.source for result in results} == {"doc1.md"}


# -- lexical + fusion -------------------------------------------------------


def test_bm25_finds_an_exact_term_and_scores_it(store):
    index = BM25Index().build(store.all_chunks())

    results = index.search("BM25 identifiers", k=3)

    assert results
    assert "BM25" in results[0].text
    assert results[0].score > 0
    assert results[0].retriever == "bm25"


def test_bm25_returns_nothing_for_unseen_terms(store):
    index = BM25Index().build(store.all_chunks())

    assert index.search("quantum chromodynamics", k=3) == []


def test_rrf_rewards_documents_ranked_well_by_both_retrievers():
    def chunk(chunk_id: str) -> Chunk:
        return Chunk(text=chunk_id, id=chunk_id, metadata={"source": f"{chunk_id}.md"})

    dense = [Retrieved(chunk("a"), 0.9, "dense", 1), Retrieved(chunk("b"), 0.8, "dense", 2)]
    sparse = [Retrieved(chunk("b"), 12.0, "bm25", 1), Retrieved(chunk("c"), 9.0, "bm25", 2)]

    fused = reciprocal_rank_fusion([dense, sparse], k=10)

    assert fused[0].chunk.id == "b", "b appears high in both lists"
    assert fused[0].retriever == "bm25+dense"
    assert [item.rank for item in fused] == [1, 2, 3]


def test_weighted_fusion_normalises_incomparable_scores():
    def chunk(chunk_id: str) -> Chunk:
        return Chunk(text=chunk_id, id=chunk_id)

    dense = [Retrieved(chunk("a"), 0.91, "dense", 1), Retrieved(chunk("b"), 0.42, "dense", 2)]
    sparse = [Retrieved(chunk("b"), 30.0, "bm25", 1), Retrieved(chunk("a"), 1.0, "bm25", 2)]

    fused = weighted_score_fusion([dense, sparse], weights=[1.0, 1.0])

    assert {item.chunk.id for item in fused} == {"a", "b"}
    assert fused[0].score >= fused[1].score


def test_deduplicate_keeps_the_best_rank():
    chunk = Chunk(text="dup", id="d1")
    results = [Retrieved(chunk, 1.0, "dense", 1), Retrieved(chunk, 0.5, "bm25", 2)]

    unique = deduplicate(results)

    assert len(unique) == 1
    assert unique[0].rank == 1


# -- reranking --------------------------------------------------------------


def test_lexical_reranker_promotes_the_better_match(store):
    candidates = [
        Retrieved(chunk, 0.5, "dense", index + 1)
        for index, chunk in enumerate(store.all_chunks())
    ]

    reranked = LexicalReranker().rerank("what does BM25 catch", candidates, top_k=2)

    assert "BM25" in reranked[0].text
    assert reranked[0].retriever.endswith("rerank:lexical")
    assert len(reranked) == 2


def test_noop_reranker_preserves_order(store):
    candidates = [Retrieved(chunk, 1.0 - i / 10, "dense", i + 1)
                  for i, chunk in enumerate(store.all_chunks())]

    reranked = NoOpReranker().rerank("anything", candidates)

    assert [item.chunk.id for item in reranked] == [item.chunk.id for item in candidates]


def test_build_reranker_falls_back_for_unknown_names():
    assert build_reranker("nonsense").name == "lexical"
    assert build_reranker("none").name == "none"


# -- query transformation ---------------------------------------------------


def test_transform_none_passes_the_question_through(echo_router):
    transformer = QueryTransformer(strategy="none", router=echo_router)

    assert transformer.transform("what is RRF") == ["what is RRF"]


def test_multi_query_produces_extra_variants(echo_router):
    transformer = QueryTransformer(strategy="multi-query", variants=3, router=echo_router)

    queries = transformer.transform("what problem does reciprocal rank fusion solve")

    assert queries[0] == "what problem does reciprocal rank fusion solve"
    assert len(queries) > 1
    assert len(set(queries)) == len(queries)


def test_rule_based_variants_are_deterministic():
    first = rule_based_variants("how does hybrid retrieval work", 3)
    second = rule_based_variants("how does hybrid retrieval work", 3)

    assert first == second
    assert first


def test_rewrite_returns_the_question_without_history(echo_router):
    transformer = QueryTransformer(strategy="rewrite", router=echo_router)

    assert transformer.rewrite("and the second one?", []) == "and the second one?"


# -- graph RAG --------------------------------------------------------------


def test_entity_extraction_picks_up_proper_nouns_and_acronyms():
    entities = extract_entities("Reciprocal Rank Fusion is used by BM25 and langchain-core.")

    assert "reciprocal rank fusion" in entities
    assert "bm25" in entities
    assert "langchain-core" in entities


def test_graph_search_links_a_query_entity_to_its_chunks(store):
    graph = KnowledgeGraph().build(store.all_chunks())

    results = graph.search("BM25", k=3)

    assert results
    assert any("BM25" in result.text for result in results)
    assert graph.node_count > 0


def test_graph_related_reports_neighbours(store):
    graph = KnowledgeGraph().build(store.all_chunks())

    related = graph.related("bm25")

    assert related["found"] is True
    assert isinstance(related["neighbours"], list)
    assert graph.related("not-an-entity")["found"] is False


# -- pipeline ---------------------------------------------------------------


def test_pipeline_answers_with_citations(pipeline):
    import re

    result = pipeline.answer("What does BM25 catch that dense retrieval misses?")

    assert result.contexts
    assert re.search(r"\[\d+\]", result.answer), "answer should carry a citation marker"
    assert "identifiers" in result.answer, "should ground in the BM25 chunk"
    assert result.queries_used


def test_hybrid_retrieval_combines_both_retrievers(store, echo_router):
    hybrid = RagPipeline(
        store=store,
        config=RagConfig(top_k=4, fetch_k=8, hybrid=True, graph=False, query_transform="none"),
        router=echo_router,
    )

    contexts, _ = hybrid.retrieve("BM25 identifiers")

    assert contexts
    assert any("bm25" in context.retriever for context in contexts)


def test_pipeline_reports_no_context_for_an_empty_index(tmp_path, echo_router):
    empty = RagPipeline(
        store=NumpyVectorStore(path=tmp_path / "empty", autoload=False),
        config=RagConfig(query_transform="none"),
        router=echo_router,
    )

    result = empty.answer("anything at all")

    assert result.contexts == []
    assert "could not find" in result.answer.lower()


def test_streaming_emits_sources_then_tokens_then_done(pipeline):
    events = list(pipeline.stream_answer("What does BM25 catch?"))
    kinds = [event["type"] for event in events]

    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"


def test_metadata_filter_narrows_the_pipeline(pipeline):
    contexts, _ = pipeline.retrieve("retrieval", where={"source": "doc0.md"})

    assert contexts
    assert {context.chunk.source for context in contexts} == {"doc0.md"}


def test_ingest_is_idempotent(pipeline):
    before = len(pipeline.store.all_chunks())
    chunks = prepare_chunks([Document(text="a new note about fusion", metadata={"source": "n.md"})])

    pipeline.ingest_chunks(chunks, save=False)
    pipeline.ingest_chunks(chunks, save=False)

    assert len(pipeline.store.all_chunks()) == before + len(chunks)


def test_basic_config_disables_every_advanced_stage():
    basic = RagConfig.from_settings().basic()

    assert basic.hybrid is False
    assert basic.graph is False
    assert basic.reranker == "none"
    assert basic.query_transform == "none"


def test_conversational_rag_persists_the_thread(pipeline):
    from agentic_studio.rag.conversational import ConversationalRag

    chat = ConversationalRag(pipeline=pipeline, rewrite_followups=False)

    chat.ask("t1", "What does BM25 catch?")
    chat.ask("t1", "And what does reranking improve?")
    history = chat.history("t1")

    assert len(history) == 4
    assert [message.role for message in history] == ["user", "assistant", "user", "assistant"]
    assert isinstance(history[0], Message)
