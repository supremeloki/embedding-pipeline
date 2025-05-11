import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from embedding_pipeline import (
    EmptyBatchError,
    EmbeddingPipeline,
    HashingEmbedder,
    PipelineError,
    chunk_text,
    cosine,
)


@pytest.fixture
def pipeline():
    return EmbeddingPipeline()


def test_embed_documents_returns_normalized_vectors(pipeline):
    docs = {"a": "hello world", "b": "another document entirely"}
    embedded = pipeline.embed_documents(docs)
    assert len(embedded) == 2
    for item in embedded:
        assert item.norm == pytest.approx(1.0, abs=1e-6)


def test_deterministic_embeddings(pipeline):
    first = pipeline.embed_query("same text")
    second = pipeline.embed_query("same text")
    assert first == second


def test_empty_batch_rejected(pipeline):
    with pytest.raises(EmptyBatchError):
        pipeline.embed_documents({})


def test_search_ranks_relevant_first(pipeline):
    docs = {
        "db": "database index query performance",
        "cooking": "chocolate cake recipe sugar butter",
    }
    pipeline.embed_documents(docs)
    hits = pipeline.search("database query index")
    assert hits[0][0] == "db"


def test_upsert_and_delete(pipeline):
    pipeline.upsert("doc:1", "some content")
    assert pipeline.indexed_count == 1
    assert pipeline.delete("doc:1") is True
    assert pipeline.delete("doc:1") is False


def test_search_before_index_raises_on_empty_query(pipeline):
    with pytest.raises(PipelineError):
        pipeline.embed_query("   ")


def test_invalid_top_k_rejected(pipeline):
    pipeline.upsert("x", "text")
    with pytest.raises(PipelineError):
        pipeline.search("query", top_k=0)


def test_batch_size_splits_large_batches(pipeline):
    small = EmbeddingPipeline(batch_size=2)
    documents = {f"doc-{i}": f"content {i}" for i in range(5)}
    results = small.embed_documents(documents)
    assert len(results) == 5


def test_chunk_text_respects_limit():
    words = ["w" * 10] * 30
    chunks = chunk_text(" ".join(words), max_chars=100)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) >= 3


def test_chunk_invalid_max_rejected():
    with pytest.raises(PipelineError):
        chunk_text("text", max_chars=0)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    first = EmbeddingPipeline(persist_path=path)
    first.upsert("persisted", "remember me")
    first.flush()

    reopened = EmbeddingPipeline(persist_path=path)
    assert "persisted" in [r[0] for r in reopened.search("remember")]


def test_corrupt_index_file_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    with pytest.raises(PipelineError):
        EmbeddingPipeline(persist_path=bad)


def test_cosine_known_values():
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 1], [1, 1]) == pytest.approx(1.0)


def test_hasher_dimension_guard():
    with pytest.raises(PipelineError):
        HashingEmbedder(dimension=4)
