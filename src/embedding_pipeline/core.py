from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


class PipelineError(Exception):
    pass


class EmptyBatchError(PipelineError):
    pass


class DimensionMismatchError(PipelineError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"expected dimension {expected}, got {actual}")


TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)
DEFAULT_BATCH_SIZE = 64
DIMENSION = 32


@dataclass(frozen=True)
class EmbeddedDocument:
    doc_id: str
    vector: tuple[float, ...]
    token_count: int

    @property
    def norm(self) -> float:
        return round(math.sqrt(sum(v * v for v in self.vector)), 6)


class EmbeddingFunction(Protocol):
    model_name: str

    def __call__(self, text: str) -> Sequence[float]: ...


class HashingEmbedder:
    model_name = "hashing-v1"

    def __init__(self, dimension: int = DIMENSION) -> None:
        if dimension < 8:
            raise PipelineError("dimension must be >= 8")
        self.dimension = dimension
        self._buckets = [0.0] * dimension
        self._seeded = False

    def _ensure_seeded(self) -> None:
        if self._seeded:
            return
        import random as _random

        rng = _random.Random(42)
        for index in range(4096):
            weight = rng.gauss(0.0, 1.0)
            self._buckets[index % self.dimension] += weight * 0.001
        self._seeded = True

    def __call__(self, text: str) -> Sequence[float]:
        self._ensure_seeded()
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            bucket = hash(token) % self.dimension
            sign = 1.0 if (hash(token + "s") % 2 == 0) else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    if max_chars < 1:
        raise PipelineError("max_chars must be >= 1")
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if length + extra > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        chunks.append(" ".join(current))
    return chunks or [""]


class EmbeddingPipeline:
    def __init__(self, embedder: EmbeddingFunction | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 persist_path: Path | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._batch_size = max(1, batch_size)
        self._index: dict[str, tuple[float, ...]] = {}
        self._persist_path = persist_path
        if persist_path is not None and persist_path.exists():
            self._load()

    @property
    def model_name(self) -> str:
        return getattr(self.embedder, "model_name", "custom")

    @property
    def indexed_count(self) -> int:
        return len(self._index)

    def embed_documents(self, documents: dict[str, str]) -> list[EmbeddedDocument]:
        if not documents:
            raise EmptyBatchError("no documents supplied")
        results: list[EmbeddedDocument] = []
        entries = sorted(documents.items())
        for start in range(0, len(entries), self._batch_size):
            batch = entries[start:start + self._batch_size]
            for doc_id, text in batch:
                vector = tuple(float(v) for v in self.embedder(text))
                results.append(EmbeddedDocument(
                    doc_id=doc_id, vector=vector, token_count=len(TOKEN_PATTERN.findall(text)),
                ))
        for embedded in results:
            self._index[embedded.doc_id] = embedded.vector
        return results

    def embed_query(self, query: str) -> tuple[float, ...]:
        if not query.strip():
            raise PipelineError("query text empty")
        return tuple(float(v) for v in self.embedder(query))

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k < 1:
            raise PipelineError("top_k must be >= 1")
        query_vector = self.embed_query(query)
        scored = [
            (doc_id, cosine(query_vector, vector))
            for doc_id, vector in self._index.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def upsert(self, doc_id: str, text: str) -> EmbeddedDocument:
        result = self.embed_documents({doc_id: text})[0]
        return result

    def delete(self, doc_id: str) -> bool:
        return self._index.pop(doc_id, None) is not None

    def flush(self) -> None:
        if self._persist_path is None:
            raise PipelineError("no persist path configured")
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("w", encoding="utf-8") as handle:
            json.dump({k: list(v) for k, v in self._index.items()},
                      handle, ensure_ascii=False)

    def _load(self) -> None:
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PipelineError(f"corrupt index file: {exc}") from exc
        self._index = {k: tuple(v) for k, v in payload.items()}


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_default_pipeline(persist_path: Path | None = None) -> EmbeddingPipeline:
    return EmbeddingPipeline(persist_path=persist_path)
