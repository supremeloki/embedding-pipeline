from .core import (
    DimensionMismatchError,
    EmbeddedDocument,
    EmbeddingFunction,
    EmbeddingPipeline,
    EmptyBatchError,
    HashingEmbedder,
    PipelineError,
    build_default_pipeline,
    chunk_text,
    cosine,
)

__all__ = [
    "DimensionMismatchError",
    "EmbeddedDocument",
    "EmbeddingFunction",
    "EmbeddingPipeline",
    "EmptyBatchError",
    "HashingEmbedder",
    "PipelineError",
    "build_default_pipeline",
    "chunk_text",
    "cosine",
]

__version__ = "0.1.0"
