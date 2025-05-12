# embedding-pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A batch embedding pipeline: chunking, deterministic hashing embedder, cosine search index, and JSON persistence — the ingestion half of every RAG system, isolated and testable.

## 🚀 Overview

`embedding-pipeline` covers everything from raw text to a searchable vector index: **chunk_text** splits long documents on word boundaries, **HashingEmbedder** maps tokens into a normalized 32-dimensional space deterministically (same text → same vector across runs), and `EmbeddingPipeline` batches documents, maintains an upsert/delete index, ranks by cosine similarity, and persists the whole index as JSON.

## ✨ Features

- **Deterministic embeddings:** seeded hashing embedder — reproducible vectors without model downloads
- **Batch processing:** configurable batch size; empty batches rejected with typed errors
- **Search index:** upsert/delete semantics with cosine top-k ranking
- **Chunker:** word-boundary splitting with strict character budgets
- **Persistence:** JSON flush/reload; corrupt files raise typed errors
- **Pluggable embedders:** any callable with a `model_name` attribute slots in
- **Zero dependencies**

## 🚧 Structure

```
embedding-pipeline/
├── src/embedding_pipeline/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/embedding-pipeline.git
cd embedding-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from pathlib import Path
from embedding_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline(persist_path=Path("index.json"))
pipeline.embed_documents({
    "doc:1": "database tuning guide",
    "doc:2": "chocolate cake recipe",
})

hits = pipeline.search("how to tune a database")
print(hits[0][0])
```

## 🔧 Error Handling

```text
PipelineError
├── EmptyBatchError          # zero-document batches rejected
└── DimensionMismatchError   # vector dimension mismatches
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen embedded documents
- Zero comments — names carry the meaning
- Determinism asserted explicitly (same text → identical vectors)

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
