"""Hybrid index builder.

Creates and populates both a Qdrant vector collection and a BM25 sparse
index. Includes a code-aware tokenizer that splits camelCase and snake_case
identifiers for better keyword matching.
"""

import pickle
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi


def code_tokenize(text: str) -> list[str]:
    """Tokenize code text, splitting snake_case and camelCase identifiers."""
    if not text:
        return []

    # Replace punctuation and special characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Split on whitespace
    tokens = text.split()

    final_tokens = []
    for token in tokens:
        # Split snake_case
        sub_tokens = token.split("_")
        for sub in sub_tokens:
            if not sub:
                continue
            # Split camelCase
            camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", sub).split()
            for part in camel_split:
                final_tokens.append(part.lower())

    return [t for t in final_tokens if t.strip()]


class HybridIndexBuilder:
    """Builder for hybrid Qdrant + BM25 index."""

    def __init__(
        self,
        collection_name: str = "reporag",
        qdrant_path: str = ":memory:",
        bm25_path: str | None = None,
    ):
        """Initialize the hybrid index builder."""
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.bm25_path = Path(bm25_path) if bm25_path else None

        if self.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=self.qdrant_path)

        self.bm25_corpus: list[list[str]] = []
        self.bm25_doc_ids: list[str] = []
        self.bm25_index: BM25Okapi | None = None

        self._load_bm25()

    def _load_bm25(self) -> None:
        """Load BM25 state from disk if available."""
        if self.bm25_path and self.bm25_path.exists():
            with open(self.bm25_path, "rb") as f:
                state = pickle.load(f)
                self.bm25_corpus = state.get("corpus", [])
                self.bm25_doc_ids = state.get("doc_ids", [])
                self._rebuild_bm25()

    def _save_bm25(self) -> None:
        """Save BM25 state to disk."""
        if self.bm25_path:
            self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(
                    {
                        "corpus": self.bm25_corpus,
                        "doc_ids": self.bm25_doc_ids,
                    },
                    f,
                )

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25Okapi object from the current corpus."""
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_index = None

    def create_collection(self) -> None:
        """Create the Qdrant collection with named vectors and payload schema."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "code": models.VectorParams(
                        size=768, distance=models.Distance.COSINE
                    ),
                    "doc": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                },
            )

            # Create payload indices
            for field in ["file", "symbol", "language"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def upsert(self, documents: list[dict[str, Any]]) -> None:
        """
        Upsert a batch of documents into Qdrant and update the BM25 index.

        Args:
            documents: List of dicts. Each must contain:
                - id (str or int): Document ID
                - text (str): The raw text for BM25
                - metadata (dict): Payload metadata
                - vectors (dict): Mapping of vector name to vector (e.g. {"code": [...], "doc": [...]})
        """
        if not documents:
            return

        points = []
        new_bm25_docs = False

        for doc in documents:
            doc_id = doc["id"]

            # Prepare Qdrant Point
            vectors = doc.get("vectors", {})
            if not isinstance(vectors, dict):
                vectors = {}

            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=vectors,
                    payload=doc.get("metadata", {}),
                )
            )

            # Update BM25 corpus (only for unique doc_ids)
            str_id = str(doc_id)
            if str_id not in self.bm25_doc_ids:
                tokens = code_tokenize(doc.get("text", ""))
                self.bm25_corpus.append(tokens)
                self.bm25_doc_ids.append(str_id)
                new_bm25_docs = True

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        # Rebuild BM25 if new docs were added
        if new_bm25_docs:
            self._rebuild_bm25()
            self._save_bm25()
