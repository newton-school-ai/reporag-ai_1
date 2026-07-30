"""Vector semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.reporag.embedding.code_embedder import CodeEmbedder
from src.reporag.embedding.doc_embedder import DocEmbedder


@dataclass(slots=True)
class RetrievalResult:
    score: float
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    chunk_text: str
    language: str | None = None
    source: str = "code"


class VectorSearcher:
    """Semantic search over Qdrant named vectors."""

    def __init__(
        self,
        collection_name: str = "reporag",
        qdrant_path: str = ":memory:",
        code_embedder: CodeEmbedder | None = None,
        doc_embedder: DocEmbedder | None = None,
    ) -> None:
        self.collection_name = collection_name

        if qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=qdrant_path)

        self.code_embedder = code_embedder or CodeEmbedder()
        self.doc_embedder = doc_embedder or DocEmbedder()

    def _build_filter(
        self,
        language: str | None,
        file_path: str | None,
        symbol_type: str | None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []

        if language:
            conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        if file_path:
            conditions.append(
                models.FieldCondition(
                    key="file",
                    match=models.MatchValue(value=file_path),
                )
            )

        if symbol_type:
            conditions.append(
                models.FieldCondition(
                    key="symbol",
                    match=models.MatchValue(value=symbol_type),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def _search_named_vector(
        self,
        vector_name: str,
        query_vector: list[float],
        top_k: int,
        search_filter: models.Filter | None,
    ) -> list[RetrievalResult]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[RetrievalResult] = []

        for hit in hits:
            payload: dict[str, Any] = hit.payload or {}

            results.append(
                RetrievalResult(
                    score=float(hit.score),
                    file_path=str(payload.get("file", "")),
                    start_line=payload.get("start_line"),
                    end_line=payload.get("end_line"),
                    symbol=payload.get("symbol"),
                    chunk_text=str(payload.get("text", "")),
                    language=payload.get("language"),
                    source=vector_name,
                )
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        file_path: str | None = None,
        symbol_type: str | None = None,
    ) -> list[RetrievalResult]:
        search_filter = self._build_filter(language, file_path, symbol_type)

        code_vector = self.code_embedder.embed_batch([query])[0].tolist()
        doc_vector = self.doc_embedder.embed_batch(
            [{"text": query, "symbol_id": "__query__"}]
        )[0]["embedding"].tolist()

        code_results = self._search_named_vector(
            vector_name="code",
            query_vector=code_vector,
            top_k=top_k,
            search_filter=search_filter,
        )

        doc_results = self._search_named_vector(
            vector_name="doc",
            query_vector=doc_vector,
            top_k=top_k,
            search_filter=search_filter,
        )

        merged: dict[
            tuple[str, int | None, int | None, str | None], RetrievalResult
        ] = {}

        for result in code_results + doc_results:
            key = (
                result.file_path,
                result.start_line,
                result.end_line,
                result.symbol,
            )

            existing = merged.get(key)
            if existing is None or result.score > existing.score:
                merged[key] = result

        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)

        return ranked[:top_k]
