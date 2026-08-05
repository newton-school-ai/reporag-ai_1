from __future__ import annotations

from collections.abc import Callable

from src.reporag.retrieval.fusion import RetrievalResult


class CrossEncoderReranker:
    # Rerank retrieval candidates using a cross-encoder style scorer

    def __init__(
        self,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self.scorer = scorer or self._default_scorer

    @staticmethod
    def _default_scorer(query: str, text: str) -> float:
        """Lightweight fallback scorer based on token overlap.

        This keeps unit tests fast and avoids requiring transformer
        models during test collection.
        """

        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())

        if not query_tokens:
            return 0.0

        overlap = len(query_tokens & text_tokens)
        return overlap / len(query_tokens)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using the configured scorer."""

        scored: list[RetrievalResult] = []

        for candidate in candidates:
            score = float(self.scorer(query, candidate.chunk_text))

            scored.append(
                RetrievalResult(
                    score=score,
                    file_path=candidate.file_path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    symbol=candidate.symbol,
                    chunk_text=candidate.chunk_text,
                    language=candidate.language,
                    source="reranker",
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)

        if top_k is not None:
            return scored[:top_k]

        return scored
