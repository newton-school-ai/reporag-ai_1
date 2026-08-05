from __future__ import annotations

from dataclasses import dataclass


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


class RRFFusion:
    """Fuse multiple ranked retrieval result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    @staticmethod
    def _result_key(
        result: RetrievalResult,
    ) -> tuple[str, int | None, int | None, str | None]:
        """Create a stable identity key for deduplication across ranked lists."""
        return (
            result.file_path,
            result.start_line,
            result.end_line,
            result.symbol,
        )

    def fuse(
        self,
        ranked_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion.

        RRF score:
            score = sum(1 / (k + rank))

        where rank is 1-indexed within each ranked list.

        Items missing from a list simply contribute nothing from that list.
        """
        fused_scores: dict[
            tuple[str, int | None, int | None, str | None],
            float,
        ] = {}

        representatives: dict[
            tuple[str, int | None, int | None, str | None],
            RetrievalResult,
        ] = {}

        for ranked in ranked_lists:
            for rank, result in enumerate(ranked, start=1):
                key = self._result_key(result)

                representatives.setdefault(key, result)

                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.k + rank)

        fused_results: list[RetrievalResult] = []

        for key, score in fused_scores.items():
            original = representatives[key]

            fused_results.append(
                RetrievalResult(
                    score=float(score),
                    file_path=original.file_path,
                    start_line=original.start_line,
                    end_line=original.end_line,
                    symbol=original.symbol,
                    chunk_text=original.chunk_text,
                    language=original.language,
                    source="fusion",
                )
            )

        fused_results.sort(key=lambda result: result.score, reverse=True)

        if top_k is not None:
            return fused_results[:top_k]

        return fused_results


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Convenience function for Reciprocal Rank Fusion."""
    return RRFFusion(k=k).fuse(
        ranked_lists=ranked_lists,
        top_k=top_k,
    )
