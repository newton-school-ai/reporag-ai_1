"""BM25 sparse keyword search.

Queries a BM25 index with code-aware tokenization. Excels at finding
exact identifier matches that vector search may miss.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


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


@dataclass(slots=True)
class BM25Document:
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    chunk_text: str
    language: str | None = None
    source: str = "code"


class BM25Searcher:
    """Sparse keyword search with code-aware tokenization."""

    def __init__(
        self,
        documents: list[BM25Document] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        exact_name_boost: float = 2.0,
    ) -> None:
        self.documents = documents or []
        self.k1 = k1
        self.b = b
        self.exact_name_boost = exact_name_boost

        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len: float = 0.0

        if self.documents:
            self._build_index()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize code-aware text.

        Splits snake_case, camelCase, PascalCase, dotted identifiers,
        and punctuation into searchable lowercase tokens.
        """

        tokens: list[str] = []

        for raw in re.split(r"[^A-Za-z0-9_.]+", text):
            if not raw:
                continue

            dotted = raw.replace(".", " ")
            for part in dotted.split():
                snake_parts = part.split("_")
                for snake in snake_parts:
                    if not snake:
                        continue

                    camel_parts = _CAMEL_RE.sub(" ", snake).split()

                    if camel_parts:
                        for cp in camel_parts:
                            tokens.append(cp.lower())
                    else:
                        tokens.append(snake.lower())

        return tokens

    def _build_index(self) -> None:
        """Build BM25 statistics from documents."""

        self.doc_tokens = []

        for doc in self.documents:
            tokens = self.tokenize(doc.chunk_text)

            if doc.symbol:
                tokens.extend(self.tokenize(doc.symbol))

            self.doc_tokens.append(tokens)

            for token in set(tokens):
                self.doc_freq[token] += 1

        if self.doc_tokens:
            self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / len(
                self.doc_tokens
            )
        else:
            self.avg_doc_len = 0.0

    def add_documents(self, documents: list[BM25Document]) -> None:

        self.documents.extend(documents)
        self._build_index()

    def _idf(self, token: str) -> float:

        n_docs = len(self.doc_tokens)
        df = self.doc_freq.get(token, 0)

        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _score_document(
        self,
        query_tokens: list[str],
        doc_index: int,
    ) -> float:
        """Compute BM25 score for a document."""

        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return 0.0

        tf = Counter(tokens)
        doc_len = len(tokens)

        score = 0.0

        for token in query_tokens:
            freq = tf.get(token, 0)

            if freq == 0:
                continue

            idf = self._idf(token)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1.0))
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        file_path: str | None = None,
        symbol_type: str | None = None,
    ) -> list[RetrievalResult]:

        if not self.documents:
            return []

        query_tokens = self.tokenize(query)
        query_lower = query.strip().lower()

        scored: list[tuple[float, BM25Document]] = []

        for index, doc in enumerate(self.documents):
            if language and doc.language != language:
                continue

            if file_path and doc.file_path != file_path:
                continue

            score = self._score_document(query_tokens, index)

            if doc.symbol and doc.symbol.lower() == query_lower:
                score *= self.exact_name_boost

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievalResult] = []

        for score, doc in scored[:top_k]:
            results.append(
                RetrievalResult(
                    score=float(score),
                    file_path=doc.file_path,
                    start_line=doc.start_line,
                    end_line=doc.end_line,
                    symbol=doc.symbol,
                    chunk_text=doc.chunk_text,
                    language=doc.language,
                    source="bm25",
                )
            )

        return results
