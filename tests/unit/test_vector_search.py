from __future__ import annotations

import pytest

from src.reporag.retrieval.vector_search import RetrievalResult, VectorSearcher


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeMatrix:
    def __init__(self, row: list[float]) -> None:
        self.row = row

    def __getitem__(self, index: int) -> FakeVector:
        assert index == 0
        return FakeVector(self.row)


class FakeCodeEmbedder:
    def embed_batch(self, texts: list[str]) -> FakeMatrix:
        return FakeMatrix([1.0] * 768)


class FakeDocEmbedder:
    def embed_batch(self, docs: list[dict[str, str]]) -> list[dict]:
        return [
            {
                "embedding": FakeVector([1.0] * 384),
                "symbol_id": docs[0]["symbol_id"],
            }
        ]


@pytest.fixture
def searcher() -> VectorSearcher:
    return VectorSearcher(
        code_embedder=FakeCodeEmbedder(),
        doc_embedder=FakeDocEmbedder(),
    )


def test_build_filter(searcher: VectorSearcher) -> None:
    search_filter = searcher._build_filter(
        language="python",
        file_path="src/reporag/parser.py",
        symbol_type="function",
    )

    assert search_filter is not None
    assert len(search_filter.must) == 3

    keys = {condition.key for condition in search_filter.must}

    assert keys == {"language", "file", "symbol"}


def test_search_merges_and_ranks_results(searcher: VectorSearcher) -> None:
    code_result = RetrievalResult(
        score=0.80,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
        symbol="parse",
        chunk_text="def parse(source): ...",
        language="python",
        source="code",
    )

    doc_result = RetrievalResult(
        score=0.95,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
        symbol="parse",
        chunk_text="Parse source code into an AST.",
        language="python",
        source="doc",
    )

    other_result = RetrievalResult(
        score=0.70,
        file_path="src/reporag/utils.py",
        start_line=5,
        end_line=15,
        symbol="helper",
        chunk_text="def helper(): ...",
        language="python",
        source="code",
    )

    def fake_search(vector_name, query_vector, top_k, search_filter):
        if vector_name == "code":
            return [code_result, other_result]
        return [doc_result]

    searcher._search_named_vector = fake_search

    results = searcher.search("parse function", top_k=5)

    assert len(results) == 2
    assert results[0].score == pytest.approx(0.95)
    assert results[0].symbol == "parse"
    assert results[1].symbol == "helper"
