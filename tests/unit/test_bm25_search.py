"""Unit tests for bm25_search module."""

from src.reporag.retrieval.bm25_search import BM25Document, BM25Searcher


def test_code_aware_tokenization() -> None:
    tokens = BM25Searcher.tokenize(
        "get_user_data camelCaseValue PascalCaseValue os.path.join"
    )

    assert "get" in tokens
    assert "user" in tokens
    assert "data" in tokens
    assert "camel" in tokens
    assert "case" in tokens
    assert "value" in tokens
    assert "pascal" in tokens
    assert "os" in tokens
    assert "path" in tokens
    assert "join" in tokens


def test_exact_identifier_ranked_first() -> None:
    docs = [
        BM25Document(
            file_path="a.py",
            start_line=1,
            end_line=2,
            symbol="helper",
            chunk_text="def helper(): pass",
            language="python",
        ),
        BM25Document(
            file_path="b.py",
            start_line=1,
            end_line=2,
            symbol="target_func",
            chunk_text="def target_func(): pass",
            language="python",
        ),
        BM25Document(
            file_path="c.py",
            start_line=1,
            end_line=2,
            symbol="other",
            chunk_text="def other(): target_func()",
            language="python",
        ),
    ]

    searcher = BM25Searcher(documents=docs)
    results = searcher.search("target_func", top_k=3)

    assert len(results) >= 1
    assert results[0].symbol == "target_func"
    assert results[0].file_path == "b.py"


def test_exact_name_boosting() -> None:
    docs = [
        BM25Document(
            file_path="def.py",
            start_line=1,
            end_line=2,
            symbol="compute",
            chunk_text="def compute(): pass",
            language="python",
        ),
        BM25Document(
            file_path="use.py",
            start_line=10,
            end_line=12,
            symbol="caller",
            chunk_text="compute() compute() compute()",
            language="python",
        ),
    ]

    searcher = BM25Searcher(documents=docs, exact_name_boost=3.0)
    results = searcher.search("compute", top_k=2)

    assert results[0].symbol == "compute"


def test_language_and_file_filters() -> None:
    docs = [
        BM25Document(
            file_path="a.py",
            start_line=1,
            end_line=2,
            symbol="foo",
            chunk_text="def foo(): pass",
            language="python",
        ),
        BM25Document(
            file_path="b.ts",
            start_line=1,
            end_line=2,
            symbol="foo",
            chunk_text="function foo() {}",
            language="typescript",
        ),
    ]

    searcher = BM25Searcher(documents=docs)

    py_results = searcher.search("foo", language="python")
    assert len(py_results) == 1
    assert py_results[0].file_path == "a.py"

    ts_results = searcher.search("foo", file_path="b.ts")
    assert len(ts_results) == 1
    assert ts_results[0].language == "typescript"
