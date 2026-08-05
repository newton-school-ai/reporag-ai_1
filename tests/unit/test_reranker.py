from src.reporag.retrieval.fusion import RetrievalResult
from src.reporag.retrieval.reranker import CrossEncoderReranker


def make_result(symbol: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        score=0.0,
        file_path=f"{symbol}.py",
        start_line=1,
        end_line=2,
        symbol=symbol,
        chunk_text=text,
        language="python",
        source="fusion",
    )


def test_reranks_by_default_score() -> None:
    reranker = CrossEncoderReranker()

    candidates = [
        make_result("foo", "helper function"),
        make_result("bar", "target function compute value"),
        make_result("baz", "unrelated text"),
    ]

    ranked = reranker.rerank("compute value", candidates)

    assert ranked[0].symbol == "bar"
    assert ranked[0].score >= ranked[1].score


def test_custom_scorer_reorders() -> None:
    scores = {
        "alpha": 0.1,
        "beta": 0.9,
        "gamma": 0.4,
    }

    def scorer(_query: str, text: str) -> float:
        return scores[text]

    reranker = CrossEncoderReranker(scorer=scorer)

    candidates = [
        make_result("a", "alpha"),
        make_result("b", "beta"),
        make_result("c", "gamma"),
    ]

    ranked = reranker.rerank("q", candidates)

    assert [r.symbol for r in ranked] == ["b", "c", "a"]


def test_top_k_limit() -> None:
    reranker = CrossEncoderReranker()

    candidates = [
        make_result("a", "one"),
        make_result("b", "one two"),
        make_result("c", "one two three"),
    ]

    ranked = reranker.rerank("one two three", candidates, top_k=2)

    assert len(ranked) == 2


def test_metadata_preserved() -> None:
    reranker = CrossEncoderReranker()

    candidate = RetrievalResult(
        score=1.0,
        file_path="src/example.py",
        start_line=10,
        end_line=20,
        symbol="example",
        chunk_text="example function body",
        language="python",
        source="fusion",
    )

    ranked = reranker.rerank("example", [candidate])

    result = ranked[0]
    assert result.file_path == "src/example.py"
    assert result.start_line == 10
    assert result.end_line == 20
    assert result.symbol == "example"
    assert result.language == "python"
    assert result.source == "reranker"
