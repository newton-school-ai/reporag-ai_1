from src.reporag.retrieval.fusion import (
    RetrievalResult,
    RRFFusion,
    reciprocal_rank_fusion,
)


def make_result(symbol: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=f"{symbol}.py",
        start_line=1,
        end_line=2,
        symbol=symbol,
        chunk_text=f"def {symbol}(): pass",
        language="python",
        source="test",
    )


def test_rrf_basic_fusion() -> None:
    vector = [
        make_result("foo"),
        make_result("bar"),
        make_result("baz"),
    ]
    bm25 = [
        make_result("bar"),
        make_result("foo"),
        make_result("qux"),
    ]

    fused = reciprocal_rank_fusion([vector, bm25], k=60)

    assert len(fused) == 4
    assert fused[0].symbol in {"foo", "bar"}
    assert fused[1].symbol in {"foo", "bar"}


def test_rrf_handles_missing_items() -> None:
    vector = [make_result("foo")]
    graph = [make_result("bar")]

    fused = reciprocal_rank_fusion([vector, graph])

    symbols = {r.symbol for r in fused}
    assert symbols == {"foo", "bar"}


def test_rrf_merges_duplicate_results() -> None:
    vector = [make_result("shared")]
    bm25 = [make_result("shared")]

    fused = reciprocal_rank_fusion([vector, bm25])

    assert len(fused) == 1
    assert fused[0].symbol == "shared"
    assert fused[0].score > 0


def test_rrf_top_k() -> None:
    vector = [
        make_result("a"),
        make_result("b"),
        make_result("c"),
    ]

    fusion = RRFFusion()
    fused = fusion.fuse([vector], top_k=2)

    assert len(fused) == 2
    assert fused[0].score >= fused[1].score
