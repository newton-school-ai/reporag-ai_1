"""Unit tests for the QueryClassifier in planner.py (Issue 20).

Tests cover:
 - Each query type (simple-lookup, multi-hop, exploratory)
 - Confidence score range validation (0-1)
 - Low-confidence fallback to multi-hop
 - Rule-based classifier patterns
 - Edge cases: empty query, unknown LLM output, JSON parsing robustness
 - 10+ representative queries across all categories
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.reporag.agent.planner import (
    ClassificationResult,
    QueryClassifier,
    QueryType,
    _rule_based_classify,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_classifier(use_llm: bool = False, threshold: float = 0.60) -> QueryClassifier:
    """Return a QueryClassifier that uses the rule-based fallback (no API key needed)."""
    return QueryClassifier(use_llm=use_llm, confidence_threshold=threshold)


def mock_llm_response(
    query_type: str, confidence: float, reasoning: str = "test"
) -> str:
    """Return a JSON string mimicking an LLM response."""
    import json

    return json.dumps(
        {"query_type": query_type, "confidence": confidence, "reasoning": reasoning}
    )


# ---------------------------------------------------------------------------
# ClassificationResult basic tests
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_confidence_clamped_below_zero(self) -> None:
        result = ClassificationResult(QueryType.SIMPLE_LOOKUP, -0.5)
        assert result.confidence == 0.0

    def test_confidence_clamped_above_one(self) -> None:
        result = ClassificationResult(QueryType.MULTI_HOP, 1.5)
        assert result.confidence == 1.0

    def test_confidence_in_range(self) -> None:
        result = ClassificationResult(QueryType.EXPLORATORY, 0.75)
        assert 0.0 <= result.confidence <= 1.0

    def test_query_type_stored(self) -> None:
        result = ClassificationResult(QueryType.MULTI_HOP, 0.8)
        assert result.query_type == QueryType.MULTI_HOP


# ---------------------------------------------------------------------------
# QueryType enum tests
# ---------------------------------------------------------------------------


class TestQueryTypeEnum:
    def test_simple_lookup_value(self) -> None:
        assert QueryType.SIMPLE_LOOKUP.value == "simple-lookup"

    def test_multi_hop_value(self) -> None:
        assert QueryType.MULTI_HOP.value == "multi-hop"

    def test_exploratory_value(self) -> None:
        assert QueryType.EXPLORATORY.value == "exploratory"


# ---------------------------------------------------------------------------
# Rule-based classifier  (10+ query tests)
# ---------------------------------------------------------------------------


class TestRuleBasedClassifier:
    """Test the rule-based fallback with representative queries across all types."""

    # Simple-lookup queries (4 examples)
    def test_simple_lookup_where_is_defined(self) -> None:
        result = _rule_based_classify("Where is the authenticate function defined?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP

    def test_simple_lookup_parameters(self) -> None:
        result = _rule_based_classify("What are the parameters of the UserModel class?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP

    def test_simple_lookup_show_me(self) -> None:
        result = _rule_based_classify("Show me the imports in config.py")
        assert result.query_type == QueryType.SIMPLE_LOOKUP

    def test_simple_lookup_what_does_return(self) -> None:
        result = _rule_based_classify("What does the calculate_score function return?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP

    # Exploratory queries (3 examples checked before multi-hop)
    def test_exploratory_architecture(self) -> None:
        result = _rule_based_classify(
            "Explain the overall architecture of this project."
        )
        assert result.query_type == QueryType.EXPLORATORY

    def test_exploratory_design_patterns(self) -> None:
        result = _rule_based_classify("What design patterns are used in this codebase?")
        assert result.query_type == QueryType.EXPLORATORY

    def test_exploratory_summarise(self) -> None:
        result = _rule_based_classify("Summarise the retrieval strategy used here.")
        assert result.query_type == QueryType.EXPLORATORY

    # Multi-hop queries (3 examples)
    def test_multihop_what_happens_when(self) -> None:
        result = _rule_based_classify(
            "What happens when a user logs in via Google OAuth?"
        )
        assert result.query_type == QueryType.MULTI_HOP

    def test_multihop_pipeline(self) -> None:
        result = _rule_based_classify("Describe the embedding pipeline step by step.")
        assert result.query_type == QueryType.MULTI_HOP

    def test_multihop_workflow(self) -> None:
        result = _rule_based_classify("Walk me through the ingestion workflow.")
        assert result.query_type == QueryType.MULTI_HOP

    # Default fallback -- ambiguous query
    def test_default_fallback_to_multihop(self) -> None:
        result = _rule_based_classify("Tell me about dependency injection here.")
        assert result.query_type == QueryType.MULTI_HOP

    # Confidence range
    def test_confidence_always_in_range(self) -> None:
        queries = [
            "Where is get_user defined?",
            "How does caching work end-to-end?",
            "Explain the overall project structure.",
            "What is the return type of parse_ast?",
            "What happens when an embedding fails?",
        ]
        for q in queries:
            result = _rule_based_classify(q)
            assert 0.0 <= result.confidence <= 1.0, f"Out-of-range confidence for: {q}"


# ---------------------------------------------------------------------------
# QueryClassifier (rule-based mode, no LLM)
# ---------------------------------------------------------------------------


class TestQueryClassifierRuleBased:
    def test_simple_lookup_classified_correctly(self) -> None:
        clf = make_classifier()
        result = clf.classify("Where is the authenticate function defined?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP
        assert 0.0 <= result.confidence <= 1.0

    def test_multi_hop_classified_correctly(self) -> None:
        clf = make_classifier()
        result = clf.classify("What happens when a user logs in?")
        assert result.query_type == QueryType.MULTI_HOP

    def test_exploratory_classified_correctly(self) -> None:
        clf = make_classifier()
        result = clf.classify("Explain the overall architecture of this project.")
        assert result.query_type == QueryType.EXPLORATORY

    def test_empty_query_returns_multihop(self) -> None:
        clf = make_classifier()
        result = clf.classify("")
        assert result.query_type == QueryType.MULTI_HOP

    def test_whitespace_only_query_returns_multihop(self) -> None:
        clf = make_classifier()
        result = clf.classify("   ")
        assert result.query_type == QueryType.MULTI_HOP

    def test_confidence_in_valid_range(self) -> None:
        clf = make_classifier()
        for query in [
            "Where is UserService defined?",
            "How does the auth pipeline work?",
            "Give me a high-level overview of the project.",
        ]:
            result = clf.classify(query)
            assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_falls_back_to_multihop(self) -> None:
        """If the rule-based score is below threshold, result must be multi-hop."""
        # Force a 0.50 confidence (below default 0.60 threshold) by using
        # an ambiguous query that hits the default branch
        clf = QueryClassifier(use_llm=False, confidence_threshold=0.60)
        result = clf.classify("Tell me something interesting about this repo.")
        # Default branch returns multi-hop at 0.50, which is below 0.60 -> stays multi-hop
        assert result.query_type == QueryType.MULTI_HOP

    def test_high_threshold_always_falls_back_to_multihop(self) -> None:
        """With threshold=1.0, every classification falls back to multi-hop."""
        clf = QueryClassifier(use_llm=False, confidence_threshold=1.0)
        result = clf.classify("Where is the parse function defined?")
        assert result.query_type == QueryType.MULTI_HOP


# ---------------------------------------------------------------------------
# QueryClassifier (LLM mode, mocked)
# ---------------------------------------------------------------------------


class TestQueryClassifierLLM:
    def test_llm_simple_lookup_parsed_correctly(self) -> None:
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = mock_llm_response("simple-lookup", 0.97)
            result = clf.classify("Where is the parse_ast function defined?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP
        assert result.confidence == pytest.approx(0.97)

    def test_llm_multi_hop_parsed_correctly(self) -> None:
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = mock_llm_response("multi-hop", 0.91)
            result = clf.classify("How does a request flow from API to database?")
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == pytest.approx(0.91)

    def test_llm_exploratory_parsed_correctly(self) -> None:
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = mock_llm_response("exploratory", 0.88)
            result = clf.classify("Explain the codebase architecture.")
        assert result.query_type == QueryType.EXPLORATORY
        assert result.confidence == pytest.approx(0.88)

    def test_llm_low_confidence_falls_back_to_multihop(self) -> None:
        """LLM returns simple-lookup at 0.40 confidence -- must fall back to multi-hop."""
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = mock_llm_response("simple-lookup", 0.40)
            result = clf.classify("Something about the codebase?")
        assert result.query_type == QueryType.MULTI_HOP

    def test_llm_markdown_fenced_json_parsed(self) -> None:
        """LLM sometimes wraps JSON in ```json fences; must still parse."""
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        fenced = '```json\n{"query_type": "simple-lookup", "confidence": 0.95, "reasoning": "test"}\n```'
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = fenced
            result = clf.classify("Where is the login function?")
        assert result.query_type == QueryType.SIMPLE_LOOKUP
        assert result.confidence == pytest.approx(0.95)

    def test_llm_invalid_json_falls_back_to_rule_based(self) -> None:
        """Unparseable LLM output must trigger rule-based fallback gracefully."""
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = "I cannot classify this query."
            result = clf.classify("Where is authenticate defined?")
        # Rule-based kicks in; result is still valid
        assert result.query_type in QueryType.__members__.values()
        assert 0.0 <= result.confidence <= 1.0

    def test_llm_unknown_query_type_falls_back_to_rule_based(self) -> None:
        """If LLM returns an unrecognised type, fall back to rule-based."""
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = mock_llm_response("unknown-type", 0.90)
            result = clf.classify("Where is parse defined?")
        assert result.query_type in QueryType.__members__.values()

    def test_llm_exception_falls_back_to_rule_based(self) -> None:
        """Network errors from LLM must trigger rule-based gracefully."""
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        with patch(
            "src.reporag.agent.planner._call_llm", side_effect=Exception("timeout")
        ):
            result = clf.classify("Where is the User class defined?")
        assert result.query_type in QueryType.__members__.values()
        assert 0.0 <= result.confidence <= 1.0

    def test_llm_raw_response_stored_on_result(self) -> None:
        clf = QueryClassifier(use_llm=True, confidence_threshold=0.60)
        raw = mock_llm_response("multi-hop", 0.85)
        with patch("src.reporag.agent.planner._call_llm") as mock_llm:
            mock_llm.return_value = raw
            result = clf.classify("How does the auth flow work end-to-end?")
        assert result.raw_response == raw


# ---------------------------------------------------------------------------
# Acceptance-criteria spot checks (10 representative queries)
# ---------------------------------------------------------------------------


ACCEPTANCE_QUERIES: list[tuple[str, QueryType]] = [
    # simple-lookup (4)
    ("Where is the authenticate function defined?", QueryType.SIMPLE_LOOKUP),
    ("What are the parameters of UserModel?", QueryType.SIMPLE_LOOKUP),
    ("Show me the imports in config.py", QueryType.SIMPLE_LOOKUP),
    ("What does calculate_score return?", QueryType.SIMPLE_LOOKUP),
    # multi-hop (3)
    ("What happens when a user logs in via Google OAuth?", QueryType.MULTI_HOP),
    ("Describe the embedding pipeline step by step.", QueryType.MULTI_HOP),
    ("Walk me through the ingestion workflow.", QueryType.MULTI_HOP),
    # exploratory (3)
    ("Explain the overall architecture of this project.", QueryType.EXPLORATORY),
    ("What design patterns are used in this codebase?", QueryType.EXPLORATORY),
    ("Summarise the retrieval strategy used here.", QueryType.EXPLORATORY),
]


@pytest.mark.parametrize("query,expected_type", ACCEPTANCE_QUERIES)
def test_acceptance_criteria_rule_based(query: str, expected_type: QueryType) -> None:
    """Each of the 10 representative queries must be classified correctly."""
    clf = QueryClassifier(use_llm=False, confidence_threshold=0.60)
    result = clf.classify(query)
    assert (
        result.query_type == expected_type
    ), f"Expected {expected_type!r} for query {query!r}, got {result.query_type!r}"
    assert 0.0 <= result.confidence <= 1.0
