"""Agentic query planner.

Contains the query classifier and query decomposer. Classifies queries
into simple-lookup / multi-hop / exploratory, then decomposes complex
queries into ordered sub-queries using a LangGraph state machine.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """The three canonical query categories."""

    SIMPLE_LOOKUP = "simple-lookup"
    MULTI_HOP = "multi-hop"
    EXPLORATORY = "exploratory"


class ClassificationResult:
    """Result of a query classification.

    Attributes:
        query_type: One of the three QueryType values.
        confidence: Float in [0, 1].  Low confidence causes fallback to
            multi-hop (the safest strategy that avoids returning nothing).
        raw_response: The raw text returned by the LLM, kept for debugging.
    """

    def __init__(
        self,
        query_type: QueryType,
        confidence: float,
        raw_response: str = "",
    ) -> None:
        self.query_type = query_type
        self.confidence: float = max(0.0, min(1.0, confidence))
        self.raw_response = raw_response

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ClassificationResult(query_type={self.query_type!r}, "
            f"confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # simple-lookup
    {
        "query": "Where is the authenticate function defined?",
        "query_type": "simple-lookup",
        "confidence": "0.97",
        "reasoning": "Asks for a single definition location -- one retrieval step suffices.",
    },
    {
        "query": "What are the parameters of the UserModel class?",
        "query_type": "simple-lookup",
        "confidence": "0.95",
        "reasoning": "Asks for a specific attribute list of a single symbol.",
    },
    {
        "query": "Show me the imports in config.py",
        "query_type": "simple-lookup",
        "confidence": "0.96",
        "reasoning": "Asks for a specific section of a single file.",
    },
    {
        "query": "What does the calculate_score function return?",
        "query_type": "simple-lookup",
        "confidence": "0.93",
        "reasoning": "Single function return type -- one lookup step needed.",
    },
    # multi-hop
    {
        "query": "How does a request flow from the API endpoint to the database?",
        "query_type": "multi-hop",
        "confidence": "0.92",
        "reasoning": "Requires tracing through multiple files/modules: API -> router -> service -> DB.",
    },
    {
        "query": "How are embeddings generated and stored in Qdrant?",
        "query_type": "multi-hop",
        "confidence": "0.90",
        "reasoning": "Needs to follow embedding pipeline through multiple modules.",
    },
    {
        "query": "What happens when a user logs in via Google OAuth?",
        "query_type": "multi-hop",
        "confidence": "0.91",
        "reasoning": "Requires tracing OAuth flow through auth, tokens, and session management.",
    },
    # exploratory
    {
        "query": "Explain the overall architecture of this project.",
        "query_type": "exploratory",
        "confidence": "0.94",
        "reasoning": "High-level architecture question needs broad retrieval across many files.",
    },
    {
        "query": "What design patterns are used in this codebase?",
        "query_type": "exploratory",
        "confidence": "0.88",
        "reasoning": "Broad question that requires surveying the entire codebase.",
    },
    {
        "query": "Summarise the retrieval strategy used in this project.",
        "query_type": "exploratory",
        "confidence": "0.85",
        "reasoning": "Broad summarisation across multiple subsystems.",
    },
]


def _build_few_shot_block() -> str:
    """Return the few-shot block as formatted text for the LLM prompt."""
    lines: list[str] = []
    for ex in _FEW_SHOT_EXAMPLES:
        lines.append(f'Query: "{ex["query"]}"')
        lines.append(
            f'Output: {{"query_type": "{ex["query_type"]}", "confidence": {ex["confidence"]}, "reasoning": "{ex["reasoning"]}"}}'
        )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query classification assistant for a code-intelligence RAG system.

Your job is to classify a user query into exactly one of three categories:

1. **simple-lookup**  -- The answer can be found with a single retrieval step.
   Examples: "where is X defined?", "what are the params of Y?", "show me Z file".

2. **multi-hop** -- The answer requires following references across multiple files
   or modules in sequence.
   Examples: "how does X flow from A to B?", "what happens when a user does Y?".

3. **exploratory** -- The query asks for a broad overview, summary, or comparison
   that requires wide retrieval across the codebase.
   Examples: "explain the architecture", "what design patterns are used?".

Return ONLY a JSON object with these keys:
  - "query_type": one of "simple-lookup", "multi-hop", "exploratory"
  - "confidence": a float between 0.0 and 1.0
  - "reasoning": a brief one-sentence explanation

Few-shot examples:
{few_shot_block}
Now classify the following query:
Query: "{query}"
Output:"""


# ---------------------------------------------------------------------------
# LLM client helpers (lazy import to keep the module importable without keys)
# ---------------------------------------------------------------------------


def _get_openai_client() -> Any:
    """Return an OpenAI client instance, or raise ImportError if unavailable."""
    try:
        from openai import OpenAI  # type: ignore[import]

        from src.reporag.config import settings

        return OpenAI(api_key=settings.openai_api_key.get_secret_value())
    except ImportError as e:
        raise ImportError("openai package is required for QueryClassifier") from e


def _call_llm(prompt: str, model: str) -> str:
    """Call the OpenAI chat API and return the response text."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Rule-based fallback classifier
# ---------------------------------------------------------------------------

# Patterns that strongly suggest simple-lookup
_SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhere\s+is\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+are\s+the\s+(params?|parameters?|arguments?|attrs?|attributes?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bshow\s+me\s+the\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+.+\s+return\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+is\s+the\s+(signature|definition|value|type)\b", re.IGNORECASE
    ),
    re.compile(r"\bfind\s+the\s+function\b", re.IGNORECASE),
    re.compile(
        r"\blist\s+(all\s+)?(methods?|functions?|classes?|imports?)\b", re.IGNORECASE
    ),
]

# Patterns that suggest multi-hop
_MULTIHOP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+does\b.+\bflow\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+happens\s+when\b", re.IGNORECASE),
    re.compile(r"\bend.to.end\b", re.IGNORECASE),
    re.compile(r"\bfrom\b.+\bto\b.+\b(through|via|using)\b", re.IGNORECASE),
    re.compile(r"\btrace\b", re.IGNORECASE),
    re.compile(r"\bpipeline\b", re.IGNORECASE),
    re.compile(r"\bworkflow\b", re.IGNORECASE),
]

# Patterns that suggest exploratory
_EXPLORATORY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bexplain\b.+(project|codebase|system|repo)\b", re.IGNORECASE),
    re.compile(r"\bsummar(ise|ize|y)\b", re.IGNORECASE),
    re.compile(r"\bdesign\s+patterns?\b", re.IGNORECASE),
    re.compile(r"\bhow\s+is\s+this\s+(project|codebase|system)\b", re.IGNORECASE),
    re.compile(r"\bhigh.level\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(design|architectural)\b", re.IGNORECASE),
]


def _rule_based_classify(query: str) -> ClassificationResult:
    """Lightweight rule-based classifier used as fallback when LLM is unavailable.

    Evaluation order: simple-lookup > exploratory > multi-hop > default(multi-hop).
    Exploratory is checked before multi-hop so that queries like "what design
    patterns are used?" are not captured by multi-hop keyword heuristics.

    Returns a low-confidence result so the caller can decide whether to trust it.
    """
    for pattern in _SIMPLE_PATTERNS:
        if pattern.search(query):
            return ClassificationResult(QueryType.SIMPLE_LOOKUP, 0.70)

    # Exploratory MUST be checked before multi-hop to avoid false positives
    for pattern in _EXPLORATORY_PATTERNS:
        if pattern.search(query):
            return ClassificationResult(QueryType.EXPLORATORY, 0.65)

    for pattern in _MULTIHOP_PATTERNS:
        if pattern.search(query):
            return ClassificationResult(QueryType.MULTI_HOP, 0.65)

    # Default: multi-hop is the safe fallback
    return ClassificationResult(QueryType.MULTI_HOP, 0.50)


# ---------------------------------------------------------------------------
# QueryClassifier
# ---------------------------------------------------------------------------


class QueryClassifier:
    """LLM-based query classifier with few-shot examples.

    Classifies a user query into one of three categories:
        - **simple-lookup**: a single retrieval step is sufficient.
        - **multi-hop**: answer requires tracing through multiple modules.
        - **exploratory**: broad question needing wide codebase coverage.

    Falls back to ``multi-hop`` when the confidence score is below
    ``confidence_threshold``, because multi-hop is the safest strategy
    (it never returns nothing).

    Args:
        model: OpenAI model name to use for classification.
        confidence_threshold: Minimum confidence to trust a non-multi-hop
            classification.  Defaults to 0.60.
        use_llm: If ``False`` the rule-based fallback is always used (useful
            for testing without API keys).

    Example::

        classifier = QueryClassifier()
        result = classifier.classify("Where is authenticate defined?")
        print(result.query_type, result.confidence)
    """

    def __init__(
        self,
        model: str | None = None,
        confidence_threshold: float = 0.60,
        use_llm: bool = True,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._use_llm = use_llm

        if model is None:
            try:
                from src.reporag.config import settings

                self._model = settings.openai_model
            except Exception:
                self._model = "gpt-4o"
        else:
            self._model = model

        self._few_shot_block = _build_few_shot_block()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query and return a :class:`ClassificationResult`.

        If the LLM is unavailable or returns an unparseable response, the
        method falls back to the rule-based classifier.  If the final
        confidence is below ``confidence_threshold``, the result is
        overridden to ``multi-hop``.

        Args:
            query: The user's natural-language query string.

        Returns:
            A :class:`ClassificationResult` with ``query_type`` and
            ``confidence`` populated.
        """
        query = query.strip()
        if not query:
            # Empty query -- safe default
            return ClassificationResult(QueryType.MULTI_HOP, 0.50)

        if self._use_llm:
            result = self._classify_with_llm(query)
        else:
            result = _rule_based_classify(query)

        # Low-confidence fallback to multi-hop
        if result.confidence < self._confidence_threshold:
            logger.debug(
                "Confidence %.2f below threshold %.2f for query %r; "
                "falling back to multi-hop.",
                result.confidence,
                self._confidence_threshold,
                query,
            )
            result.query_type = QueryType.MULTI_HOP

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_with_llm(self, query: str) -> ClassificationResult:
        """Call the LLM and parse the JSON response.

        Falls back to the rule-based classifier on any error.
        """
        prompt = _SYSTEM_PROMPT.format(
            few_shot_block=self._few_shot_block,
            query=query,
        )
        try:
            raw = _call_llm(prompt, self._model)
            return self._parse_llm_response(raw, query)
        except Exception as exc:
            logger.warning(
                "LLM classification failed (%s); falling back to rule-based.", exc
            )
            return _rule_based_classify(query)

    def _parse_llm_response(self, raw: str, query: str) -> ClassificationResult:
        """Parse the LLM JSON response into a :class:`ClassificationResult`.

        Handles cases where the model wraps the JSON in markdown fences.
        Falls back to the rule-based classifier if parsing fails.
        """
        # Strip markdown fences if present
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM response: %r", raw)
                    return _rule_based_classify(query)
            else:
                logger.warning("No JSON found in LLM response: %r", raw)
                return _rule_based_classify(query)

        # Validate fields
        raw_type = data.get("query_type", "").strip().lower()
        try:
            query_type = QueryType(raw_type)
        except ValueError:
            logger.warning(
                "Unknown query_type %r from LLM; using rule-based.", raw_type
            )
            return _rule_based_classify(query)

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        return ClassificationResult(
            query_type=query_type,
            confidence=confidence,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# QueryDecomposer -- placeholder (Issue 21)
# ---------------------------------------------------------------------------

# TODO: Implement in Issue 21
#
# QueryDecomposer:
# - LangGraph state machine for decomposition
# - Input: complex query + repo context (modules, key symbols)
# - Output: ordered list of SubQuery objects with dependency edges
# - Each SubQuery: text, expected_answer_type, context_from (prior IDs)
# - Handles queries that do not need decomposition (single step)
