"""Agentic query planner.

Contains the query classifier and query decomposer. Classifies queries
into simple-lookup / multi-hop / exploratory, then decomposes complex
queries into ordered sub-queries using a LangGraph state machine.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
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


class AnswerType(str, Enum):
    """Expected answer format for a sub-query."""

    CODE = "code"
    EXPLANATION = "explanation"
    LIST = "list"


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


# ---------------------------------------------------------------------------
# Issue 21 -- QueryDecomposer (LangGraph state machine)
# ---------------------------------------------------------------------------


@dataclass
class SubQuery:
    """A single step in a decomposed query plan.

    Attributes:
        id: Zero-based integer step index (e.g. 0, 1, 2).
        query: The natural-language sub-query text.
        expected_answer_type: What kind of answer is expected
            (``code``, ``explanation``, or ``list``).
        depends_on: List of step IDs whose results this step needs as context.
            Empty for the first step.
        context_from: Alias for ``depends_on`` used in prompt templates --
            the IDs of prior steps whose retrieved context should be injected.
    """

    id: int
    query: str
    expected_answer_type: AnswerType = AnswerType.EXPLANATION
    depends_on: list[int] = field(default_factory=list)

    @property
    def context_from(self) -> list[int]:
        """Alias for ``depends_on`` -- prior step IDs to inject as context."""
        return self.depends_on

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SubQuery(id={self.id}, query={self.query!r}, "
            f"answer_type={self.expected_answer_type.value!r}, "
            f"depends_on={self.depends_on!r})"
        )


@dataclass
class DecompositionPlan:
    """The output of :class:`QueryDecomposer`.

    Attributes:
        original_query: The raw query that was decomposed.
        steps: Ordered list of :class:`SubQuery` objects.  Steps are ordered
            so that all dependencies appear before the step that needs them.
    """

    original_query: str
    steps: list[SubQuery] = field(default_factory=list)

    @property
    def is_single_step(self) -> bool:
        """True when the query did not need decomposition."""
        return len(self.steps) == 1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DecompositionPlan(steps={len(self.steps)}, query={self.original_query!r})"
        )


# -- LangGraph state ---------------------------------------------------------

# DecomposerState is a plain dict used as the LangGraph state object.
# Keys:
#   query        : str  -- original query
#   repo_context : dict -- module names, key symbols, etc.
#   raw_json     : str  -- LLM response (set by decompose_node)
#   plan         : DecompositionPlan | None  -- set by parse_node
#   error        : str | None -- set if any node fails


def _make_initial_state(
    query: str, repo_context: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "query": query,
        "repo_context": repo_context or {},
        "raw_json": "",
        "plan": None,
        "error": None,
    }


# -- Prompt ------------------------------------------------------------------

_DECOMPOSER_SYSTEM_PROMPT = """\
You are a query decomposition assistant for a code-intelligence RAG system.

Your task: break a complex multi-hop query about a software repository into
2-5 ordered sub-queries. Each sub-query should retrieve one specific piece
of information needed to answer the full question.

Repository context:
  modules   : {modules}
  key_symbols: {key_symbols}

Rules:
1. Return between 1 and 5 sub-queries. Return exactly 1 if the query is
   simple enough to answer in a single retrieval step.
2. Sub-queries must be ordered so that earlier steps provide context for
   later ones.
3. Each sub-query must specify:
     - "id"                  : integer starting at 0
     - "query"               : the sub-query text
     - "expected_answer_type": one of "code", "explanation", "list"
     - "depends_on"          : list of step IDs this step needs as context
                               (empty list [] for step 0)
4. Return ONLY a JSON object:
   {{
     "steps": [ ... ]
   }}

Few-shot examples:

Query: "How does a request flow from the API endpoint to the database?"
Output:
{{
  "steps": [
    {{"id": 0, "query": "Find the API endpoint handler function for the main request path.", "expected_answer_type": "code", "depends_on": []}},
    {{"id": 1, "query": "Trace how the handler calls the service or business logic layer.", "expected_answer_type": "code", "depends_on": [0]}},
    {{"id": 2, "query": "Find how the service layer interacts with the database ORM or query layer.", "expected_answer_type": "code", "depends_on": [1]}},
    {{"id": 3, "query": "Summarise the full request-to-database flow.", "expected_answer_type": "explanation", "depends_on": [0, 1, 2]}}
  ]
}}

Query: "Where is the UserModel class defined?"
Output:
{{
  "steps": [
    {{"id": 0, "query": "Where is the UserModel class defined?", "expected_answer_type": "code", "depends_on": []}}
  ]
}}

Query: "How are embeddings generated and indexed into Qdrant?"
Output:
{{
  "steps": [
    {{"id": 0, "query": "Find the embedding generation function or class.", "expected_answer_type": "code", "depends_on": []}},
    {{"id": 1, "query": "Find where embeddings are inserted or upserted into Qdrant.", "expected_answer_type": "code", "depends_on": [0]}},
    {{"id": 2, "query": "Explain the end-to-end embedding and indexing pipeline.", "expected_answer_type": "explanation", "depends_on": [0, 1]}}
  ]
}}

Now decompose this query:
Query: "{query}"
Output:"""

# -- LangGraph nodes ---------------------------------------------------------


def _build_langgraph_app() -> Any:
    """Build and return the compiled LangGraph application.

    The graph has three nodes:
      decompose  -- calls the LLM to produce a raw JSON decomposition
      parse      -- parses the raw JSON into a DecompositionPlan
      fallback   -- used when decompose fails; produces a single-step plan

    State transitions:
      START -> decompose -> parse -> END
                         -> fallback -> END  (on parse error)
      START -> fallback -> END               (on LLM error)
    """
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("langgraph is required for QueryDecomposer") from exc

    def decompose_node(state: dict[str, Any]) -> dict[str, Any]:
        """Node: call LLM to decompose the query."""
        query: str = state["query"]
        repo_context: dict[str, Any] = state.get("repo_context", {})
        modules = repo_context.get("modules", [])
        key_symbols = repo_context.get("key_symbols", [])

        prompt = _DECOMPOSER_SYSTEM_PROMPT.format(
            modules=", ".join(modules) if modules else "unknown",
            key_symbols=", ".join(key_symbols) if key_symbols else "unknown",
            query=query,
        )
        try:
            raw = _call_llm(prompt, "gpt-4o")
            return {**state, "raw_json": raw}
        except Exception as exc:
            logger.warning("Decomposer LLM call failed (%s); using fallback.", exc)
            return {**state, "raw_json": "", "error": str(exc)}

    def parse_node(state: dict[str, Any]) -> dict[str, Any]:
        """Node: parse raw JSON into a DecompositionPlan."""
        raw: str = state.get("raw_json", "")
        query: str = state["query"]

        if not raw or state.get("error"):
            plan = _single_step_plan(query)
            return {**state, "plan": plan}

        plan = _parse_decomposition_json(raw, query)
        return {**state, "plan": plan}

    graph = StateGraph(dict)
    graph.add_node("decompose", decompose_node)
    graph.add_node("parse", parse_node)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "parse")
    graph.add_edge("parse", END)

    return graph.compile()


# -- Parsing helpers ---------------------------------------------------------


def _parse_decomposition_json(raw: str, original_query: str) -> DecompositionPlan:
    """Parse the LLM decomposition JSON into a :class:`DecompositionPlan`."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse decomposition JSON: %r", raw)
                return _single_step_plan(original_query)
        else:
            logger.warning("No JSON in decomposition response: %r", raw)
            return _single_step_plan(original_query)

    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        return _single_step_plan(original_query)

    steps: list[SubQuery] = []
    for i, s in enumerate(raw_steps[:5]):  # cap at 5
        try:
            answer_type = AnswerType(s.get("expected_answer_type", "explanation"))
        except ValueError:
            answer_type = AnswerType.EXPLANATION

        depends_on = s.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []

        steps.append(
            SubQuery(
                id=i,
                query=str(s.get("query", original_query)),
                expected_answer_type=answer_type,
                depends_on=[int(d) for d in depends_on if isinstance(d, int | float)],
            )
        )

    if not steps:
        return _single_step_plan(original_query)

    return DecompositionPlan(original_query=original_query, steps=steps)


def _single_step_plan(query: str) -> DecompositionPlan:
    """Return a no-op plan with a single step (no decomposition needed)."""
    return DecompositionPlan(
        original_query=query,
        steps=[
            SubQuery(
                id=0,
                query=query,
                expected_answer_type=AnswerType.EXPLANATION,
                depends_on=[],
            )
        ],
    )


# Multi-hop hint patterns for the rule-based decomposer
_MULTIHOP_DECOMPOSE_RE = re.compile(
    r"\b(flow|pipeline|workflow|happen|end.to.end|trace|step.by.step"
    r"|generat|stored|indexed|ingestion|authentication|go\s+from)\b",
    re.IGNORECASE,
)


def _rule_based_decompose(
    query: str, repo_context: dict[str, Any] | None
) -> DecompositionPlan:
    """Simple rule-based decomposer used when LLM is unavailable.

    Produces a 2-step plan for multi-hop queries: locate then explain.
    Returns a single-step plan for all others.
    """
    ctx = repo_context or {}
    modules: list[str] = ctx.get("modules", [])

    if _MULTIHOP_DECOMPOSE_RE.search(query):
        locate_query = f"Find the relevant functions and classes involved in: {query}"
        if modules:
            locate_query += f" Focus on modules: {', '.join(modules[:4])}."

        return DecompositionPlan(
            original_query=query,
            steps=[
                SubQuery(
                    id=0,
                    query=locate_query,
                    expected_answer_type=AnswerType.CODE,
                    depends_on=[],
                ),
                SubQuery(
                    id=1,
                    query=f"Explain how the components work together to answer: {query}",
                    expected_answer_type=AnswerType.EXPLANATION,
                    depends_on=[0],
                ),
            ],
        )

    return _single_step_plan(query)


# -- Public API --------------------------------------------------------------


class QueryDecomposer:
    """LangGraph-based query decomposer.

    Breaks a complex multi-hop query into 1-5 ordered :class:`SubQuery` steps
    with explicit dependency edges.  Each step carries the sub-query text,
    the expected answer type (``code`` / ``explanation`` / ``list``), and the
    IDs of prior steps whose retrieved context should be injected.

    For simple queries (e.g. single symbol lookups), the decomposer returns a
    :class:`DecompositionPlan` with exactly one step -- no decomposition needed.

    The LangGraph state machine has three nodes:
        - **decompose**: call the LLM with a few-shot prompt
        - **parse**: convert raw JSON into a :class:`DecompositionPlan`
        - **fallback**: produce a single-step plan on any error

    When ``use_llm=False`` (testing without API keys), a lightweight rule-based
    heuristic is used instead.

    Args:
        model: OpenAI model used for decomposition. Defaults to ``gpt-4o``.
        use_llm: Set to ``False`` to use the rule-based fallback (no API key
            needed -- useful for unit tests).
        max_steps: Hard cap on the number of sub-queries (1-5). Defaults to 5.

    Example::

        decomposer = QueryDecomposer()
        plan = decomposer.decompose(
            "How does a request go from the API endpoint to the database?",
            repo_context={"modules": ["api", "routes", "db", "models"]},
        )
        for step in plan.steps:
            print(step.id, step.query, step.depends_on)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        use_llm: bool = True,
        max_steps: int = 5,
    ) -> None:
        self._model = model
        self._use_llm = use_llm
        self._max_steps = max(1, min(5, max_steps))
        self._app: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        query: str,
        repo_context: dict[str, Any] | None = None,
    ) -> DecompositionPlan:
        """Decompose ``query`` into an ordered list of sub-queries.

        Args:
            query: The user's natural-language question.
            repo_context: Optional dict with keys such as ``modules`` (list of
                module names) and ``key_symbols`` (list of important identifiers).
                Used to inform the LLM prompt.

        Returns:
            A :class:`DecompositionPlan` with 1-5 ordered :class:`SubQuery` steps.
        """
        query = query.strip()
        if not query:
            return _single_step_plan("")

        if not self._use_llm:
            return _rule_based_decompose(query, repo_context)

        return self._decompose_with_langgraph(query, repo_context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose_with_langgraph(
        self,
        query: str,
        repo_context: dict[str, Any] | None,
    ) -> DecompositionPlan:
        """Run the LangGraph state machine and return a :class:`DecompositionPlan`."""
        try:
            if self._app is None:
                self._app = _build_langgraph_app()
            initial_state = _make_initial_state(query, repo_context)
            final_state = self._app.invoke(initial_state)
            plan: DecompositionPlan | None = final_state.get("plan")
            if plan is None:
                logger.warning("LangGraph returned no plan; using fallback.")
                plan = _rule_based_decompose(query, repo_context)
            # Enforce max_steps cap
            plan.steps = plan.steps[: self._max_steps]
            return plan
        except Exception as exc:
            logger.warning(
                "LangGraph decomposition failed (%s); using rule-based fallback.", exc
            )
            plan = _rule_based_decompose(query, repo_context)
            plan.steps = plan.steps[: self._max_steps]
            return plan
