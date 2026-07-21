"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python and JavaScript
out of the box, with a language-agnostic interface extensible to any
tree-sitter grammar.

Handles syntax errors gracefully by returning a partial AST and flagging
the error nodes -- tree-sitter never raises on bad input.

Usage::

    from src.reporag.ingestion.parser import ASTParser

    parser = ASTParser()
    tree = parser.parse('def hello():\\n    return 42\\n', language='python')
    print(tree.root_node.children)

    # Walk into structured node data
    nodes = parser.walk(tree)
    for node in nodes:
        print(node.node_type, node.start_line, node.text[:40])
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy language registry
# ---------------------------------------------------------------------------

#: Maps canonical language name -> callable that returns a tree-sitter
#: Language object.  Callables are evaluated lazily on first use so that
#: importing this module does not hard-require every grammar package.
_LANGUAGE_LOADERS: dict[str, Callable[[], Language]] = {}


def _load_python() -> Language:
    import tree_sitter_python as _tspy  # type: ignore[import-untyped]

    return Language(_tspy.language())


def _load_javascript() -> Language:
    import tree_sitter_javascript as _tsjs  # type: ignore[import-untyped]

    return Language(_tsjs.language())


def _load_typescript() -> Language:
    import tree_sitter_javascript as _tsjs  # type: ignore[import-untyped]

    # tree-sitter-javascript ships TypeScript as a second grammar
    return Language(_tsjs.language_typescript())


_LANGUAGE_LOADERS["python"] = _load_python
_LANGUAGE_LOADERS["javascript"] = _load_javascript
_LANGUAGE_LOADERS["typescript"] = _load_typescript


def register_language(name: str, loader: Callable[[], Language]) -> None:
    """Register a custom tree-sitter language loader.

    Parameters:
        name: Canonical language name (lowercase), e.g. ``"go"``.
        loader: Zero-argument callable that returns a :class:`Language`.

    Example::

        import tree_sitter_go as tsgo
        from tree_sitter import Language
        from src.reporag.ingestion.parser import register_language

        register_language("go", lambda: Language(tsgo.language()))
    """
    _LANGUAGE_LOADERS[name] = loader


def supported_languages() -> list[str]:
    """Return the names of all currently registered languages."""
    return sorted(_LANGUAGE_LOADERS.keys())


# ---------------------------------------------------------------------------
# Structured node data
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    """Structured representation of a single tree-sitter AST node.

    Attributes:
        node_type: Grammar node type (e.g. ``"function_definition"``).
        is_named: ``True`` for named nodes, ``False`` for anonymous/literal
            nodes such as punctuation or keywords stored as exact text.
        is_error: ``True`` when this node is an ERROR recovery node.
        has_error: ``True`` when this subtree contains at least one ERROR node.
        is_missing: ``True`` when a required token was absent (tree-sitter
            inserts a zero-width missing node to keep the AST valid).
        text: Source text covered by this node, decoded as UTF-8.
        start_line: 0-indexed line of the node's first character.
        end_line: 0-indexed line of the node's last character.
        start_col: 0-indexed column of the first character.
        end_col: 0-indexed column of the last character (exclusive).
        start_byte: Byte offset of the first character.
        end_byte: Byte offset of the last character (exclusive).
        depth: Depth in the AST (root = 0).
        children: Immediate child :class:`NodeInfo` records (populated by
            :meth:`ASTParser.walk` when ``recursive=True``).
    """

    node_type: str
    is_named: bool
    is_error: bool
    has_error: bool
    is_missing: bool
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    start_byte: int
    end_byte: int
    depth: int
    children: list[NodeInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse result
# ---------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Output of a single :meth:`ASTParser.parse` call.

    Attributes:
        tree: The raw :class:`tree_sitter.Tree` object returned by the parser.
            Always present -- even for completely invalid input tree-sitter
            produces a partial AST with ERROR nodes.
        language: The canonical language name that was used (e.g. ``"python"``).
        source_bytes: The UTF-8 encoded source that was parsed.
        has_error: ``True`` when the root node reports a parse error.
        error_count: Number of ERROR nodes found in the tree (shallow scan of
            direct children of root; use :meth:`ASTParser.walk` for a full
            deep scan).
    """

    tree: Tree
    language: str
    source_bytes: bytes
    has_error: bool
    error_count: int

    @property
    def root_node(self) -> Node:
        """Convenience accessor for ``tree.root_node``."""
        return self.tree.root_node


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------


class ASTParser:
    """Language-agnostic tree-sitter AST parser.

    Internally caches one :class:`tree_sitter.Parser` instance per language
    so repeated calls for the same language are cheap.

    Parameters:
        extra_languages: Optional mapping of ``{name: loader}`` passed to
            :func:`register_language` at construction time.
    """

    def __init__(
        self,
        extra_languages: dict[str, Callable[[], Language]] | None = None,
    ) -> None:
        # Cache: language name -> Parser instance
        self._parsers: dict[str, Parser] = {}
        # Cache: language name -> Language instance
        self._languages: dict[str, Language] = {}

        if extra_languages:
            for name, loader in extra_languages.items():
                register_language(name, loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        source: str | bytes,
        language: str = "python",
    ) -> ParseResult:
        """Parse *source* and return a :class:`ParseResult`.

        tree-sitter never raises on malformed input: syntax errors are
        represented as ``ERROR`` nodes in the returned tree, giving callers
        a partial but usable AST.

        Parameters:
            source: Source code as a ``str`` or ``bytes``.  Strings are
                encoded to UTF-8 before parsing.
            language: Canonical language name.  Must be registered in
                :data:`_LANGUAGE_LOADERS` (or via :func:`register_language`).

        Returns:
            A :class:`ParseResult` with the tree, metadata, and error flag.

        Raises:
            ValueError: If *language* is not registered.
        """
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        parser = self._get_parser(language)

        tree = parser.parse(source_bytes)
        root = tree.root_node

        # Count ERROR children one level deep (fast heuristic)
        error_count = sum(1 for c in root.children if c.is_error)

        if root.has_error:
            logger.debug(
                "Parse completed with errors (language=%s, error_nodes=%d)",
                language,
                error_count,
            )

        return ParseResult(
            tree=tree,
            language=language,
            source_bytes=source_bytes,
            has_error=root.has_error,
            error_count=error_count,
        )

    def parse_file(
        self,
        path: str,
        language: str = "python",
        encoding: str = "utf-8",
    ) -> ParseResult:
        """Read *path* from disk and parse it.

        Parameters:
            path: Absolute or relative path to a source file.
            language: Canonical language name.
            encoding: File encoding (default ``"utf-8"``).

        Returns:
            A :class:`ParseResult` for the file contents.
        """
        with open(path, encoding=encoding, errors="replace") as fh:
            source = fh.read()
        logger.debug("Parsing file: %s (language=%s)", path, language)
        return self.parse(source, language=language)

    def walk(
        self,
        result: ParseResult,
        named_only: bool = False,
        max_depth: int | None = None,
    ) -> list[NodeInfo]:
        """Walk the AST and return a flat list of :class:`NodeInfo` records.

        The list is ordered by a pre-order depth-first traversal, matching
        the natural reading order of the source file.

        Parameters:
            result: A :class:`ParseResult` produced by :meth:`parse`.
            named_only: When ``True`` only named grammar nodes are included
                (anonymous punctuation/keyword nodes are skipped).
            max_depth: Maximum traversal depth (root = 0).  ``None`` means
                unlimited.

        Returns:
            Flat pre-order list of :class:`NodeInfo`.
        """
        nodes: list[NodeInfo] = []
        self._walk_node(
            node=result.root_node,
            source_bytes=result.source_bytes,
            depth=0,
            named_only=named_only,
            max_depth=max_depth,
            accumulator=nodes,
        )
        return nodes

    def node_info(self, node: Node, source_bytes: bytes, depth: int = 0) -> NodeInfo:
        """Convert a raw :class:`tree_sitter.Node` into a :class:`NodeInfo`.

        Parameters:
            node: The tree-sitter node to convert.
            source_bytes: The original UTF-8 encoded source bytes.
            depth: Depth hint for the resulting record.

        Returns:
            A populated :class:`NodeInfo` (without children filled in).
        """
        raw_text = source_bytes[node.start_byte : node.end_byte]
        return NodeInfo(
            node_type=node.type,
            is_named=node.is_named,
            is_error=node.is_error,
            has_error=node.has_error,
            is_missing=node.is_missing,
            text=raw_text.decode("utf-8", errors="replace"),
            start_line=node.start_point.row,
            end_line=node.end_point.row,
            start_col=node.start_point.column,
            end_col=node.end_point.column,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_parser(self, language: str) -> Parser:
        """Return a cached :class:`Parser` for *language*, creating if needed."""
        if language not in self._parsers:
            if language not in _LANGUAGE_LOADERS:
                raise ValueError(
                    f"Unsupported language {language!r}. "
                    f"Available: {supported_languages()}. "
                    "Use register_language() to add more."
                )
            lang_obj = _LANGUAGE_LOADERS[language]()
            self._languages[language] = lang_obj
            self._parsers[language] = Parser(lang_obj)
            logger.debug("Loaded tree-sitter grammar for %r", language)
        return self._parsers[language]

    def _walk_node(
        self,
        node: Node,
        source_bytes: bytes,
        depth: int,
        named_only: bool,
        max_depth: int | None,
        accumulator: list[NodeInfo],
    ) -> None:
        """Recursive DFS walk; appends :class:`NodeInfo` to *accumulator*."""
        if max_depth is not None and depth > max_depth:
            return

        if named_only and not node.is_named:
            # Still recurse so we don't miss named descendants
            for child in node.children:
                self._walk_node(
                    child, source_bytes, depth + 1, named_only, max_depth, accumulator
                )
            return

        info = self.node_info(node, source_bytes, depth)
        accumulator.append(info)

        for child in node.children:
            self._walk_node(
                child, source_bytes, depth + 1, named_only, max_depth, accumulator
            )
