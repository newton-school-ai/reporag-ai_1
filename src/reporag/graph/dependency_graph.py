"""Import dependency graph builder.

Builds directed edges representing module-level import relationships.
Resolves relative imports, handles star imports, detects circular
dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


@dataclass
class DependencyEdge:
    """Directed edge representing a module dependency."""

    source: str
    target: str
    import_type: str
    imported_names: list[str]


class DependencyGraphBuilder:
    """Build a directed module dependency graph."""

    def __init__(self, parser: ASTParser | None = None) -> None:
        self.parser = parser or ASTParser()

    def extract_from_file(
        self,
        file_path: str | Path,
        language: str = "python",
    ) -> list[DependencyEdge]:
        """Parse a file and extract dependency edges."""
        file_path = str(file_path)
        result = self.parser.parse_file(file_path, language=language)

        return self.extract(
            result.root_node,
            result.source_bytes,
            file_path,
            language,
        )

    def extract(
        self,
        root_node: Node,
        source_bytes: bytes,
        file_path: str,
        language: str = "python",
    ) -> list[DependencyEdge]:
        """Extract dependency edges from a syntax tree."""
        if language != "python":
            logger.warning(
                "Dependency graph extraction currently supports only Python."
            )
            return []

        edges: list[DependencyEdge] = []

        module_name = self._module_name(file_path)

        self._walk_python(
            node=root_node,
            source_bytes=source_bytes,
            source_module=module_name,
            file_path=file_path,
            edges=edges,
        )

        return edges

    def _walk_python(
        self,
        node: Node,
        source_bytes: bytes,
        source_module: str,
        file_path: str,
        edges: list[DependencyEdge],
    ) -> None:
        """Recursively walk the AST."""

        if node.type == "import_statement":
            self._handle_import_statement(
                node=node,
                source_bytes=source_bytes,
                source_module=source_module,
                edges=edges,
            )
            return

        if node.type == "import_from_statement":
            self._handle_import_from_statement(
                node=node,
                source_bytes=source_bytes,
                source_module=source_module,
                file_path=file_path,
                edges=edges,
            )
            return

        for child in node.children:
            self._walk_python(
                child,
                source_bytes,
                source_module,
                file_path,
                edges,
            )

    def _handle_import_statement(
        self,
        node: Node,
        source_bytes: bytes,
        source_module: str,
        edges: list[DependencyEdge],
    ) -> None:
        """Handle 'import x' statements."""

        for child in node.children:
            if child.type not in ("dotted_name", "aliased_import"):
                continue

            if child.type == "aliased_import":
                text = self._get_text(child, source_bytes)
                module = text.split(" as ")[0].strip()
            else:
                module = self._get_text(child, source_bytes)

            edges.append(
                DependencyEdge(
                    source=source_module,
                    target=module,
                    import_type="import",
                    imported_names=[],
                )
            )

    def _handle_import_from_statement(
        self,
        node: Node,
        source_bytes: bytes,
        source_module: str,
        file_path: str,
        edges: list[DependencyEdge],
    ) -> None:
        """Handle 'from x import y' statements."""

        module_node = node.child_by_field_name("module_name")

        module_name = self._get_text(module_node, source_bytes) if module_node else ""

        relative_level = len(module_name) - len(module_name.lstrip("."))

        module_name = self._resolve_relative_import(
            module_name.lstrip("."),
            file_path,
            relative_level,
        )
        imported_names: list[str] = []
        star_import = False

        for child in node.children:
            if child == module_node:
                continue

            if child.type == "wildcard_import":
                star_import = True
                imported_names.append("*")
                continue

            if child.type == "dotted_name":
                # Skip the module itself (already handled above)
                if child == module_node:
                    continue
                imported_names.append(self._get_text(child, source_bytes))

            elif child.type == "aliased_import":
                text = self._get_text(child, source_bytes)
                imported_names.append(text.split(" as ")[0].strip())

        if star_import:
            logger.warning(
                "Star import encountered in %s",
                source_module,
            )

        edges.append(
            DependencyEdge(
                source=source_module,
                target=module_name,
                import_type="from",
                imported_names=imported_names,
            )
        )

    def _module_name(self, file_path: str) -> str:
        """Convert a file path into a Python module name."""

        path = Path(file_path).with_suffix("")

        parts = list(path.parts)

        if "reporag" in parts:
            parts = parts[parts.index("reporag") :]

        return ".".join(parts)

    def _resolve_relative_import(
        self,
        module: str,
        file_path: str,
        level: int,
    ) -> str:
        """Resolve a relative import into an absolute module."""

        if level == 0:
            return module

        current = self._module_name(file_path).split(".")

        if current:
            current.pop()

        if level > 1:
            current = current[: -(level - 1)]

        if module:
            current.extend(module.split("."))

        return ".".join(current)

    def detect_circular_dependencies(
        self,
        edges: list[DependencyEdge],
    ) -> list[list[str]]:
        """Detect circular dependency chains."""

        graph: dict[str, set[str]] = {}

        for edge in edges:
            graph.setdefault(edge.source, set()).add(edge.target)

        visited: set[str] = set()
        stack: list[str] = []
        on_stack: set[str] = set()
        cycles: list[list[str]] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.append(node)
            on_stack.add(node)

            for neighbour in graph.get(node, set()):
                if neighbour not in visited:
                    dfs(neighbour)
                elif neighbour in on_stack:
                    try:
                        start = stack.index(neighbour)
                        cycle = stack[start:] + [neighbour]
                        if cycle not in cycles:
                            cycles.append(cycle)
                    except ValueError:
                        pass

            stack.pop()
            on_stack.remove(node)

        for module in graph:
            if module not in visited:
                dfs(module)

        return cycles

    def build_graph(
        self,
        file_paths: list[str],
        language: str = "python",
    ) -> tuple[list[DependencyEdge], list[list[str]]]:
        """Build the dependency graph for multiple files."""

        edges: list[DependencyEdge] = []

        for file_path in file_paths:
            edges.extend(
                self.extract_from_file(
                    file_path=file_path,
                    language=language,
                )
            )

        cycles = self.detect_circular_dependencies(edges)

        if cycles:
            for cycle in cycles:
                logger.warning(
                    "Circular dependency detected: %s",
                    " -> ".join(cycle),
                )

        return edges, cycles

    def _get_text(
        self,
        node: Node,
        source_bytes: bytes,
    ) -> str:
        """Extract text corresponding to a tree-sitter node."""

        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8",
            errors="replace",
        )
