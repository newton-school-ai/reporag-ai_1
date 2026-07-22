"""Call graph builder.

Walks ASTs to identify function call expressions and resolves them to
target symbols. Builds directed edges: caller -> callee with call site
metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolType

logger = logging.getLogger(__name__)


@dataclass
class CallEdge:
    """Directed edge representing a function call."""
    caller: str
    callee: str
    call_site_file: str
    call_site_line: int


class CallGraphBuilder:
    """Builds a call graph by walking the AST of source files."""

    def __init__(
        self,
        parser: ASTParser | None = None,
        known_symbols: list[Symbol] | None = None,
    ) -> None:
        self.parser = parser or ASTParser()
        self.known_symbols = known_symbols or []

        # Build a quick lookup for functions/methods by name
        # We store them by their short name for best-effort matching
        self._symbol_map: dict[str, list[Symbol]] = {}
        for sym in self.known_symbols:
            if sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
                if sym.name not in self._symbol_map:
                    self._symbol_map[sym.name] = []
                self._symbol_map[sym.name].append(sym)

    def extract_from_file(
        self, file_path: str | Path, language: str = "python"
    ) -> list[CallEdge]:
        """Parse a file and extract all call edges."""
        file_path = str(file_path)
        result = self.parser.parse_file(file_path, language=language)
        return self.extract(result.root_node, result.source_bytes, file_path, language)

    def extract(
        self,
        root_node: Node,
        source_bytes: bytes,
        file_path: str,
        language: str = "python",
    ) -> list[CallEdge]:
        """Walk the AST and extract call edges."""
        if language != "python":
            logger.warning(
                "Call graph extraction is currently only fully supported for Python."
            )
            return []

        edges: list[CallEdge] = []
        
        # Track active imports in this file to help resolve cross-file calls
        # Mapping from short name (or alias) -> fully qualified target
        imports: dict[str, str] = {}
        
        self._walk_python(
            root_node,
            source_bytes,
            file_path,
            edges,
            imports,
            caller_stack=[],
        )
        return edges

    def _walk_python(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        edges: list[CallEdge],
        imports: dict[str, str],
        caller_stack: list[str],
    ) -> None:
        """Recursively walk the Python AST to find calls."""
        node_type = node.type

        # Track imports for basic resolution
        if node_type == "import_statement":
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    name = self._get_text(child, source_bytes)
                    imports[name.split(".")[-1]] = name
        elif node_type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module_name = self._get_text(module_node, source_bytes) if module_node else ""
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import") and child != module_node:
                    name = self._get_text(child, source_bytes)
                    imports[name] = f"{module_name}.{name}"
                    
        # Maintain caller context
        if node_type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            name = self._get_text(name_node, source_bytes) if name_node else "<unknown>"
            
            # If we are inside a class, prefix the method name with the class name
            if node_type == "function_definition" and caller_stack and caller_stack[-1].isidentifier():
                # Rough check if parent is a class by seeing if we just appended a valid identifier and we are in its block
                pass # We'll just append it to the stack and join with dots later
                
            caller_stack.append(name)
            
            # Walk children with new stack
            for child in node.children:
                self._walk_python(child, source_bytes, file_path, edges, imports, caller_stack)
                
            caller_stack.pop()
            return

        # Extract calls
        if node_type == "call":
            function_node = node.child_by_field_name("function")
            if function_node:
                callee_text = self._get_text(function_node, source_bytes)
                
                # Best effort resolution
                resolved_callee = self._resolve_target(callee_text, imports)
                
                caller = ".".join(caller_stack) if caller_stack else "<module>"
                
                edges.append(CallEdge(
                    caller=caller,
                    callee=resolved_callee,
                    call_site_file=file_path,
                    call_site_line=node.start_point.row + 1,  # 1-indexed
                ))

        # Traverse children
        for child in node.children:
            self._walk_python(child, source_bytes, file_path, edges, imports, caller_stack)

    def _resolve_target(self, callee_text: str, imports: dict[str, str]) -> str:
        """Attempt to resolve the fully qualified target."""
        # 1. Direct import match
        if callee_text in imports:
            return imports[callee_text]
            
        # 2. Attribute match against imports (e.g., os.path.join -> import os)
        parts = callee_text.split(".")
        if parts[0] in imports:
            return f"{imports[parts[0]]}.{'.'.join(parts[1:])}"
            
        # 3. Match against known symbols (best effort by short name)
        # Note: If there are multiple matches, we just pick the first or use the text itself
        # Since we don't have full type inference, this is a heuristic.
        method_name = parts[-1]
        if method_name in self._symbol_map:
            # We found a matching function or method in the project
            # If it's a direct function call (no dots), we might try to match the exact name
            if len(parts) == 1:
                # Prioritize matching functions over methods if no dot
                funcs = [s for s in self._symbol_map[method_name] if s.symbol_type == SymbolType.FUNCTION]
                if funcs:
                    return funcs[0].name
            else:
                # It's a method call like `obj.method()`. We match the method name.
                # In a real system we'd infer `obj` type. Here we just keep `callee_text`.
                pass

        return callee_text

    def _get_text(self, node: Node, source_bytes: bytes) -> str:
        """Extract text for a node."""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
