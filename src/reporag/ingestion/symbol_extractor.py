"""Symbol extractor.

Walks a tree-sitter AST and extracts meaningful code entities: functions,
classes, methods, imports. Each symbol carries metadata (line range,
signature, docstring, decorators).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


class SymbolType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"


@dataclass
class Symbol:
    """Metadata for an extracted code symbol."""

    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int

    # Function / Method fields
    signature: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    return_type_hint: str | None = None

    # Method fields
    parent_class: str | None = None

    # Class fields
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


class SymbolExtractor:
    """Extracts high-level symbols from an AST.

    Currently supports Python.
    """

    def __init__(self, parser: ASTParser | None = None) -> None:
        self.parser = parser or ASTParser()

    def extract_from_file(
        self, file_path: str | Path, language: str = "python"
    ) -> list[Symbol]:
        """Parse a file and extract all its symbols."""
        file_path = str(file_path)
        result = self.parser.parse_file(file_path, language=language)
        return self.extract(result.root_node, result.source_bytes, file_path, language)

    def extract(
        self,
        root_node: Node,
        source_bytes: bytes,
        file_path: str,
        language: str = "python",
    ) -> list[Symbol]:
        """Walk the AST and extract symbols."""
        if language != "python":
            # For now, only python is implemented. Future issues can extend this.
            logger.warning(
                "Symbol extraction is currently only fully supported for Python."
            )
            return []

        symbols: list[Symbol] = []
        self._walk_python(
            root_node, source_bytes, file_path, symbols, parent_class=None
        )
        return symbols

    def _walk_python(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
    ) -> None:
        """Recursively walk the Python AST to find classes, functions, and imports."""
        # Check node type
        node_type = node.type

        # Handle decorators by looking at decorated_definition
        # tree-sitter parses `@dec\ndef foo():` as `decorated_definition` containing `decorator` and `function_definition`
        if node_type == "decorated_definition":
            # We will extract decorators and pass them down
            decorators = []
            definition_node = None
            for child in node.children:
                if child.type == "decorator":
                    decorators.append(self._get_text(child, source_bytes))
                elif child.type in ("function_definition", "class_definition"):
                    definition_node = child

            if definition_node:
                if definition_node.type == "function_definition":
                    self._extract_python_function(
                        definition_node,
                        source_bytes,
                        file_path,
                        symbols,
                        parent_class,
                        decorators,
                    )
                elif definition_node.type == "class_definition":
                    self._extract_python_class(
                        definition_node, source_bytes, file_path, symbols, decorators
                    )
            return

        elif node_type == "function_definition":
            self._extract_python_function(
                node, source_bytes, file_path, symbols, parent_class, []
            )
            return

        elif node_type == "class_definition":
            self._extract_python_class(node, source_bytes, file_path, symbols, [])
            return

        elif node_type in ("import_statement", "import_from_statement"):
            self._extract_python_import(node, source_bytes, file_path, symbols)
            return

        # If not one of the above, traverse children
        for child in node.children:
            self._walk_python(child, source_bytes, file_path, symbols, parent_class)

    def _extract_python_function(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
        decorators: list[str],
    ) -> None:
        """Extract a function or method."""
        name_node = node.child_by_field_name("name")
        name = self._get_text(name_node, source_bytes) if name_node else "<unknown>"

        params_node = node.child_by_field_name("parameters")
        signature = self._get_text(params_node, source_bytes) if params_node else "()"

        return_type_node = node.child_by_field_name("return_type")
        return_type_hint = (
            self._get_text(return_type_node, source_bytes) if return_type_node else None
        )

        # Check for async
        is_async = any(c.type == "async" for c in node.children)
        if is_async and not signature.startswith("async "):
            # decorators aren't in signature, async might be
            pass  # async is handled by the caller knowing it's async, or we prepend it to signature
            # Let's just keep signature as the parameter list

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, source_bytes)

        # Decide if function or method
        sym_type = SymbolType.METHOD if parent_class else SymbolType.FUNCTION

        sym = Symbol(
            name=name,
            symbol_type=sym_type,
            file_path=file_path,
            start_line=node.start_point.row,
            end_line=node.end_point.row,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            return_type_hint=return_type_hint,
            parent_class=parent_class,
        )
        symbols.append(sym)

        # Traverse body for nested functions/classes
        if body_node:
            for child in body_node.children:
                self._walk_python(
                    child, source_bytes, file_path, symbols, parent_class=None
                )

    def _extract_python_class(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        decorators: list[str],
    ) -> None:
        """Extract a class definition."""
        name_node = node.child_by_field_name("name")
        name = self._get_text(name_node, source_bytes) if name_node else "<unknown>"

        bases_node = node.child_by_field_name("superclasses")
        bases = []
        if bases_node:
            # bases_node is argument_list like `(Base1, Base2)`
            for child in bases_node.children:
                if child.is_named:
                    bases.append(self._get_text(child, source_bytes))

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, source_bytes)

        class_sym = Symbol(
            name=name,
            symbol_type=SymbolType.CLASS,
            file_path=file_path,
            start_line=node.start_point.row,
            end_line=node.end_point.row,
            docstring=docstring,
            decorators=decorators,
            bases=bases,
        )
        symbols.append(class_sym)

        # Traverse body for methods (and nested classes)
        # We need to collect method names to populate class_sym.methods
        if body_node:
            before_count = len(symbols)
            for child in body_node.children:
                self._walk_python(
                    child, source_bytes, file_path, symbols, parent_class=name
                )

            # Find all methods added for this class
            for added_sym in symbols[before_count:]:
                if (
                    added_sym.symbol_type == SymbolType.METHOD
                    and added_sym.parent_class == name
                ):
                    class_sym.methods.append(added_sym.name)

    def _extract_python_import(
        self, node: Node, source_bytes: bytes, file_path: str, symbols: list[Symbol]
    ) -> None:
        """Extract import statements."""
        if node.type == "import_statement":
            # e.g. import os, sys
            for child in node.children:
                if child.type == "dotted_name" or child.type == "aliased_import":
                    name = self._get_text(child, source_bytes)
                    symbols.append(
                        Symbol(
                            name=name,
                            symbol_type=SymbolType.IMPORT,
                            file_path=file_path,
                            start_line=node.start_point.row,
                            end_line=node.end_point.row,
                            signature=f"import {name}",
                        )
                    )
        elif node.type == "import_from_statement":
            # e.g. from sys import argv
            module_node = node.child_by_field_name("module_name")
            module_name = (
                self._get_text(module_node, source_bytes) if module_node else ""
            )

            # find imported names
            for child in node.children:
                if (
                    child.type in ("dotted_name", "aliased_import", "wildcard_import")
                    and child != module_node
                ):
                    name = self._get_text(child, source_bytes)
                    symbols.append(
                        Symbol(
                            name=name,
                            symbol_type=SymbolType.IMPORT,
                            file_path=file_path,
                            start_line=node.start_point.row,
                            end_line=node.end_point.row,
                            signature=f"from {module_name} import {name}",
                        )
                    )

    def _extract_docstring(
        self, body_node: Node | None, source_bytes: bytes
    ) -> str | None:
        """Extract docstring from a block body node."""
        if not body_node:
            return None

        for child in body_node.children:
            if child.type == "expression_statement":
                # Check if the expression statement contains a string
                for expr_child in child.children:
                    if expr_child.type == "string":
                        text = self._get_text(expr_child, source_bytes)
                        # Strip quotes
                        if text.startswith('"""') or text.startswith("'''"):
                            return text[3:-3]
                        if text.startswith('"') or text.startswith("'"):
                            return text[1:-1]
                        return text
                # Only the first expression statement can be a docstring
                break
            elif child.type != "comment":
                # Non-comment, non-expression node means no docstring
                break
        return None

    def _get_text(self, node: Node, source_bytes: bytes) -> str:
        """Extract text for a node."""
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )
