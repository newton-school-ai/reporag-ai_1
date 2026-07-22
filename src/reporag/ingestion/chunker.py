"""Semantic code chunker.

AST-aware chunking that respects function/class boundaries. Never splits
a function mid-body. Large functions are split at logical points with
function signature overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser


@dataclass(slots=True)
class Chunk:
    """Represents one semantic chunk of code."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    parent_symbol: str | None
    language: str
    token_count: int


class CodeChunker:
    """AST-aware code chunker."""

    PYTHON_SYMBOLS = {
        "function_definition",
        "class_definition",
    }

    def __init__(
        self,
        max_tokens: int = 512,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.max_tokens = max_tokens
        self.parser = ASTParser()
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Return token count."""
        return len(self.encoding.encode(text))

    def chunk(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> list[Chunk]:
        """Split a source file into semantic chunks."""

        result = self.parser.parse(source, language)

        chunks: list[Chunk] = []

        self._walk(
            node=result.root_node,
            source_bytes=result.source_bytes,
            language=language,
            file_path=file_path,
            chunks=chunks,
            parent_symbol=None,
        )

        return chunks

    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        language: str,
        file_path: str,
        chunks: list[Chunk],
        parent_symbol: str | None,
    ) -> None:
        """Walk the AST recursively."""

        current_parent = parent_symbol

        if language == "python" and node.type in self.PYTHON_SYMBOLS:
            self._process_node(
                node=node,
                source_bytes=source_bytes,
                language=language,
                file_path=file_path,
                chunks=chunks,
                parent_symbol=parent_symbol,
            )

            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    current_parent = self._get_text(
                        name_node,
                        source_bytes,
                    )

        for child in node.children:
            self._walk(
                child,
                source_bytes,
                language,
                file_path,
                chunks,
                current_parent,
            )

    def _process_node(
        self,
        node: Node,
        source_bytes: bytes,
        language: str,
        file_path: str,
        chunks: list[Chunk],
        parent_symbol: str | None,
    ) -> None:
        """Convert a symbol node into one or more chunks."""

        text = self._get_text(node, source_bytes)

        token_count = self.count_tokens(text)

        name_node = node.child_by_field_name("name")

        symbol_name = (
            self._get_text(name_node, source_bytes) if name_node else parent_symbol
        )

        if token_count <= self.max_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent_symbol=symbol_name,
                    language=language,
                    token_count=token_count,
                )
            )
            return

        self._split_large_node(
            node=node,
            source_bytes=source_bytes,
            language=language,
            file_path=file_path,
            chunks=chunks,
            parent_symbol=symbol_name,
        )

    def _split_large_node(
        self,
        node: Node,
        source_bytes: bytes,
        language: str,
        file_path: str,
        chunks: list[Chunk],
        parent_symbol: str | None,
    ) -> None:
        """Split an oversized symbol into multiple semantic chunks."""

        body = node.child_by_field_name("body")

        if body is None:
            text = self._get_text(node, source_bytes)
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent_symbol=parent_symbol,
                    language=language,
                    token_count=self.count_tokens(text),
                )
            )
            return

        signature = source_bytes[node.start_byte : body.start_byte].decode(
            "utf-8",
            errors="replace",
        )

        current_lines: list[str] = [signature]
        current_start = node.start_point.row + 1
        current_end = current_start

        for child in body.children:
            statement = self._get_text(child, source_bytes)

            candidate = "".join(current_lines) + statement

            if self.count_tokens(candidate) <= self.max_tokens:
                current_lines.append(statement)
                current_end = child.end_point.row + 1
                continue

            self._emit_chunk(
                chunks=chunks,
                text="".join(current_lines),
                file_path=file_path,
                start_line=current_start,
                end_line=current_end,
                parent_symbol=parent_symbol,
                language=language,
            )

            current_lines = [signature, statement]
            current_start = child.start_point.row + 1
            current_end = child.end_point.row + 1

        if "".join(current_lines).strip():
            self._emit_chunk(
                chunks=chunks,
                text="".join(current_lines),
                file_path=file_path,
                start_line=current_start,
                end_line=current_end,
                parent_symbol=parent_symbol,
                language=language,
            )

    def _emit_chunk(
        self,
        *,
        chunks: list[Chunk],
        text: str,
        file_path: str,
        start_line: int,
        end_line: int,
        parent_symbol: str | None,
        language: str,
    ) -> None:
        """Create a chunk object."""

        chunks.append(
            Chunk(
                text=text,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                parent_symbol=parent_symbol,
                language=language,
                token_count=self.count_tokens(text),
            )
        )

    def _get_text(
        self,
        node: Node,
        source_bytes: bytes,
    ) -> str:
        """Extract node text."""

        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8",
            errors="replace",
        )

    def chunk_file(
        self,
        file_path: str,
        language: str = "python",
    ) -> list[Chunk]:
        """Chunk a source file."""

        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        return self.chunk(
            source=source,
            language=language,
            file_path=file_path,
        )


__all__ = [
    "Chunk",
    "CodeChunker",
]
