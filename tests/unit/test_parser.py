"""Unit tests for tree-sitter AST parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporag.ingestion.parser import ASTParser


@pytest.fixture
def parser() -> ASTParser:
    """Fixture providing a fresh ASTParser."""
    return ASTParser()


def test_parse_empty_file(parser: ASTParser) -> None:
    result = parser.parse("", language="python")
    assert not result.has_error
    assert result.error_count == 0
    assert result.root_node.type == "module"
    assert len(result.root_node.children) == 0


def test_parse_single_function(parser: ASTParser) -> None:
    source = "def hello():\n    return 42\n"
    result = parser.parse(source, language="python")

    assert not result.has_error
    assert result.root_node.type == "module"

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    assert "function_definition" in types
    assert "return_statement" in types


def test_parse_class_with_methods(parser: ASTParser) -> None:
    source = """
class MyClass:
    def __init__(self):
        self.val = 1

    def get_val(self):
        return self.val
"""
    result = parser.parse(source, language="python")
    assert not result.has_error

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    assert "class_definition" in types
    assert types.count("function_definition") == 2


def test_parse_async_function(parser: ASTParser) -> None:
    source = "async def fetch_data():\n    await asyncio.sleep(1)\n"
    result = parser.parse(source, language="python")
    assert not result.has_error

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    assert "function_definition" in types
    assert "await" in types


def test_parse_syntax_error_returns_partial_ast(parser: ASTParser) -> None:
    source = "def broken_func(\n    return 42\n"
    result = parser.parse(source, language="python")

    assert result.has_error
    assert result.error_count > 0

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    # Even with syntax error, we should get some partial AST
    assert "ERROR" in types


def test_parse_nested_classes(parser: ASTParser) -> None:
    source = """
class Outer:
    class Inner:
        pass
"""
    result = parser.parse(source, language="python")
    assert not result.has_error

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    assert types.count("class_definition") == 2


def test_language_agnostic_interface(parser: ASTParser) -> None:
    js_source = "const x = 1 + 2;"
    result = parser.parse(js_source, language="javascript")

    assert not result.has_error
    assert result.language == "javascript"

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]

    assert "lexical_declaration" in types


def test_parse_file(parser: ASTParser, tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("x = 10")

    result = parser.parse_file(str(test_file), language="python")
    assert not result.has_error

    nodes = parser.walk(result)
    types = [n.node_type for n in nodes]
    assert "expression_statement" in types


def test_unsupported_language(parser: ASTParser) -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        parser.parse("test", language="nonexistent")
