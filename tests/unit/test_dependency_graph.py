"""Unit tests for dependency_graph module."""

import pytest

from src.reporag.graph.dependency_graph import (
    DependencyEdge,
    DependencyGraphBuilder,
)
from src.reporag.ingestion.parser import ASTParser


@pytest.fixture
def parser() -> ASTParser:
    return ASTParser()


def test_absolute_imports(parser: ASTParser) -> None:
    source = """
import os
import sys
"""

    result = parser.parse(source, language="python")
    builder = DependencyGraphBuilder(parser=parser)

    edges = builder.extract(
        result.root_node,
        result.source_bytes,
        "src/reporag/sample.py",
        "python",
    )

    assert len(edges) == 2

    assert edges[0].source == "reporag.sample"
    assert edges[0].target == "os"
    assert edges[0].import_type == "import"

    assert edges[1].target == "sys"


def test_from_import(parser: ASTParser) -> None:
    source = """
from pathlib import Path
"""

    result = parser.parse(source, language="python")
    builder = DependencyGraphBuilder(parser=parser)

    edges = builder.extract(
        result.root_node,
        result.source_bytes,
        "src/reporag/sample.py",
        "python",
    )

    assert len(edges) == 1
    assert edges[0].target == "pathlib"
    assert edges[0].imported_names == ["Path"]


def test_relative_import(parser: ASTParser) -> None:
    source = """
from .utils import helper
"""

    result = parser.parse(source, language="python")
    builder = DependencyGraphBuilder(parser=parser)

    edges = builder.extract(
        result.root_node,
        result.source_bytes,
        "src/reporag/graph/sample.py",
        "python",
    )

    assert len(edges) == 1
    assert edges[0].target.endswith("graph.utils")
    assert edges[0].imported_names == ["helper"]


def test_star_import(parser: ASTParser) -> None:
    source = """
from math import *
"""

    result = parser.parse(source, language="python")
    builder = DependencyGraphBuilder(parser=parser)

    edges = builder.extract(
        result.root_node,
        result.source_bytes,
        "src/reporag/sample.py",
        "python",
    )

    assert len(edges) == 1
    assert edges[0].target == "math"
    assert edges[0].imported_names == ["*"]


def test_circular_dependency_detection() -> None:
    builder = DependencyGraphBuilder()

    edges = [
        DependencyEdge(
            source="a",
            target="b",
            import_type="import",
            imported_names=[],
        ),
        DependencyEdge(
            source="b",
            target="c",
            import_type="import",
            imported_names=[],
        ),
        DependencyEdge(
            source="c",
            target="a",
            import_type="import",
            imported_names=[],
        ),
    ]

    cycles = builder.detect_circular_dependencies(edges)

    assert len(cycles) == 1
    assert cycles[0][0] == "a"
    assert cycles[0][-1] == "a"
