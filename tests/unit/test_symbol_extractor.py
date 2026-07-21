"""Unit tests for symbol extractor."""

from __future__ import annotations

import pytest

from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import SymbolExtractor, SymbolType


@pytest.fixture
def extractor() -> SymbolExtractor:
    parser = ASTParser()
    return SymbolExtractor(parser)


def test_extract_function(extractor: SymbolExtractor) -> None:
    source = '''
@my_decorator
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello {name}"
'''
    result = extractor.parser.parse(source)
    symbols = extractor.extract(result.root_node, result.source_bytes, "test.py")

    assert len(symbols) == 1
    func = symbols[0]
    assert func.name == "hello"
    assert func.symbol_type == SymbolType.FUNCTION
    assert func.signature == "(name: str)"
    assert func.return_type_hint == "str"
    assert func.docstring == "Say hello."
    assert func.decorators == ["@my_decorator"]
    assert func.parent_class is None


def test_extract_class_and_methods(extractor: SymbolExtractor) -> None:
    source = '''
class MyClass(Base1, Base2):
    """My custom class."""

    def __init__(self):
        pass

    @property
    def my_prop(self) -> int:
        return 1
'''
    result = extractor.parser.parse(source)
    symbols = extractor.extract(result.root_node, result.source_bytes, "test.py")

    assert len(symbols) == 3

    cls = next(s for s in symbols if s.symbol_type == SymbolType.CLASS)
    assert cls.name == "MyClass"
    assert cls.bases == ["Base1", "Base2"]
    assert cls.docstring == "My custom class."
    assert "__init__" in cls.methods
    assert "my_prop" in cls.methods

    init_method = next(s for s in symbols if s.name == "__init__")
    assert init_method.symbol_type == SymbolType.METHOD
    assert init_method.parent_class == "MyClass"
    assert init_method.signature == "(self)"

    prop_method = next(s for s in symbols if s.name == "my_prop")
    assert prop_method.symbol_type == SymbolType.METHOD
    assert prop_method.parent_class == "MyClass"
    assert prop_method.decorators == ["@property"]


def test_extract_imports(extractor: SymbolExtractor) -> None:
    source = """
import os, sys
import numpy as np
from collections import defaultdict, Counter
from typing import *
"""
    result = extractor.parser.parse(source)
    symbols = extractor.extract(result.root_node, result.source_bytes, "test.py")

    assert len(symbols) == 6
    for s in symbols:
        assert s.symbol_type == SymbolType.IMPORT

    names = [s.name for s in symbols]
    assert "os" in names
    assert "sys" in names
    assert "numpy as np" in names
    assert "defaultdict" in names
    assert "Counter" in names
    assert "*" in names

    # check signatures
    np_sym = next(s for s in symbols if s.name == "numpy as np")
    assert np_sym.signature == "import numpy as np"

    dd_sym = next(s for s in symbols if s.name == "defaultdict")
    assert dd_sym.signature == "from collections import defaultdict"


def test_extract_nested_and_async_functions(extractor: SymbolExtractor) -> None:
    source = """
async def fetch_data():
    def inner_helper():
        pass
"""
    result = extractor.parser.parse(source)
    symbols = extractor.extract(result.root_node, result.source_bytes, "test.py")

    assert len(symbols) == 2

    fetch = next(s for s in symbols if s.name == "fetch_data")
    assert fetch.symbol_type == SymbolType.FUNCTION

    inner = next(s for s in symbols if s.name == "inner_helper")
    assert inner.symbol_type == SymbolType.FUNCTION
    assert inner.parent_class is None  # It's a function, not a method


def test_extract_docstring_edge_cases(extractor: SymbolExtractor) -> None:
    source = '''
def func1():
    # comment before
    """docstring"""
    pass

def func2():
    pass

def func3():
    """line1
    line2"""
    pass
'''
    result = extractor.parser.parse(source)
    symbols = extractor.extract(result.root_node, result.source_bytes, "test.py")

    f1 = next(s for s in symbols if s.name == "func1")
    assert f1.docstring == "docstring"

    f2 = next(s for s in symbols if s.name == "func2")
    assert f2.docstring is None

    f3 = next(s for s in symbols if s.name == "func3")
    assert f3.docstring == "line1\n    line2"
