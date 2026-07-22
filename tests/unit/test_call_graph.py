"""Unit tests for call_graph module."""

import pytest

from src.reporag.graph.call_graph import CallEdge, CallGraphBuilder
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolType

@pytest.fixture
def parser() -> ASTParser:
    return ASTParser()

def test_direct_function_calls(parser: ASTParser) -> None:
    source = """
def foo():
    pass

def bar():
    foo()
    baz()
"""
    result = parser.parse(source, language="python")
    builder = CallGraphBuilder(parser=parser)
    edges = builder.extract(result.root_node, result.source_bytes, "test.py", "python")
    
    assert len(edges) == 2
    assert edges[0].caller == "bar"
    assert edges[0].callee == "foo"
    assert edges[0].call_site_line == 6
    assert edges[0].call_site_file == "test.py"
    
    assert edges[1].caller == "bar"
    assert edges[1].callee == "baz"
    assert edges[1].call_site_line == 7

def test_method_calls_and_chained(parser: ASTParser) -> None:
    source = """
class A:
    def method_a(self):
        self.method_b()
        
    def method_b(self):
        pass

def main():
    a = A()
    a.method_a()
    A().method_b().chain()
"""
    result = parser.parse(source, language="python")
    builder = CallGraphBuilder(parser=parser)
    edges = builder.extract(result.root_node, result.source_bytes, "test.py", "python")
    
    assert len(edges) == 6 # self.method_b(), A(), a.method_a(), A(), method_b(), chain() -> actually 6 calls
    # Let's verify specific calls
    
    # In A.method_a
    method_a_calls = [e for e in edges if e.caller == "A.method_a"]
    assert len(method_a_calls) == 1
    assert method_a_calls[0].callee == "self.method_b"
    
    # In main
    main_calls = [e for e in edges if e.caller == "main"]
    assert len(main_calls) == 5
    callees = [e.callee for e in main_calls]
    assert "A" in callees # Constructor
    assert "a.method_a" in callees
    assert "A" in callees # Second constructor
    assert "A().method_b" in callees
    assert "A().method_b().chain" in callees

def test_cross_file_calls_with_imports(parser: ASTParser) -> None:
    source = """
import os
from sys import argv
from my_module import my_func

def main():
    os.path.join("a", "b")
    print(argv)
    my_func()
"""
    result = parser.parse(source, language="python")
    builder = CallGraphBuilder(parser=parser)
    edges = builder.extract(result.root_node, result.source_bytes, "test.py", "python")
    
    assert len(edges) == 3
    callees = [e.callee for e in edges]
    assert "os.path.join" in callees  # Handled by attribute match against imports
    assert "print" in callees
    assert "my_module.my_func" in callees # Resolved by import_from

def test_recursive_calls(parser: ASTParser) -> None:
    source = """
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
"""
    result = parser.parse(source, language="python")
    builder = CallGraphBuilder(parser=parser)
    edges = builder.extract(result.root_node, result.source_bytes, "test.py", "python")
    
    assert len(edges) == 1
    assert edges[0].caller == "fact"
    assert edges[0].callee == "fact"

def test_known_symbols_resolution(parser: ASTParser) -> None:
    source = """
def main():
    target_func()
"""
    known = [
        Symbol(name="target_func", symbol_type=SymbolType.FUNCTION, file_path="other.py", start_line=1, end_line=2)
    ]
    result = parser.parse(source, language="python")
    builder = CallGraphBuilder(parser=parser, known_symbols=known)
    edges = builder.extract(result.root_node, result.source_bytes, "test.py", "python")
    
    assert len(edges) == 1
    assert edges[0].caller == "main"
    assert edges[0].callee == "target_func" # Remains target_func, but verified to match through heuristic
