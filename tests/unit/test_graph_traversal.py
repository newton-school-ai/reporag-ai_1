import networkx as nx

from src.reporag.retrieval.graph_traversal import GraphTraversalEngine


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()

    graph.add_node("A", file_path="a.py", chunk_text="class A")
    graph.add_node("B", file_path="b.py", chunk_text="class B")
    graph.add_node("C", file_path="c.py", chunk_text="class C")
    graph.add_node("D", file_path="d.py", chunk_text="class D")

    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")

    return graph


def test_one_hop_neighbors() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.neighbors("A", hops=1)

    assert len(results) == 1
    assert results[0].symbol == "B"


def test_two_hop_neighbors() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.neighbors("A", hops=2)

    symbols = {r.symbol for r in results}

    assert symbols == {"B", "C"}


def test_shortest_path() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.shortest_path("A", "D")

    assert [r.symbol for r in results] == ["A", "B", "C", "D"]


def test_shortest_path_missing() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.shortest_path("A", "X")

    assert results == []


def test_subgraph_extraction() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.extract_subgraph(["B"], hops=1)

    symbols = {r.symbol for r in results}

    assert symbols == {"A", "B", "C"}


def test_subgraph_multiple_symbols() -> None:
    engine = GraphTraversalEngine(build_graph())

    results = engine.extract_subgraph(["A", "D"], hops=1)

    symbols = {r.symbol for r in results}

    assert symbols == {"A", "B", "C", "D"}
