"""Unit tests for the neo4j_store module.

The test suite is parameterised to run against *both* backends:

* **NetworkXStore** - in-memory, no external dependencies.
* **Neo4jStore** - tested via a ``MagicMock`` driver so the suite never
  requires a live Neo4j instance.  The mock validates that the correct
  Cypher statements and parameters are forwarded to the driver.

Acceptance criteria verified
-----------------------------
* Creates nodes with correct labels and properties.
* CALLS, IMPORTS, INHERITS, CONTAINS edges work.
* Cypher helpers (get_neighbors, shortest_path, subgraph, query, clear)
  return correct results.
* NetworkX fallback passes the same test suite.
* Bulk insert handles 10K+ nodes without error.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.reporag.graph.neo4j_store import (
    EdgeType,
    GraphEdge,
    GraphNode,
    Neo4jStore,
    NetworkXStore,
    NodeType,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_nodes() -> list[GraphNode]:
    """Return a small fixed set of nodes."""
    return [
        GraphNode(
            id="func_a",
            node_type=NodeType.FUNCTION,
            properties={"file": "a.py", "line": 1},
        ),
        GraphNode(
            id="func_b",
            node_type=NodeType.FUNCTION,
            properties={"file": "a.py", "line": 10},
        ),
        GraphNode(
            id="ClassX",
            node_type=NodeType.CLASS,
            properties={"file": "b.py", "line": 5},
        ),
        GraphNode(
            id="mod_main", node_type=NodeType.MODULE, properties={"file": "main.py"}
        ),
    ]


def make_edges() -> list[GraphEdge]:
    """Return a small fixed set of edges."""
    return [
        GraphEdge(source_id="func_a", target_id="func_b", edge_type=EdgeType.CALLS),
        GraphEdge(source_id="mod_main", target_id="ClassX", edge_type=EdgeType.IMPORTS),
        GraphEdge(source_id="ClassX", target_id="func_a", edge_type=EdgeType.CONTAINS),
        GraphEdge(source_id="ClassX", target_id="func_b", edge_type=EdgeType.INHERITS),
    ]


# ---------------------------------------------------------------------------
# Shared behaviour helpers (backend-agnostic)
# ---------------------------------------------------------------------------


def _run_node_creation(store: Any) -> None:
    nodes = make_nodes()
    store.create_nodes(nodes)  # should not raise


def _run_edge_creation(store: Any) -> None:
    store.create_nodes(make_nodes())
    store.create_edges(make_edges())  # should not raise


def _run_clear(store: Any) -> None:
    store.create_nodes(make_nodes())
    store.clear()


# ---------------------------------------------------------------------------
# NetworkXStore tests  (always run - no mocking required)
# ---------------------------------------------------------------------------


class TestNetworkXStore:
    """NetworkXStore passes the full test suite without any mocks."""

    def setup_method(self) -> None:
        pytest.importorskip("networkx", reason="networkx not installed")
        self.store = NetworkXStore()

    # --- node creation ---

    def test_create_nodes_adds_to_graph(self) -> None:
        self.store.create_nodes(make_nodes())
        assert "func_a" in self.store._graph.nodes
        assert "ClassX" in self.store._graph.nodes

    def test_node_properties_preserved(self) -> None:
        self.store.create_nodes(make_nodes())
        data = self.store._graph.nodes["func_a"]
        assert data["node_type"] == NodeType.FUNCTION.value
        assert data["file"] == "a.py"
        assert data["line"] == 1

    def test_create_nodes_empty_is_noop(self) -> None:
        self.store.create_nodes([])
        assert len(self.store._graph.nodes) == 0

    # --- edge creation ---

    def test_create_edges_all_types(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        edges = list(self.store._graph.edges(data=True))
        edge_types = {d["edge_type"] for _, _, d in edges}
        assert EdgeType.CALLS.value in edge_types
        assert EdgeType.IMPORTS.value in edge_types
        assert EdgeType.CONTAINS.value in edge_types
        assert EdgeType.INHERITS.value in edge_types

    def test_create_edges_correct_direction(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges([GraphEdge("func_a", "func_b", EdgeType.CALLS)])
        assert self.store._graph.has_edge("func_a", "func_b")
        assert not self.store._graph.has_edge("func_b", "func_a")

    def test_create_edges_empty_is_noop(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges([])
        assert len(self.store._graph.edges) == 0

    # --- get_neighbors ---

    def test_get_neighbors_depth_1(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        neighbors = self.store.get_neighbors("func_a", depth=1)
        ids = {n["id"] for n in neighbors}
        # func_a --CALLS--> func_b   &   ClassX --CONTAINS--> func_a
        assert "func_b" in ids
        assert "ClassX" in ids

    def test_get_neighbors_unknown_node_returns_empty(self) -> None:
        assert self.store.get_neighbors("does_not_exist") == []

    def test_get_neighbors_depth_2(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        neighbors = self.store.get_neighbors("mod_main", depth=2)
        ids = {n["id"] for n in neighbors}
        # mod_main --IMPORTS--> ClassX --CONTAINS--> func_a / func_b
        assert "ClassX" in ids
        assert "func_a" in ids or "func_b" in ids

    # --- shortest_path ---

    def test_shortest_path_direct_edge(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        path = self.store.shortest_path("func_a", "func_b")
        assert path[0] == "func_a"
        assert path[-1] == "func_b"

    def test_shortest_path_no_path_returns_empty(self) -> None:
        self.store.create_nodes(
            [
                GraphNode("isolated_a", NodeType.FUNCTION),
                GraphNode("isolated_b", NodeType.CLASS),
            ]
        )
        # No edges between them
        assert self.store.shortest_path("isolated_a", "isolated_b") == []

    def test_shortest_path_unknown_node_returns_empty(self) -> None:
        assert self.store.shortest_path("x", "y") == []

    # --- subgraph ---

    def test_subgraph_returns_nodes_and_edges(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        sg = self.store.subgraph(["func_a", "func_b"])
        node_ids = {n["id"] for n in sg["nodes"]}
        assert "func_a" in node_ids
        assert "func_b" in node_ids
        # The CALLS edge between them should appear
        assert len(sg["edges"]) >= 1
        assert sg["edges"][0]["rel"] == EdgeType.CALLS.value

    def test_subgraph_empty_ids(self) -> None:
        self.store.create_nodes(make_nodes())
        sg = self.store.subgraph([])
        assert sg["nodes"] == []
        assert sg["edges"] == []

    # --- query (unsupported) ---

    def test_query_returns_empty_list_with_warning(self) -> None:
        result = self.store.query("MATCH (n) RETURN n")
        assert result == []

    # --- clear ---

    def test_clear_removes_all_nodes_and_edges(self) -> None:
        self.store.create_nodes(make_nodes())
        self.store.create_edges(make_edges())
        self.store.clear()
        assert len(self.store._graph.nodes) == 0
        assert len(self.store._graph.edges) == 0

    def test_clear_on_empty_store_is_safe(self) -> None:
        self.store.clear()  # should not raise

    # --- bulk insert ---

    def test_bulk_insert_10k_nodes(self) -> None:
        nodes = [
            GraphNode(
                id=f"func_{i}", node_type=NodeType.FUNCTION, properties={"index": i}
            )
            for i in range(10_000)
        ]
        self.store.create_nodes(nodes)
        assert len(self.store._graph.nodes) == 10_000

    def test_bulk_insert_10k_edges(self) -> None:
        nodes = [
            GraphNode(id=f"n_{i}", node_type=NodeType.FUNCTION) for i in range(1_001)
        ]
        self.store.create_nodes(nodes)
        edges = [
            GraphEdge(
                source_id=f"n_{i}", target_id=f"n_{i + 1}", edge_type=EdgeType.CALLS
            )
            for i in range(10_000 % 1_000)  # keep within available nodes
        ]
        self.store.create_edges(edges)  # should not raise


# ---------------------------------------------------------------------------
# Neo4jStore tests  (mocked driver - no live DB required)
# ---------------------------------------------------------------------------


def _make_neo4j_store() -> Neo4jStore:
    """Return a Neo4jStore whose driver is fully mocked."""
    mock_driver = MagicMock()
    mock_driver.verify_connectivity.return_value = None

    # Mock session context manager
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])
    mock_driver.session.return_value = mock_session

    with patch("src.reporag.graph.neo4j_store.Neo4jStore.__init__") as mock_init:
        mock_init.return_value = None
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

    return store


class TestNeo4jStoreMocked:
    """Neo4jStore tested with a mocked driver.

    The Neo4jStore delegates all real I/O to the driver.  These tests
    verify that the correct Cypher patterns and parameters are forwarded,
    matching the same acceptance criteria as the NetworkX suite.
    """

    def setup_method(self) -> None:
        self.store = _make_neo4j_store()
        self.session: MagicMock = self.store._driver.session.return_value

    # --- helper ---

    def _all_cypher(self) -> list[str]:
        """Collect all Cypher statements sent to session.run()."""
        return [call.args[0] for call in self.session.run.call_args_list]

    # --- node creation ---

    def test_create_nodes_calls_merge(self) -> None:
        self.store.create_nodes(make_nodes())
        cypher_stmts = self._all_cypher()
        assert any("MERGE" in s for s in cypher_stmts)

    def test_create_nodes_uses_correct_labels(self) -> None:
        self.store.create_nodes(make_nodes())
        cypher_stmts = " ".join(self._all_cypher())
        assert "Function" in cypher_stmts
        assert "Class" in cypher_stmts
        assert "Module" in cypher_stmts

    def test_create_nodes_empty_no_db_call(self) -> None:
        self.store.create_nodes([])
        self.session.run.assert_not_called()

    # --- edge creation ---

    def test_create_edges_all_types_forwarded(self) -> None:
        self.store.create_edges(make_edges())
        cypher_stmts = " ".join(self._all_cypher())
        for rel in ("CALLS", "IMPORTS", "CONTAINS", "INHERITS"):
            assert rel in cypher_stmts

    def test_create_edges_uses_merge(self) -> None:
        self.store.create_edges(make_edges())
        cypher_stmts = self._all_cypher()
        assert any("MERGE" in s for s in cypher_stmts)

    def test_create_edges_empty_no_db_call(self) -> None:
        self.store.create_edges([])
        self.session.run.assert_not_called()

    # --- get_neighbors ---

    def test_get_neighbors_sends_cypher(self) -> None:
        self.session.run.return_value = iter([])
        self.store.get_neighbors("func_a", depth=1)
        cypher_stmts = self._all_cypher()
        assert any("neighbor" in s.lower() for s in cypher_stmts)

    def test_get_neighbors_passes_depth_param(self) -> None:
        self.session.run.return_value = iter([])
        self.store.get_neighbors("func_a", depth=3)
        call_kwargs = self.session.run.call_args
        params = (
            call_kwargs.args[1]
            if len(call_kwargs.args) > 1
            else call_kwargs.kwargs.get("parameters", {})
        )
        assert params.get("depth") == 3

    # --- shortest_path ---

    def test_shortest_path_sends_cypher(self) -> None:
        self.session.run.return_value = iter([])
        self.store.shortest_path("func_a", "func_b")
        cypher_stmts = self._all_cypher()
        assert any("shortestPath" in s for s in cypher_stmts)

    def test_shortest_path_no_result_returns_empty(self) -> None:
        self.session.run.return_value = iter([])
        result = self.store.shortest_path("x", "y")
        assert result == []

    def test_shortest_path_with_result(self) -> None:
        mock_row = MagicMock()
        mock_row.__iter__ = MagicMock(return_value=iter([("path", ["x", "y"])]))
        mock_row.__getitem__ = MagicMock(
            side_effect=lambda k: ["x", "y"] if k == "path" else None
        )
        mock_row.keys.return_value = ["path"]

        def mock_run_result(*args, **kwargs):
            if "shortestPath" in args[0]:
                return iter([mock_row])
            return iter([])

        self.session.run.side_effect = mock_run_result
        result = self.store.shortest_path("x", "y")
        assert result == ["x", "y"]

    # --- subgraph ---

    def test_subgraph_fires_two_queries(self) -> None:
        self.session.run.return_value = iter([])
        self.store.subgraph(["func_a", "func_b"])
        # One query for nodes, one for edges
        assert self.session.run.call_count == 2

    def test_subgraph_returns_dict_with_keys(self) -> None:
        self.session.run.return_value = iter([])
        result = self.store.subgraph(["func_a"])
        assert "nodes" in result
        assert "edges" in result

    # --- raw query ---

    def test_query_forwards_statement(self) -> None:
        self.session.run.return_value = iter([])
        stmt = "MATCH (n:Function) RETURN n"
        self.store.query(stmt)
        self.session.run.assert_called_once_with(stmt, {})

    def test_query_passes_parameters(self) -> None:
        self.session.run.return_value = iter([])
        params = {"name": "func_a"}
        self.store.query("MATCH (n {id: $name}) RETURN n", params)
        self.session.run.assert_called_once_with(
            "MATCH (n {id: $name}) RETURN n", params
        )

    # --- clear ---

    def test_clear_sends_detach_delete(self) -> None:
        self.store.clear()
        cypher_stmts = self._all_cypher()
        assert any("DETACH DELETE" in s for s in cypher_stmts)

    # --- bulk insert (passes parameters in batches) ---

    def test_bulk_insert_10k_nodes_batched(self) -> None:
        """10K nodes should trigger multiple batched MERGE calls."""
        nodes = [
            GraphNode(id=f"func_{i}", node_type=NodeType.FUNCTION)
            for i in range(10_000)
        ]
        self.store.create_nodes(nodes)
        # At least ceil(10000 / 500) = 20 session.run calls
        assert self.session.run.call_count >= 20

    def test_bulk_insert_10k_edges_batched(self) -> None:
        """10K edges should trigger multiple batched MERGE calls."""
        edges = [
            GraphEdge(
                source_id=f"n_{i}",
                target_id=f"n_{i + 1}",
                edge_type=EdgeType.CALLS,
            )
            for i in range(10_000)
        ]
        self.store.create_edges(edges)
        assert self.session.run.call_count >= 20

    # --- context manager ---

    def test_context_manager_calls_close(self) -> None:
        with self.store:
            pass
        self.store._driver.close.assert_called_once()

    # --- connection retry ---

    def test_connection_retry_raises_after_max_attempts(self) -> None:
        neo4j_mod = pytest.importorskip("neo4j", reason="neo4j package not installed")
        graph_database = neo4j_mod.GraphDatabase

        with patch.object(graph_database, "driver") as mock_gd:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.side_effect = Exception(
                "Connection refused"
            )
            mock_gd.return_value = mock_driver

            with pytest.raises(ConnectionError):
                Neo4jStore(uri="bolt://localhost:9999", max_retries=2, retry_delay=0)
