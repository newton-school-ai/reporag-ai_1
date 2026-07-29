"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j. Provides Cypher query helpers for neighbors, shortest
path, and subgraph extraction. Includes NetworkX fallback for testing.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node / Edge type enumerations
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Supported graph node labels."""

    FUNCTION = "Function"
    CLASS = "Class"
    MODULE = "Module"


class EdgeType(str, Enum):
    """Supported relationship types."""

    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS = "INHERITS"
    CONTAINS = "CONTAINS"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the code knowledge graph."""

    id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge in the code knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface (shared by Neo4j and NetworkX backends)
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """Abstract interface for graph store backends."""

    @abstractmethod
    def create_nodes(self, nodes: list[GraphNode]) -> None:
        """Bulk-create or upsert nodes."""

    @abstractmethod
    def create_edges(self, edges: list[GraphEdge]) -> None:
        """Bulk-create or upsert edges."""

    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return neighbor nodes up to *depth* hops away."""

    @abstractmethod
    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Return ordered list of node IDs forming the shortest path."""

    @abstractmethod
    def subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """Return nodes and edges induced by *node_ids*."""

    @abstractmethod
    def query(
        self, statement: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw query and return results as a list of dicts."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all nodes and edges from the store."""


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------

_NEO4J_BATCH_SIZE = 500


class Neo4jStore(GraphStore):
    """Neo4j graph store with Cypher query helpers."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Connect to Neo4j with optional retry logic."""
        from neo4j import GraphDatabase  # local import to avoid hard dependency

        self._driver = None
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password))
                driver.verify_connectivity()
                self._driver = driver
                logger.info("Connected to Neo4j at %s (attempt %d)", uri, attempt)
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Neo4j connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)

        if self._driver is None:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {uri} after {max_retries} attempts."
            ) from last_exc

        self._ensure_constraints()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints for each node label."""
        with self._driver.session() as session:
            for label in NodeType:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label.value}) REQUIRE n.id IS UNIQUE"
                )

    def _run(
        self, statement: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(statement, parameters or {})
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # GraphStore interface
    # ------------------------------------------------------------------

    def create_nodes(self, nodes: list[GraphNode]) -> None:
        """Bulk upsert nodes using MERGE, batched for 10K+ scale."""
        if not nodes:
            return

        for i in range(0, len(nodes), _NEO4J_BATCH_SIZE):
            batch = nodes[i : i + _NEO4J_BATCH_SIZE]

            # Group by node type so we can use the correct label
            by_label: dict[str, list[dict[str, Any]]] = {}
            for node in batch:
                label = node.node_type.value
                by_label.setdefault(label, [])
                props = {"id": node.id, **node.properties}
                by_label[label].append(props)

            with self._driver.session() as session:
                for label, rows in by_label.items():
                    session.run(
                        f"""
                        UNWIND $rows AS props
                        MERGE (n:{label} {{id: props.id}})
                        SET n += props
                        """,
                        {"rows": rows},
                    )

        logger.debug("Upserted %d nodes.", len(nodes))

    def create_edges(self, edges: list[GraphEdge]) -> None:
        """Bulk upsert edges, batched for 10K+ scale."""
        if not edges:
            return

        # Group by edge type
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            rel = edge.edge_type.value
            by_rel.setdefault(rel, [])
            by_rel[rel].append(
                {
                    "src": edge.source_id,
                    "tgt": edge.target_id,
                    **edge.properties,
                }
            )

        for rel, rows in by_rel.items():
            for i in range(0, len(rows), _NEO4J_BATCH_SIZE):
                batch = rows[i : i + _NEO4J_BATCH_SIZE]
                with self._driver.session() as session:
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (a {{id: row.src}})
                        MATCH (b {{id: row.tgt}})
                        MERGE (a)-[r:{rel}]->(b)
                        SET r += row
                        """,
                        {"rows": batch},
                    )

        logger.debug("Upserted %d edges.", len(edges))

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return nodes reachable from *node_id* within *depth* hops."""
        return self._run(
            """
            MATCH (n {id: $id})-[*1..$depth]-(neighbor)
            RETURN DISTINCT neighbor.id AS id, labels(neighbor) AS labels,
                   properties(neighbor) AS props
            """,
            {"id": node_id, "depth": depth},
        )

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Return ordered node IDs forming the shortest path."""
        rows = self._run(
            """
            MATCH (src {id: $src}), (tgt {id: $tgt})
            MATCH path = shortestPath((src)-[*]-(tgt))
            RETURN [n IN nodes(path) | n.id] AS path
            """,
            {"src": source_id, "tgt": target_id},
        )
        return rows[0]["path"] if rows else []

    def subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """Return nodes and edges induced by *node_ids*."""
        nodes = self._run(
            """
            MATCH (n)
            WHERE n.id IN $ids
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
            """,
            {"ids": node_ids},
        )
        edges = self._run(
            """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id AS source, b.id AS target, type(r) AS rel, properties(r) AS props
            """,
            {"ids": node_ids},
        )
        return {"nodes": nodes, "edges": edges}

    def query(
        self, statement: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher statement."""
        return self._run(statement, parameters)

    def clear(self) -> None:
        """Delete all nodes and relationships."""
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j store cleared.")

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            self._driver.close()

    def __enter__(self) -> Neo4jStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# NetworkX fallback (identical interface, no external DB required)
# ---------------------------------------------------------------------------


class NetworkXStore(GraphStore):
    """In-memory NetworkX backend implementing the same GraphStore interface."""

    def __init__(self) -> None:
        import networkx as nx  # lazy import: optional dependency

        self._graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # GraphStore interface
    # ------------------------------------------------------------------

    def create_nodes(self, nodes: list[GraphNode]) -> None:
        """Add or update nodes in the in-memory graph."""
        for node in nodes:
            self._graph.add_node(
                node.id,
                node_type=node.node_type.value,
                **node.properties,
            )
        logger.debug("Added %d nodes to NetworkX graph.", len(nodes))

    def create_edges(self, edges: list[GraphEdge]) -> None:
        """Add or update edges in the in-memory graph."""
        for edge in edges:
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                **edge.properties,
            )
        logger.debug("Added %d edges to NetworkX graph.", len(edges))

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return neighbors up to *depth* hops using BFS."""
        if node_id not in self._graph:
            return []

        visited: set[str] = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for nb in self._graph.successors(n):
                    if nb not in visited:
                        next_frontier.add(nb)
                for nb in self._graph.predecessors(n):
                    if nb not in visited:
                        next_frontier.add(nb)
            frontier = next_frontier
            visited |= frontier

        result = []
        for nid in visited - {node_id}:
            data = dict(self._graph.nodes[nid])
            result.append(
                {"id": nid, "labels": [data.pop("node_type", "")], "props": data}
            )
        return result

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Return shortest path using NetworkX."""
        import networkx as nx  # lazy import

        try:
            return nx.shortest_path(self._graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """Return the subgraph induced by *node_ids*."""
        sg = self._graph.subgraph(node_ids)
        nodes = []
        for nid, data in sg.nodes(data=True):
            d = dict(data)
            nodes.append({"id": nid, "labels": [d.pop("node_type", "")], "props": d})
        edges = []
        for src, tgt, data in sg.edges(data=True):
            d = dict(data)
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "rel": d.pop("edge_type", ""),
                    "props": d,
                }
            )
        return {"nodes": nodes, "edges": edges}

    def query(
        self, statement: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Not supported by the NetworkX backend; always returns an empty list."""
        logger.warning(
            "NetworkXStore does not support raw Cypher queries. Statement ignored: %s",
            statement,
        )
        return []

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._graph.clear()
        logger.info("NetworkX store cleared.")
