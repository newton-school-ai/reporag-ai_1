from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx


@dataclass(slots=True)
class RetrievalResult:
    score: float
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    chunk_text: str
    language: str | None = None
    source: str = "graph"


class GraphTraversalEngine:
    """Graph traversal queries over a directed graph."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.graph = graph or nx.DiGraph()

    def neighbors(self, symbol: str, hops: int = 1) -> list[RetrievalResult]:
        """Return all neighbors reachable within `hops` steps."""

        if symbol not in self.graph:
            return []

        visited = {symbol}
        queue = deque([(symbol, 0)])
        results: list[RetrievalResult] = []

        while queue:
            node, depth = queue.popleft()

            if depth == hops:
                continue

            for neighbor in self.graph.successors(node):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                results.append(self._node_to_result(neighbor))

        return results

    def _node_to_result(self, node: str) -> RetrievalResult:

        data = self.graph.nodes.get(node, {})

        return RetrievalResult(
            score=1.0,
            file_path=data.get("file_path", ""),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            symbol=node,
            chunk_text=data.get("chunk_text", node),
            language=data.get("language"),
            source="graph",
        )

    def shortest_path(
        self,
        source: str,
        target: str,
    ) -> list[RetrievalResult]:
        """Return the shortest path between two symbols."""

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return [self._node_to_result(node) for node in path]

    def extract_subgraph(
        self,
        symbols: list[str],
        hops: int = 1,
    ) -> list[RetrievalResult]:

        if not symbols:
            return []

        visited: set[str] = set()

        for symbol in symbols:
            if symbol not in self.graph:
                continue

            queue = deque([(symbol, 0)])
            visited.add(symbol)

            while queue:
                node, depth = queue.popleft()

                if depth == hops:
                    continue

                neighbors = set(self.graph.successors(node)) | set(
                    self.graph.predecessors(node)
                )

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return [self._node_to_result(node) for node in visited]
