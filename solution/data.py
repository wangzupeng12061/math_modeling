from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Node:
    """Representation of a graph node lifted from the JSON schema."""

    id: int
    op: str
    pipe: Optional[str] = None
    cycles: int = 0
    bufs: Tuple[int, ...] = ()
    buf_id: Optional[int] = None
    size: int = 0
    type: Optional[str] = None

    @property
    def is_alloc(self) -> bool:
        return self.op.upper() == "ALLOC"

    @property
    def is_free(self) -> bool:
        return self.op.upper() == "FREE"

    @property
    def is_memory_op(self) -> bool:
        return self.is_alloc or self.is_free


@dataclass
class Graph:
    """In-memory representation of the DAG with quick lookup helpers."""

    nodes: Dict[int, Node]
    edges_out: Dict[int, List[int]]
    edges_in: Dict[int, List[int]]
    topo_order: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Guarantee we can iterate deterministically even if JSON omits entries.
        for node_id in self.nodes:
            self.edges_out.setdefault(node_id, [])
            self.edges_in.setdefault(node_id, [])

    def compute_topological_order(self) -> List[int]:
        if self.topo_order:
            return list(self.topo_order)
        indegree = {nid: len(self.edges_in[nid]) for nid in self.nodes}
        ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
        order: List[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for nbr in self.edges_out[nid]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    ready.append(nbr)
                    ready.sort()
        if len(order) != len(self.nodes):
            missing = set(self.nodes) - set(order)
            raise ValueError(f"Graph contains cycles or disconnected nodes: {missing}")
        self.topo_order = order
        return list(order)

    def bottom_levels(self, weight: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        """Return length of the longest path from each node to exit."""

        if weight is None:
            weight = {nid: 1.0 for nid in self.nodes}
        order = self.compute_topological_order()[::-1]
        levels: Dict[int, float] = {nid: weight.get(nid, 1.0) for nid in self.nodes}
        for nid in order:
            longest_child = 0.0
            for child in self.edges_out[nid]:
                longest_child = max(longest_child, levels[child])
            levels[nid] = weight.get(nid, 1.0) + longest_child
        return levels

    @property
    def sources(self) -> List[int]:
        return [nid for nid, parents in self.edges_in.items() if not parents]

    @property
    def sinks(self) -> List[int]:
        return [nid for nid, children in self.edges_out.items() if not children]

    def buffer_allocators(self) -> Dict[int, int]:
        allocators: Dict[int, int] = {}
        for node in self.nodes.values():
            if node.is_alloc and node.buf_id is not None:
                allocators[node.buf_id] = node.id
        return allocators

    def buffer_frees(self) -> Dict[int, int]:
        frees: Dict[int, int] = {}
        for node in self.nodes.values():
            if node.is_free and node.buf_id is not None:
                frees[node.buf_id] = node.id
        return frees


def load_graph(path: Path) -> Graph:
    """Load a graph description from JSON or CSV file."""

    if path.suffix.lower() == ".json":
        return _load_from_json(path)
    if path.suffix.lower() == ".csv":
        raise NotImplementedError("CSV graph format is not implemented yet")
    raise ValueError(f"Unsupported graph file: {path}")


def _load_from_json(path: Path) -> Graph:
    payload = json.loads(path.read_text())
    nodes: Dict[int, Node] = {}
    for raw_node in payload["Nodes"]:
        node = Node(
            id=raw_node["Id"],
            op=raw_node["Op"],
            pipe=raw_node.get("Pipe"),
            cycles=int(raw_node.get("Cycles", 0) or 0),
            bufs=tuple(raw_node.get("Bufs", [])),
            buf_id=raw_node.get("BufId"),
            size=int(raw_node.get("Size", 0) or 0),
            type=raw_node.get("Type"),
        )
        nodes[node.id] = node
    edges_out: Dict[int, List[int]] = {}
    edges_in: Dict[int, List[int]] = {}
    for src, dst in payload["Edges"]:
        edges_out.setdefault(src, []).append(dst)
        edges_in.setdefault(dst, []).append(src)
    return Graph(nodes=nodes, edges_out=edges_out, edges_in=edges_in)
