"""Graph parsing and analysis utilities."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .common import BufferInterval, Node


class ComputeGraph:
    def __init__(self, json_data: Dict):
        self.nodes: Dict[int, Node] = {}
        self.successors: Dict[int, List[int]] = defaultdict(list)
        self.predecessors: Dict[int, List[int]] = defaultdict(list)
        self.in_degree: Dict[int, int] = defaultdict(int)

        self.buf_alloc: Dict[int, int] = {}
        self.buf_free: Dict[int, int] = {}
        self.buf_consumers: Dict[int, List[int]] = defaultdict(list)
        self.buf_size: Dict[int, int] = {}
        self.buf_type: Dict[int, Optional[str]] = {}

        self._parse(json_data)
        self._index_buffers()

    # ------------------------------------------------------------------ parsing
    def _parse(self, data: Dict) -> None:
        for node_data in data["Nodes"]:
            node = Node(
                id=node_data["Id"],
                op=node_data["Op"],
                pipe=node_data.get("Pipe"),
                cycles=int(node_data.get("Cycles", 1) or 1),
                bufs=list(node_data.get("Bufs", [])),
                buf_id=node_data.get("BufId"),
                size=int(node_data.get("Size", 0) or 0),
                cache_type=node_data.get("Type"),
            )
            self.nodes[node.id] = node

        for src, dst in data["Edges"]:
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            self.in_degree[dst] += 1
            if src not in self.in_degree:
                self.in_degree[src] = 0

    def _index_buffers(self) -> None:
        for node in self.nodes.values():
            if node.op == "ALLOC" and node.buf_id is not None:
                self.buf_alloc[node.buf_id] = node.id
                self.buf_size[node.buf_id] = node.size
                self.buf_type[node.buf_id] = node.cache_type
            elif node.op == "FREE" and node.buf_id is not None:
                self.buf_free[node.buf_id] = node.id

            if node.op not in {"ALLOC", "FREE"}:
                for buf in node.bufs:
                    self.buf_consumers[buf].append(node.id)

    # ---------------------------------------------------------------- analysis
    def topological_order(self) -> List[int]:
        order: List[int] = []
        in_deg = dict(self.in_degree)
        ready = deque([nid for nid, deg in in_deg.items() if deg == 0])
        while ready:
            node_id = ready.popleft()
            order.append(node_id)
            for succ in self.successors[node_id]:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    ready.append(succ)
        if len(order) != len(self.nodes):
            missing = set(self.nodes) - set(order)
            raise ValueError(f"graph has cycle or disconnected components: {sorted(missing)[:8]}")
        return order

    def buffer_intervals(self) -> Dict[int, BufferInterval]:
        topo_index = {nid: idx for idx, nid in enumerate(self.topological_order())}
        intervals: Dict[int, BufferInterval] = {}
        for buf_id, alloc_node in self.buf_alloc.items():
            free_node = self.buf_free.get(buf_id)
            if free_node is None:
                continue
            consumers = sorted(self.buf_consumers.get(buf_id, []), key=lambda n: topo_index[n])
            intervals[buf_id] = BufferInterval(
                buf_id=buf_id,
                cache_type=self.buf_type.get(buf_id),
                size=self.buf_size.get(buf_id, 0),
                alloc_node=alloc_node,
                free_node=free_node,
                consumers=consumers,
            )
        return intervals

    def __len__(self) -> int:
        return len(self.nodes)


def load_graph(path: Path) -> ComputeGraph:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return ComputeGraph(data)


__all__ = ["ComputeGraph", "load_graph"]
