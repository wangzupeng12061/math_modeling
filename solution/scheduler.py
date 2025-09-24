from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Graph, Node


@dataclass
class ScheduleResult:
    order: List[int]
    max_resident: int
    residency_trace: List[Tuple[int, int]]


class GreedyMemoryAwareScheduler:
    """Heuristic scheduler that prefers nodes releasing memory early."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        weight = {nid: graph.nodes[nid].cycles or 1 for nid in graph.nodes}
        self.bottom_level = graph.bottom_levels(weight=weight)

    def run(self) -> ScheduleResult:
        indegree = {nid: len(self.graph.edges_in[nid]) for nid in self.graph.nodes}
        ready_heap: List[Tuple[Tuple[float, ...], int]] = []
        for nid, deg in indegree.items():
            if deg == 0:
                heappush(ready_heap, (self._priority_key(nid), nid))
        order: List[int] = []
        current_resident = 0
        max_resident = 0
        trace: List[Tuple[int, int]] = []

        while ready_heap:
            _, nid = heappop(ready_heap)
            order.append(nid)
            node = self.graph.nodes[nid]

            if node.is_alloc:
                current_resident += node.size
            elif node.is_free:
                current_resident -= node.size
            max_resident = max(max_resident, current_resident)
            trace.append((nid, current_resident))

            for nbr in self.graph.edges_out[nid]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    heappush(ready_heap, (self._priority_key(nbr), nbr))

        if len(order) != len(self.graph.nodes):
            missing = set(self.graph.nodes) - set(order)
            raise RuntimeError(f"Scheduling failed, remaining nodes: {missing}")

        if current_resident != 0:
            # Mem accounting guard: free nodes should balance alloc nodes.
            raise RuntimeError(
                "Residency accounting imbalance detected: {}".format(current_resident)
            )

        return ScheduleResult(order=order, max_resident=max_resident, residency_trace=trace)

    def _priority_key(self, nid: int) -> Tuple[float, ...]:
        node = self.graph.nodes[nid]
        class_rank = self._class_rank(node)
        release = node.size if node.is_free else 0
        alloc = node.size if node.is_alloc else 0
        bottom_level = self.bottom_level.get(nid, 0.0)
        # heapq is min-heap, so use negative for max-style ordering fields.
        return (
            -class_rank,
            -release,
            alloc,
            -bottom_level,
            node.id,
        )

    @staticmethod
    def _class_rank(node: Node) -> int:
        op = node.op.upper()
        if op == "FREE":
            return 3
        if op in {"COPY_OUT", "COPY_IN", "MOVE", "SPILL_OUT", "SPILL_IN"}:
            return 2
        if op == "ALLOC":
            return 1
        return 2
