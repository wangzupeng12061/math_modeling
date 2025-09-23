"""Baseline scheduler focusing on minimizing peak L1/UB residency."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .common import Node, compute_memory_profile
from .graph import ComputeGraph, load_graph

TRACKED_TYPES: Set[str] = {"L1", "UB"}


@dataclass
class ScheduleResult:
    task_name: str
    order: List[int]
    peak_memory: int


class MemoryAwareScheduler:
    def __init__(self, graph: ComputeGraph):
        self.graph = graph
        self.release_gain = self._compute_release_gain()
        self.fan_out = {nid: len(graph.successors[nid]) for nid in graph.nodes}

    def _compute_release_gain(self) -> Dict[int, int]:
        gain: Dict[int, int] = {}
        intervals = self.graph.buffer_intervals()
        for interval in intervals.values():
            size = interval.size
            for consumer in interval.consumers:
                gain[consumer] = gain.get(consumer, 0) + size
            free_node = interval.free_node
            gain[free_node] = gain.get(free_node, 0) + size
        return gain

    def _priority(self, node_id: int) -> int:
        node = self.graph.nodes[node_id]
        score = self.fan_out[node_id]
        score += self.release_gain.get(node_id, 0)
        if node.op == "FREE" and node.cache_type in TRACKED_TYPES:
            score += node.size * 4
        elif node.op == "ALLOC" and node.cache_type in TRACKED_TYPES:
            score -= node.size * 2
        return score

    def schedule(self) -> List[int]:
        in_degree = dict(self.graph.in_degree)
        ready: List[Tuple[int, int]] = []
        for nid, deg in in_degree.items():
            if deg == 0:
                heapq.heappush(ready, (-self._priority(nid), nid))

        order: List[int] = []
        while ready:
            _, node_id = heapq.heappop(ready)
            order.append(node_id)
            for succ in self.graph.successors[node_id]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    heapq.heappush(ready, (-self._priority(succ), succ))

        if len(order) != len(self.graph):
            missing = set(self.graph.nodes) - set(order)
            raise RuntimeError(f"Scheduling incomplete: {sorted(missing)[:8]}")
        return order


def schedule_graph(graph: ComputeGraph) -> ScheduleResult:
    scheduler = MemoryAwareScheduler(graph)
    order = scheduler.schedule()
    nodes = [graph.nodes[nid] for nid in order if graph.nodes[nid].cache_type in TRACKED_TYPES or graph.nodes[nid].op not in {"ALLOC", "FREE"}]
    peak, _ = compute_memory_profile(nodes)
    return ScheduleResult(task_name="", order=order, peak_memory=peak)


def solve_files(paths: Sequence[Path]) -> List[ScheduleResult]:
    results: List[ScheduleResult] = []
    for path in paths:
        graph = load_graph(path)
        result = schedule_graph(graph)
        result.task_name = path.stem
        results.append(result)
    return results


__all__ = ["ScheduleResult", "schedule_graph", "solve_files", "MemoryAwareScheduler"]
