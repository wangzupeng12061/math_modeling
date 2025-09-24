from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .data import Graph, Node


@dataclass
class PipelineScheduleResult:
    order: List[int]
    start_times: Dict[int, int]
    finish_times: Dict[int, int]
    makespan: int


class HeftPipelineScheduler:
    """Simplified HEFT-like scheduler optimized for total runtime."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        weight = {nid: max(graph.nodes[nid].cycles, 1) for nid in graph.nodes}
        self.bottom_level = graph.bottom_levels(weight=weight)

    def run(self) -> PipelineScheduleResult:
        indegree = {nid: len(self.graph.edges_in[nid]) for nid in self.graph.nodes}
        ready: List[int] = [nid for nid, deg in indegree.items() if deg == 0]
        order: List[int] = []
        resource_available: Dict[str, int] = defaultdict(int)
        start_times: Dict[int, int] = {}
        finish_times: Dict[int, int] = {}

        while ready:
            candidate = self._select_candidate(ready, finish_times, resource_available)
            ready.remove(candidate)
            node = self.graph.nodes[candidate]

            dep_ready = 0
            if self.graph.edges_in[candidate]:
                dep_ready = max(finish_times[parent] for parent in self.graph.edges_in[candidate])
            pipe_ready = 0
            if node.pipe:
                pipe_ready = resource_available[node.pipe]
            start = max(dep_ready, pipe_ready)
            duration = max(node.cycles, 0)
            finish = start + duration

            order.append(candidate)
            start_times[candidate] = start
            finish_times[candidate] = finish
            if node.pipe:
                resource_available[node.pipe] = finish

            for nbr in self.graph.edges_out[candidate]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    ready.append(nbr)

        makespan = max(finish_times.values(), default=0)
        return PipelineScheduleResult(
            order=order,
            start_times=start_times,
            finish_times=finish_times,
            makespan=makespan,
        )

    def _select_candidate(
        self,
        ready: List[int],
        finish_times: Dict[int, int],
        resource_available: Dict[str, int],
    ) -> int:
        best_id = None
        best_key = None
        for nid in ready:
            node = self.graph.nodes[nid]
            dep_ready = 0
            if self.graph.edges_in[nid]:
                dep_ready = max(finish_times[parent] for parent in self.graph.edges_in[nid])
            pipe_ready = resource_available[node.pipe] if node.pipe else 0
            est = max(dep_ready, pipe_ready)
            key = (est, -self.bottom_level.get(nid, 0), nid)
            if best_key is None or key < best_key:
                best_key = key
                best_id = nid
        assert best_id is not None
        return best_id


def simulate_timeline(graph: Graph, order: List[int]) -> PipelineScheduleResult:
    resource_available: Dict[str, int] = defaultdict(int)
    start_times: Dict[int, int] = {}
    finish_times: Dict[int, int] = {}
    for nid in order:
        node = graph.nodes[nid]
        dep_ready = 0
        if graph.edges_in[nid]:
            dep_ready = max(finish_times[parent] for parent in graph.edges_in[nid])
        pipe_ready = resource_available[node.pipe] if node.pipe else 0
        start = max(dep_ready, pipe_ready)
        duration = max(node.cycles, 0)
        finish = start + duration
        start_times[nid] = start
        finish_times[nid] = finish
        if node.pipe:
            resource_available[node.pipe] = finish
    makespan = max(finish_times.values(), default=0)
    return PipelineScheduleResult(order=order, start_times=start_times, finish_times=finish_times, makespan=makespan)
