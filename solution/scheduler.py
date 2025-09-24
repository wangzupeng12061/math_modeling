from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, item: int) -> int:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        if pa < pb:
            self.parent[pb] = pa
        else:
            self.parent[pa] = pb

from .data import Graph, Node

DEFAULT_CAPACITIES: Dict[str, int] = {
    "L1": 4096,
    "UB": 1024,
    "L0A": 256,
    "L0B": 256,
    "L0C": 512,
}


@dataclass
class SchedulerConfig:
    capacities: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CAPACITIES))
    capacity_slack: float = 1.0
    strategy: str = "auto"


@dataclass
class ScheduleResult:
    order: List[int]
    max_resident: int
    residency_trace: List[Tuple[int, int]]


class GreedyMemoryAwareScheduler:
    """List scheduler emphasising lifetime compression and operator structure."""

    def __init__(self, graph: Graph, config: Optional[SchedulerConfig] = None) -> None:
        self.graph = graph
        self.config = config or SchedulerConfig()
        self.strategy = self._infer_strategy(self.config.strategy)

        weight = {nid: graph.nodes[nid].cycles or 1 for nid in graph.nodes}
        self.bottom_level = graph.bottom_levels(weight=weight)

        # Buffer metadata.
        self.buf_types: Dict[int, str] = {}
        self.buf_sizes: Dict[int, int] = {}
        self.buf_alloc_index: Dict[int, int] = {}
        for index, nid in enumerate(sorted(graph.nodes)):
            node = graph.nodes[nid]
            if node.is_alloc and node.buf_id is not None:
                self.buf_types[node.buf_id] = node.type or "L1"
                self.buf_sizes[node.buf_id] = node.size
                self.buf_alloc_index[node.buf_id] = index

        self.total_allocs = max(self.buf_alloc_index.values(), default=0) + 1

        # Remaining consumer counts per buffer.
        self.remaining_uses: Dict[int, int] = {}
        for node in graph.nodes.values():
            if node.is_free:
                continue
            for buf_id in node.bufs or ():
                if buf_id is None:
                    continue
                self.remaining_uses[buf_id] = self.remaining_uses.get(buf_id, 0) + 1

        self.buffer_users: Dict[int, List[int]] = {}
        for node in graph.nodes.values():
            for buf_id in self._node_buffers(node):
                self.buffer_users.setdefault(buf_id, []).append(node.id)

        # Capacity bookkeeping (needed for tile heuristics).
        slack_factor = max(0.1, self.config.capacity_slack * self._strategy_slack_factor())
        self.capacities = {
            cache: max(1, int(cap * slack_factor))
            for cache, cap in self.config.capacities.items()
        }
        self.global_limit = sum(self.capacities.values()) if self.capacities else None
        self.current_usage: Dict[str, int] = {cache: 0 for cache in self.capacities}
        self.total_usage = 0

        # Tile bookkeeping.
        self.tile_types = self._tile_types_for_strategy()
        self.tile_of_buf: Dict[int, int] = self._compute_tile_components()
        self.node_tile: Dict[int, Optional[int]] = {}
        self.tile_remaining: Dict[int, int] = {}
        for nid, node in graph.nodes.items():
            tile = self._infer_tile(node)
            self.node_tile[nid] = tile
            if tile is not None:
                self.tile_remaining[tile] = self.tile_remaining.get(tile, 0) + 1
        self.focus_tile: Optional[int] = None
        self.tile_sequence: List[int] = sorted(
            self.tile_remaining,
            key=lambda tile: self.buf_alloc_index.get(tile, self.total_allocs),
        )
        self.tile_cursor = 0
        self.tile_sizes: Dict[int, int] = self._compute_tile_sizes()
        self.conv_mode: Optional[str] = self._select_conv_mode()

    # ------------------------------------------------------------------
    # Public API

    def run(self) -> ScheduleResult:
        indegree = {nid: len(self.graph.edges_in[nid]) for nid in self.graph.nodes}
        ready: List[int] = sorted(nid for nid, deg in indegree.items() if deg == 0)

        order: List[int] = []
        current_resident = 0
        max_resident = 0
        trace: List[Tuple[int, int]] = []

        while ready:
            nid = self._pick_best(ready, current_resident)
            ready.remove(nid)
            order.append(nid)

            node = self.graph.nodes[nid]
            delta = 0
            if node.is_alloc:
                self._apply_alloc(node)
                delta += node.size
            if node.is_free:
                self._apply_free(node)
                delta -= node.size

            current_resident += delta
            max_resident = max(max_resident, current_resident)
            trace.append((nid, current_resident))

            self._after_execute(nid)

            for nbr in self.graph.edges_out[nid]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    ready.append(nbr)

        if len(order) != len(self.graph.nodes):
            missing = set(self.graph.nodes) - set(order)
            raise RuntimeError(f"Scheduling failed, remaining nodes: {missing}")
        if current_resident != 0:
            raise RuntimeError(f"Residency accounting imbalance detected: {current_resident}")

        return ScheduleResult(order=order, max_resident=max_resident, residency_trace=trace)

    # ------------------------------------------------------------------
    # Selection logic

    def _pick_best(self, ready: List[int], current_resident: int) -> int:
        best_id: Optional[int] = None
        best_key: Optional[Tuple[float, ...]] = None
        fallback_id: Optional[int] = None
        fallback_key: Optional[Tuple[float, ...]] = None

        current_tile = self._current_tile()
        candidate_pool = self._tile_filtered_ready(ready, current_resident, current_tile)

        for nid in candidate_pool:
            node = self.graph.nodes[nid]
            score_key = self._score_tuple(nid, current_resident)
            if not self._violates_capacity(node):
                if best_key is None or score_key > best_key:
                    best_key = score_key
                    best_id = nid
            else:
                if fallback_key is None or score_key > fallback_key:
                    fallback_key = score_key
                    fallback_id = nid

        if best_id is not None:
            return best_id
        if fallback_id is not None:
            return fallback_id
        return max(ready, key=lambda nid: self._score_tuple(nid, current_resident))

    def _tile_filtered_ready(
        self,
        ready: List[int],
        current_resident: int,
        current_tile: Optional[int],
    ) -> List[int]:
        if self.strategy == "conv":
            return ready
        if current_tile is None:
            return ready
        tile_candidates = [
            nid
            for nid in ready
            if self.node_tile.get(nid) == current_tile or self.graph.nodes[nid].is_free
        ]
        return tile_candidates if tile_candidates else ready

    def _current_tile(self) -> Optional[int]:
        if self.strategy == "conv":
            active_tiles = [tile for tile, remain in self.tile_remaining.items() if remain > 0]
            if not active_tiles:
                return None
            if self.conv_mode == "depth":
                return max(
                    active_tiles,
                    key=lambda tile: (
                        self.tile_sizes.get(tile, 0),
                        -self.buf_alloc_index.get(tile, self.total_allocs),
                    ),
                )
            return min(active_tiles, key=lambda tile: self.buf_alloc_index.get(tile, self.total_allocs))
        while self.tile_cursor < len(self.tile_sequence):
            tile = self.tile_sequence[self.tile_cursor]
            if self.tile_remaining.get(tile, 0) > 0:
                return tile
            self.tile_cursor += 1
        return None

    def _score_tuple(self, nid: int, current_resident: int) -> Tuple[float, ...]:
        node = self.graph.nodes[nid]
        alloc = node.size if node.is_alloc else 0
        release = node.size if node.is_free else 0
        future_release = self._future_release(node)

        net_after = current_resident + alloc - release
        predicted_peak = max(current_resident + alloc, net_after)

        class_rank = self._class_rank(node, future_release)
        tile_bonus = self._tile_bonus(nid)
        alloc_priority = self._alloc_priority(node)
        bottom = self.bottom_level.get(nid, 0.0)
        conv_bonus = 0
        if self.strategy == "conv":
            if node.is_free and node.buf_id is not None:
                if self.buf_types.get(node.buf_id) == "L1":
                    conv_bonus = self.buf_sizes.get(node.buf_id, 0)
            elif node.is_alloc and node.buf_id is not None:
                if self.buf_types.get(node.buf_id) == "L1":
                    conv_bonus = -self.buf_sizes.get(node.buf_id, 0)
        flash_bonus = 0
        if self.strategy == "flashattention" and node.buf_id is not None:
            buf_type = self.buf_types.get(node.buf_id)
            if buf_type in {"UB", "L0C", "L1"}:
                size = self.buf_sizes.get(node.buf_id, node.size)
                if node.is_free:
                    flash_bonus = size
                elif node.is_alloc:
                    flash_bonus = -size

        return (
            -predicted_peak,
            -net_after,
            future_release,
            release,
            -alloc,
            class_rank,
            tile_bonus,
            alloc_priority,
            conv_bonus,
            flash_bonus,
            bottom,
            -node.id,
        )

    def _future_release(self, node: Node) -> int:
        total = 0
        for buf_id in node.bufs or ():
            if buf_id is None:
                continue
            if self.remaining_uses.get(buf_id, 0) == 1:
                total += self.buf_sizes.get(buf_id, 0)
        return total

    @staticmethod
    def _class_rank(node: Node, future_release: int) -> int:
        if node.is_free:
            return 4
        if future_release > 0:
            return 3
        if node.is_alloc:
            return 1
        return 2

    def _tile_bonus(self, nid: int) -> int:
        tile = self.node_tile.get(nid)
        if tile is None:
            return 0
        if self.focus_tile is None:
            return 1
        return 3 if tile == self.focus_tile else 0

    def _alloc_priority(self, node: Node) -> int:
        buffers = self._node_buffers(node)
        indices = [self.buf_alloc_index.get(buf) for buf in buffers if buf in self.buf_alloc_index]
        if not indices:
            return 0
        return self.total_allocs - min(indices)

    def _node_buffers(self, node: Node) -> List[int]:
        buffers: List[int] = []
        if node.buf_id is not None:
            buffers.append(node.buf_id)
        for buf_id in node.bufs or ():
            if buf_id is not None:
                buffers.append(buf_id)
        return buffers

    # ------------------------------------------------------------------
    # Capacity management

    def _violates_capacity(self, node: Node) -> bool:
        if not node.is_alloc:
            return False
        buf_type = node.type or self.buf_types.get(node.buf_id or -1)
        if buf_type is None:
            return False
        capacity = self.capacities.get(buf_type)
        if capacity and (self.current_usage.get(buf_type, 0) + node.size) > capacity:
            return True
        if self.global_limit is not None and (self.total_usage + node.size) > self.global_limit:
            return True
        return False

    def _apply_alloc(self, node: Node) -> None:
        buf_id = node.buf_id
        if buf_id is None:
            return
        buf_type = node.type or self.buf_types.get(buf_id) or "L1"
        self.current_usage[buf_type] = self.current_usage.get(buf_type, 0) + node.size
        self.total_usage += node.size

    def _apply_free(self, node: Node) -> None:
        buf_id = node.buf_id
        if buf_id is None:
            return
        buf_type = self.buf_types.get(buf_id) or node.type or "L1"
        size = self.buf_sizes.get(buf_id, node.size)
        if buf_type in self.current_usage:
            self.current_usage[buf_type] = max(0, self.current_usage.get(buf_type, 0) - size)
        self.total_usage = max(0, self.total_usage - size)
        self.remaining_uses[buf_id] = 0

    # ------------------------------------------------------------------
    # State updates

    def _after_execute(self, nid: int) -> None:
        node = self.graph.nodes[nid]
        for buf_id in node.bufs or ():
            if buf_id is None:
                continue
            remaining = self.remaining_uses.get(buf_id)
            if remaining is not None and remaining > 0:
                self.remaining_uses[buf_id] = remaining - 1

        tile = self.node_tile.get(nid)
        if tile is not None:
            remaining = self.tile_remaining.get(tile, 0)
            if remaining > 0:
                remaining -= 1
                if remaining <= 0:
                    self.tile_remaining.pop(tile, None)
                    if self.focus_tile == tile:
                        self.focus_tile = None
                    if self.tile_cursor < len(self.tile_sequence) and self.tile_sequence[self.tile_cursor] == tile:
                        self.tile_cursor += 1
                else:
                    self.tile_remaining[tile] = remaining
            if not node.is_alloc and not node.is_free:
                self.focus_tile = tile

    # ------------------------------------------------------------------
    # Strategy helpers

    def _tile_types_for_strategy(self) -> Tuple[str, ...]:
        if self.strategy == "matmul":
            return ("L0C",)
        if self.strategy == "flashattention":
            return ("UB",)
        if self.strategy == "conv":
            return ("L1",)
        return tuple()

    def _strategy_slack_factor(self) -> float:
        if self.strategy == "flashattention":
            return 0.6
        if self.strategy == "matmul":
            return 0.8
        return 1.0

    def _compute_tile_components(self) -> Dict[int, int]:
        if not self.tile_types:
            return {}
        tile_buffers = [
            buf_id
            for buf_id, buf_type in self.buf_types.items()
            if buf_type in self.tile_types
        ]
        if not tile_buffers:
            return {}

        dsu = DisjointSet()
        for buf_id in tile_buffers:
            dsu.find(buf_id)

        for node in self.graph.nodes.values():
            buffers = [
                buf
                for buf in self._node_buffers(node)
                if self.buf_types.get(buf) in self.tile_types
            ]
            if len(buffers) < 2:
                continue
            base = buffers[0]
            for other in buffers[1:]:
                dsu.union(base, other)

        return {buf: dsu.find(buf) for buf in tile_buffers}

    def _propagate_tile_ids(self, assignment: Dict[int, int]) -> Dict[int, int]:
        if not assignment:
            return assignment
        queue: List[int] = list(assignment.keys())
        while queue:
            buf = queue.pop()
            tile = assignment[buf]
            for node_id in self.buffer_users.get(buf, []):
                node = self.graph.nodes[node_id]
                for other in self._node_buffers(node):
                    if other not in assignment:
                        assignment[other] = tile
                        queue.append(other)
        return assignment

    def _compute_tile_sizes(self) -> Dict[int, int]:
        sizes: Dict[int, int] = {}
        if not self.tile_types:
            return sizes
        for buf_id, size in self.buf_sizes.items():
            buf_type = self.buf_types.get(buf_id)
            if buf_type in self.tile_types:
                tile = self.tile_of_buf.get(buf_id, buf_id)
                sizes[tile] = sizes.get(tile, 0) + size
        return sizes

    def _select_conv_mode(self) -> Optional[str]:
        if self.strategy != "conv" or not self.tile_sizes:
            return None
        sizes = list(self.tile_sizes.values())
        if not sizes:
            return None
        max_size = max(sizes)
        min_size = min(sizes)
        ratio = max_size / max(1, min_size)
        if ratio > 4:
            return "breadth"
        if ratio < 1.5:
            return "depth"
        return "breadth"

    def _infer_tile(self, node: Node) -> Optional[int]:
        if not self.tile_types:
            return None
        for buf_id in self._node_buffers(node):
            if buf_id in self.tile_of_buf:
                return self.tile_of_buf[buf_id]
            buf_type = self.buf_types.get(buf_id)
            if buf_type in self.tile_types:
                return buf_id
        return None

    def _infer_strategy(self, strategy: str) -> str:
        if strategy != "auto":
            return strategy
        ops = {node.op.upper() for node in self.graph.nodes.values()}
        pipes = {node.pipe.upper() for node in self.graph.nodes.values() if node.pipe}
        if "CONV" in ops:
            return "conv"
        if "MATMUL" in ops and "VECTOR" in pipes:
            return "flashattention"
        if "MATMUL" in ops:
            return "matmul"
        return "generic"
