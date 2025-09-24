from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .data import Graph, Node


@dataclass
class SpillRecord:
    buf_id: int
    new_offset: str
    op: str  # SPILL_OUT or SPILL_IN


@dataclass
class MemoryPlan:
    extended_nodes: Dict[int, Node]
    schedule: List[int]
    initial_offsets: Dict[int, int]
    spill_records: List[SpillRecord]
    extra_data_movement: int


class MemoryPlanner:
    """Allocate buffers with spill handling under capacity constraints."""

    def __init__(
        self,
        graph: Graph,
        schedule: List[int],
        capacities: Dict[str, int],
    ) -> None:
        self.graph = graph
        self.schedule = schedule
        self.capacities = capacities
        self.next_node_id = max(graph.nodes) + 1 if graph.nodes else 0

        # Buffer bookkeeping
        self.initial_offsets: Dict[int, int] = {}
        self.current_offsets: Dict[int, Optional[int]] = {}
        self.buffer_sizes: Dict[int, int] = {}
        self.buffer_types: Dict[int, str] = {}
        self.buffer_live: Dict[int, bool] = {}
        self.buffer_spilled: Dict[int, bool] = {}

        # Free lists per cache type
        self.free_lists: Dict[str, List[Tuple[int, int]]] = {
            cache_type: [(0, capacity)]
            for cache_type, capacity in capacities.items()
        }

        self.spill_records: List[SpillRecord] = []
        self.extra_data_movement = 0

        # Extended node mapping initially contains the original nodes.
        self.extended_nodes: Dict[int, Node] = dict(graph.nodes)
        self.future_uses = self._compute_future_uses(schedule)
        self.extended_schedule: List[int] = []

    def plan(self) -> MemoryPlan:
        for position, node_id in enumerate(self.schedule):
            node = self.extended_nodes[node_id]

            self._ensure_inputs_available(node)
            if node.is_alloc:
                self._handle_alloc(node)

            self.extended_schedule.append(node_id)
            self._consume_future_use(node, position)

            if node.is_free:
                self._handle_free(node)

        return MemoryPlan(
            extended_nodes=self.extended_nodes,
            schedule=self.extended_schedule,
            initial_offsets=self.initial_offsets,
            spill_records=list(self.spill_records),
            extra_data_movement=self.extra_data_movement,
        )

    # -- Internal helpers -------------------------------------------------

    def _handle_alloc(self, node: Node) -> None:
        buf_id = node.buf_id
        assert buf_id is not None
        cache_type = node.type or "L1"
        size = node.size
        self.buffer_sizes[buf_id] = size
        self.buffer_types[buf_id] = cache_type

        offset = self._allocate_region(cache_type, size, forbidden={buf_id})
        self.current_offsets[buf_id] = offset
        self.buffer_live[buf_id] = True
        self.buffer_spilled[buf_id] = False
        if buf_id not in self.initial_offsets:
            self.initial_offsets[buf_id] = offset

    def _handle_free(self, node: Node) -> None:
        buf_id = node.buf_id
        assert buf_id is not None
        cache_type = self.buffer_types.get(buf_id)
        if cache_type is None:
            return
        offset = self.current_offsets.get(buf_id)
        size = self.buffer_sizes.get(buf_id, 0)
        if offset is not None:
            self._release_region(cache_type, offset, size)
        self.current_offsets[buf_id] = None
        self.buffer_live[buf_id] = False

    def _ensure_inputs_available(self, node: Node) -> None:
        bufs = node.bufs or ()
        needed = [buf_id for buf_id in bufs if self.buffer_spilled.get(buf_id)]
        for buf_id in needed:
            self._perform_spill_in(buf_id)

    def _perform_spill_in(self, buf_id: int) -> None:
        cache_type = self.buffer_types.get(buf_id)
        if cache_type is None:
            return
        size = self.buffer_sizes.get(buf_id, 0)
        offset = self._allocate_region(cache_type, size, forbidden={buf_id})
        self.current_offsets[buf_id] = offset
        self.buffer_spilled[buf_id] = False

        spill_node_id = self._create_spill_node(buf_id, cache_type, size, "SPILL_IN")
        self.extended_schedule.append(spill_node_id)
        self.extra_data_movement += size
        self.spill_records.append(SpillRecord(buf_id=buf_id, new_offset=str(offset), op="SPILL_IN"))

    def _perform_spill_out(self, buf_id: int) -> None:
        cache_type = self.buffer_types.get(buf_id)
        if cache_type is None:
            return
        offset = self.current_offsets.get(buf_id)
        size = self.buffer_sizes.get(buf_id, 0)
        if offset is not None:
            self._release_region(cache_type, offset, size)
        self.current_offsets[buf_id] = None
        self.buffer_spilled[buf_id] = True

        spill_node_id = self._create_spill_node(buf_id, cache_type, size, "SPILL_OUT")
        self.extended_schedule.append(spill_node_id)
        self.extra_data_movement += size
        self.spill_records.append(SpillRecord(buf_id=buf_id, new_offset="DDR", op="SPILL_OUT"))

    def _create_spill_node(self, buf_id: int, cache_type: str, size: int, op: str) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        node = Node(
            id=node_id,
            op=op,
            pipe="MTE",
            cycles=size,
            bufs=(buf_id,),
            buf_id=buf_id,
            size=size,
            type=cache_type,
        )
        self.extended_nodes[node_id] = node
        return node_id

    def _allocate_region(self, cache_type: str, size: int, forbidden: set[int]) -> int:
        free_list = self.free_lists[cache_type]
        slot_index = self._find_first_fit(free_list, size)
        while slot_index is None:
            candidate = self._choose_spill_candidate(cache_type, forbidden)
            if candidate is None:
                raise RuntimeError(
                    f"Insufficient {cache_type} capacity for size {size}"
                )
            self._perform_spill_out(candidate)
            free_list = self.free_lists[cache_type]
            slot_index = self._find_first_fit(free_list, size)
        start, end = free_list.pop(slot_index)
        if end - start > size:
            free_list.insert(slot_index, (start + size, end))
        return start

    def _release_region(self, cache_type: str, start: int, size: int) -> None:
        free_list = self.free_lists[cache_type]
        end = start + size
        free_list.append((start, end))
        free_list.sort()
        merged: List[Tuple[int, int]] = []
        for seg_start, seg_end in free_list:
            if not merged:
                merged.append((seg_start, seg_end))
            else:
                last_start, last_end = merged[-1]
                if seg_start <= last_end:
                    merged[-1] = (last_start, max(last_end, seg_end))
                else:
                    merged.append((seg_start, seg_end))
        self.free_lists[cache_type] = merged

    def _find_first_fit(self, free_list: List[Tuple[int, int]], size: int) -> Optional[int]:
        for idx, (start, end) in enumerate(free_list):
            if end - start >= size:
                return idx
        return None

    def _choose_spill_candidate(self, cache_type: str, forbidden: set[int]) -> Optional[int]:
        active_buffers = [
            buf_id
            for buf_id, live in self.buffer_live.items()
            if live
            and not self.buffer_spilled.get(buf_id)
            and self.buffer_types.get(buf_id) == cache_type
            and buf_id not in forbidden
        ]
        if not active_buffers:
            return None
        return max(active_buffers, key=self._next_use_position)

    def _next_use_position(self, buf_id: int) -> int:
        uses = self.future_uses.get(buf_id)
        if not uses:
            return 10**12
        return uses[0]

    def _consume_future_use(self, node: Node, position: int) -> None:
        for buf_id in node.bufs or ():
            uses = self.future_uses.get(buf_id)
            if uses and uses[0] == position:
                uses.pop(0)
        if node.is_free and node.buf_id is not None:
            uses = self.future_uses.get(node.buf_id)
            if uses and uses[0] == position:
                uses.pop(0)

    def _compute_future_uses(self, schedule: List[int]) -> Dict[int, List[int]]:
        future: Dict[int, List[int]] = {}
        for idx, node_id in enumerate(schedule):
            node = self.graph.nodes[node_id]
            for buf_id in node.bufs or ():
                future.setdefault(buf_id, []).append(idx)
            if node.is_free and node.buf_id is not None:
                future.setdefault(node.buf_id, []).append(idx)
        return future
