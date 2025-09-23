"""Linear scan allocator utilities for buffer reuse experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Allocation:
    offset: int
    size: int


class LinearAllocator:
    """Simple first-fit allocator over a 1-D memory space."""

    def __init__(self, capacity: Optional[int] = None) -> None:
        self.capacity = capacity if capacity and capacity > 0 else None
        self.free_segments: List[Tuple[int, int]] = []
        self.allocations: Dict[int, Allocation] = {}
        self.next_offset = 0
        if self.capacity is not None:
            self.free_segments.append((0, self.capacity))

    def allocate(self, key: int, size: int) -> Optional[int]:
        if key in self.allocations:
            return self.allocations[key].offset

        if self.capacity is None:
            offset = self.next_offset
            self.next_offset += size
            self.allocations[key] = Allocation(offset, size)
            return offset

        for idx, (start, length) in enumerate(self.free_segments):
            if length >= size:
                offset = start
                remaining = length - size
                if remaining:
                    self.free_segments[idx] = (start + size, remaining)
                else:
                    self.free_segments.pop(idx)
                self.allocations[key] = Allocation(offset, size)
                return offset
        return None

    def free(self, key: int) -> None:
        allocation = self.allocations.pop(key, None)
        if allocation is None or self.capacity is None:
            return
        self._insert_segment(allocation.offset, allocation.size)

    def _insert_segment(self, start: int, size: int) -> None:
        idx = 0
        while idx < len(self.free_segments) and self.free_segments[idx][0] < start:
            idx += 1
        self.free_segments.insert(idx, (start, size))
        self._merge()

    def _merge(self) -> None:
        merged: List[Tuple[int, int]] = []
        for seg in self.free_segments:
            if not merged:
                merged.append(seg)
                continue
            last_start, last_size = merged[-1]
            current_start, current_size = seg
            if last_start + last_size == current_start:
                merged[-1] = (last_start, last_size + current_size)
            else:
                merged.append(seg)
        self.free_segments = merged


__all__ = ["LinearAllocator", "Allocation"]
