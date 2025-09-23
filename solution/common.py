"""Common utilities shared across scheduling components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Node:
    id: int
    op: str
    pipe: Optional[str] = None
    cycles: int = 1
    bufs: List[int] = field(default_factory=list)
    buf_id: Optional[int] = None
    size: int = 0
    cache_type: Optional[str] = None

    def duration(self) -> int:
        if self.op in {"ALLOC", "FREE"}:
            return 1
        return max(1, self.cycles)


@dataclass
class BufferInterval:
    buf_id: int
    cache_type: Optional[str]
    size: int
    alloc_node: int
    free_node: int
    consumers: List[int]


def compute_memory_profile(sequence: Iterable[Node]) -> Tuple[int, List[int]]:
    usage = 0
    peak = 0
    timeline: List[int] = []
    for node in sequence:
        if node.op == "ALLOC" and node.size:
            usage += node.size
        elif node.op == "FREE" and node.size:
            usage -= node.size
        peak = max(peak, usage)
        timeline.append(usage)
    return peak, timeline


__all__ = ["Node", "BufferInterval", "compute_memory_profile"]
