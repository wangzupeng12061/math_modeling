from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .data import Graph, load_graph
from .memory import MemoryPlanner, MemoryPlan
from .runtime import HeftPipelineScheduler, simulate_timeline
from .scheduler import GreedyMemoryAwareScheduler

DEFAULT_CAPACITIES = {
    "L1": 4096,
    "UB": 1024,
    "L0A": 256,
    "L0B": 256,
    "L0C": 512,
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NPU kernel scheduling toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schedule_parser = subparsers.add_parser(
        "schedule", help="Compute a memory-aware schedule (problem 1)."
    )
    _add_common_graph_args(schedule_parser)
    schedule_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/p1"),
        help="Directory to place schedule file",
    )

    memory_parser = subparsers.add_parser(
        "memory", help="Allocate buffers with spills (problem 2)."
    )
    _add_common_graph_args(memory_parser)
    memory_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/p2"),
        help="Directory to place memory planning outputs",
    )

    optimize_parser = subparsers.add_parser(
        "optimize", help="Pipeline optimization with runtime focus (problem 3)."
    )
    _add_common_graph_args(optimize_parser)
    optimize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/p3"),
        help="Directory to place optimized outputs",
    )
    optimize_parser.add_argument(
        "--movement-tolerance",
        type=float,
        default=0.05,
        help="Maximum relative increase in extra data movement allowed",
    )

    args = parser.parse_args(argv)

    if args.command == "schedule":
        return command_schedule(args.graph, args.output_dir, args.name)
    if args.command == "memory":
        return command_memory(args.graph, args.output_dir, args.name)
    if args.command == "optimize":
        return command_optimize(args.graph, args.output_dir, args.name, args.movement_tolerance)

    parser.error(f"Unknown command {args.command}")
    return 1


def _add_common_graph_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("graph", type=Path, help="Path to graph JSON file")
    subparser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional task name, defaults to graph stem",
    )


def command_schedule(graph_path: Path, output_dir: Path, name: Optional[str]) -> int:
    graph = _load_graph_or_exit(graph_path)
    scheduler = GreedyMemoryAwareScheduler(graph)
    result = scheduler.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = name or graph_path.stem
    schedule_path = output_dir / f"{task_name}_schedule.txt"
    with schedule_path.open("w", encoding="utf-8") as fh:
        for nid in result.order:
            fh.write(f"{nid}\n")

    print(
        f"[problem1] Schedule written to {schedule_path} (max resident: {result.max_resident})"
    )
    return 0


def command_memory(graph_path: Path, output_dir: Path, name: Optional[str]) -> int:
    graph = _load_graph_or_exit(graph_path)
    scheduler = GreedyMemoryAwareScheduler(graph)
    schedule_result = scheduler.run()

    planner = MemoryPlanner(graph, schedule_result.order, DEFAULT_CAPACITIES)
    plan = planner.plan()

    task_name = name or graph_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_memory_outputs(task_name, output_dir, plan)

    print(
        "[problem2] Outputs written to"
        f" schedule={output_dir / (task_name + '_schedule.txt')},"
        f" memory={output_dir / (task_name + '_memory.txt')},"
        f" spill={output_dir / (task_name + '_spill.txt')};"
        f" extra data movement={plan.extra_data_movement}"
    )
    return 0


def command_optimize(
    graph_path: Path,
    output_dir: Path,
    name: Optional[str],
    tolerance: float,
) -> int:
    graph = _load_graph_or_exit(graph_path)
    task_name = name or graph_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (problem 1) schedule metrics.
    baseline_scheduler = GreedyMemoryAwareScheduler(graph)
    baseline_schedule = baseline_scheduler.run()
    baseline_plan = MemoryPlanner(graph, baseline_schedule.order, DEFAULT_CAPACITIES).plan()
    baseline_timeline = simulate_timeline(graph, baseline_schedule.order)

    # Runtime-optimized schedule.
    heft_scheduler = HeftPipelineScheduler(graph)
    optimized = heft_scheduler.run()
    optimized_plan = MemoryPlanner(graph, optimized.order, DEFAULT_CAPACITIES).plan()

    movement_limit = baseline_plan.extra_data_movement * (1.0 + tolerance)
    if optimized_plan.extra_data_movement > movement_limit:
        chosen_order = baseline_schedule.order
        chosen_plan = baseline_plan
        chosen_timeline = baseline_timeline
        strategy = "baseline"
    else:
        chosen_order = optimized.order
        chosen_plan = optimized_plan
        chosen_timeline = optimized
        strategy = "optimized"

    _write_memory_outputs(task_name, output_dir, chosen_plan)

    summary_path = output_dir / f"{task_name}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "baseline_max_resident={} baseline_extra_movement={} baseline_makespan={}\n".format(
                baseline_schedule.max_resident,
                baseline_plan.extra_data_movement,
                baseline_timeline.makespan,
            )
        )
        fh.write(
            "chosen_strategy={} chosen_extra_movement={} chosen_makespan={}\n".format(
                strategy,
                chosen_plan.extra_data_movement,
                chosen_timeline.makespan,
            )
        )

    print(
        "[problem3] Strategy={} schedule written to {}; extra movement={}; makespan={}".format(
            strategy,
            output_dir / (task_name + '_schedule.txt'),
            chosen_plan.extra_data_movement,
            chosen_timeline.makespan,
        )
    )
    return 0


def _write_memory_outputs(task_name: str, output_dir: Path, plan: MemoryPlan) -> None:
    schedule_path = output_dir / f"{task_name}_schedule.txt"
    with schedule_path.open("w", encoding="utf-8") as fh:
        for nid in plan.schedule:
            fh.write(f"{nid}\n")

    memory_path = output_dir / f"{task_name}_memory.txt"
    with memory_path.open("w", encoding="utf-8") as fh:
        for buf_id in sorted(plan.initial_offsets):
            fh.write(f"{buf_id}:{plan.initial_offsets[buf_id]}\n")

    spill_path = output_dir / f"{task_name}_spill.txt"
    with spill_path.open("w", encoding="utf-8") as fh:
        for record in plan.spill_records:
            fh.write(f"{record.buf_id}:{record.new_offset}\n")


def _load_graph_or_exit(graph_path: Path) -> Graph:
    if not graph_path.exists():
        print(f"Graph file not found: {graph_path}", file=sys.stderr)
        raise SystemExit(2)
    return load_graph(graph_path)


if __name__ == "__main__":
    raise SystemExit(main())
