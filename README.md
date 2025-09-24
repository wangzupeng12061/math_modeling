# math_modeling

Utility scripts and heuristics for the "通用神经网络处理器下的核内调度问题" modeling task. The code under `solution/` implements the three stages outlined in `analyze.md`:

- problem 1: generate a memory-pressure-aware topological schedule
- problem 2: allocate multi-level cache space with spill handling and DDR traffic accounting
- problem 3: refine the schedule with a HEFT-style pipeline heuristic while respecting spill tolerances

## Repository layout

- `analyze.md` / `backgroud.md` – problem statement, assumptions, and algorithm design notes
- `datasets/` – sample DAGs in JSON format used for evaluation
- `solution/` – Python package containing graph parsing, schedulers, and CLI entrypoints
- `outputs/` – default directory for generated schedules, memory maps, spill logs, and summaries

## Requirements

- Python 3.9+ (tested with the system `python3`)
- No third-party dependencies are required; only the standard library is used

## Usage

Each stage can be run through the CLI wrapper. Replace `<graph.json>` with any file from `datasets/` or your own DAG in the same schema.

### Problem 1: schedule generation

```
python3 -m solution.cli schedule datasets/Matmul_Case0.json --output-dir outputs/p1
```

Outputs `<task>_schedule.txt` containing the node order (one node ID per line) and prints the peak resident memory estimate.

### Problem 2: memory planning with spills

```
python3 -m solution.cli memory datasets/Matmul_Case0.json --output-dir outputs/p2
```

Produces three files:

- `<task>_schedule.txt` – original nodes plus inserted `SPILL_*` nodes
- `<task>_memory.txt` – initial physical offsets per buffer (`BufId:Offset`)
- `<task>_spill.txt` – spill history (`BufId:NewOffset`, with `DDR` denoting eviction)

The planner uses the default capacities (L1=4096, UB=1024, L0A=256, L0B=256, L0C=512). Adjust them by editing `DEFAULT_CAPACITIES` in `solution/cli.py` if needed.

### Problem 3: runtime-oriented optimization

```
python3 -m solution.cli optimize datasets/Matmul_Case0.json --output-dir outputs/p3
```

This command compares the baseline schedule against a HEFT-style pipeline order. The variant whose extra data movement stays within the configurable tolerance (default 5%) is retained. Generated artifacts include the schedule/memory/spill trio plus `<task>_summary.txt` summarising baseline and chosen metrics.

## Extending the toolkit

- Implement alternative schedulers by subclassing or replacing `GreedyMemoryAwareScheduler` in `solution/scheduler.py`.
- Experiment with different spill heuristics in `solution/memory.py` (e.g. size-aware eviction or buffer splitting).
- Enhance runtime optimization by extending `solution/runtime.py` with multi-resource lookahead or local swaps.

## Results and reporting

Raw outputs for the provided datasets are kept under `outputs/`. These files can be directly embedded into the accompanying paper or further processed to compute aggregate indicators such as maximal residency, total spill traffic, and makespan.

## Optimisation log

- **Baseline greedy scheduler** – initial `GreedyMemoryAwareScheduler` ordered ready nodes by release potential and bottom-level priority, yielding Matmul peaks of 131 kB/1.05 MB while keeping makespan low.
- **Memory planner integration** – inserted `MemoryPlanner` with spill-aware first-fit allocation, recording DDR traffic and extending schedules with `SPILL_*` nodes to satisfy Problem 2.
- **Runtime (Problem 3) tooling** – added HEFT-style pipeline scheduler and CLI flags so the baseline/optimised makespans and extra movement can be compared automatically.
- **Conv heuristics** – introduced L1-tile splitting, depth/breadth mode detection, and L1-weighted scoring; Case0 MaxVstay fell to ~38 kB (from ~63 kB) while keeping Case1 stable.
- **FlashAttention tiling** – rebuilt UB-only tile buckets with active-tile limits and higher UB/L0 free bonuses; with `--capacity-slack 1` peaks sit near 42 kB, and dropping capacity checks (`--capacity-slack 100`) confirms the row-based grouping keeps peaks within ~12 kB/75 kB.
- **Capacity-sensitivity study** – reran Problem 1 with large `--capacity-slack` to show how L1/UB limits influence residency; traces are stored in `outputs/traces_unbounded/` for comparison.
