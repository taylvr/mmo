
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from core import (
    DEFAULT_CONFIG,
    DEFAULT_DATASETS,
    load_problem,
    load_run_config,
)
from solvers import SOLVER_ORDER, run_solver
from utils import print_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVMO"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", choices=("tpch", "tpcds"))
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--max-ops", type=int)
    parser.add_argument(
        "--solver",
        nargs="+",
        choices=SOLVER_ORDER,
        help="Algorithms to run (default: all three)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print each selected migration plan",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print StructDP optimization statistics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    overrides = {}

    if args.dataset is not None:
        overrides["dataset"] = args.dataset
        if args.csv_path is None:
            overrides["csv_path"] = DEFAULT_DATASETS[args.dataset]
    if args.csv_path is not None:
        overrides["csv_path"] = args.csv_path.expanduser().resolve()
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.lookahead is not None:
        overrides["lookahead"] = args.lookahead
    if args.max_ops is not None:
        overrides["max_ops"] = args.max_ops
    config = replace(config, **overrides)

    problem = load_problem(
        config.csv_path,
        dataset=config.dataset,
        n_curr=config.n_curr,
        n_target=config.n_target,
        n_common=config.n_common,
        budget=config.budget,
        delta=config.delta,
        seed=config.seed,
    )

    selected = args.solver or SOLVER_ORDER
    operation_count = len(problem.to_drop) + len(problem.to_create)
    print(
        f"Dataset: {config.dataset.upper()} | "
        f"queries={len(problem.queries)} | "
        f"operations={operation_count} | "
        f"budget={problem.budget + problem.delta:.2f} MB"
    )

    results = {}
    for name in selected:
        started = time.perf_counter()
        loss, sequence, stats = run_solver(
            problem,
            name,
            lookahead=config.lookahead,
            max_ops=config.max_ops,
        )
        results[name] = {
            "loss": loss,
            "sequence": sequence,
            "seconds": time.perf_counter() - started,
            "stats": stats,
        }

    print(f"\n{'Algorithm':<12} {'Loss':>16} {'Time':>12}")
    print("-" * 42)
    for name in selected:
        result = results[name]
        loss = result["loss"]
        loss_text = "skipped" if loss == float("inf") else f"{loss:.4f}"
        print(f"{name:<12} {loss_text:>16} {result['seconds']:>11.3f}s")

    if (
        args.stats
        and "StructDP" in results
        and results["StructDP"]["stats"] is not None
    ):
        print(f"\n{results['StructDP']['stats'].summary()}")

    if args.trace or config.print_trace:
        for name in selected:
            sequence = results[name]["sequence"]
            if sequence:
                print_trace(problem, sequence, name)


if __name__ == "__main__":
    main()
