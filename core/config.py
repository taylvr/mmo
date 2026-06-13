

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf.ini"
DEFAULT_DATASETS = {
    "tpch": PROJECT_ROOT / "data/tpch.csv",
    "tpcds": PROJECT_ROOT / "data/tpcds.csv",
}


@dataclass
class RunConfig:
    dataset: str = "tpch"
    csv_path: Path | None = None
    n_curr: int = 10
    n_target: int = 10
    n_common: int = 5
    budget: float | None = None
    delta: float = 8.0
    seed: int = 42
    max_ops: int = 2
    lookahead: int = 2
    print_trace: bool = False


def resolve_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_run_config(path: str | Path = DEFAULT_CONFIG) -> RunConfig:
    config_path = Path(path).expanduser().resolve()
    parser = configparser.ConfigParser()
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    dataset = parser.get("problem", "dataset", fallback="tpch").lower()
    csv_path = DEFAULT_DATASETS.get(dataset)
    configured_csv = parser.get("problem", "csv_path", fallback="").strip()
    if configured_csv:
        csv_path = resolve_path(configured_csv, config_path)

    budget_value = parser.getfloat("problem", "budget", fallback=0.0)
    return RunConfig(
        dataset=dataset,
        csv_path=csv_path,
        n_curr=parser.getint("problem", "n_curr", fallback=10),
        n_target=parser.getint("problem", "n_target", fallback=10),
        n_common=parser.getint("problem", "n_common", fallback=5),
        budget=budget_value or None,
        delta=parser.getfloat("problem", "delta", fallback=8.0),
        seed=parser.getint("problem", "seed", fallback=42),
        max_ops=parser.getint("solver", "max_ops", fallback=26),
        lookahead=parser.getint("solver", "lookahead", fallback=2),
        print_trace=parser.getboolean("display", "print_trace", fallback=False),
    )
