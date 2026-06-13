# MVMO

This repository contains the core implementation of MVMO. It plans an online transition from a current
materialized-view configuration to a target configuration while minimizing
cumulative time-weighted query regret under a storage constraint.

- **StructDP**: the exact structure-aware dynamic programming algorithm.
- **MCGreedy**: the marginal-contribution greedy algorithm.
- **RSGreedy**: the rolling-simulation greedy algorithm.
- The MVMO data model and cost functions.


## Cost Model

A migration plan interleaves:

- `DROP(v)` for views that occur only in the current configuration.
- `CREATE(v)` for views that occur only in the target configuration.

While a view is being built, the active configuration
incurs workload regret relative to the target configuration:

```text
total loss = sum(state regret before CREATE * build duration)
```

Every intermediate configuration must satisfy the storage limit
`budget + delta`.

## Environment

The project uses the Python 3.11.

Run selected algorithms:

```bash
python run.py --solver MCGreedy RSGreedy
python run.py --solver StructDP
```

Run on TPC-DS:

```bash
python run.py --dataset tpcds
```

Print migration operations:

```bash
python run.py --solver StructDP --trace
```

## Configuration

The default settings are in `conf.ini`:

```ini
[problem]
dataset = tpch
csv_path = data/tpch.csv
n_curr = 8
n_target = 8
n_common = 4
budget = 0
delta = 8.0
seed = 42

[solver]
max_ops = 24
lookahead = 2
```

Set `budget = 0` to derive a valid storage budget from the selected current and
target configurations.

The CSV file must contain:

```text
new_q,new_v,q_exec_time_s,r_exec_time_s,mv_size_MB,v_exec_time_s
```

## Python API

```python
from core import load_problem
from solvers import solve_mcgreedy, solve_rsgreedy, solve_structdp

problem = load_problem(
    "data/tpch.csv",
    dataset="tpch",
    n_curr=8,
    n_target=8,
    n_common=4,
    seed=42,
)

loss, plan = solve_mcgreedy(problem)
```

`solve_structdp` additionally returns optimization statistics:

```python
loss, plan, stats = solve_structdp(problem, max_ops=24)
```
