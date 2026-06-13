

from core.models import Problem

from .heuristics import solve_mcgreedy, solve_rsgreedy
from .structdp import solve_structdp

SOLVER_ORDER = ["MCGreedy", "RSGreedy", "StructDP"]


def run_solver(
    problem: Problem,
    name: str,
    *,
    lookahead: int = 2,
    max_ops: int = 24,
):
    
    if name == "MCGreedy":
        loss, sequence = solve_mcgreedy(problem)
        return loss, sequence, None
    if name == "RSGreedy":
        loss, sequence = solve_rsgreedy(
            problem,
            lookahead=lookahead,
        )
        return loss, sequence, None
    if name == "StructDP":
        return solve_structdp(problem, max_ops=max_ops)
    raise ValueError(f"Unknown solver: {name}")
