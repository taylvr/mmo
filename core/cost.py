

from typing import Dict, List, Set, Tuple

from .models import MV, Problem, Query

Operation = Tuple[str, str]


def query_cost(query: Query, active_mvs: Set[str]) -> float:
    
    return min(
        [query.original_cost]
        + [cost for mv_id, cost in query.mv_costs.items() if mv_id in active_mvs]
    )


def state_regret(
    queries: Dict[str, Query],
    active_mvs: Set[str],
    target_mvs: Set[str],
) -> float:
    
    return sum(
        query.frequency
        * max(
            0.0,
            query_cost(query, active_mvs) - query_cost(query, target_mvs),
        )
        for query in queries.values()
    )


def storage_used(mvs: Dict[str, MV], active_mvs: Set[str]) -> float:
    
    return sum(mvs[mv_id].size for mv_id in active_mvs)


def operation_duration(action: str, mv_id: str, mvs: Dict[str, MV]) -> float:
    
    return 0.0 if action == "DROP" else mvs[mv_id].build_time


def total_loss(
    problem: Problem,
    op_sequence: List[Operation],
) -> Tuple[float, List[dict]]:
    
    current_mvs = set(problem.s_curr)
    total = 0.0
    trace = []

    for step_idx, (action, mv_id) in enumerate(op_sequence):
        duration = operation_duration(action, mv_id, problem.mvs)
        regret = state_regret(problem.queries, current_mvs, problem.s_target)
        step_loss = regret * duration
        wait_state = set(current_mvs)
        total += step_loss

        if action == "DROP":
            current_mvs.discard(mv_id)
        elif action == "CREATE":
            current_mvs.add(mv_id)
        else:
            raise ValueError(f"Unknown migration action: {action}")

        trace.append(
            {
                "step": step_idx + 1,
                "action": f"{action}({mv_id})",
                "wait_state": wait_state,
                "after_state": set(current_mvs),
                "duration": duration,
                "regret_rate": regret,
                "loss": step_loss,
                "cumulative_loss": total,
            }
        )

    return total, trace


def is_feasible(problem: Problem, op_sequence: List[Operation]) -> bool:
    
    current_mvs = set(problem.s_curr)
    limit = problem.budget + problem.delta

    for action, mv_id in op_sequence:
        if action == "CREATE":
            current_mvs.add(mv_id)
        elif action == "DROP":
            current_mvs.discard(mv_id)
        else:
            return False
        if storage_used(problem.mvs, current_mvs) > limit + 1e-9:
            return False
    return True


def marginal_regret_reduction(
    problem: Problem,
    current_mvs: Set[str],
    mv_id: str,
    action: str,
) -> float:
    
    regret_before = state_regret(problem.queries, current_mvs, problem.s_target)
    if action == "CREATE":
        next_mvs = current_mvs | {mv_id}
    elif action == "DROP":
        next_mvs = current_mvs - {mv_id}
    else:
        raise ValueError(f"Unknown migration action: {action}")
    regret_after = state_regret(problem.queries, next_mvs, problem.s_target)
    return regret_before - regret_after


def potential_speedup(problem: Problem, mv_id: str) -> float:
    
    return sum(
        query.frequency
        * max(0.0, query.original_cost - query.mv_costs.get(mv_id, query.original_cost))
        for query in problem.queries.values()
    )


def can_create(problem: Problem, current_mvs: Set[str], mv_id: str) -> bool:
    
    limit = problem.budget + problem.delta
    return storage_used(problem.mvs, current_mvs | {mv_id}) <= limit + 1e-9
