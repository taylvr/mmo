

from __future__ import annotations

from core.cost import can_create, operation_duration, state_regret
from core.models import Problem

Operation = tuple[str, str]


def solve_mcgreedy(problem: Problem) -> tuple[float, list[Operation]]:
    
    drops_left = set(problem.to_drop)
    creates_left = set(problem.to_create)
    current_mvs = set(problem.s_curr)
    sequence: list[Operation] = []
    total_loss = 0.0

    while drops_left or creates_left:
        current_regret = state_regret(
            problem.queries,
            current_mvs,
            problem.s_target,
        )
        remaining_build_time = sum(
            problem.mvs[mv_id].build_time for mv_id in creates_left
        )
        best_operation: Operation | None = None
        best_score = float("inf")

        for mv_id in sorted(drops_left):
            next_regret = state_regret(
                problem.queries,
                current_mvs - {mv_id},
                problem.s_target,
            )
            score = (next_regret - current_regret) * remaining_build_time
            if score < best_score:
                best_score = score
                best_operation = ("DROP", mv_id)

        for mv_id in sorted(creates_left):
            if not can_create(problem, current_mvs, mv_id):
                continue
            duration = problem.mvs[mv_id].build_time
            next_regret = state_regret(
                problem.queries,
                current_mvs | {mv_id},
                problem.s_target,
            )
            future_benefit = (
                current_regret - next_regret
            ) * (remaining_build_time - duration)
            score = current_regret * duration - future_benefit
            if score < best_score:
                best_score = score
                best_operation = ("CREATE", mv_id)

        if best_operation is None:
            if not drops_left:
                raise ValueError("No feasible migration operation remains")
            mv_id = max(drops_left, key=lambda item: problem.mvs[item].size)
            best_operation = ("DROP", mv_id)

        action, mv_id = best_operation
        duration = operation_duration(action, mv_id, problem.mvs)
        total_loss += current_regret * duration
        _apply_operation(current_mvs, drops_left, creates_left, best_operation)
        sequence.append(best_operation)

    return total_loss, sequence


def solve_rsgreedy(
    problem: Problem,
    lookahead: int = 2,
) -> tuple[float, list[Operation]]:
    
    if lookahead < 1:
        raise ValueError("lookahead must be at least 1")

    drops_left = set(problem.to_drop)
    creates_left = set(problem.to_create)
    current_mvs = set(problem.s_curr)
    sequence: list[Operation] = []
    total_loss = 0.0

    def estimate_remaining(mvs: set[str], creates: set[str]) -> float:
        regret = state_regret(problem.queries, mvs, problem.s_target)
        build_time = sum(problem.mvs[mv_id].build_time for mv_id in creates)
        return regret * build_time * 0.5

    def search(
        mvs: set[str],
        drops: set[str],
        creates: set[str],
        depth: int,
    ) -> tuple[float, list[Operation]]:
        if depth == 0 or (not drops and not creates):
            return estimate_remaining(mvs, creates), []

        candidates = [("DROP", mv_id) for mv_id in sorted(drops)]
        candidates.extend(
            ("CREATE", mv_id)
            for mv_id in sorted(creates)
            if can_create(problem, mvs, mv_id)
        )
        if not candidates:
            return float("inf"), []

        best_loss = float("inf")
        best_sequence: list[Operation] = []
        current_regret = state_regret(problem.queries, mvs, problem.s_target)

        for operation in candidates:
            action, mv_id = operation
            step_loss = (
                current_regret * operation_duration(action, mv_id, problem.mvs)
            )
            next_mvs = set(mvs)
            next_drops = set(drops)
            next_creates = set(creates)
            _apply_operation(
                next_mvs,
                next_drops,
                next_creates,
                operation,
            )
            future_loss, future_sequence = search(
                next_mvs,
                next_drops,
                next_creates,
                depth - 1,
            )
            candidate_loss = step_loss + future_loss
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_sequence = [operation, *future_sequence]

        return best_loss, best_sequence

    while drops_left or creates_left:
        depth = min(lookahead, len(drops_left) + len(creates_left))
        _, simulated_sequence = search(
            current_mvs,
            drops_left,
            creates_left,
            depth,
        )
        if not simulated_sequence:
            raise ValueError("No feasible migration plan exists")

        operation = simulated_sequence[0]
        action, mv_id = operation
        regret = state_regret(problem.queries, current_mvs, problem.s_target)
        total_loss += regret * operation_duration(action, mv_id, problem.mvs)
        _apply_operation(current_mvs, drops_left, creates_left, operation)
        sequence.append(operation)

    return total_loss, sequence


def _apply_operation(
    current_mvs: set[str],
    drops_left: set[str],
    creates_left: set[str],
    operation: Operation,
) -> None:
    action, mv_id = operation
    if action == "DROP":
        current_mvs.discard(mv_id)
        drops_left.discard(mv_id)
    else:
        current_mvs.add(mv_id)
        creates_left.discard(mv_id)
