from dataclasses import dataclass

from core.cost import state_regret
from core.models import Problem


@dataclass
class OptimizationStats:
    total_possible_states: int = 0
    total_states_explored: int = 0
    harmless_drops: int = 0
    storage_prunes: int = 0
    regret_cache_size: int = 0
    memoized_states: int = 0

    def summary(self) -> str:
        ratio = 0.0
        if self.total_possible_states:
            ratio = (
                self.total_states_explored
                / self.total_possible_states
                * 100
            )
        return "\n".join(
            [
                "=" * 48,
                "StructDP optimization statistics",
                "=" * 48,
                f"State space: {self.total_possible_states}",
                f"States explored: {self.total_states_explored}",
                f"Explored ratio: {ratio:.4f}%",
                f"Harmless drops: {self.harmless_drops}",
                f"Storage prunes: {self.storage_prunes}",
                f"Regret cache size: {self.regret_cache_size}",
                f"Memoized states: {self.memoized_states}",
                "=" * 48,
            ]
        )


def solve_structdp(
    problem: Problem,
    max_ops: int = 24,
) -> tuple[float, list[tuple[str, str]], OptimizationStats]:
    drops = [("DROP", mv_id) for mv_id in problem.to_drop]
    creates = [("CREATE", mv_id) for mv_id in problem.to_create]
    operations = drops + creates
    operation_count = len(operations)
    stats = OptimizationStats(total_possible_states=1 << operation_count)

    if operation_count > max_ops:
        return float("inf"), [], stats

    mv_ids = sorted(problem.s_curr | problem.s_target)
    mv_index = {mv_id: index for index, mv_id in enumerate(mv_ids)}
    initial_mask = _set_to_mask(problem.s_curr, mv_index)
    target_mask = _set_to_mask(problem.s_target, mv_index)
    full_operation_mask = (1 << operation_count) - 1
    storage_limit = problem.budget + problem.delta

    operation_bits = [mv_index[mv_id] for _, mv_id in operations]
    operation_is_create = [
        action == "CREATE" for action, _ in operations
    ]
    operation_durations = [
        problem.mvs[mv_id].build_time if action == "CREATE" else 0.0
        for action, mv_id in operations
    ]
    operation_sizes = [
        problem.mvs[mv_id].size if action == "CREATE"
        else -problem.mvs[mv_id].size
        for action, mv_id in operations
    ]

    regret_cache: dict[int, float] = {}
    memo: dict[int, tuple[float, list[int]]] = {}

    def mask_to_set(mask: int) -> set[str]:
        return {
            mv_id
            for mv_id, index in mv_index.items()
            if mask & (1 << index)
        }

    def regret(mask: int) -> float:
        if mask not in regret_cache:
            regret_cache[mask] = state_regret(
                problem.queries,
                mask_to_set(mask),
                mask_to_set(target_mask),
            )
        return regret_cache[mask]

    def close_harmless_drops(
        operation_mask: int,
        active_mask: int,
        storage: float,
    ) -> tuple[int, int, float, list[int]]:
        forced: list[int] = []
        changed = True
        while changed:
            changed = False
            current_regret = regret(active_mask)
            for index in range(len(drops)):
                if operation_mask & (1 << index):
                    continue
                bit = operation_bits[index]
                next_active = active_mask & ~(1 << bit)
                if regret(next_active) <= current_regret + 1e-12:
                    operation_mask |= 1 << index
                    active_mask = next_active
                    storage += operation_sizes[index]
                    forced.append(index)
                    stats.harmless_drops += 1
                    changed = True
                    break
        return operation_mask, active_mask, storage, forced

    def search(
        operation_mask: int,
        active_mask: int,
        storage: float,
    ) -> tuple[float, list[int]]:
        operation_mask, active_mask, storage, forced = (
            close_harmless_drops(
                operation_mask,
                active_mask,
                storage,
            )
        )

        if operation_mask == full_operation_mask:
            return 0.0, forced
        if operation_mask in memo:
            cached_loss, cached_path = memo[operation_mask]
            return cached_loss, forced + cached_path

        stats.total_states_explored += 1
        current_regret = regret(active_mask)
        best_loss = float("inf")
        best_path: list[int] = []

        for index in range(operation_count):
            if operation_mask & (1 << index):
                continue

            next_storage = storage + operation_sizes[index]
            if (
                operation_is_create[index]
                and next_storage > storage_limit + 1e-9
            ):
                stats.storage_prunes += 1
                continue

            bit = operation_bits[index]
            if operation_is_create[index]:
                next_active = active_mask | (1 << bit)
            else:
                next_active = active_mask & ~(1 << bit)

            step_loss = (
                current_regret * operation_durations[index]
            )
            future_loss, future_path = search(
                operation_mask | (1 << index),
                next_active,
                next_storage,
            )
            total_loss = step_loss + future_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_path = [index] + future_path

        memo[operation_mask] = (best_loss, best_path)
        return best_loss, forced + best_path

    initial_storage = sum(
        problem.mvs[mv_id].size for mv_id in problem.s_curr
    )
    loss, path_indices = search(0, initial_mask, initial_storage)
    stats.regret_cache_size = len(regret_cache)
    stats.memoized_states = len(memo)
    path = [operations[index] for index in path_indices]
    return loss, path, stats


def _set_to_mask(values: set[str], index: dict[str, int]) -> int:
    mask = 0
    for value in values:
        mask |= 1 << index[value]
    return mask
