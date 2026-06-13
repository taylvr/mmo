# ============== solvers/dp.py ==============
from typing import List, Tuple
from core.models import Problem


def solve_dp(problem: Problem, max_ops: int = 200) -> Tuple[float, List[Tuple[str, str]]]:

    drops = [('DROP', m) for m in problem.to_drop]
    creates = [('CREATE', m) for m in problem.to_create]
    all_ops = drops + creates
    k = len(all_ops)

    if k > max_ops:
        return float('inf'), []

    limit = problem.budget + problem.delta

    all_mv_ids = sorted(set(list(problem.s_curr) + list(problem.s_target)))
    mv_idx = {mid: i for i, mid in enumerate(all_mv_ids)}

    def mvset_to_mask(s):
        mask = 0
        for mid in s:
            mask |= (1 << mv_idx[mid])
        return mask

    init_mv = mvset_to_mask(problem.s_curr)
    target_mv = mvset_to_mask(problem.s_target)
    mv_sizes_by_bit = {mv_idx[mid]: problem.mvs[mid].size for mid in all_mv_ids}

    def mask_storage(mv_mask):
        return sum(sz for bit_i, sz in mv_sizes_by_bit.items()
                   if mv_mask & (1 << bit_i))

    q_info = []
    target_costs = []
    for q in problem.queries.values():
        mv_list = []
        for mid, cost in q.mv_costs.items():
            if mid in mv_idx:
                mv_list.append((mv_idx[mid], cost))
        q_info.append((q.original_cost, q.frequency, mv_list))
        best_t = q.original_cost
        for idx, cost in mv_list:
            if target_mv & (1 << idx) and cost < best_t:
                best_t = cost
        target_costs.append(best_t)

    n_queries = len(q_info)

    regret_cache = {}

    def compute_regret(mv_mask):
        if mv_mask in regret_cache:
            return regret_cache[mv_mask]
        total = 0.0
        for q_idx in range(n_queries):
            orig, freq, mv_list = q_info[q_idx]
            best_now = orig
            for idx, cost in mv_list:
                if mv_mask & (1 << idx) and cost < best_now:
                    best_now = cost
            loss = max(0.0, best_now - target_costs[q_idx])
            total += freq * loss
        regret_cache[mv_mask] = total
        return total

    op_mv_bit = []
    op_is_create = []
    op_duration = []
    op_size_delta = []
    for action, mid in all_ops:
        bit_i = mv_idx[mid]
        is_create = (action == 'CREATE')
        op_mv_bit.append(bit_i)
        op_is_create.append(is_create)
        op_duration.append(0.0 if not is_create else problem.mvs[mid].build_time)
        op_size_delta.append(problem.mvs[mid].size if is_create
                             else -problem.mvs[mid].size)

    full = (1 << k) - 1
    INF = float('inf')
    dp = [INF] * (full + 1)
    dp_mv = [0] * (full + 1)
    dp_stor = [0.0] * (full + 1)
    parent = [-1] * (full + 1)
    par_op = [-1] * (full + 1)

    dp[0] = 0.0
    dp_mv[0] = init_mv
    dp_stor[0] = mask_storage(init_mv)

    for op_mask in range(full):
        if dp[op_mask] >= INF:
            continue
        curr_mv = dp_mv[op_mask]
        curr_stor = dp_stor[op_mask]
        curr_regret = None  #

        for i in range(k):
            if op_mask & (1 << i):
                continue

            bit_i = op_mv_bit[i]
            is_create = op_is_create[i]

            if is_create:
                new_mv = curr_mv | (1 << bit_i)
                new_stor = curr_stor + op_size_delta[i]
                if new_stor > limit + 1e-9:
                    continue
                if curr_regret is None:
                    curr_regret = compute_regret(curr_mv)
                wait_loss = curr_regret * op_duration[i]
            else:
                wait_loss = 0.0
                new_mv = curr_mv & ~(1 << bit_i)
                new_stor = curr_stor + op_size_delta[i]

            new_op_mask = op_mask | (1 << i)
            new_total = dp[op_mask] + wait_loss

            if new_total < dp[new_op_mask]:
                dp[new_op_mask] = new_total
                dp_mv[new_op_mask] = new_mv
                dp_stor[new_op_mask] = new_stor
                parent[new_op_mask] = op_mask
                par_op[new_op_mask] = i

    if dp[full] >= INF:
        return INF, []

    path = []
    mask = full
    while mask != 0:
        path.append(all_ops[par_op[mask]])
        mask = parent[mask]
    path.reverse()

    print(f"[DP] TotalLoss = {dp[full]:.2f}")
    return dp[full], path