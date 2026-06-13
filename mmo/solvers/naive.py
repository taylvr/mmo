from typing import List, Tuple
from core.models import Problem
from core.cost import can_create, is_feasible, potential_speedup, total_loss


def _fallback_interleave(problem: Problem,
                         creates_order: list) -> Tuple[float, List[Tuple[str, str]]]:
    current_mvs = set(problem.s_curr)
    drops_left = set(problem.to_drop)
    seq = []

    for mid in creates_order:
        while not can_create(problem, current_mvs, mid) and drops_left:
            d = max(drops_left, key=lambda m: problem.mvs[m].size)
            seq.append(('DROP', d))
            current_mvs.discard(d)
            drops_left.discard(d)
        seq.append(('CREATE', mid))
        current_mvs.add(mid)

    for d in sorted(drops_left):
        seq.append(('DROP', d))
        current_mvs.discard(d)

    if is_feasible(problem, seq):
        loss, _ = total_loss(problem, seq)
        return loss, seq
    return float('inf'), []

def solve_g1_fastest_build(problem: Problem) -> Tuple[float, List[Tuple[str, str]]]:
    drops = [('DROP', m) for m in problem.to_drop]
    creates_sorted = sorted(problem.to_create,
                            key=lambda m: problem.mvs[m].build_time)
    seq = drops + [('CREATE', m) for m in creates_sorted]

    if not is_feasible(problem, seq):
        return _fallback_interleave(problem, creates_sorted)

    loss, _ = total_loss(problem, seq)
    print(f"[G1-FastestBuild] TotalLoss = {loss:.2f}")
    return loss, seq

def solve_g2_potential_speedup(problem: Problem) -> Tuple[float, List[Tuple[str, str]]]:
    drops = [('DROP', m) for m in problem.to_drop]
    speedups = {m: potential_speedup(problem, m) for m in problem.to_create}
    creates_sorted = sorted(problem.to_create, key=lambda m: -speedups[m])
    seq = drops + [('CREATE', m) for m in creates_sorted]

    if not is_feasible(problem, seq):
        return _fallback_interleave(problem, creates_sorted)

    loss, _ = total_loss(problem, seq)
    print(f"[G2-PotentialSpeedup] TotalLoss = {loss:.2f}")
    return loss, seq

def solve_g3_smallest_size(problem: Problem) -> Tuple[float, List[Tuple[str, str]]]:
    drops = [('DROP', m) for m in problem.to_drop]
    creates_sorted = sorted(problem.to_create,
                            key=lambda m: problem.mvs[m].size)
    seq = drops + [('CREATE', m) for m in creates_sorted]

    if not is_feasible(problem, seq):
        return _fallback_interleave(problem, creates_sorted)

    loss, _ = total_loss(problem, seq)
    print(f"[G3-SmallestSize] TotalLoss = {loss:.2f}")
    return loss, seq
