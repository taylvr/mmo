
from typing import List, Tuple
from core.models import Problem
from core.cost import total_loss, is_feasible


def solve_drop_first(problem: Problem) -> Tuple[float, List[Tuple[str, str]]]:
    seq = ([('DROP', m) for m in problem.to_drop] +
           [('CREATE', m) for m in problem.to_create])
    if not is_feasible(problem, seq):
        return float('inf'), []
    loss, _ = total_loss(problem, seq)
    print(f"[DropFirst] TotalLoss = {loss:.2f}")
    return loss, seq


def solve_create_first(problem: Problem) -> Tuple[float, List[Tuple[str, str]]]:
    seq = ([('CREATE', m) for m in problem.to_create] +
           [('DROP', m) for m in problem.to_drop])
    if not is_feasible(problem, seq):
        return float('inf'), []
    loss, _ = total_loss(problem, seq)
    print(f"[CreateFirst] TotalLoss = {loss:.2f}")
    return loss, seq