

from core.cost import total_loss
from core.models import Problem


def print_trace(
    problem: Problem,
    sequence: list[tuple[str, str]],
    label: str,
) -> None:
    
    loss, trace = total_loss(problem, sequence)
    print(f"\n{label} migration plan (loss={loss:.4f})")
    print(
        f"{'Step':>4}  {'Operation':<18}  {'Duration':>10}  "
        f"{'Regret':>12}  {'Step loss':>12}"
    )
    print("-" * 68)
    for entry in trace:
        print(
            f"{entry['step']:>4}  {entry['action']:<18}  "
            f"{entry['duration']:>9.2f}s  "
            f"{entry['regret_rate']:>12.2f}  "
            f"{entry['loss']:>12.4f}"
        )
