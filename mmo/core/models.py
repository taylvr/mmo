

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class Query:
    

    qid: str
    original_cost: float
    frequency: float
    mv_costs: Dict[str, float] = field(default_factory=dict)


@dataclass
class MV:
    

    mv_id: str
    size: float
    build_time: float


@dataclass
class Problem:
    

    queries: Dict[str, Query]
    mvs: Dict[str, MV]
    s_curr: Set[str]
    s_target: Set[str]
    budget: float
    delta: float

    @property
    def to_drop(self) -> list[str]:
        return sorted(self.s_curr - self.s_target)

    @property
    def to_create(self) -> list[str]:
        return sorted(self.s_target - self.s_curr)

    @property
    def keep(self) -> Set[str]:
        return self.s_curr & self.s_target
