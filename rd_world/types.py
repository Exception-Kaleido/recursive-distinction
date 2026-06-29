from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Position = Tuple[int, int]
Action = Tuple[int, int]


@dataclass(frozen=True)
class SensoryItem:
    color_r: float
    color_g: float
    color_b: float
    rel_x: int
    rel_y: int

    def features(self) -> Dict[str, float]:
        return {
            "color_r": self.color_r,
            "color_g": self.color_g,
            "color_b": self.color_b,
            "rel_x": float(self.rel_x),
            "rel_y": float(self.rel_y),
        }


@dataclass
class Trace:
    features: Dict[str, float]
    action: Action
    resistance: float
    delta_viability: float
    selected_features: List[str] = field(default_factory=list)
    source_step: int = 0
    commit_step: int = 0
    temporal_delay: int = 0
    temporal_eligibility: float = 1.0

    @property
    def outcome(self) -> Tuple[float, float]:
        return self.resistance, self.delta_viability


@dataclass
class StepResult:
    moved: bool
    resistance: float
    delta_viability: float
    viability: float
