from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List

from .types import Trace


FEATURES = (
    "color_r",
    "color_g",
    "color_b",
    "rel_x",
    "rel_y",
    "resistance",
    "delta_viability",
)


class BoundaryGate:
    """A competitive feature gate updated by predictive re-entry."""

    def __init__(
        self,
        reentry: bool = True,
        reentry_strength: float = 1.0,
        learning_rate: float = 2.0,
        threshold: float = 0.10,
        temperature: float = 1.0,
    ) -> None:
        self.reentry_strength = max(0.0, reentry_strength)
        self.reentry = reentry and self.reentry_strength > 0.0
        self.learning_rate = learning_rate * self.reentry_strength
        self.threshold = threshold
        self.temperature = temperature
        self.base_logits: Dict[str, float] = {
            "color_r": 0.0,
            "color_g": 0.0,
            "color_b": 0.0,
            "rel_x": -0.15,
            "rel_y": -0.15,
            "resistance": 0.35,
            "delta_viability": 0.35,
        }
        self.logits = self.base_logits.copy()
        self.utility_ema: Dict[str, float] = {name: 0.0 for name in FEATURES}
        self.weights = self._softmax()
        self.history: List[Dict[str, float]] = [self.weights.copy()]

    def select(self, features: Dict[str, float]) -> List[str]:
        available = [name for name in features if self.weights.get(name, 0.0) >= self.threshold]
        if available:
            return available
        return [max(features, key=lambda name: self.weights.get(name, 0.0))]

    def update_from_reentry(self, current: Trace, previous: Iterable[Trace]) -> None:
        if not self.reentry:
            self.history.append(self.weights.copy())
            return

        recent = list(previous)[-40:]
        if len(recent) < 3:
            self.history.append(self.weights.copy())
            return

        comparisons = recent + [current]
        outcome_values = [
            (trace.resistance, trace.delta_viability) for trace in comparisons
        ]
        if len(set(outcome_values)) < 2:
            self.history.append(self.weights.copy())
            return

        utilities = {
            name: self._predictive_utility(name, comparisons) for name in FEATURES
        }
        for name, utility in utilities.items():
            self.utility_ema[name] = 0.88 * self.utility_ema[name] + 0.12 * utility

        mean_utility = sum(self.utility_ema.values()) / len(self.utility_ema)
        for name, utility in utilities.items():
            # Centering creates direct competition: above-average predictors rise
            # while below-average predictors fall.
            self.logits[name] = self.base_logits[name] + self.learning_rate * (
                self.utility_ema[name] - mean_utility
            )

        self.weights = self._softmax()
        self.history.append(self.weights.copy())

    def _softmax(self) -> Dict[str, float]:
        peak = max(self.logits.values())
        exponentials = {
            name: math.exp((value - peak) / self.temperature)
            for name, value in self.logits.items()
        }
        total = sum(exponentials.values())
        return {name: value / total for name, value in exponentials.items()}

    @classmethod
    def _predictive_utility(cls, name: str, traces: List[Trace]) -> float:
        feature_differences: List[float] = []
        outcome_differences: List[float] = []
        for index, left in enumerate(traces):
            for right in traces[index + 1 :]:
                feature_differences.append(cls._feature_difference(name, left, right))
                outcome_differences.append(cls._outcome_difference(left, right))

        if (
            len(feature_differences) < 3
            or statistics.pstdev(feature_differences) < 1e-6
            or statistics.pstdev(outcome_differences) < 1e-6
        ):
            return 0.0
        correlation = statistics.correlation(feature_differences, outcome_differences)
        return max(0.0, correlation)

    @staticmethod
    def _outcome_difference(a: Trace, b: Trace) -> float:
        viability = min(1.0, abs(a.delta_viability - b.delta_viability) / 8.0)
        blockage = abs(a.resistance - b.resistance)
        return min(1.0, 0.6 * viability + 0.4 * blockage)

    @staticmethod
    def _feature_difference(name: str, a: Trace, b: Trace) -> float:
        if name == "resistance":
            return abs(a.resistance - b.resistance)
        if name == "delta_viability":
            return min(1.0, abs(a.delta_viability - b.delta_viability) / 8.0)
        value_a = a.features.get(name, 0.0)
        value_b = b.features.get(name, 0.0)
        scale = 4.0 if name.startswith("rel_") else 1.0
        return min(1.0, abs(value_a - value_b) / scale)
