from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .objecthood import ObjecthoodStabilizer
from .types import Action, SensoryItem, Trace


ACTIONS: Tuple[Action, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


class AgentPolicy:
    def __init__(
        self,
        stabilizer: ObjecthoodStabilizer,
        seed: int = 7,
        exploration: float = 0.12,
        policy_reentry: bool = False,
    ) -> None:
        self.stabilizer = stabilizer
        self.rng = random.Random(seed)
        self.exploration = exploration
        self.policy_reentry = policy_reentry
        self.preference_values: Dict[tuple[int, int, int, Action], float] = {}
        self.preference_counts: Dict[tuple[int, int, int, Action], int] = {}

    def choose(self, patch: List[SensoryItem]) -> Action:
        adjacent: Dict[Action, SensoryItem] = {
            (item.rel_x, item.rel_y): item
            for item in patch
            if abs(item.rel_x) + abs(item.rel_y) == 1
        }
        scores = []
        for action in ACTIONS:
            item = adjacent[action]
            delta, resistance, confidence = self.stabilizer.predict(
                item, action, record_policy_influence=True
            )
            score = confidence * (1.8 * delta - 8.0 * resistance)
            if self.policy_reentry:
                score += self.preference_values.get(
                    self._preference_key(item, action), 0.0
                )
            score += 0.08 * (1.0 - confidence)
            scores.append((score, self.rng.random(), action))

        target_action, target_bonus = self._positive_target(patch)
        if target_action is not None:
            scores = [
                (
                    score + (target_bonus if action == target_action else 0.0),
                    tie_breaker,
                    action,
                )
                for score, tie_breaker, action in scores
            ]

        no_guidance = (
            not self.stabilizer.candidates
            and not self.policy_reentry
        )
        if self.rng.random() < self.exploration or no_guidance:
            safe = [entry for entry in scores if entry[0] > -0.25]
            return self.rng.choice(safe or scores)[2]
        return max(scores)[2]

    def update_from_trace(self, trace: Trace) -> None:
        if not self.policy_reentry:
            return
        item = SensoryItem(
            trace.features["color_r"],
            trace.features["color_g"],
            trace.features["color_b"],
            int(trace.features["rel_x"]),
            int(trace.features["rel_y"]),
        )
        key = self._preference_key(item, trace.action)
        count = self.preference_counts.get(key, 0) + 1
        target = 1.5 * trace.delta_viability - 6.0 * trace.resistance
        old = self.preference_values.get(key, 0.0)
        rate = min(0.35, 1.0 / count)
        self.preference_values[key] = old + rate * (target - old)
        self.preference_counts[key] = count

    @staticmethod
    def _preference_key(
        item: SensoryItem, action: Action
    ) -> tuple[int, int, int, Action]:
        return (
            round(item.color_r * 4),
            round(item.color_g * 4),
            round(item.color_b * 4),
            action,
        )

    def _positive_target(
        self, patch: List[SensoryItem]
    ) -> Tuple[Action | None, float]:
        targets = []
        for item in patch:
            delta, resistance, confidence = self.stabilizer.predict_identity(
                item, record_policy_influence=True
            )
            if delta <= 0.5 or resistance >= 0.5 or confidence <= 0.05:
                continue
            distance = abs(item.rel_x) + abs(item.rel_y)
            utility = confidence * delta / max(1, distance)
            targets.append((utility, item))
        if not targets:
            return None, 0.0

        utility, target = max(targets, key=lambda pair: pair[0])
        horizontal = (1 if target.rel_x > 0 else -1, 0)
        vertical = (0, 1 if target.rel_y > 0 else -1)
        if target.rel_x == 0:
            action = vertical
        elif target.rel_y == 0:
            action = horizontal
        else:
            action = horizontal if abs(target.rel_x) >= abs(target.rel_y) else vertical
        return action, min(6.0, utility)


class FixedWeightPolicy:
    """Memoryless baseline using fixed numeric sensory preferences."""

    def __init__(self, seed: int = 7, exploration: float = 0.15) -> None:
        self.rng = random.Random(seed)
        self.exploration = exploration

    def choose(self, patch: List[SensoryItem]) -> Action:
        adjacent = {
            (item.rel_x, item.rel_y): item
            for item in patch
            if abs(item.rel_x) + abs(item.rel_y) == 1
        }
        if self.rng.random() < self.exploration:
            return self.rng.choice(ACTIONS)
        scored = []
        for action in ACTIONS:
            item = adjacent[action]
            brightness = (item.color_r + item.color_g + item.color_b) / 3.0
            contrast = max(item.color_r, item.color_g, item.color_b) - min(
                item.color_r, item.color_g, item.color_b
            )
            scored.append((0.7 * brightness + 0.3 * contrast, self.rng.random(), action))
        return max(scored)[2]


class RandomPolicy:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def choose(self, patch: List[SensoryItem]) -> Action:
        return self.rng.choice(ACTIONS)
