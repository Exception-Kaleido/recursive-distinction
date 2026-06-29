from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .types import Action, Position, SensoryItem, StepResult


COLORS = {
    "empty": (0.92, 0.92, 0.92),
    "wall": (0.25, 0.25, 0.28),
    "food": (0.20, 0.86, 0.22),
    "hazard": (0.86, 0.20, 0.20),
    # Deliberately close to food, but not identical.
    "decoy": (0.24, 0.82, 0.23),
}
CHANGED_HAZARD_COLOR = (0.22, 0.38, 0.88)


@dataclass
class Entity:
    kind: str
    color_override: tuple[float, float, float] | None = None
    entity_id: int = -1


@dataclass(frozen=True)
class HiddenTransition:
    """Evaluator-only state. Never passed to the agent or trace field."""

    target: Position
    target_kind: str | None
    environment_changed: bool = False


class GridWorld:
    """World semantics are hidden behind numeric sensory observations."""

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        seed: int = 7,
        counts: Dict[str, int] | None = None,
        ambiguous_rgb: bool = False,
        delayed_outcomes: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.counts = counts or {"wall": 15, "food": 10, "hazard": 8, "decoy": 10}
        self.ambiguous_rgb = ambiguous_rgb
        self.delayed_outcomes = delayed_outcomes
        self.colors = COLORS.copy()
        self.role_effects = {"food": 8.0, "decoy": 0.0, "hazard": -7.0}
        self.roles_reversed = False
        self.step_index = 0
        self.pending_outcomes: List[tuple[int, float, str, int]] = []
        self.last_delivered_kinds: List[str] = []
        self._next_entity_id = 0
        self.entities: Dict[Position, Entity] = {}
        self.agent_position: Position = (width // 2, height // 2)
        self.viability = 50.0
        self.last_hidden_transition = HiddenTransition(self.agent_position, None)
        self._populate()

    def _populate(self) -> None:
        available = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.agent_position
        ]
        self.rng.shuffle(available)
        for kind, count in self.counts.items():
            for _ in range(count):
                if not available:
                    return
                self.entities[available.pop()] = self._make_entity(kind)

    def _make_entity(self, kind: str) -> Entity:
        color_override = None
        if self.ambiguous_rgb and kind in ("food", "decoy", "hazard"):
            center = (0.28, 0.72, 0.28)
            color_override = tuple(
                min(1.0, max(0.0, value + self.rng.uniform(-0.12, 0.12)))
                for value in center
            )
        entity = Entity(kind, color_override, self._next_entity_id)
        self._next_entity_id += 1
        return entity

    def observe(self, radius: int = 2) -> List[SensoryItem]:
        ax, ay = self.agent_position
        patch: List[SensoryItem] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = ax + dx, ay + dy
                if not self.in_bounds((x, y)):
                    color = self.colors["wall"]
                else:
                    entity = self.entities.get((x, y))
                    color = (
                        entity.color_override or self.colors[entity.kind]
                        if entity
                        else self.colors["empty"]
                    )
                patch.append(SensoryItem(*color, dx, dy))
        return patch

    def step(self, action: Action) -> StepResult:
        self.step_index += 1
        ax, ay = self.agent_position
        target = (ax + action[0], ay + action[1])
        entity = self.entities.get(target)
        touched_kind = entity.kind if entity else None
        resistance = 0.0
        delta = -0.05  # small metabolic cost makes viability meaningful over time
        moved = False
        self.last_delivered_kinds = []

        due = [
            event for event in self.pending_outcomes if event[0] <= self.step_index
        ]
        self.pending_outcomes = [
            event for event in self.pending_outcomes if event[0] > self.step_index
        ]
        for _, effect, kind, _ in due:
            delta += effect
            self.last_delivered_kinds.append(kind)

        if not self.in_bounds(target) or (entity and entity.kind == "wall"):
            resistance = 1.0
        else:
            self.agent_position = target
            moved = True
            if entity and entity.kind in self.role_effects:
                effect = self.role_effects[entity.kind]
                if effect:
                    if self.delayed_outcomes:
                        delay = 3 + (entity.entity_id % 3)
                        self.pending_outcomes.append(
                            (
                                self.step_index + delay,
                                effect,
                                entity.kind,
                                entity.entity_id,
                            )
                        )
                    else:
                        delta += effect
                        self.last_delivered_kinds.append(entity.kind)
                if effect > 0.0:
                    del self.entities[target]

        self.viability = max(0.0, min(100.0, self.viability + delta))
        self.last_hidden_transition = HiddenTransition(target, touched_kind)
        return StepResult(moved, resistance, delta, self.viability)

    def apply_role_reversal(self, replenish_each: int = 6) -> Dict[str, int]:
        self.role_effects["food"], self.role_effects["decoy"] = (
            self.role_effects["decoy"],
            self.role_effects["food"],
        )
        self.roles_reversed = not self.roles_reversed
        available = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.agent_position and (x, y) not in self.entities
        ]
        self.rng.shuffle(available)
        added = {"food": 0, "decoy": 0}
        for kind in ("food", "decoy"):
            existing = sum(entity.kind == kind for entity in self.entities.values())
            while existing < replenish_each and available:
                self.entities[available.pop()] = self._make_entity(kind)
                existing += 1
                added[kind] += 1
        return {
            "roles_reversed": int(self.roles_reversed),
            "food_added": added["food"],
            "decoy_added": added["decoy"],
        }

    def current_effect(self, kind: str | None) -> float:
        return self.role_effects.get(kind or "", 0.0)

    def apply_environment_change(
        self,
        food_to_decoy: int = 4,
        change_hazard_color: bool = True,
    ) -> Dict[str, int]:
        food_positions = [
            position
            for position, entity in self.entities.items()
            if entity.kind == "food"
        ]
        self.rng.shuffle(food_positions)
        converted = 0
        for position in food_positions[:food_to_decoy]:
            # Preserve food appearance so only the action outcome reveals change.
            self.entities[position] = Entity("decoy", color_override=self.colors["food"])
            converted += 1
        available = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.agent_position and (x, y) not in self.entities
        ]
        self.rng.shuffle(available)
        while converted < food_to_decoy and available:
            self.entities[available.pop()] = Entity(
                "decoy", color_override=self.colors["food"]
            )
            converted += 1

        remaining_food = sum(
            entity.kind == "food" for entity in self.entities.values()
        )
        replenished = 0
        while remaining_food < 2 and available:
            self.entities[available.pop()] = Entity("food")
            remaining_food += 1
            replenished += 1
        if change_hazard_color:
            self.colors["hazard"] = CHANGED_HAZARD_COLOR
        self.last_hidden_transition = HiddenTransition(
            self.agent_position, None, environment_changed=True
        )
        return {
            "food_to_decoy": converted,
            "food_replenished": replenished,
            "hazard_color_changed": int(change_hazard_color),
        }

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def iter_entities(self) -> Iterable[tuple[Position, Entity]]:
        return self.entities.items()
