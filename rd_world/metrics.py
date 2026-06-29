from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .objecthood import ObjectCandidate, ObjecthoodStabilizer
from .types import Action, Position, StepResult
from .world import COLORS, GridWorld


CANDIDATE_TYPES = ("food_like", "decoy_like", "wall_like", "hazard_like")


@dataclass
class CandidateGroupStats:
    count: int = 0
    observations: int = 0
    mean_confidence: float = 0.0
    mean_delta_viability: float = 0.0
    mean_resistance: float = 0.0


@dataclass
class Metrics:
    viability: List[float] = field(default_factory=list)
    mean_confidence: List[float] = field(default_factory=list)
    boundary_weights: List[Dict[str, float]] = field(default_factory=list)
    food_approach_history: List[float] = field(default_factory=list)
    decoy_false_positive_history: List[float] = field(default_factory=list)
    normalized_decoy_false_positive_history: List[float] = field(default_factory=list)
    hazard_avoidance_history: List[float] = field(default_factory=list)
    wall_collision_history: List[float] = field(default_factory=list)
    candidate_stability: List[float] = field(default_factory=list)
    split_events: List[int] = field(default_factory=list)
    merge_events: List[int] = field(default_factory=list)
    destabilization_events: List[int] = field(default_factory=list)
    candidate_confidence_by_type: Dict[str, List[float]] = field(
        default_factory=lambda: {name: [] for name in CANDIDATE_TYPES}
    )
    food_approaches: int = 0
    food_opportunities: int = 0
    decoy_false_positives: int = 0
    decoy_opportunities: int = 0
    green_food_approaches: int = 0
    green_decoy_approaches: int = 0
    hazard_avoidances: int = 0
    hazard_opportunities: int = 0
    wall_collisions: int = 0
    steps: int = 0
    environment_change_step: int | None = None
    candidates_at_change: int = 0
    created_at_change: int = 0
    splits_at_change: int = 0
    merges_at_change: int = 0
    destabilizations_at_change: int = 0
    adaptation_step: int | None = None
    _previous_candidates: Dict[int, tuple[float, float, float]] = field(
        default_factory=dict, repr=False
    )

    @property
    def food_approach_rate(self) -> float:
        return self._rate(self.food_approaches, self.food_opportunities)

    @property
    def decoy_false_positive_rate(self) -> float:
        return self._rate(self.decoy_false_positives, self.decoy_opportunities)

    @property
    def normalized_decoy_false_positive_rate(self) -> float:
        return self._rate(
            self.green_decoy_approaches,
            self.green_food_approaches + self.green_decoy_approaches,
        )

    @property
    def hazard_avoidance_rate(self) -> float:
        return self._rate(self.hazard_avoidances, self.hazard_opportunities)

    @property
    def wall_collision_rate(self) -> float:
        return self._rate(self.wall_collisions, self.steps)

    @property
    def wall_collision_reduction(self) -> float:
        if len(self.wall_collision_history) < 8:
            return 0.0
        quarter = max(1, len(self.wall_collision_history) // 4)
        early = sum(self.wall_collision_history[:quarter]) / quarter
        late = sum(self.wall_collision_history[-quarter:]) / quarter
        return early - late

    @property
    def post_change_split_events(self) -> int:
        final = self.split_events[-1] if self.split_events else 0
        return final - self.splits_at_change

    @property
    def post_change_merge_events(self) -> int:
        final = self.merge_events[-1] if self.merge_events else 0
        return final - self.merges_at_change

    @property
    def post_change_new_candidates(self) -> int:
        # Filled by the experiment summary from stabilizer.total_created.
        return getattr(self, "_final_created", 0) - self.created_at_change

    @property
    def post_change_destabilization_events(self) -> int:
        final = self.destabilization_events[-1] if self.destabilization_events else 0
        return final - self.destabilizations_at_change

    @property
    def pre_change_destabilization_events(self) -> int:
        return self.destabilizations_at_change

    @property
    def adaptation_latency(self) -> int:
        if self.environment_change_step is None:
            return 0
        end_step = max(self.environment_change_step, self.steps - 1)
        if self.adaptation_step is None:
            return end_step - self.environment_change_step + 1
        return max(0, self.adaptation_step - self.environment_change_step)

    @property
    def stability_plasticity_score(self) -> float:
        if self.environment_change_step is None:
            return 0.0
        post_horizon = max(1, self.steps - self.environment_change_step)
        responsiveness = max(0.0, 1.0 - self.adaptation_latency / post_horizon)
        pre_rate = self.pre_change_destabilization_events / max(
            1, self.environment_change_step
        )
        stability_penalty = 1.0 / (1.0 + 5.0 * pre_rate)
        return responsiveness * stability_penalty

    @staticmethod
    def _rate(successes: int, opportunities: int) -> float:
        return successes / opportunities if opportunities else 0.0


class HiddenEvaluator:
    """Computes validation metrics without exposing labels to the agent."""

    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def record_decision(self, world: GridWorld, action: Action, radius: int = 2) -> None:
        ax, ay = world.agent_position
        visible_food = [
            position
            for position, entity in world.iter_entities()
            if entity.kind == "food"
            and max(abs(position[0] - ax), abs(position[1] - ay)) <= radius
        ]
        if visible_food:
            self.metrics.food_opportunities += 1
            old_distance = min(self._manhattan((ax, ay), target) for target in visible_food)
            proposed = (ax + action[0], ay + action[1])
            new_distance = min(self._manhattan(proposed, target) for target in visible_food)
            if new_distance < old_distance:
                self.metrics.food_approaches += 1

        target = (ax + action[0], ay + action[1])
        adjacent = {
            position: entity.kind
            for position, entity in world.iter_entities()
            if self._manhattan((ax, ay), position) == 1
        }
        if "decoy" in adjacent.values():
            self.metrics.decoy_opportunities += 1
            if adjacent.get(target) == "decoy":
                self.metrics.decoy_false_positives += 1
        selected_kind = adjacent.get(target)
        if selected_kind == "food":
            self.metrics.green_food_approaches += 1
        elif selected_kind == "decoy":
            self.metrics.green_decoy_approaches += 1
        if "hazard" in adjacent.values():
            self.metrics.hazard_opportunities += 1
            if adjacent.get(target) != "hazard":
                self.metrics.hazard_avoidances += 1

    def record_step(
        self,
        result: StepResult,
        stabilizer: ObjecthoodStabilizer,
        boundary_weights: Dict[str, float],
        split_events: int,
        merge_events: int,
        destabilization_events: int,
    ) -> None:
        metrics = self.metrics
        metrics.steps += 1
        collision = float(result.resistance >= 0.5)
        metrics.wall_collisions += int(collision)
        metrics.wall_collision_history.append(collision)
        metrics.viability.append(result.viability)
        metrics.boundary_weights.append(boundary_weights.copy())

        candidates = stabilizer.candidates
        metrics.mean_confidence.append(
            sum(candidate.objecthood_score for candidate in candidates) / len(candidates)
            if candidates
            else 0.0
        )
        metrics.candidate_stability.append(
            self._candidate_stability(
                candidates,
                split_events,
                merge_events,
                destabilization_events,
            )
        )
        metrics.split_events.append(split_events)
        metrics.merge_events.append(merge_events)
        metrics.destabilization_events.append(destabilization_events)

        groups = group_candidates(candidates)
        for name in CANDIDATE_TYPES:
            group = groups[name]
            metrics.candidate_confidence_by_type[name].append(
                group.mean_confidence if group.count else 0.0
            )

        metrics.food_approach_history.append(metrics.food_approach_rate)
        metrics.decoy_false_positive_history.append(metrics.decoy_false_positive_rate)
        metrics.normalized_decoy_false_positive_history.append(
            metrics.normalized_decoy_false_positive_rate
        )
        metrics.hazard_avoidance_history.append(metrics.hazard_avoidance_rate)
        self._detect_adaptation(candidates)

    def mark_environment_change(
        self,
        step: int,
        stabilizer: ObjecthoodStabilizer,
        split_events: int,
        merge_events: int,
        created_events: int,
        destabilization_events: int,
    ) -> None:
        self.metrics.environment_change_step = step
        self.metrics.candidates_at_change = len(stabilizer.candidates)
        self.metrics.created_at_change = created_events
        self.metrics.splits_at_change = split_events
        self.metrics.merges_at_change = merge_events
        self.metrics.destabilizations_at_change = destabilization_events

    def _candidate_stability(
        self,
        candidates: Iterable[ObjectCandidate],
        split_events: int,
        merge_events: int,
        destabilization_events: int,
    ) -> float:
        current = {
            candidate.candidate_id: (
                candidate.objecthood_score,
                candidate.predicted_delta_viability,
                candidate.predicted_resistance,
            )
            for candidate in candidates
        }
        previous = self.metrics._previous_candidates
        if not previous:
            stability = 0.0
        else:
            common = set(previous) & set(current)
            if common:
                change = sum(
                    abs(current[key][0] - previous[key][0])
                    + abs(current[key][1] - previous[key][1]) / 8.0
                    + abs(current[key][2] - previous[key][2])
                    for key in common
                ) / (3.0 * len(common))
            else:
                change = 1.0
            churn = len(set(previous) ^ set(current)) / max(
                1, len(set(previous) | set(current))
            )
            previous_splits = self.metrics.split_events[-1] if self.metrics.split_events else 0
            previous_merges = self.metrics.merge_events[-1] if self.metrics.merge_events else 0
            previous_destabilizations = (
                self.metrics.destabilization_events[-1]
                if self.metrics.destabilization_events
                else 0
            )
            event_cost = (
                0.20 * max(0, split_events - previous_splits)
                + 0.15 * max(0, merge_events - previous_merges)
                + 0.05
                * max(0, destabilization_events - previous_destabilizations)
            )
            stability = max(
                0.0, 1.0 - min(1.0, change + churn + event_cost)
            )
        self.metrics._previous_candidates = current
        return stability

    def _detect_adaptation(self, candidates: Iterable[ObjectCandidate]) -> None:
        metrics = self.metrics
        if (
            metrics.environment_change_step is None
            or metrics.adaptation_step is not None
        ):
            return
        for candidate in candidates:
            if candidate.candidate_id < metrics.created_at_change:
                continue
            rgb = tuple(
                candidate.centroid[name]
                for name in ("color_r", "color_g", "color_b")
            )
            if (
                _rgb_distance(rgb, COLORS["food"]) <= 0.03
                and abs(candidate.predicted_delta_viability + 0.05) <= 0.25
                and candidate.observations >= 2
            ):
                metrics.adaptation_step = metrics.steps - 1
                return

    @staticmethod
    def _manhattan(left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])


def candidate_type(candidate: ObjectCandidate) -> str:
    if candidate.predicted_resistance >= 0.5:
        return "wall_like"
    if candidate.predicted_delta_viability >= 0.5:
        return "food_like"
    if candidate.predicted_delta_viability <= -0.5:
        return "hazard_like"
    rgb = tuple(candidate.centroid[name] for name in ("color_r", "color_g", "color_b"))
    green_distance = min(
        _rgb_distance(rgb, COLORS["food"]),
        _rgb_distance(rgb, COLORS["decoy"]),
    )
    if green_distance <= 0.15:
        return "decoy_like"
    return "background"


def group_candidates(
    candidates: Iterable[ObjectCandidate],
) -> Dict[str, CandidateGroupStats]:
    grouped: Dict[str, List[ObjectCandidate]] = {
        name: [] for name in CANDIDATE_TYPES
    }
    for candidate in candidates:
        name = candidate_type(candidate)
        if name in grouped:
            grouped[name].append(candidate)

    result: Dict[str, CandidateGroupStats] = {}
    for name, group in grouped.items():
        count = len(group)
        result[name] = CandidateGroupStats(
            count=count,
            observations=sum(candidate.observations for candidate in group),
            mean_confidence=(
                sum(candidate.objecthood_score for candidate in group) / count
                if count
                else 0.0
            ),
            mean_delta_viability=(
                sum(candidate.predicted_delta_viability for candidate in group) / count
                if count
                else 0.0
            ),
            mean_resistance=(
                sum(candidate.predicted_resistance for candidate in group) / count
                if count
                else 0.0
            ),
        )
    return result


def _rgb_distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / 3.0
