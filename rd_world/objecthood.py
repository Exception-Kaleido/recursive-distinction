from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .boundary import BoundaryGate
from .types import Action, SensoryItem, Trace


RGB_FEATURES = ("color_r", "color_g", "color_b")
VISUAL_FEATURES = RGB_FEATURES + ("rel_x", "rel_y")
METABOLIC_COST = -0.05

CONFLICT_PROFILES = {
    "low": {
        "viability": 2.0,
        "resistance": 0.30,
        "rgb": 0.18,
        "position": 0.30,
        "temporal": 0.40,
    },
    "medium": {
        "viability": 3.5,
        "resistance": 0.50,
        "rgb": 0.12,
        "position": 0.20,
        "temporal": 0.60,
    },
    "high": {
        "viability": 5.0,
        "resistance": 0.70,
        "rgb": 0.08,
        "position": 0.10,
        "temporal": 0.80,
    },
}

PARENT_DECAY_FACTORS = {
    "none": 1.0,
    "slow": 0.85,
    "fast": 0.55,
}

ACTIVATION_THRESHOLDS = {
    "low": 0.45,
    "medium": 0.60,
    "high": 0.75,
}

STATUS_POLICY_INFLUENCE = {
    "active": 1.0,
    "validated": 0.75,
    "provisional": 0.15,
    "obsolete": 0.0,
}


@dataclass
class ConflictAttribution:
    log_id: int
    step: int
    trace_source_step: int
    candidate_id: int
    child_candidate_id: int | None
    rgb_distance: float
    relative_position_distance: float
    action_distance: float
    candidate_action: Action
    trace_action: Action
    predicted_delta_viability: float
    predicted_resistance: float
    observed_delta_viability: float
    observed_resistance: float
    temporal_eligibility: float
    temporal_delay: int
    viability_conflict: bool
    resistance_conflict: bool
    evidence_count: int
    threshold_passed: bool
    split_triggered: bool
    parent_decay_applied: bool = False
    suppression_applied: bool = False


@dataclass
class BranchEvidence:
    observations: int = 0
    support: int = 0
    error_sum: float = 0.0
    consistency_sum: float = 0.0
    viability_accuracy_sum: float = 0.0
    resistance_accuracy_sum: float = 0.0

    def record(self, candidate: ObjectCandidate, trace: Trace) -> None:
        viability_error = abs(
            trace.delta_viability - candidate.predicted_delta_viability
        )
        resistance_error = abs(
            trace.resistance - candidate.predicted_resistance
        )
        error = min(
            1.0, 0.7 * viability_error / 8.0 + 0.3 * resistance_error
        )
        consistency = float(
            candidate.action == trace.action
            and Destabilizer._outcome_bin(
                candidate.predicted_delta_viability
            )
            == Destabilizer._outcome_bin(trace.delta_viability)
            and (candidate.predicted_resistance >= 0.5)
            == (trace.resistance >= 0.5)
        )
        self.observations += 1
        self.support += int(error <= 0.25)
        self.error_sum += error
        self.consistency_sum += consistency
        self.viability_accuracy_sum += max(
            0.0, 1.0 - viability_error / 8.0
        )
        self.resistance_accuracy_sum += max(
            0.0, 1.0 - resistance_error
        )

    @property
    def score(self) -> float:
        if not self.observations:
            return 0.0
        count = self.observations
        return (
            0.30 * self.support / count
            + 0.25 * self.consistency_sum / count
            + 0.25 * self.viability_accuracy_sum / count
            + 0.20 * self.resistance_accuracy_sum / count
        )


@dataclass
class ProvisionalFamily:
    parent_id: int
    child_ids: List[int]
    split_step: int
    window_observations: int = 0
    evidence: Dict[int, BranchEvidence] = field(default_factory=dict)
    parent_resolved_step: int | None = None


@dataclass
class ObjectCandidate:
    candidate_id: int
    centroid: Dict[str, float]
    action: Action
    predicted_resistance: float
    predicted_delta_viability: float
    observations: int = 1
    confidence: float = 0.0
    objecthood_score: float = 0.0
    reidentifiability: float = 0.0
    failures: int = 0
    contradictions: int = 0
    inhibition: float = 1.0
    status: str = "active"
    parent_candidate_id: int | None = None
    required_evidence: int = 1
    parent_decay_applied: bool = False
    support_count: int = 0
    conflict_count: int = 0
    prediction_error: List[float] = field(default_factory=list)
    action_outcome_consistency: List[float] = field(default_factory=list)
    viability_prediction_accuracy: List[float] = field(default_factory=list)
    resistance_prediction_accuracy: List[float] = field(default_factory=list)
    created_step: int = 0
    resolved_step: int | None = None
    won_parent_competition: bool = False
    encounter_signatures: List[tuple[int, ...]] = field(default_factory=list)
    encounter_steps: List[int] = field(default_factory=list)
    validated_compatible_streak: int = 0
    ever_validated: bool = False
    ever_activated: bool = False
    outcome_history: List[float] = field(default_factory=list)
    resistance_history: List[float] = field(default_factory=list)
    visual_error_history: List[float] = field(default_factory=list)

    def update(self, trace: Trace, record_evidence: bool = True) -> None:
        if record_evidence:
            self.record_evidence(trace)
        self.observations += 1
        rate = 1.0 / min(self.observations, 20)
        old_centroid = self.centroid.copy()
        for name in VISUAL_FEATURES:
            self.centroid[name] += rate * (trace.features[name] - self.centroid[name])
        self.predicted_resistance += rate * (trace.resistance - self.predicted_resistance)
        self.predicted_delta_viability += rate * (
            trace.delta_viability - self.predicted_delta_viability
        )
        self.outcome_history.append(trace.delta_viability)
        self.outcome_history = self.outcome_history[-20:]
        self.resistance_history.append(trace.resistance)
        self.resistance_history = self.resistance_history[-20:]
        visual_error = sum(
            abs(trace.features[name] - old_centroid[name]) for name in RGB_FEATURES
        ) / len(RGB_FEATURES)
        self.visual_error_history.append(visual_error)
        self.visual_error_history = self.visual_error_history[-20:]
        self._update_scores()

    def initialize_scores(self) -> None:
        self._update_scores()

    @property
    def action_relevance(self) -> float:
        viability_effect = min(
            1.0, abs(self.predicted_delta_viability - METABOLIC_COST) / 7.0
        )
        return max(viability_effect, self.predicted_resistance)

    def _update_scores(self) -> None:
        outcome_consistency = 1.0 / (
            1.0 + 2.0 * self._variance(self.outcome_history)
            + self._variance(self.resistance_history)
        )
        mean_visual_error = (
            sum(self.visual_error_history) / len(self.visual_error_history)
            if self.visual_error_history
            else 0.25
        )
        self.reidentifiability = max(0.0, 1.0 - 4.0 * mean_visual_error)
        evidence = min(1.0, self.observations / 4.0)
        viability_relevance = min(
            1.0, abs(self.predicted_delta_viability - METABOLIC_COST) / 7.0
        )
        blockage = min(1.0, self.predicted_resistance)

        # Frequency is only a small evidence term. Neutral, freely traversed
        # background remains low-objecthood even after many observations.
        relevance = max(viability_relevance, blockage)
        self.objecthood_score = max(
            0.0,
            min(
                1.0,
                relevance
                * (0.40 * outcome_consistency + 0.35 * self.reidentifiability + 0.25 * evidence),
            ),
        )
        self.objecthood_score *= self.inhibition
        if self.status == "obsolete":
            self.objecthood_score = 0.0
        self.confidence = self.objecthood_score

    def inhibit(self, factor: float) -> None:
        self.inhibition = max(0.0, min(1.0, self.inhibition * factor))
        self._update_scores()

    @property
    def provisional(self) -> bool:
        return self.status == "provisional"

    @provisional.setter
    def provisional(self, value: bool) -> None:
        if value:
            self.status = "provisional"
        elif self.status == "provisional":
            self.status = "active"

    @property
    def obsolete(self) -> bool:
        return self.status == "obsolete"

    @obsolete.setter
    def obsolete(self, value: bool) -> None:
        if value:
            self.status = "obsolete"
        elif self.status == "obsolete":
            self.status = "active"

    def record_evidence(self, trace: Trace) -> float:
        viability_error = abs(
            trace.delta_viability - self.predicted_delta_viability
        )
        resistance_error = abs(
            trace.resistance - self.predicted_resistance
        )
        normalized_error = min(
            1.0, 0.7 * viability_error / 8.0 + 0.3 * resistance_error
        )
        viability_accuracy = max(0.0, 1.0 - viability_error / 8.0)
        resistance_accuracy = max(0.0, 1.0 - resistance_error)
        consistency = float(
            self.action == trace.action
            and Destabilizer._outcome_bin(self.predicted_delta_viability)
            == Destabilizer._outcome_bin(trace.delta_viability)
            and (self.predicted_resistance >= 0.5)
            == (trace.resistance >= 0.5)
        )
        if normalized_error <= 0.25:
            self.support_count += 1
        else:
            self.conflict_count += 1
        self._append_evidence(self.prediction_error, normalized_error)
        self._append_evidence(
            self.action_outcome_consistency, consistency
        )
        self._append_evidence(
            self.viability_prediction_accuracy, viability_accuracy
        )
        self._append_evidence(
            self.resistance_prediction_accuracy, resistance_accuracy
        )
        signature = (
            round(trace.features["color_r"] * 8),
            round(trace.features["color_g"] * 8),
            round(trace.features["color_b"] * 8),
            int(trace.features["rel_x"]),
            int(trace.features["rel_y"]),
            trace.action[0],
            trace.action[1],
            min(5, trace.temporal_delay),
        )
        self.encounter_signatures.append(signature)
        self.encounter_signatures = self.encounter_signatures[-20:]
        self.encounter_steps.append(trace.source_step)
        self.encounter_steps = self.encounter_steps[-20:]
        return normalized_error

    @property
    def predictive_usefulness(self) -> float:
        evidence = self.support_count + self.conflict_count
        support_rate = self.support_count / evidence if evidence else 0.0
        return (
            0.30 * support_rate
            + 0.25 * self._mean(self.action_outcome_consistency)
            + 0.25 * self._mean(self.viability_prediction_accuracy)
            + 0.20 * self._mean(self.resistance_prediction_accuracy)
        ) * max(0.05, self.objecthood_score)

    @property
    def evidence_diversity(self) -> float:
        contexts = len(set(self.encounter_signatures))
        encounters = len(set(self.encounter_steps))
        return min(1.0, contexts / 3.0) * min(1.0, encounters / 3.0)

    @property
    def calibrated_activation_score(self) -> float:
        evidence = self.support_count + self.conflict_count
        support_rate = self.support_count / evidence if evidence else 0.0
        mean_error = self._mean(self.prediction_error)
        return max(
            0.0,
            min(
                1.0,
                0.20 * support_rate
                + 0.15 * (1.0 - mean_error)
                + 0.20 * self._mean(self.action_outcome_consistency)
                + 0.18 * self._mean(self.viability_prediction_accuracy)
                + 0.17 * self._mean(self.resistance_prediction_accuracy)
                + 0.10 * self.evidence_diversity,
            ),
        )

    @property
    def policy_influence(self) -> float:
        return STATUS_POLICY_INFLUENCE[self.status]

    @staticmethod
    def _append_evidence(buffer: List[float], value: float) -> None:
        buffer.append(value)
        del buffer[:-20]

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.5
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)


class ObjecthoodStabilizer:
    """Builds recurring sensorimotor prototypes without semantic labels."""

    def __init__(self, gate: BoundaryGate, match_threshold: float = 0.16) -> None:
        self.gate = gate
        self.match_threshold = match_threshold
        self.candidates: List[ObjectCandidate] = []
        self._next_id = 0
        self.total_created = 0
        self.total_merged = 0
        self.total_pruned = 0
        self.policy_influence_totals = {
            status: 0.0
            for status in ("provisional", "validated", "active")
        }
        self.policy_prediction_calls = 0

    def process(self, trace: Trace) -> Tuple[ObjectCandidate, float]:
        """Backward-compatible outcome-sensitive prototype update."""
        candidate, distance = self.nearest_trace(trace)
        if candidate is None or distance > self.match_threshold:
            candidate = self.create_candidate(trace)
            return candidate, 0.0
        candidate.update(trace)
        return candidate, distance

    def formation_update(
        self,
        trace: Trace,
        preferred: ObjectCandidate | None = None,
        record_evidence: bool = True,
    ) -> Tuple[ObjectCandidate, float]:
        """Create or update a basic visual-action prototype.

        Outcome conflicts are averaged here. Splitting a prototype because its
        outcome prediction failed belongs to the revision mechanism.
        """
        if preferred is not None:
            preferred.update(trace, record_evidence=record_evidence)
            return preferred, 0.0

        candidate, distance = self.nearest(trace.features, trace.action)
        if candidate is None or distance > self.match_threshold:
            candidate = self.create_candidate(trace)
            return candidate, 0.0
        candidate.update(trace)
        return candidate, distance

    def create_candidate(self, trace: Trace) -> ObjectCandidate:
        candidate = ObjectCandidate(
            candidate_id=self._next_id,
            centroid={name: trace.features[name] for name in VISUAL_FEATURES},
            action=trace.action,
            predicted_resistance=trace.resistance,
            predicted_delta_viability=trace.delta_viability,
            support_count=1,
            prediction_error=[0.0],
            action_outcome_consistency=[1.0],
            viability_prediction_accuracy=[1.0],
            resistance_prediction_accuracy=[1.0],
            outcome_history=[trace.delta_viability],
            resistance_history=[trace.resistance],
        )
        candidate.initialize_scores()
        self._next_id += 1
        self.total_created += 1
        self.candidates.append(candidate)
        return candidate

    def candidate_by_id(self, candidate_id: int) -> ObjectCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == candidate_id
            ),
            None,
        )

    def merge_redundant(self) -> int:
        merges = 0
        kept: List[ObjectCandidate] = []
        for candidate in sorted(
            self.candidates, key=lambda item: item.observations, reverse=True
        ):
            target = next(
                (
                    existing
                    for existing in kept
                    if existing.action == candidate.action
                    and self.identity_distance(candidate.centroid, existing) < 0.015
                    and abs(
                        existing.predicted_delta_viability
                        - candidate.predicted_delta_viability
                    )
                    < 0.4
                    and abs(
                        existing.predicted_resistance
                        - candidate.predicted_resistance
                    )
                    < 0.2
                ),
                None,
            )
            if target is None:
                kept.append(candidate)
                continue
            total = target.observations + candidate.observations
            target.predicted_delta_viability = (
                target.predicted_delta_viability * target.observations
                + candidate.predicted_delta_viability * candidate.observations
            ) / total
            target.predicted_resistance = (
                target.predicted_resistance * target.observations
                + candidate.predicted_resistance * candidate.observations
            ) / total
            target.observations = total
            target.outcome_history.extend(candidate.outcome_history)
            target.outcome_history = target.outcome_history[-20:]
            target.resistance_history.extend(candidate.resistance_history)
            target.resistance_history = target.resistance_history[-20:]
            target.initialize_scores()
            merges += 1
        self.candidates = kept
        self.total_merged += merges
        return merges

    def prune(self, candidate: ObjectCandidate) -> bool:
        if candidate not in self.candidates:
            return False
        self.candidates.remove(candidate)
        self.total_pruned += 1
        return True

    def nearest(
        self, features: Dict[str, float], action: Action
    ) -> Tuple[ObjectCandidate | None, float]:
        matching = [
            candidate
            for candidate in self.candidates
            if candidate.action == action and not candidate.obsolete
        ]
        if not matching:
            return None, math.inf
        scored = [
            (self.visual_distance(features, candidate), candidate)
            for candidate in matching
        ]
        distance, candidate = min(scored, key=lambda pair: pair[0])
        return candidate, distance

    def nearest_trace(self, trace: Trace) -> Tuple[ObjectCandidate | None, float]:
        matching = [
            candidate
            for candidate in self.candidates
            if candidate.action == trace.action and not candidate.obsolete
        ]
        if not matching:
            return None, math.inf
        scored = [
            (self.trace_distance(trace, candidate), candidate)
            for candidate in matching
        ]
        distance, candidate = min(scored, key=lambda pair: pair[0])
        return candidate, distance

    def predict(
        self,
        item: SensoryItem,
        action: Action,
        record_policy_influence: bool = False,
    ) -> Tuple[float, float, float]:
        return self._aggregate_prediction(
            item.features(),
            self.candidates,
            action,
            record_policy_influence=record_policy_influence,
        )

    def predict_identity(
        self,
        item: SensoryItem,
        record_policy_influence: bool = False,
    ) -> Tuple[float, float, float]:
        return self._aggregate_prediction(
            item.features(),
            self.candidates,
            None,
            record_policy_influence=record_policy_influence,
        )

    def predict_stabilized(
        self,
        item: SensoryItem,
        action: Action,
    ) -> Tuple[float, float, float]:
        active = [
            candidate
            for candidate in self.candidates
            if candidate.status == "active"
        ]
        return self._aggregate_prediction(
            item.features(),
            active,
            action,
        )

    def _aggregate_prediction(
        self,
        features: Dict[str, float],
        candidates: List[ObjectCandidate],
        action: Action | None,
        record_policy_influence: bool = False,
    ) -> Tuple[float, float, float]:
        candidates = [candidate for candidate in candidates if not candidate.obsolete]
        if not candidates:
            return 0.0, 0.0, 0.0
        distances = [
            (self.identity_distance(features, candidate), candidate)
            for candidate in candidates
        ]
        best_distance = min(distance for distance, _ in distances)
        nearby = [
            (distance, candidate)
            for distance, candidate in distances
            if distance <= best_distance + 0.02
        ]
        weighted = [
            (
                candidate.observations
                * max(0.03, candidate.objecthood_score)
                * (
                    candidate.policy_influence
                    if record_policy_influence
                    else 1.0
                )
                * (1.0 if action is None or candidate.action == action else 0.35),
                distance,
                candidate,
            )
            for distance, candidate in nearby
        ]
        total_weight = sum(weight for weight, _, _ in weighted)
        if total_weight <= 1e-12:
            return 0.0, 0.0, 0.0
        if record_policy_influence:
            self.policy_prediction_calls += 1
            for status in self.policy_influence_totals:
                self.policy_influence_totals[status] += (
                    sum(
                        weight
                        for weight, _, candidate in weighted
                        if candidate.status == status
                    )
                    / total_weight
                )
        delta = sum(
            weight * candidate.predicted_delta_viability
            for weight, _, candidate in weighted
        ) / total_weight
        resistance = sum(
            weight * candidate.predicted_resistance
            for weight, _, candidate in weighted
        ) / total_weight
        confidence = sum(
            weight * candidate.objecthood_score
            for weight, _, candidate in weighted
        ) / total_weight
        closeness = max(0.0, 1.0 - best_distance / 0.30)
        disagreement = min(
            1.0,
            (
                max(candidate.predicted_delta_viability for _, _, candidate in weighted)
                - min(candidate.predicted_delta_viability for _, _, candidate in weighted)
            )
            / 8.0,
        )
        return (
            delta,
            resistance,
            confidence * closeness * (1.0 - 0.5 * disagreement),
        )

    def _best_prediction(
        self,
        features: Dict[str, float],
        candidates: List[ObjectCandidate],
    ) -> Tuple[ObjectCandidate | None, float]:
        if not candidates:
            return None, math.inf
        scored = [
            (
                self.identity_distance(features, candidate)
                - 0.002 * candidate.objecthood_score,
                candidate,
            )
            for candidate in candidates
        ]
        adjusted_distance, candidate = min(scored, key=lambda pair: pair[0])
        return candidate, max(0.0, adjusted_distance)

    def identity_distance(
        self, features: Dict[str, float], candidate: ObjectCandidate
    ) -> float:
        total = 0.0
        weight_total = 0.0
        for name in RGB_FEATURES:
            weight = self.gate.weights[name]
            total += weight * abs(features[name] - candidate.centroid[name])
            weight_total += weight
        return total / max(weight_total, 1e-9)

    def visual_distance(
        self, features: Dict[str, float], candidate: ObjectCandidate
    ) -> float:
        total = 0.0
        weight_total = 0.0
        for name in VISUAL_FEATURES:
            weight = self.gate.weights[name]
            scale = 4.0 if name.startswith("rel_") else 1.0
            difference = abs(features[name] - candidate.centroid[name]) / scale
            total += weight * min(1.0, difference)
            weight_total += weight
        return total / max(weight_total, 1e-9)

    def trace_distance(self, trace: Trace, candidate: ObjectCandidate) -> float:
        visual = self.visual_distance(trace.features, candidate)
        viability = min(
            1.0,
            abs(trace.delta_viability - candidate.predicted_delta_viability) / 8.0,
        )
        resistance = abs(trace.resistance - candidate.predicted_resistance)
        return 0.55 * visual + 0.30 * viability + 0.15 * resistance


class Destabilizer:
    def __init__(self, failure_threshold: float = 2.5, split_after: int = 1) -> None:
        self.failure_threshold = failure_threshold
        self.split_after = split_after
        self.total_splits = 0
        self.total_destabilizations = 0
        self.total_confidence_updates = 0
        self.total_suppressions = 0
        self.total_obsolete_marked = 0
        self.conflict_logs: List[ConflictAttribution] = []
        self._next_log_id = 0
        self._targeted_evidence: Dict[
            tuple[int, str], List[Trace]
        ] = {}
        self._targeted_children: Dict[tuple[int, str], int] = {}
        self._provisional_families: Dict[int, ProvisionalFamily] = {}
        self.total_provisional_children = 0
        self.total_child_wins = 0
        self.total_child_validated = 0
        self.total_child_activated = 0
        self.total_child_pruned = 0
        self.total_parent_decays = 0
        self.total_merged_children = 0
        self.parent_survival_times: List[int] = []

    def inspect(
        self,
        candidate: ObjectCandidate,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
    ) -> bool:
        split = self.revise(
            candidate,
            trace,
            stabilizer,
            confidence_update=True,
            split=True,
            suppression=False,
            obsolete_marking=False,
        )
        return split is not None

    def revise(
        self,
        candidate: ObjectCandidate,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        *,
        confidence_update: bool,
        split: bool,
        suppression: bool,
        obsolete_marking: bool,
    ) -> ObjectCandidate | None:
        if not self.is_conflict(candidate, trace):
            candidate.failures = max(0, candidate.failures - 1)
            candidate.contradictions = max(0, candidate.contradictions - 1)
            return None

        candidate.failures += 1
        candidate.contradictions += 1
        self.total_destabilizations += 1
        if confidence_update:
            candidate.inhibit(0.65)
            self.total_confidence_updates += 1

        split_candidate = None
        if split and candidate.failures >= self.split_after:
            candidate.failures = 0
            split_candidate = stabilizer.create_candidate(trace)
            self.total_splits += 1

        if suppression:
            self._suppress(candidate, stabilizer, obsolete_marking)
        return split_candidate

    def is_conflict(self, candidate: ObjectCandidate, trace: Trace) -> bool:
        error = abs(trace.delta_viability - candidate.predicted_delta_viability)
        error += abs(trace.resistance - candidate.predicted_resistance) * 4.0
        return error > self.failure_threshold

    def log_baseline_conflict(
        self,
        candidate: ObjectCandidate,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        step: int,
        split_triggered: bool,
    ) -> None:
        viability_error = abs(
            trace.delta_viability - candidate.predicted_delta_viability
        )
        resistance_error = abs(
            trace.resistance - candidate.predicted_resistance
        )
        self._append_conflict_log(
            step,
            trace,
            candidate,
            stabilizer,
            viability_conflict=viability_error > self.failure_threshold,
            resistance_conflict=(
                resistance_error * 4.0 > self.failure_threshold
            ),
            evidence_count=1,
            threshold_passed=True,
            split_triggered=split_triggered,
            child_candidate_id=(
                stabilizer.candidates[-1].candidate_id
                if split_triggered
                else None
            ),
        )

    def targeted_split(
        self,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        *,
        step: int,
        threshold: str,
        evidence_window: int,
    ) -> tuple[ObjectCandidate | None, bool]:
        profile = CONFLICT_PROFILES[threshold]
        selected = self._select_targeted_conflict(trace, stabilizer, profile)
        if selected is None:
            return None, False
        (
            candidate,
            conflict_kind,
            viability_conflict,
            resistance_conflict,
            attribution_passed,
        ) = selected
        if not attribution_passed:
            self._append_conflict_log(
                step,
                trace,
                candidate,
                stabilizer,
                viability_conflict=viability_conflict,
                resistance_conflict=resistance_conflict,
                evidence_count=0,
                threshold_passed=False,
                split_triggered=False,
                child_candidate_id=None,
            )
            return None, False

        key = (candidate.candidate_id, conflict_kind)
        evidence = self._targeted_evidence.setdefault(key, [])
        if evidence and not self._compatible_evidence(
            evidence[-1], trace, profile["rgb"], conflict_kind
        ):
            evidence.clear()
        evidence.append(trace)
        evidence[:] = evidence[-evidence_window:]
        evidence_count = len(evidence)
        threshold_passed = evidence_count >= evidence_window
        self.total_destabilizations += 1
        candidate.contradictions += 1

        child = None
        child_id = self._targeted_children.get(key)
        if child_id is not None:
            child = stabilizer.candidate_by_id(child_id)
        split_triggered = False
        if threshold_passed and child is None:
            child = stabilizer.create_candidate(trace)
            child.parent_candidate_id = candidate.candidate_id
            child.provisional = True
            child.required_evidence = max(3, evidence_window)
            self._targeted_children[key] = child.candidate_id
            self.total_splits += 1
            split_triggered = True

        self._append_conflict_log(
            step,
            trace,
            candidate,
            stabilizer,
            viability_conflict=viability_conflict,
            resistance_conflict=resistance_conflict,
            evidence_count=evidence_count,
            threshold_passed=True,
            split_triggered=split_triggered,
            child_candidate_id=child.candidate_id if child else None,
        )
        return child, True

    def mature_targeted_child(
        self,
        child: ObjectCandidate,
        stabilizer: ObjecthoodStabilizer,
        *,
        parent_decay: str,
        suppression: bool,
    ) -> None:
        if (
            not child.provisional
            or child.observations < child.required_evidence
        ):
            return
        child.provisional = False
        parent_id = child.parent_candidate_id
        if parent_id is None:
            return
        parent = stabilizer.candidate_by_id(parent_id)
        if parent is None:
            return

        decay_applied = False
        factor = PARENT_DECAY_FACTORS[parent_decay]
        if not parent.parent_decay_applied and factor < 1.0:
            parent.inhibit(factor)
            parent.parent_decay_applied = True
            decay_applied = True

        suppression_applied = False
        if suppression:
            self._suppress(parent, stabilizer, obsolete_marking=False)
            suppression_applied = True

        for log in reversed(self.conflict_logs):
            if log.child_candidate_id != child.candidate_id:
                continue
            log.parent_decay_applied = decay_applied
            log.suppression_applied = suppression_applied
            break

    def provisional_split(
        self,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        *,
        step: int,
        selection_enabled: bool,
        selection_window: int,
        calibrated_activation: bool = False,
        validation_threshold: str = "medium",
        activation_threshold: str = "medium",
        activation_delay: int = 3,
        selection_margin: float = 0.08,
    ) -> tuple[ObjectCandidate | None, bool]:
        parent, visual_distance = self._nearest_root_candidate(
            trace, stabilizer
        )
        if parent is None or visual_distance > 0.20:
            return None, False

        family = self._provisional_families.get(parent.candidate_id)
        members = self._family_members(parent, family, stabilizer)
        for candidate in members:
            candidate.record_evidence(trace)
            if family is not None:
                family.evidence.setdefault(
                    candidate.candidate_id, BranchEvidence()
                ).record(candidate, trace)

        conflict = self.is_conflict(parent, trace)
        child = self._compatible_child(trace, members)
        if conflict and child is None:
            child = stabilizer.create_candidate(trace)
            child.parent_candidate_id = parent.candidate_id
            child.status = "provisional"
            child.required_evidence = selection_window
            child.created_step = step
            self.total_splits += 1
            self.total_destabilizations += 1
            self.total_provisional_children += 1
            if family is None:
                family = ProvisionalFamily(
                    parent_id=parent.candidate_id,
                    child_ids=[],
                    split_step=step,
                )
                self._provisional_families[parent.candidate_id] = family
                family.evidence[parent.candidate_id] = BranchEvidence()
            family.child_ids.append(child.candidate_id)
            family.evidence[child.candidate_id] = BranchEvidence()
            self._append_conflict_log(
                step,
                trace,
                parent,
                stabilizer,
                viability_conflict=abs(
                    trace.delta_viability
                    - parent.predicted_delta_viability
                )
                > self.failure_threshold,
                resistance_conflict=abs(
                    trace.resistance - parent.predicted_resistance
                )
                * 4.0
                > self.failure_threshold,
                evidence_count=1,
                threshold_passed=True,
                split_triggered=True,
                child_candidate_id=child.candidate_id,
            )
            return child, True

        if family is None:
            return parent, True

        family.window_observations += 1
        if calibrated_activation:
            self._advance_validated_children(
                family,
                members,
                trace,
                activation_threshold=activation_threshold,
                activation_delay=activation_delay,
            )
        preferred = min(
            members,
            key=lambda candidate: self._instant_prediction_error(
                candidate, trace
            ),
        )
        if family.window_observations >= selection_window:
            if selection_enabled:
                if calibrated_activation:
                    self._select_calibrated_family(
                        family,
                        stabilizer,
                        step=step,
                        margin=selection_margin,
                        validation_threshold=validation_threshold,
                    )
                else:
                    self._select_family(
                        family,
                        stabilizer,
                        step=step,
                        margin=selection_margin,
                    )
            else:
                self._promote_supported_children(
                    family, stabilizer, selection_window
                )
            family.window_observations = 0
            family.evidence = {
                candidate.candidate_id: BranchEvidence()
                for candidate in self._family_members(
                    parent, family, stabilizer
                )
            }
        if stabilizer.candidate_by_id(preferred.candidate_id) is None:
            preferred = parent
        return preferred, True

    def merge_redundant_children(
        self,
        family: ProvisionalFamily,
        stabilizer: ObjecthoodStabilizer,
    ) -> int:
        merges = 0
        child_ids = list(family.child_ids)
        for index, left_id in enumerate(child_ids):
            left = stabilizer.candidate_by_id(left_id)
            if left is None:
                continue
            for right_id in child_ids[index + 1:]:
                right = stabilizer.candidate_by_id(right_id)
                if right is None:
                    continue
                if (
                    left.action != right.action
                    or stabilizer.identity_distance(
                        left.centroid, right
                    )
                    >= 0.02
                    or abs(
                        left.predicted_delta_viability
                        - right.predicted_delta_viability
                    )
                    >= 0.5
                    or abs(
                        left.predicted_resistance
                        - right.predicted_resistance
                    )
                    >= 0.2
                ):
                    continue
                if right.predictive_usefulness > left.predictive_usefulness:
                    left, right = right, left
                left.observations += right.observations
                left.support_count += right.support_count
                left.conflict_count += right.conflict_count
                left.outcome_history.extend(right.outcome_history)
                left.outcome_history = left.outcome_history[-20:]
                left.resistance_history.extend(right.resistance_history)
                left.resistance_history = left.resistance_history[-20:]
                left.initialize_scores()
                stabilizer.prune(right)
                stabilizer.total_merged += 1
                family.child_ids.remove(right.candidate_id)
                family.evidence.pop(right.candidate_id, None)
                self.total_child_pruned += 1
                self.total_merged_children = getattr(
                    self, "total_merged_children", 0
                ) + 1
                merges += 1
        return merges

    def enforce_candidate_budget(
        self,
        stabilizer: ObjecthoodStabilizer,
        budget: int | None,
        step: int,
    ) -> int:
        if budget is None:
            return 0
        pruned = 0
        protected_parents = {
            family.parent_id
            for family in self._provisional_families.values()
            if family.parent_resolved_step is None
        }
        while len(stabilizer.candidates) > budget:
            removable = [
                candidate
                for candidate in stabilizer.candidates
                if candidate.status != "obsolete"
                and candidate.candidate_id not in protected_parents
            ]
            if len(removable) <= 1:
                break
            candidate = min(
                removable,
                key=lambda item: (
                    item.predictive_usefulness,
                    item.objecthood_score,
                ),
            )
            candidate.status = "obsolete"
            candidate.resolved_step = step
            stabilizer.prune(candidate)
            if candidate.parent_candidate_id is not None:
                self.total_child_pruned += 1
            pruned += 1
        return pruned

    def parent_survival_time(self, current_step: int) -> float:
        values = list(self.parent_survival_times)
        for family in self._provisional_families.values():
            if family.parent_resolved_step is None:
                values.append(max(0, current_step - family.split_step))
        return sum(values) / len(values) if values else 0.0

    @property
    def child_survival_rate(self) -> float:
        if not self.total_provisional_children:
            return 0.0
        survived = (
            self.total_provisional_children - self.total_child_pruned
        )
        return max(0.0, survived / self.total_provisional_children)

    @property
    def child_win_rate(self) -> float:
        if not self.total_provisional_children:
            return 0.0
        return self.total_child_wins / self.total_provisional_children

    @property
    def child_validation_rate(self) -> float:
        if not self.total_provisional_children:
            return 0.0
        return self.total_child_validated / self.total_provisional_children

    @property
    def child_activation_rate(self) -> float:
        if not self.total_provisional_children:
            return 0.0
        return self.total_child_activated / self.total_provisional_children

    def _advance_validated_children(
        self,
        family: ProvisionalFamily,
        members: List[ObjectCandidate],
        trace: Trace,
        *,
        activation_threshold: str,
        activation_delay: int,
    ) -> None:
        threshold = ACTIVATION_THRESHOLDS[activation_threshold]
        for child in members:
            if child.status != "validated":
                continue
            compatible = (
                child.action == trace.action
                and self._instant_prediction_error(child, trace) <= 0.25
            )
            child.validated_compatible_streak = (
                child.validated_compatible_streak + 1
                if compatible
                else 0
            )
            if (
                child.validated_compatible_streak < activation_delay
                or child.calibrated_activation_score < threshold
            ):
                continue
            child.status = "active"
            child.ever_activated = True
            child.resolved_step = trace.commit_step
            self.total_child_activated += 1

    def _select_calibrated_family(
        self,
        family: ProvisionalFamily,
        stabilizer: ObjecthoodStabilizer,
        *,
        step: int,
        margin: float,
        validation_threshold: str,
    ) -> None:
        parent = stabilizer.candidate_by_id(family.parent_id)
        if parent is None:
            return
        self.merge_redundant_children(family, stabilizer)
        children = [
            candidate
            for child_id in family.child_ids
            if (candidate := stabilizer.candidate_by_id(child_id))
            is not None
        ]
        if not children:
            return
        parent_evidence = family.evidence.get(
            parent.candidate_id, BranchEvidence()
        )
        parent_score = parent_evidence.score
        best_child = max(
            children,
            key=lambda candidate: family.evidence.get(
                candidate.candidate_id, BranchEvidence()
            ).score,
        )
        best_evidence = family.evidence.get(
            best_child.candidate_id, BranchEvidence()
        )
        best_score = best_evidence.score
        validation_cutoff = ACTIVATION_THRESHOLDS[
            validation_threshold
        ]

        if (
            best_child.status == "provisional"
            and best_score >= parent_score + margin
            and best_child.calibrated_activation_score
            >= validation_cutoff
            and best_evidence.support
            >= max(1, best_evidence.observations // 2)
        ):
            best_child.status = "validated"
            best_child.ever_validated = True
            best_child.validated_compatible_streak = 0
            self.total_child_validated += 1
            if not parent.parent_decay_applied:
                # A validated child has enough evidence to reduce the stale
                # parent's policy influence, but not enough to obsolete it.
                decay = max(
                    0.40,
                    0.75 - 0.35 * best_child.calibrated_activation_score,
                )
                parent.inhibit(decay)
                parent.parent_decay_applied = True
                self.total_parent_decays += 1
                for log in reversed(self.conflict_logs):
                    if log.child_candidate_id == best_child.candidate_id:
                        log.parent_decay_applied = True
                        break

        active_children = [
            child for child in children if child.status == "active"
        ]
        if active_children:
            winner = max(
                active_children,
                key=lambda candidate: family.evidence.get(
                    candidate.candidate_id, BranchEvidence()
                ).score,
            )
            winner_evidence = family.evidence.get(
                winner.candidate_id, BranchEvidence()
            )
            if (
                winner_evidence.score >= parent_score + margin
                and winner_evidence.support
                >= max(1, winner_evidence.observations // 2)
            ):
                winner.won_parent_competition = True
                winner.parent_candidate_id = None
                self.total_child_wins += 1
                parent.status = "obsolete"
                parent.resolved_step = step
                family.parent_resolved_step = step
                self.parent_survival_times.append(
                    max(0, step - family.split_step)
                )
                stabilizer.prune(parent)

        for child in list(children):
            evidence = family.evidence.get(
                child.candidate_id, BranchEvidence()
            )
            if child.status != "provisional":
                continue
            weak = evidence.score + margin < max(
                parent_score, best_score
            )
            unsupported = (
                evidence.observations > 0 and evidence.support == 0
            )
            if not weak and not unsupported:
                continue
            child.status = "obsolete"
            child.resolved_step = step
            stabilizer.prune(child)
            family.child_ids.remove(child.candidate_id)
            family.evidence.pop(child.candidate_id, None)
            self.total_child_pruned += 1

    def _select_family(
        self,
        family: ProvisionalFamily,
        stabilizer: ObjecthoodStabilizer,
        *,
        step: int,
        margin: float,
    ) -> None:
        parent = stabilizer.candidate_by_id(family.parent_id)
        if parent is None:
            return
        self.merge_redundant_children(family, stabilizer)
        children = [
            candidate
            for child_id in family.child_ids
            if (candidate := stabilizer.candidate_by_id(child_id))
            is not None
        ]
        if not children:
            return
        parent_score = family.evidence.get(
            parent.candidate_id, BranchEvidence()
        ).score
        best_child = max(
            children,
            key=lambda candidate: family.evidence.get(
                candidate.candidate_id, BranchEvidence()
            ).score,
        )
        best_score = family.evidence.get(
            best_child.candidate_id, BranchEvidence()
        ).score
        best_evidence = family.evidence.get(
            best_child.candidate_id, BranchEvidence()
        )

        if (
            best_score >= parent_score + margin
            and best_evidence.support
            >= max(1, best_evidence.observations // 2)
        ):
            best_child.status = "active"
            best_child.resolved_step = step
            if not best_child.won_parent_competition:
                best_child.won_parent_competition = True
                self.total_child_wins += 1
            if not parent.parent_decay_applied:
                parent.inhibit(0.75)
                parent.parent_decay_applied = True
                parent.resolved_step = step
                self.total_parent_decays += 1
                family.parent_resolved_step = step
                self.parent_survival_times.append(
                    max(0, step - family.split_step)
                )
                for log in reversed(self.conflict_logs):
                    if log.child_candidate_id == best_child.candidate_id:
                        log.parent_decay_applied = True
                        break

        for child in list(children):
            evidence = family.evidence.get(
                child.candidate_id, BranchEvidence()
            )
            score = evidence.score
            weak = (
                child is not best_child
                and score + margin < max(parent_score, best_score)
            )
            unsupported = (
                evidence.observations > 0 and evidence.support == 0
            )
            if not weak and not unsupported:
                continue
            child.status = "obsolete"
            child.resolved_step = step
            stabilizer.prune(child)
            family.child_ids.remove(child.candidate_id)
            family.evidence.pop(child.candidate_id, None)
            self.total_child_pruned += 1

    @staticmethod
    def _promote_supported_children(
        family: ProvisionalFamily,
        stabilizer: ObjecthoodStabilizer,
        selection_window: int,
    ) -> None:
        for child_id in family.child_ids:
            child = stabilizer.candidate_by_id(child_id)
            evidence = family.evidence.get(
                child_id, BranchEvidence()
            )
            if (
                child is not None
                and child.status == "provisional"
                and evidence.observations >= selection_window
                and evidence.support > 0
            ):
                child.status = "active"

    @staticmethod
    def _instant_prediction_error(
        candidate: ObjectCandidate, trace: Trace
    ) -> float:
        return (
            abs(
                trace.delta_viability
                - candidate.predicted_delta_viability
            )
            / 8.0
            + abs(trace.resistance - candidate.predicted_resistance)
        )

    @staticmethod
    def _compatible_child(
        trace: Trace,
        members: List[ObjectCandidate],
    ) -> ObjectCandidate | None:
        children = [
            candidate
            for candidate in members
            if candidate.parent_candidate_id is not None
            and candidate.action == trace.action
            and abs(
                candidate.predicted_delta_viability
                - trace.delta_viability
            )
            < 1.0
            and abs(
                candidate.predicted_resistance - trace.resistance
            )
            < 0.25
        ]
        return (
            min(
                children,
                key=lambda candidate: Destabilizer._instant_prediction_error(
                    candidate, trace
                ),
            )
            if children
            else None
        )

    @staticmethod
    def _family_members(
        parent: ObjectCandidate,
        family: ProvisionalFamily | None,
        stabilizer: ObjecthoodStabilizer,
    ) -> List[ObjectCandidate]:
        members = [parent]
        if family is None:
            return members
        members.extend(
            candidate
            for child_id in family.child_ids
            if (candidate := stabilizer.candidate_by_id(child_id))
            is not None
        )
        return members

    @staticmethod
    def _nearest_root_candidate(
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
    ) -> tuple[ObjectCandidate | None, float]:
        roots = [
            candidate
            for candidate in stabilizer.candidates
            if candidate.parent_candidate_id is None
            and candidate.status == "active"
            and candidate.action == trace.action
        ]
        if not roots:
            return None, math.inf
        scored = [
            (stabilizer.visual_distance(trace.features, candidate), candidate)
            for candidate in roots
        ]
        distance, candidate = min(scored, key=lambda pair: pair[0])
        return candidate, distance

    def _select_targeted_conflict(
        self,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        profile: Dict[str, float],
    ) -> tuple[ObjectCandidate, str, bool, bool, bool] | None:
        scored = []
        for candidate in stabilizer.candidates:
            if candidate.obsolete or candidate.provisional:
                continue
            action_distance = float(candidate.action != trace.action)
            if action_distance > 0.0:
                continue
            rgb_distance = stabilizer.identity_distance(
                trace.features, candidate
            )
            position_distance = self._position_distance(trace, candidate)
            if rgb_distance > 0.25 or position_distance > 0.30:
                continue

            viability_error = abs(
                trace.delta_viability - candidate.predicted_delta_viability
            )
            resistance_error = abs(
                trace.resistance - candidate.predicted_resistance
            )
            viability_conflict = (
                viability_error >= CONFLICT_PROFILES["low"]["viability"]
                and resistance_error <= 0.25
            )
            resistance_conflict = (
                resistance_error >= CONFLICT_PROFILES["low"]["resistance"]
                and viability_error <= 1.0
            )
            if not viability_conflict and not resistance_conflict:
                continue
            attribution_passed = (
                rgb_distance <= profile["rgb"]
                and position_distance <= profile["position"]
                and trace.temporal_eligibility >= profile["temporal"]
                and (
                    (
                        viability_conflict
                        and viability_error >= profile["viability"]
                    )
                    or (
                        resistance_conflict
                        and resistance_error >= profile["resistance"]
                    )
                )
            )
            viability_score = viability_error / profile["viability"]
            resistance_score = resistance_error / profile["resistance"]
            conflict_kind = (
                "viability"
                if viability_conflict
                and (not resistance_conflict or viability_score >= resistance_score)
                else "resistance"
            )
            conflict_score = max(viability_score, resistance_score)
            score = (
                conflict_score
                + 0.5 * trace.temporal_eligibility
                - rgb_distance
                - position_distance
            )
            scored.append(
                (
                    score,
                    candidate,
                    conflict_kind,
                    viability_conflict,
                    resistance_conflict,
                    attribution_passed,
                )
            )
        if not scored:
            return None
        (
            _,
            candidate,
            conflict_kind,
            viability_conflict,
            resistance_conflict,
            attribution_passed,
        ) = max(scored, key=lambda item: item[0])
        return (
            candidate,
            conflict_kind,
            viability_conflict,
            resistance_conflict,
            attribution_passed,
        )

    @staticmethod
    def _compatible_evidence(
        previous: Trace,
        current: Trace,
        rgb_threshold: float,
        conflict_kind: str,
    ) -> bool:
        if previous.action != current.action:
            return False
        rgb_distance = sum(
            abs(previous.features[name] - current.features[name])
            for name in RGB_FEATURES
        ) / len(RGB_FEATURES)
        if rgb_distance > rgb_threshold:
            return False
        if conflict_kind == "viability":
            return Destabilizer._outcome_bin(
                previous.delta_viability
            ) == Destabilizer._outcome_bin(current.delta_viability)
        return (previous.resistance >= 0.5) == (current.resistance >= 0.5)

    @staticmethod
    def _outcome_bin(value: float) -> int:
        if value >= 0.5:
            return 1
        if value <= -0.5:
            return -1
        return 0

    def _append_conflict_log(
        self,
        step: int,
        trace: Trace,
        candidate: ObjectCandidate,
        stabilizer: ObjecthoodStabilizer,
        *,
        viability_conflict: bool,
        resistance_conflict: bool,
        evidence_count: int,
        threshold_passed: bool,
        split_triggered: bool,
        child_candidate_id: int | None,
    ) -> None:
        self.conflict_logs.append(
            ConflictAttribution(
                log_id=self._next_log_id,
                step=step,
                trace_source_step=trace.source_step,
                candidate_id=candidate.candidate_id,
                child_candidate_id=child_candidate_id,
                rgb_distance=stabilizer.identity_distance(
                    trace.features, candidate
                ),
                relative_position_distance=self._position_distance(
                    trace, candidate
                ),
                action_distance=float(candidate.action != trace.action),
                candidate_action=candidate.action,
                trace_action=trace.action,
                predicted_delta_viability=candidate.predicted_delta_viability,
                predicted_resistance=candidate.predicted_resistance,
                observed_delta_viability=trace.delta_viability,
                observed_resistance=trace.resistance,
                temporal_eligibility=trace.temporal_eligibility,
                temporal_delay=trace.temporal_delay,
                viability_conflict=viability_conflict,
                resistance_conflict=resistance_conflict,
                evidence_count=evidence_count,
                threshold_passed=threshold_passed,
                split_triggered=split_triggered,
            )
        )
        self._next_log_id += 1

    @staticmethod
    def _position_distance(
        trace: Trace, candidate: ObjectCandidate
    ) -> float:
        return (
            abs(trace.features["rel_x"] - candidate.centroid["rel_x"])
            + abs(trace.features["rel_y"] - candidate.centroid["rel_y"])
        ) / 4.0

    def suppress_related(
        self,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        excluded: tuple[ObjectCandidate, ...] = (),
        obsolete_marking: bool = False,
    ) -> int:
        suppressed = 0
        for candidate in list(stabilizer.candidates):
            if candidate in excluded:
                continue
            if stabilizer.identity_distance(trace.features, candidate) > 0.025:
                continue
            if not self.is_conflict(candidate, trace):
                continue
            candidate.contradictions += 1
            self.total_destabilizations += 1
            self._suppress(candidate, stabilizer, obsolete_marking)
            suppressed += 1
        return suppressed

    def _suppress(
        self,
        candidate: ObjectCandidate,
        stabilizer: ObjecthoodStabilizer,
        obsolete_marking: bool,
    ) -> None:
        candidate.inhibit(0.55)
        self.total_suppressions += 1
        if candidate.contradictions < 2 and candidate.inhibition >= 0.20:
            return
        if obsolete_marking:
            candidate.obsolete = True
            candidate.initialize_scores()
            self.total_obsolete_marked += 1
        stabilizer.prune(candidate)

    def weaken_related(
        self,
        trace: Trace,
        stabilizer: ObjecthoodStabilizer,
        excluded: ObjectCandidate | None = None,
    ) -> int:
        weakened = 0
        for candidate in stabilizer.candidates:
            if candidate is excluded:
                continue
            if stabilizer.identity_distance(trace.features, candidate) > 0.025:
                continue
            if not self.is_conflict(candidate, trace):
                continue
            candidate.inhibit(0.78)
            weakened += 1
        self.total_destabilizations += weakened
        return weakened
