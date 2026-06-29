from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .objecthood import ConflictAttribution
from .sharp_validation import MeanCI, SharpProbe
from .simulation import Simulation, SimulationConfig
from .types import Action, SensoryItem, StepResult


TARGETED_SPLIT_MODES = (
    "split_only",
    "targeted_split",
    "targeted_split_plus_suppression",
)
TARGETING_METRICS = (
    "adaptation_lag",
    "obsolete_persistence",
    "objecthood_f1",
    "false_stabilization_rate",
    "candidate_count",
    "candidate_churn",
    "destabilization_precision",
    "target_precision",
    "target_recall",
    "wrong_target_rate",
    "useful_candidate_damage_rate",
    "obsolete_candidate_hit_rate",
)


@dataclass(frozen=True)
class HiddenTraceContext:
    expected_delta_viability: float
    expected_resistance: float
    after_change: bool


@dataclass
class EvaluatedConflictAttribution:
    mode: str
    seed: int
    log: ConflictAttribution
    trace_valid: bool
    actual_wrong: bool
    actually_obsolete: bool


@dataclass
class TargetedSplitRunMetrics:
    adaptation_lag: float
    obsolete_persistence: float
    objecthood_f1: float
    false_stabilization_rate: float
    candidate_count: float
    candidate_churn: float
    destabilization_precision: float
    target_precision: float
    target_recall: float
    wrong_target_rate: float
    useful_candidate_damage_rate: float
    obsolete_candidate_hit_rate: float


@dataclass
class TargetedSplitCondition:
    label: str
    metrics: Dict[str, MeanCI]
    paired_differences: Dict[str, MeanCI]


@dataclass
class TargetedSplitSuiteResult:
    conditions: List[TargetedSplitCondition]
    evaluated_logs: List[EvaluatedConflictAttribution]
    seeds: List[int]
    steps: int
    change_step: int


class ConflictTargetingProbe(SharpProbe):
    def __init__(
        self,
        change_step: int,
        steps: int,
        mode: str,
        seed: int,
    ) -> None:
        super().__init__(change_step, steps)
        self.mode = mode
        self.seed = seed
        self.hidden_contexts: Dict[int, HiddenTraceContext] = {}
        self.prechange_candidate_ids: set[int] = set()
        self.evaluated_logs: List[EvaluatedConflictAttribution] = []
        self.post_change_candidate_counts: List[int] = []
        self.created_at_change = 0
        self.merged_at_change = 0
        self.pruned_at_change = 0
        self._conflict_log_count = 0

    def on_change(self, step: int, simulation: Simulation) -> None:
        super().on_change(step, simulation)
        self.prechange_candidate_ids = {
            candidate.candidate_id
            for candidate in simulation.stabilizer.candidates
        }
        self.created_at_change = simulation.stabilizer.total_created
        self.merged_at_change = simulation.stabilizer.total_merged
        self.pruned_at_change = simulation.stabilizer.total_pruned

    def before_step(
        self,
        step: int,
        simulation: Simulation,
        patch: List[SensoryItem],
        action: Action,
    ) -> None:
        super().before_step(step, simulation, patch, action)
        world = simulation.world
        ax, ay = world.agent_position
        target = (ax + action[0], ay + action[1])
        entity = world.entities.get(target)
        blocked = not world.in_bounds(target) or (
            entity is not None and entity.kind == "wall"
        )
        effect = 0.0 if blocked else world.current_effect(
            entity.kind if entity is not None else None
        )
        self.hidden_contexts[step + 1] = HiddenTraceContext(
            expected_delta_viability=effect - 0.05,
            expected_resistance=float(blocked),
            after_change=step >= self.change_step,
        )

    def after_step(
        self,
        step: int,
        simulation: Simulation,
        result: StepResult,
        destabilization_delta: int,
    ) -> None:
        super().after_step(step, simulation, result, destabilization_delta)
        if step >= self.change_step:
            self.post_change_candidate_counts.append(
                len(simulation.stabilizer.candidates)
            )
        logs = simulation.destabilizer.conflict_logs
        for log in logs[self._conflict_log_count:]:
            context = self.hidden_contexts.get(log.trace_source_step)
            trace_valid = False
            actual_wrong = False
            actually_obsolete = False
            if context is not None:
                trace_valid = (
                    abs(
                        log.observed_delta_viability
                        - context.expected_delta_viability
                    )
                    <= 0.75
                    and abs(
                        log.observed_resistance
                        - context.expected_resistance
                    )
                    <= 0.25
                )
                actual_wrong = trace_valid and (
                    abs(
                        log.predicted_delta_viability
                        - context.expected_delta_viability
                    )
                    >= 2.5
                    or abs(
                        log.predicted_resistance
                        - context.expected_resistance
                    )
                    >= 0.5
                )
                actually_obsolete = (
                    actual_wrong
                    and context.after_change
                    and log.candidate_id in self.prechange_candidate_ids
                )
            self.evaluated_logs.append(
                EvaluatedConflictAttribution(
                    mode=self.mode,
                    seed=self.seed,
                    log=log,
                    trace_valid=trace_valid,
                    actual_wrong=actual_wrong,
                    actually_obsolete=actually_obsolete,
                )
            )
        self._conflict_log_count = len(logs)

    def finalize_targeting(
        self, simulation: Simulation
    ) -> TargetedSplitRunMetrics:
        sharp = super().finalize()
        split_logs = [
            item for item in self.evaluated_logs if item.log.split_triggered
        ]
        correct_splits = [item for item in split_logs if item.actual_wrong]
        wrong_splits = [item for item in split_logs if not item.actual_wrong]
        wrong_candidate_ids = {
            item.log.candidate_id
            for item in self.evaluated_logs
            if item.actual_wrong
        }
        hit_candidate_ids = {
            item.log.candidate_id
            for item in correct_splits
        }
        obsolete_candidate_ids = {
            item.log.candidate_id
            for item in self.evaluated_logs
            if item.actually_obsolete
        }
        obsolete_hit_ids = {
            item.log.candidate_id
            for item in correct_splits
            if item.actually_obsolete
        }
        damage_logs = [
            item
            for item in self.evaluated_logs
            if item.log.parent_decay_applied or item.log.suppression_applied
            if item.trace_valid
        ]
        useful_damage = [
            item for item in damage_logs if not item.actual_wrong
        ]
        churn = (
            simulation.stabilizer.total_created
            - self.created_at_change
            + simulation.stabilizer.total_merged
            - self.merged_at_change
            + simulation.stabilizer.total_pruned
            - self.pruned_at_change
        )
        return TargetedSplitRunMetrics(
            adaptation_lag=sharp.adaptation_lag,
            obsolete_persistence=sharp.obsolete_persistence,
            objecthood_f1=sharp.objecthood_f1,
            false_stabilization_rate=sharp.false_stabilization_rate,
            candidate_count=(
                statistics.fmean(self.post_change_candidate_counts)
                if self.post_change_candidate_counts
                else 0.0
            ),
            candidate_churn=float(max(0, churn)),
            destabilization_precision=sharp.destabilization_precision,
            target_precision=_optional_rate(
                len(correct_splits), len(split_logs)
            ),
            target_recall=_rate(
                len(hit_candidate_ids), len(wrong_candidate_ids)
            ),
            wrong_target_rate=_optional_rate(
                len(wrong_splits), len(split_logs)
            ),
            useful_candidate_damage_rate=_rate(
                len(useful_damage), len(damage_logs)
            ),
            obsolete_candidate_hit_rate=_rate(
                len(obsolete_hit_ids), len(obsolete_candidate_ids)
            ),
        )


def run_targeted_split_suite(
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
    conflict_threshold: str = "medium",
    evidence_window: int = 3,
    parent_decay: str = "slow",
) -> TargetedSplitSuiteResult:
    conditions = [
        ("split_only", "split_only", "medium", 1, "none"),
        (
            "targeted_split",
            "targeted_split",
            conflict_threshold,
            evidence_window,
            parent_decay,
        ),
        (
            "targeted_split_plus_suppression",
            "targeted_split_plus_suppression",
            conflict_threshold,
            evidence_window,
            parent_decay,
        ),
    ]
    return _run_conditions(
        conditions, runs, seed_start, steps, change_step
    )


def run_targeted_split_sweep(
    runs: int = 20,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
    thresholds: Iterable[str] = ("low", "medium", "high"),
    evidence_windows: Iterable[int] = (1, 3, 5),
    parent_decays: Iterable[str] = ("none", "slow", "fast"),
) -> TargetedSplitSuiteResult:
    conditions = [("split_only", "split_only", "medium", 1, "none")]
    for threshold in thresholds:
        for evidence_window in evidence_windows:
            for parent_decay in parent_decays:
                label = f"{threshold}/w{evidence_window}/{parent_decay}"
                conditions.append(
                    (
                        label,
                        "targeted_split",
                        threshold,
                        evidence_window,
                        parent_decay,
                    )
                )
    return _run_conditions(
        conditions, runs, seed_start, steps, change_step
    )


def _run_conditions(
    conditions: List[tuple[str, str, str, int, str]],
    runs: int,
    seed_start: int,
    steps: int,
    change_step: int,
) -> TargetedSplitSuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = list(range(seed_start, seed_start + runs))
    raw = {
        label: {name: [] for name in TARGETING_METRICS}
        for label, _, _, _, _ in conditions
    }
    evaluated_logs: List[EvaluatedConflictAttribution] = []
    for label, submode, threshold, evidence_window, parent_decay in conditions:
        for seed in seeds:
            probe = ConflictTargetingProbe(
                change_step, steps, label, seed
            )
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    feature_reentry_enabled=False,
                    candidate_formation_enabled=True,
                    candidate_revision_enabled=True,
                    candidate_revision_submode=submode,
                    conflict_threshold=threshold,
                    conflict_evidence_window=evidence_window,
                    parent_decay=parent_decay,
                    policy_reentry_enabled=False,
                    environment_change_step=change_step,
                    role_reversal=True,
                    ambiguous_rgb=True,
                    delayed_outcomes=True,
                    width=10,
                    height=10,
                    counts={
                        "wall": 8,
                        "food": 18,
                        "decoy": 18,
                        "hazard": 10,
                    },
                    exploration=0.25,
                ),
                probe=probe,
            )
            simulation.run()
            run_metrics = probe.finalize_targeting(simulation)
            evaluated_logs.extend(probe.evaluated_logs)
            for name in TARGETING_METRICS:
                raw[label][name].append(getattr(run_metrics, name))

    baseline = raw["split_only"]
    summary_conditions = []
    for label, _, _, _, _ in conditions:
        summary_conditions.append(
            TargetedSplitCondition(
                label=label,
                metrics={
                    name: _mean_ci(values)
                    for name, values in raw[label].items()
                },
                paired_differences={
                    name: _mean_ci(
                        [
                            value - baseline_value
                            for value, baseline_value in zip(
                                raw[label][name], baseline[name]
                            )
                            if math.isfinite(value)
                            and math.isfinite(baseline_value)
                        ]
                    )
                    for name in TARGETING_METRICS
                },
            )
        )
    return TargetedSplitSuiteResult(
        summary_conditions,
        evaluated_logs,
        seeds,
        steps,
        change_step,
    )


def format_targeted_split_summary(result: TargetedSplitSuiteResult) -> str:
    lines = [
        f"targeted split audit: {len(result.seeds)} paired seeds, "
        f"role reversal at {result.change_step}",
        (
            "condition | lag | obsolete | F1 | false stabilization | count | "
            "churn | destab precision | target precision | target recall | "
            "wrong target | useful damage | obsolete hit"
        ),
    ]
    for condition in result.conditions:
        metrics = condition.metrics
        lines.append(
            f"{condition.label:34s} | "
            f"{_ci(metrics['adaptation_lag'])} | "
            f"{_ci(metrics['obsolete_persistence'], True)} | "
            f"{_ci(metrics['objecthood_f1'])} | "
            f"{_ci(metrics['false_stabilization_rate'], True)} | "
            f"{_ci(metrics['candidate_count'])} | "
            f"{_ci(metrics['candidate_churn'])} | "
            f"{_ci(metrics['destabilization_precision'], True)} | "
            f"{_ci(metrics['target_precision'], True)} | "
            f"{_ci(metrics['target_recall'], True)} | "
            f"{_ci(metrics['wrong_target_rate'], True)} | "
            f"{_ci(metrics['useful_candidate_damage_rate'], True)} | "
            f"{_ci(metrics['obsolete_candidate_hit_rate'], True)}"
        )
        if condition.label != "split_only":
            differences = condition.paired_differences
            lines.append(
                "  paired vs split_only: "
                f"lag={_ci(differences['adaptation_lag'])}, "
                f"obsolete={_ci(differences['obsolete_persistence'], True)}, "
                f"F1={_ci(differences['objecthood_f1'])}, "
                f"false-stable="
                f"{_ci(differences['false_stabilization_rate'], True)}, "
                f"count={_ci(differences['candidate_count'])}, "
                f"churn={_ci(differences['candidate_churn'])}, "
                f"wrong-target="
                f"{_ci(differences['wrong_target_rate'], True)}"
            )
    return "\n".join(lines)


def write_conflict_attribution_logs(
    result: TargetedSplitSuiteResult, output: str
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in result.evaluated_logs:
            payload = {
                "mode": item.mode,
                "seed": item.seed,
                **asdict(item.log),
                "evaluator_trace_valid": item.trace_valid,
                "evaluator_actual_wrong": item.actual_wrong,
                "evaluator_actually_obsolete": item.actually_obsolete,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _mean_ci(values: List[float]) -> MeanCI:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return MeanCI(0.0, 0.0)
    mean = statistics.fmean(finite)
    if len(finite) < 2:
        return MeanCI(mean, 0.0)
    return MeanCI(
        mean,
        1.96 * statistics.stdev(finite) / math.sqrt(len(finite)),
    )


def _ci(value: MeanCI, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{value.mean * scale:.3f}+/-{value.ci95 * scale:.3f}{suffix}"
