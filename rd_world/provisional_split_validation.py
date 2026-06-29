from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List

from .sharp_validation import MeanCI, SharpProbe
from .simulation import Simulation, SimulationConfig
from .targeted_split_validation import ConflictTargetingProbe


PROVISIONAL_SPLIT_MODES = (
    "split_only",
    "targeted_split",
    "provisional_split",
    "provisional_split_plus_selection",
    "provisional_split_budgeted",
)
PROVISIONAL_SPLIT_METRICS = (
    "adaptation_lag",
    "obsolete_persistence",
    "objecthood_f1",
    "false_stabilization_rate",
    "candidate_count",
    "candidate_churn",
    "target_precision",
    "target_recall",
    "wrong_target_rate",
    "useful_candidate_damage_rate",
    "parent_survival_time",
    "child_survival_rate",
    "child_win_rate",
    "recovery_slope",
)


@dataclass
class ProvisionalSplitRunMetrics:
    adaptation_lag: float
    obsolete_persistence: float
    objecthood_f1: float
    false_stabilization_rate: float
    candidate_count: float
    candidate_churn: float
    target_precision: float
    target_recall: float
    wrong_target_rate: float
    useful_candidate_damage_rate: float
    parent_survival_time: float
    child_survival_rate: float
    child_win_rate: float
    recovery_slope: float


@dataclass
class ProvisionalSplitCondition:
    mode: str
    metrics: Dict[str, MeanCI]
    paired_differences: Dict[str, MeanCI]


@dataclass
class ProvisionalSplitSuiteResult:
    conditions: List[ProvisionalSplitCondition]
    contrasts: Dict[str, Dict[str, MeanCI]]
    seeds: List[int]
    steps: int
    change_step: int
    selection_window: int
    candidate_budget: int | None


class ProvisionalSplitProbe(ConflictTargetingProbe):
    def finalize_provisional(
        self, simulation: Simulation
    ) -> ProvisionalSplitRunMetrics:
        targeting = self.finalize_targeting(simulation)
        sharp = SharpProbe.finalize(self)
        return ProvisionalSplitRunMetrics(
            adaptation_lag=targeting.adaptation_lag,
            obsolete_persistence=targeting.obsolete_persistence,
            objecthood_f1=targeting.objecthood_f1,
            false_stabilization_rate=targeting.false_stabilization_rate,
            candidate_count=targeting.candidate_count,
            candidate_churn=targeting.candidate_churn,
            target_precision=targeting.target_precision,
            target_recall=targeting.target_recall,
            wrong_target_rate=targeting.wrong_target_rate,
            useful_candidate_damage_rate=(
                targeting.useful_candidate_damage_rate
            ),
            parent_survival_time=simulation.destabilizer.parent_survival_time(
                simulation.config.steps - 1
            ),
            child_survival_rate=(
                simulation.destabilizer.child_survival_rate
            ),
            child_win_rate=simulation.destabilizer.child_win_rate,
            recovery_slope=sharp.recovery_slope,
        )


def run_provisional_split_suite(
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
    selection_window: int = 5,
    candidate_budget: int | None = 15,
) -> ProvisionalSplitSuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if selection_window not in (3, 5, 10):
        raise ValueError("selection_window must be 3, 5, or 10")
    if candidate_budget not in (None, 10, 15, 20):
        raise ValueError("candidate_budget must be None, 10, 15, or 20")

    seeds = list(range(seed_start, seed_start + runs))
    raw = {
        mode: {name: [] for name in PROVISIONAL_SPLIT_METRICS}
        for mode in PROVISIONAL_SPLIT_MODES
    }
    for mode in PROVISIONAL_SPLIT_MODES:
        for seed in seeds:
            probe = ProvisionalSplitProbe(change_step, steps, mode, seed)
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    feature_reentry_enabled=False,
                    candidate_formation_enabled=True,
                    candidate_revision_enabled=True,
                    candidate_revision_submode=mode,
                    conflict_threshold="medium",
                    conflict_evidence_window=3,
                    parent_decay="none",
                    post_split_selection_window=selection_window,
                    candidate_budget=(
                        candidate_budget
                        if mode == "provisional_split_budgeted"
                        else None
                    ),
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
            metrics = probe.finalize_provisional(simulation)
            for name in PROVISIONAL_SPLIT_METRICS:
                raw[mode][name].append(getattr(metrics, name))

    baseline = raw["split_only"]
    conditions = []
    for mode in PROVISIONAL_SPLIT_MODES:
        conditions.append(
            ProvisionalSplitCondition(
                mode=mode,
                metrics={
                    name: _mean_ci(values)
                    for name, values in raw[mode].items()
                },
                paired_differences={
                    name: _mean_ci(
                        [
                            value - baseline_value
                            for value, baseline_value in zip(
                                raw[mode][name], baseline[name]
                            )
                            if math.isfinite(value)
                            and math.isfinite(baseline_value)
                        ]
                    )
                    for name in PROVISIONAL_SPLIT_METRICS
                },
            )
        )
    contrast_pairs = {
        "selection_vs_provisional": (
            "provisional_split_plus_selection",
            "provisional_split",
        ),
        "budget_vs_selection": (
            "provisional_split_budgeted",
            "provisional_split_plus_selection",
        ),
        "selection_vs_targeted": (
            "provisional_split_plus_selection",
            "targeted_split",
        ),
        "budget_vs_targeted": (
            "provisional_split_budgeted",
            "targeted_split",
        ),
    }
    contrasts = {
        label: {
            name: _mean_ci(
                [
                    value - reference
                    for value, reference in zip(
                        raw[left][name], raw[right][name]
                    )
                    if math.isfinite(value) and math.isfinite(reference)
                ]
            )
            for name in PROVISIONAL_SPLIT_METRICS
        }
        for label, (left, right) in contrast_pairs.items()
    }
    return ProvisionalSplitSuiteResult(
        conditions,
        contrasts,
        seeds,
        steps,
        change_step,
        selection_window,
        candidate_budget,
    )


def format_provisional_split_summary(
    result: ProvisionalSplitSuiteResult,
) -> str:
    lines = [
        f"provisional split audit: {len(result.seeds)} paired seeds, "
        f"role reversal at {result.change_step}, "
        f"selection window {result.selection_window}, "
        f"budget {result.candidate_budget}",
        (
            "mode | lag | obsolete | F1 | false stabilization | count | churn | "
            "target precision | target recall | wrong target | useful damage | "
            "parent survival | child survival | child win | recovery slope"
        ),
    ]
    for condition in result.conditions:
        metrics = condition.metrics
        lines.append(
            f"{condition.mode:34s} | "
            f"{_ci(metrics['adaptation_lag'])} | "
            f"{_ci(metrics['obsolete_persistence'], True)} | "
            f"{_ci(metrics['objecthood_f1'])} | "
            f"{_ci(metrics['false_stabilization_rate'], True)} | "
            f"{_ci(metrics['candidate_count'])} | "
            f"{_ci(metrics['candidate_churn'])} | "
            f"{_ci(metrics['target_precision'], True)} | "
            f"{_ci(metrics['target_recall'], True)} | "
            f"{_ci(metrics['wrong_target_rate'], True)} | "
            f"{_ci(metrics['useful_candidate_damage_rate'], True)} | "
            f"{_ci(metrics['parent_survival_time'])} | "
            f"{_ci(metrics['child_survival_rate'], True)} | "
            f"{_ci(metrics['child_win_rate'], True)} | "
            f"{_ci(metrics['recovery_slope'])}"
        )
        if condition.mode != "split_only":
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
                f"{_ci(differences['wrong_target_rate'], True)}, "
                f"useful-damage="
                f"{_ci(differences['useful_candidate_damage_rate'], True)}, "
                f"recovery={_ci(differences['recovery_slope'])}"
            )
    lines.append("incremental paired contrasts:")
    for label, metrics in result.contrasts.items():
        lines.append(
            f"  {label}: "
            f"lag={_ci(metrics['adaptation_lag'])}, "
            f"false-stable="
            f"{_ci(metrics['false_stabilization_rate'], True)}, "
            f"count={_ci(metrics['candidate_count'])}, "
            f"F1={_ci(metrics['objecthood_f1'])}, "
            f"useful-damage="
            f"{_ci(metrics['useful_candidate_damage_rate'], True)}, "
            f"recovery={_ci(metrics['recovery_slope'])}"
        )
    return "\n".join(lines)


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
