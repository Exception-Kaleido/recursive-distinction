from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .provisional_split_validation import ProvisionalSplitProbe
from .sharp_validation import MeanCI, SharpProbe
from .simulation import Simulation, SimulationConfig


CALIBRATED_MODES = (
    "split_only",
    "provisional_split_plus_selection",
    "calibrated_provisional_split",
    "calibrated_provisional_split_budgeted",
)
CALIBRATED_METRICS = (
    "adaptation_lag",
    "objecthood_f1",
    "false_stabilization_rate",
    "candidate_count",
    "candidate_churn",
    "useful_candidate_damage_rate",
    "parent_survival_time",
    "child_validation_rate",
    "child_activation_rate",
    "child_win_rate",
    "provisional_policy_influence",
    "validated_policy_influence",
    "active_policy_influence",
    "recovery_slope",
)


@dataclass
class CalibratedRunMetrics:
    adaptation_lag: float
    objecthood_f1: float
    false_stabilization_rate: float
    candidate_count: float
    candidate_churn: float
    useful_candidate_damage_rate: float
    parent_survival_time: float
    child_validation_rate: float
    child_activation_rate: float
    child_win_rate: float
    provisional_policy_influence: float
    validated_policy_influence: float
    active_policy_influence: float
    recovery_slope: float


@dataclass
class CalibratedCondition:
    label: str
    metrics: Dict[str, MeanCI]
    paired_vs_split_only: Dict[str, MeanCI]
    paired_vs_provisional_selection: Dict[str, MeanCI]


@dataclass
class CalibratedSuiteResult:
    conditions: List[CalibratedCondition]
    seeds: List[int]
    steps: int
    change_step: int
    validation_threshold: str
    activation_threshold: str
    activation_delay: int
    selection_window: int
    candidate_budget: int | None


class CalibratedActivationProbe(ProvisionalSplitProbe):
    def finalize_calibrated(
        self, simulation: Simulation
    ) -> CalibratedRunMetrics:
        provisional = self.finalize_provisional(simulation)
        sharp = SharpProbe.finalize(self)
        calls = max(1, simulation.stabilizer.policy_prediction_calls)
        influence = simulation.stabilizer.policy_influence_totals
        return CalibratedRunMetrics(
            adaptation_lag=provisional.adaptation_lag,
            objecthood_f1=provisional.objecthood_f1,
            false_stabilization_rate=provisional.false_stabilization_rate,
            candidate_count=provisional.candidate_count,
            candidate_churn=provisional.candidate_churn,
            useful_candidate_damage_rate=(
                provisional.useful_candidate_damage_rate
            ),
            parent_survival_time=provisional.parent_survival_time,
            child_validation_rate=(
                simulation.destabilizer.child_validation_rate
            ),
            child_activation_rate=(
                simulation.destabilizer.child_activation_rate
            ),
            child_win_rate=simulation.destabilizer.child_win_rate,
            provisional_policy_influence=(
                influence["provisional"] / calls
            ),
            validated_policy_influence=influence["validated"] / calls,
            active_policy_influence=influence["active"] / calls,
            recovery_slope=sharp.recovery_slope,
        )


def run_calibrated_activation_suite(
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
    validation_threshold: str = "medium",
    activation_threshold: str = "medium",
    activation_delay: int = 3,
    selection_window: int = 5,
    candidate_budget: int | None = 15,
) -> CalibratedSuiteResult:
    conditions = [
        ("split_only", "split_only"),
        (
            "provisional_split_plus_selection",
            "provisional_split_plus_selection",
        ),
        (
            "calibrated_provisional_split",
            "calibrated_provisional_split",
        ),
        (
            "calibrated_provisional_split_budgeted",
            "calibrated_provisional_split_budgeted",
        ),
    ]
    return _run_conditions(
        conditions,
        runs,
        seed_start,
        steps,
        change_step,
        validation_threshold,
        activation_threshold,
        activation_delay,
        selection_window,
        candidate_budget,
    )


def run_calibrated_activation_sweep(
    runs: int = 20,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
    validation_thresholds: Iterable[str] = ("low", "medium", "high"),
    activation_thresholds: Iterable[str] = ("low", "medium", "high"),
    activation_delays: Iterable[int] = (1, 3, 5),
    selection_window: int = 5,
) -> CalibratedSuiteResult:
    conditions = [
        ("split_only", "split_only"),
        (
            "provisional_split_plus_selection",
            "provisional_split_plus_selection",
        ),
    ]
    for validation in validation_thresholds:
        for activation in activation_thresholds:
            for delay in activation_delays:
                conditions.append(
                    (
                        f"{validation}/{activation}/k{delay}",
                        "calibrated_provisional_split",
                    )
                )
    result = _run_conditions(
        conditions,
        runs,
        seed_start,
        steps,
        change_step,
        "medium",
        "medium",
        3,
        selection_window,
        None,
        condition_parameters=True,
    )
    return result


def _run_conditions(
    conditions: List[tuple[str, str]],
    runs: int,
    seed_start: int,
    steps: int,
    change_step: int,
    validation_threshold: str,
    activation_threshold: str,
    activation_delay: int,
    selection_window: int,
    candidate_budget: int | None,
    condition_parameters: bool = False,
) -> CalibratedSuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = list(range(seed_start, seed_start + runs))
    raw = {
        label: {name: [] for name in CALIBRATED_METRICS}
        for label, _ in conditions
    }
    for label, mode in conditions:
        condition_validation = validation_threshold
        condition_activation = activation_threshold
        condition_delay = activation_delay
        if condition_parameters and "/" in label:
            condition_validation, condition_activation, delay = label.split(
                "/"
            )
            condition_delay = int(delay.removeprefix("k"))
        for seed in seeds:
            probe = CalibratedActivationProbe(
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
                    candidate_revision_submode=mode,
                    conflict_threshold="medium",
                    conflict_evidence_window=3,
                    parent_decay="none",
                    post_split_selection_window=selection_window,
                    candidate_budget=(
                        candidate_budget
                        if mode
                        == "calibrated_provisional_split_budgeted"
                        else None
                    ),
                    validation_threshold=condition_validation,
                    activation_threshold=condition_activation,
                    activation_delay=condition_delay,
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
            metrics = probe.finalize_calibrated(simulation)
            for name in CALIBRATED_METRICS:
                raw[label][name].append(getattr(metrics, name))

    split_baseline = raw["split_only"]
    provisional_baseline = raw["provisional_split_plus_selection"]
    summaries = []
    for label, _ in conditions:
        summaries.append(
            CalibratedCondition(
                label=label,
                metrics={
                    name: _mean_ci(values)
                    for name, values in raw[label].items()
                },
                paired_vs_split_only=_paired_difference(
                    raw[label], split_baseline
                ),
                paired_vs_provisional_selection=_paired_difference(
                    raw[label], provisional_baseline
                ),
            )
        )
    return CalibratedSuiteResult(
        summaries,
        seeds,
        steps,
        change_step,
        validation_threshold,
        activation_threshold,
        activation_delay,
        selection_window,
        candidate_budget,
    )


def format_calibrated_activation_summary(
    result: CalibratedSuiteResult,
) -> str:
    lines = [
        f"calibrated activation audit: {len(result.seeds)} paired seeds, "
        f"role reversal at {result.change_step}",
        (
            "condition | lag | F1 | false stabilization | count | churn | "
            "useful damage | parent survival | validation | activation | "
            "child win | policy P/V/A | recovery"
        ),
    ]
    for condition in result.conditions:
        metrics = condition.metrics
        lines.append(
            f"{condition.label:38s} | "
            f"{_ci(metrics['adaptation_lag'])} | "
            f"{_ci(metrics['objecthood_f1'])} | "
            f"{_ci(metrics['false_stabilization_rate'], True)} | "
            f"{_ci(metrics['candidate_count'])} | "
            f"{_ci(metrics['candidate_churn'])} | "
            f"{_ci(metrics['useful_candidate_damage_rate'], True)} | "
            f"{_ci(metrics['parent_survival_time'])} | "
            f"{_ci(metrics['child_validation_rate'], True)} | "
            f"{_ci(metrics['child_activation_rate'], True)} | "
            f"{_ci(metrics['child_win_rate'], True)} | "
            f"{_ci(metrics['provisional_policy_influence'], True)}/"
            f"{_ci(metrics['validated_policy_influence'], True)}/"
            f"{_ci(metrics['active_policy_influence'], True)} | "
            f"{_ci(metrics['recovery_slope'])}"
        )
        if condition.label not in (
            "split_only",
            "provisional_split_plus_selection",
        ):
            split = condition.paired_vs_split_only
            provisional = condition.paired_vs_provisional_selection
            lines.append(
                "  paired vs split_only: "
                f"lag={_ci(split['adaptation_lag'])}, "
                f"F1={_ci(split['objecthood_f1'])}, "
                f"false-stable="
                f"{_ci(split['false_stabilization_rate'], True)}, "
                f"count={_ci(split['candidate_count'])}, "
                f"damage="
                f"{_ci(split['useful_candidate_damage_rate'], True)}"
            )
            lines.append(
                "  paired vs provisional selection: "
                f"lag={_ci(provisional['adaptation_lag'])}, "
                f"F1={_ci(provisional['objecthood_f1'])}, "
                f"false-stable="
                f"{_ci(provisional['false_stabilization_rate'], True)}, "
                f"count={_ci(provisional['candidate_count'])}, "
                f"damage="
                f"{_ci(provisional['useful_candidate_damage_rate'], True)}"
            )
    return "\n".join(lines)


def _paired_difference(
    values: Dict[str, List[float]],
    baseline: Dict[str, List[float]],
) -> Dict[str, MeanCI]:
    return {
        name: _mean_ci(
            [
                value - reference
                for value, reference in zip(values[name], baseline[name])
                if math.isfinite(value) and math.isfinite(reference)
            ]
        )
        for name in CALIBRATED_METRICS
    }


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
