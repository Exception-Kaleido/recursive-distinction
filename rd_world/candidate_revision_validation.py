from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List

from .sharp_validation import MeanCI, SharpProbe
from .simulation import Simulation, SimulationConfig
from .types import StepResult


CANDIDATE_REVISION_MODES = (
    "no_candidate_revision",
    "confidence_only",
    "split_only",
    "suppression_only",
    "split_plus_suppression",
    "full_candidate_revision",
)
CANDIDATE_REVISION_METRICS = (
    "adaptation_lag",
    "obsolete_persistence",
    "objecthood_f1",
    "false_stabilization_rate",
    "destabilization_precision",
    "candidate_count",
    "candidate_churn",
    "recovery_slope",
)


@dataclass
class CandidateRevisionRunMetrics:
    adaptation_lag: float
    obsolete_persistence: float
    objecthood_f1: float
    false_stabilization_rate: float
    destabilization_precision: float
    candidate_count: float
    candidate_churn: float
    recovery_slope: float


@dataclass
class CandidateRevisionCondition:
    mode: str
    metrics: Dict[str, MeanCI]
    paired_differences: Dict[str, MeanCI]


@dataclass
class CandidateRevisionSuiteResult:
    conditions: List[CandidateRevisionCondition]
    contrasts: Dict[str, Dict[str, MeanCI]]
    seeds: List[int]
    steps: int
    change_step: int


class CandidateRevisionProbe(SharpProbe):
    def __init__(self, change_step: int, steps: int) -> None:
        super().__init__(change_step, steps)
        self.post_change_candidate_counts: List[int] = []
        self.created_at_change = 0
        self.merged_at_change = 0
        self.pruned_at_change = 0

    def on_change(self, step: int, simulation: Simulation) -> None:
        super().on_change(step, simulation)
        self.created_at_change = simulation.stabilizer.total_created
        self.merged_at_change = simulation.stabilizer.total_merged
        self.pruned_at_change = simulation.stabilizer.total_pruned

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

    def finalize_revision(
        self, simulation: Simulation
    ) -> CandidateRevisionRunMetrics:
        sharp = super().finalize()
        churn = (
            simulation.stabilizer.total_created
            - self.created_at_change
            + simulation.stabilizer.total_merged
            - self.merged_at_change
            + simulation.stabilizer.total_pruned
            - self.pruned_at_change
        )
        return CandidateRevisionRunMetrics(
            adaptation_lag=sharp.adaptation_lag,
            obsolete_persistence=sharp.obsolete_persistence,
            objecthood_f1=sharp.objecthood_f1,
            false_stabilization_rate=sharp.false_stabilization_rate,
            destabilization_precision=sharp.destabilization_precision,
            candidate_count=(
                statistics.fmean(self.post_change_candidate_counts)
                if self.post_change_candidate_counts
                else 0.0
            ),
            candidate_churn=float(max(0, churn)),
            recovery_slope=sharp.recovery_slope,
        )


def run_candidate_revision_suite(
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
) -> CandidateRevisionSuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = list(range(seed_start, seed_start + runs))
    raw: Dict[str, Dict[str, List[float]]] = {
        mode: {name: [] for name in CANDIDATE_REVISION_METRICS}
        for mode in CANDIDATE_REVISION_MODES
    }
    for mode in CANDIDATE_REVISION_MODES:
        revision_enabled = mode != "no_candidate_revision"
        submode = mode if revision_enabled else None
        for seed in seeds:
            probe = CandidateRevisionProbe(change_step, steps)
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    feature_reentry_enabled=False,
                    candidate_formation_enabled=True,
                    candidate_revision_enabled=revision_enabled,
                    candidate_revision_submode=submode,
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
            run_metrics = probe.finalize_revision(simulation)
            for name in CANDIDATE_REVISION_METRICS:
                raw[mode][name].append(getattr(run_metrics, name))

    baseline = raw["no_candidate_revision"]
    conditions = []
    for mode in CANDIDATE_REVISION_MODES:
        conditions.append(
            CandidateRevisionCondition(
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
                        ]
                    )
                    for name in CANDIDATE_REVISION_METRICS
                },
            )
        )
    contrast_pairs = {
        "split_vs_formation": ("split_only", "no_candidate_revision"),
        "suppression_vs_formation": (
            "suppression_only",
            "no_candidate_revision",
        ),
        "suppression_added_to_split": (
            "split_plus_suppression",
            "split_only",
        ),
        "full_added_to_split_suppression": (
            "full_candidate_revision",
            "split_plus_suppression",
        ),
    }
    contrasts = {
        label: {
            name: _mean_ci(
                [
                    value - reference_value
                    for value, reference_value in zip(
                        raw[left][name], raw[right][name]
                    )
                ]
            )
            for name in CANDIDATE_REVISION_METRICS
        }
        for label, (left, right) in contrast_pairs.items()
    }
    return CandidateRevisionSuiteResult(
        conditions, contrasts, seeds, steps, change_step
    )


def format_candidate_revision_summary(
    result: CandidateRevisionSuiteResult,
) -> str:
    lines = [
        f"candidate revision audit: {len(result.seeds)} paired seeds, "
        f"role reversal at {result.change_step}",
        (
            "mode | adaptation lag | obsolete persistence | objecthood F1 | "
            "false stabilization | destabilization precision | candidate count | "
            "candidate churn | recovery slope"
        ),
    ]
    for condition in result.conditions:
        metrics = condition.metrics
        lines.append(
            f"{condition.mode:24s} | "
            f"{_ci(metrics['adaptation_lag'])} | "
            f"{_ci(metrics['obsolete_persistence'], True)} | "
            f"{_ci(metrics['objecthood_f1'])} | "
            f"{_ci(metrics['false_stabilization_rate'], True)} | "
            f"{_ci(metrics['destabilization_precision'], True)} | "
            f"{_ci(metrics['candidate_count'])} | "
            f"{_ci(metrics['candidate_churn'])} | "
            f"{_ci(metrics['recovery_slope'])}"
        )
        if condition.mode != "no_candidate_revision":
            differences = condition.paired_differences
            lines.append(
                "  paired vs no_candidate_revision: "
                f"lag={_ci(differences['adaptation_lag'])}, "
                f"obsolete={_ci(differences['obsolete_persistence'], True)}, "
                f"F1={_ci(differences['objecthood_f1'])}, "
                f"false-stable="
                f"{_ci(differences['false_stabilization_rate'], True)}, "
                f"destab-precision="
                f"{_ci(differences['destabilization_precision'], True)}, "
                f"count={_ci(differences['candidate_count'])}, "
                f"churn={_ci(differences['candidate_churn'])}, "
                f"recovery={_ci(differences['recovery_slope'])}"
            )
    lines.append("incremental paired contrasts:")
    for label, metrics in result.contrasts.items():
        lines.append(
            f"  {label}: "
            f"lag={_ci(metrics['adaptation_lag'])}, "
            f"obsolete={_ci(metrics['obsolete_persistence'], True)}, "
            f"F1={_ci(metrics['objecthood_f1'])}, "
            f"false-stable={_ci(metrics['false_stabilization_rate'], True)}, "
            f"count={_ci(metrics['candidate_count'])}, "
            f"churn={_ci(metrics['candidate_churn'])}"
        )
    return "\n".join(lines)


def _mean_ci(values: List[float]) -> MeanCI:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return MeanCI(mean, 0.0)
    return MeanCI(
        mean,
        1.96 * statistics.stdev(values) / math.sqrt(len(values)),
    )


def _ci(value: MeanCI, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{value.mean * scale:.3f}+/-{value.ci95 * scale:.3f}{suffix}"
