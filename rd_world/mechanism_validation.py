from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .sharp_validation import MeanCI, SharpProbe
from .simulation import MECHANISM_MODES, Simulation, SimulationConfig
from .types import Action, SensoryItem, StepResult


AUDIT_METRICS = (
    "feature_weight_change",
    "distinction_change_rate",
    "candidate_revision_count",
    "policy_distribution_change",
    "adaptation_lag",
    "obsolete_persistence",
    "objecthood_f1",
    "recovery_slope",
)


@dataclass
class MechanismRunMetrics:
    feature_weight_change: float
    distinction_change_rate: float
    candidate_revision_count: float
    policy_distribution_change: float
    adaptation_lag: float
    obsolete_persistence: float
    objecthood_f1: float
    recovery_slope: float


@dataclass
class MechanismCondition:
    mode: str
    metrics: Dict[str, MeanCI]
    paired_differences: Dict[str, MeanCI]


@dataclass
class MechanismSuiteResult:
    conditions: List[MechanismCondition]
    seeds: List[int]
    steps: int
    change_step: int


class MechanismAuditProbe(SharpProbe):
    def __init__(self, change_step: int, steps: int, window: int = 50) -> None:
        super().__init__(change_step, steps)
        self.window = window
        self.weights_at_change: Dict[str, float] = {}
        self.actions_pre: Counter[Action] = Counter()
        self.actions_post: Counter[Action] = Counter()
        self.selected_pre: List[frozenset[str]] = []
        self.selected_post: List[frozenset[str]] = []
        self._trace_count = 0
        self.revisions_at_change = 0

    def on_change(self, step: int, simulation: Simulation) -> None:
        super().on_change(step, simulation)
        self.weights_at_change = simulation.gate.weights.copy()
        self.revisions_at_change = self._revision_total(simulation)

    def before_step(
        self,
        step: int,
        simulation: Simulation,
        patch: List[SensoryItem],
        action: Action,
    ) -> None:
        super().before_step(step, simulation, patch, action)
        if self.change_step - self.window <= step < self.change_step:
            self.actions_pre[action] += 1
        elif self.change_step <= step < self.change_step + self.window:
            self.actions_post[action] += 1

    def after_step(
        self,
        step: int,
        simulation: Simulation,
        result: StepResult,
        destabilization_delta: int,
    ) -> None:
        super().after_step(step, simulation, result, destabilization_delta)
        traces = simulation.trace_field.traces
        for trace in traces[self._trace_count :]:
            selected = frozenset(trace.selected_features)
            if step < self.change_step:
                self.selected_pre.append(selected)
            else:
                self.selected_post.append(selected)
        self._trace_count = len(traces)

    def finalize_audit(self, simulation: Simulation) -> MechanismRunMetrics:
        sharp = super().finalize()
        feature_change = sum(
            abs(simulation.gate.weights[name] - self.weights_at_change.get(name, 0.0))
            for name in simulation.gate.weights
        )
        baseline_distinction = self._dominant(self.selected_pre)
        distinction_change = (
            sum(selected != baseline_distinction for selected in self.selected_post)
            / len(self.selected_post)
            if self.selected_post
            else 0.0
        )
        revisions = self._revision_total(simulation) - self.revisions_at_change
        return MechanismRunMetrics(
            feature_weight_change=feature_change,
            distinction_change_rate=distinction_change,
            candidate_revision_count=float(max(0, revisions)),
            policy_distribution_change=self._policy_tv(),
            adaptation_lag=sharp.adaptation_lag,
            obsolete_persistence=sharp.obsolete_persistence,
            objecthood_f1=sharp.objecthood_f1,
            recovery_slope=sharp.recovery_slope,
        )

    def _policy_tv(self) -> float:
        pre_total = sum(self.actions_pre.values())
        post_total = sum(self.actions_post.values())
        if not pre_total or not post_total:
            return 0.0
        actions = set(self.actions_pre) | set(self.actions_post)
        return 0.5 * sum(
            abs(
                self.actions_pre[action] / pre_total
                - self.actions_post[action] / post_total
            )
            for action in actions
        )

    @staticmethod
    def _dominant(values: List[frozenset[str]]) -> frozenset[str]:
        return Counter(values).most_common(1)[0][0] if values else frozenset()

    @staticmethod
    def _revision_total(simulation: Simulation) -> int:
        return (
            simulation.destabilizer.total_destabilizations
            + simulation.destabilizer.total_splits
            + simulation.stabilizer.total_merged
        )


def run_mechanism_suite(
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 300,
    change_step: int = 150,
) -> MechanismSuiteResult:
    seeds = list(range(seed_start, seed_start + runs))
    raw: Dict[str, Dict[str, List[float]]] = {
        mode: {name: [] for name in AUDIT_METRICS}
        for mode in MECHANISM_MODES
    }
    for mode in MECHANISM_MODES:
        for seed in seeds:
            probe = MechanismAuditProbe(change_step, steps)
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    mechanism_mode=mode,
                    reentry_strength=1.0,
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
            audit = probe.finalize_audit(simulation)
            for name in AUDIT_METRICS:
                raw[mode][name].append(getattr(audit, name))

    baseline = raw["no_reentry"]
    conditions = []
    for mode in MECHANISM_MODES:
        conditions.append(
            MechanismCondition(
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
                    for name in AUDIT_METRICS
                },
            )
        )
    return MechanismSuiteResult(conditions, seeds, steps, change_step)


def format_mechanism_summary(result: MechanismSuiteResult) -> str:
    lines = [
        f"mechanism audit: {len(result.seeds)} paired seeds, "
        f"role reversal at {result.change_step}",
        (
            "mode | weight change | distinction change | candidate revisions | "
            "policy distribution change | adaptation lag | obsolete persistence | "
            "objecthood F1 | recovery slope"
        ),
    ]
    for condition in result.conditions:
        m = condition.metrics
        lines.append(
            f"{condition.mode:18s} | "
            f"{_ci(m['feature_weight_change'])} | "
            f"{_ci(m['distinction_change_rate'], True)} | "
            f"{_ci(m['candidate_revision_count'])} | "
            f"{_ci(m['policy_distribution_change'], True)} | "
            f"{_ci(m['adaptation_lag'])} | "
            f"{_ci(m['obsolete_persistence'], True)} | "
            f"{_ci(m['objecthood_f1'])} | "
            f"{_ci(m['recovery_slope'])}"
        )
        if condition.mode != "no_reentry":
            d = condition.paired_differences
            lines.append(
                f"  paired vs no_reentry: "
                f"weights={_ci(d['feature_weight_change'])}, "
                f"distinctions={_ci(d['distinction_change_rate'], True)}, "
                f"revisions={_ci(d['candidate_revision_count'])}, "
                f"policy={_ci(d['policy_distribution_change'], True)}, "
                f"lag={_ci(d['adaptation_lag'])}, "
                f"obsolete={_ci(d['obsolete_persistence'], True)}, "
                f"F1={_ci(d['objecthood_f1'])}, "
                f"recovery={_ci(d['recovery_slope'])}"
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
