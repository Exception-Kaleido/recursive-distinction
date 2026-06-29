from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .simulation import Simulation, SimulationConfig
from .types import Action, SensoryItem, StepResult


SHARP_STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)
SHARP_METRICS = (
    "adaptation_lag",
    "obsolete_persistence",
    "destabilization_precision",
    "false_stabilization_rate",
    "objecthood_f1",
    "normalized_decoy_false_positive",
    "recovery_slope",
)
ACTIONS: tuple[Action, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class SharpRunMetrics:
    adaptation_lag: float = 0.0
    obsolete_persistence: float = 0.0
    destabilization_precision: float = 0.0
    false_stabilization_rate: float = 0.0
    objecthood_f1: float = 0.0
    normalized_decoy_false_positive: float = 0.0
    recovery_slope: float = 0.0
    obsolete_history: List[float] = field(default_factory=list)
    viability: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class MeanCI:
    mean: float
    ci95: float


@dataclass
class SharpCondition:
    strength: float
    metrics: Dict[str, MeanCI]
    paired_differences: Dict[str, MeanCI]


@dataclass
class SharpSuiteResult:
    conditions: List[SharpCondition]
    seeds: List[int]
    steps: int
    change_step: int


class SharpProbe:
    def __init__(self, change_step: int, steps: int) -> None:
        self.change_step = change_step
        self.steps = steps
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.neutral_predictions = 0
        self.neutral_probes = 0
        self.green_food_approaches = 0
        self.green_decoy_approaches = 0
        self.precise_destabilizations = 0
        self.total_destabilizations = 0
        self.obsolete_history: List[float] = []
        self.viability: List[float] = []
        self.adaptation_step: int | None = None
        self.role_probes: List[tuple[str, SensoryItem]] = []

    def on_change(self, step: int, simulation: Simulation) -> None:
        grouped: Dict[str, List[SensoryItem]] = {"food": [], "decoy": []}
        for entity in simulation.world.entities.values():
            if entity.kind not in grouped:
                continue
            color = entity.color_override or simulation.world.colors[entity.kind]
            grouped[entity.kind].append(SensoryItem(*color, 1, 0))
        count = min(len(grouped["food"]), len(grouped["decoy"]))
        self.role_probes = [
            (kind, item)
            for kind in ("food", "decoy")
            for item in grouped[kind][:count]
        ]

    def before_step(
        self,
        step: int,
        simulation: Simulation,
        patch: List[SensoryItem],
        action: Action,
    ) -> None:
        world = simulation.world
        ax, ay = world.agent_position
        adjacent_items = {
            (item.rel_x, item.rel_y): item
            for item in patch
            if abs(item.rel_x) + abs(item.rel_y) == 1
        }
        for probe_action in ACTIONS:
            position = (ax + probe_action[0], ay + probe_action[1])
            entity = world.entities.get(position)
            true_relevant = (
                not world.in_bounds(position)
                or (entity is not None and entity.kind == "wall")
                or (
                    entity is not None
                    and abs(world.current_effect(entity.kind)) >= 0.5
                )
            )
            delta, resistance, confidence = simulation.stabilizer.predict(
                adjacent_items[probe_action], probe_action
            )
            predicted = confidence >= 0.15 and (
                abs(delta + 0.05) >= 0.5 or resistance >= 0.5
            )
            if predicted and true_relevant:
                self.tp += 1
            elif predicted and not true_relevant:
                self.fp += 1
            elif true_relevant and not predicted:
                self.fn += 1
            if entity is not None and entity.kind in ("food", "decoy"):
                if abs(world.current_effect(entity.kind)) < 0.5:
                    stable_delta, stable_resistance, stable_confidence = (
                        simulation.stabilizer.predict_stabilized(
                            adjacent_items[probe_action], probe_action
                        )
                    )
                    stabilized = (
                        stable_confidence >= 0.15
                        and (
                            abs(stable_delta + 0.05) >= 0.5
                            or stable_resistance >= 0.5
                        )
                    )
                    self.neutral_probes += 1
                    self.neutral_predictions += int(stabilized)

        target = (ax + action[0], ay + action[1])
        selected = world.entities.get(target)
        if selected is not None:
            if selected.kind == "food":
                self.green_food_approaches += 1
            elif selected.kind == "decoy":
                self.green_decoy_approaches += 1

        if step >= self.change_step:
            obsolete = self._obsolete_fraction(simulation)
            self.obsolete_history.append(obsolete)
            if (
                self.adaptation_step is None
                and len(self.obsolete_history) >= 5
                and statistics.fmean(self.obsolete_history[-5:]) <= 0.40
            ):
                self.adaptation_step = step

    def after_step(
        self,
        step: int,
        simulation: Simulation,
        result: StepResult,
        destabilization_delta: int,
    ) -> None:
        self.viability.append(result.viability)
        if destabilization_delta <= 0:
            return
        self.total_destabilizations += destabilization_delta
        relevant = (
            step >= self.change_step
            and any(
                kind in ("food", "decoy")
                for kind in simulation.world.last_delivered_kinds
            )
        )
        if relevant:
            self.precise_destabilizations += destabilization_delta

    def finalize(self) -> SharpRunMetrics:
        precision = (
            self.precise_destabilizations / self.total_destabilizations
            if self.total_destabilizations
            else 0.0
        )
        f1_denominator = 2 * self.tp + self.fp + self.fn
        f1 = 2 * self.tp / f1_denominator if f1_denominator else 0.0
        green_total = self.green_food_approaches + self.green_decoy_approaches
        normalized_fp = (
            self.green_decoy_approaches / green_total if green_total else 0.0
        )
        adaptation_lag = (
            self.adaptation_step - self.change_step
            if self.adaptation_step is not None
            else self.steps - self.change_step
        )
        return SharpRunMetrics(
            adaptation_lag=float(adaptation_lag),
            obsolete_persistence=(
                statistics.fmean(self.obsolete_history)
                if self.obsolete_history
                else 1.0
            ),
            destabilization_precision=precision,
            false_stabilization_rate=(
                self.neutral_predictions / self.neutral_probes
                if self.neutral_probes
                else 0.0
            ),
            objecthood_f1=f1,
            normalized_decoy_false_positive=normalized_fp,
            recovery_slope=self._recovery_slope(),
            obsolete_history=self.obsolete_history,
            viability=self.viability,
        )

    def _obsolete_fraction(
        self, simulation: Simulation
    ) -> float:
        stale = 0
        probes = 0
        for kind, item in self.role_probes:
            delta, _, confidence = simulation.stabilizer.predict_identity(item)
            if kind == "food":
                # Food was positive before reversal and is now neutral.
                obsolete = confidence >= 0.10 and delta >= 0.5
            else:
                # Decoy was neutral before reversal and is now positive.
                obsolete = confidence < 0.10 or delta < 0.5
            stale += int(obsolete)
            probes += 1
        return stale / probes if probes else 1.0

    def _recovery_slope(self, horizon: int = 75) -> float:
        values = self.viability[
            self.change_step : min(len(self.viability), self.change_step + horizon)
        ]
        if len(values) < 2:
            return 0.0
        x_mean = (len(values) - 1) / 2.0
        y_mean = statistics.fmean(values)
        numerator = sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(values)
        )
        denominator = sum(
            (index - x_mean) ** 2 for index in range(len(values))
        )
        return numerator / denominator if denominator else 0.0


def run_sharp_suite(
    strengths: Iterable[float] = SHARP_STRENGTHS,
    runs: int = 50,
    seed_start: int = 0,
    steps: int = 400,
    change_step: int = 200,
) -> SharpSuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = list(range(seed_start, seed_start + runs))
    raw: Dict[float, Dict[str, List[float]]] = {}
    for strength in strengths:
        raw[float(strength)] = {name: [] for name in SHARP_METRICS}
        for seed in seeds:
            probe = SharpProbe(change_step, steps)
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    reentry_strength=float(strength),
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
            run_metrics = probe.finalize()
            for name in SHARP_METRICS:
                raw[float(strength)][name].append(getattr(run_metrics, name))

    baseline = raw[0.0]
    conditions = []
    for strength in strengths:
        strength = float(strength)
        conditions.append(
            SharpCondition(
                strength=strength,
                metrics={
                    name: _mean_ci(values)
                    for name, values in raw[strength].items()
                },
                paired_differences={
                    name: _mean_ci(
                        [
                            value - baseline_value
                            for value, baseline_value in zip(
                                raw[strength][name], baseline[name]
                            )
                        ]
                    )
                    for name in SHARP_METRICS
                },
            )
        )
    return SharpSuiteResult(conditions, seeds, steps, change_step)


def format_sharp_summary(result: SharpSuiteResult) -> str:
    lines = [
        f"sharp paired suite: {len(result.seeds)} seeds, "
        f"{result.steps} steps, role reversal at {result.change_step}",
        (
            "strength | adaptation lag | obsolete persistence | destabilization "
            "precision | false stabilization | objecthood F1 | normalized decoy FP "
            "| recovery slope"
        ),
    ]
    for condition in result.conditions:
        m = condition.metrics
        lines.append(
            f"{condition.strength:8.2f} | "
            f"{_ci(m['adaptation_lag'])} | "
            f"{_ci(m['obsolete_persistence'], True)} | "
            f"{_ci(m['destabilization_precision'], True)} | "
            f"{_ci(m['false_stabilization_rate'], True)} | "
            f"{_ci(m['objecthood_f1'])} | "
            f"{_ci(m['normalized_decoy_false_positive'], True)} | "
            f"{_ci(m['recovery_slope'])}"
        )
        if condition.strength != 0.0:
            d = condition.paired_differences
            lines.append(
                f"         paired vs 0.0: lag={_ci(d['adaptation_lag'])}, "
                f"obsolete={_ci(d['obsolete_persistence'], True)}, "
                f"destab-precision={_ci(d['destabilization_precision'], True)}, "
                f"false-stable={_ci(d['false_stabilization_rate'], True)}, "
                f"F1={_ci(d['objecthood_f1'])}, "
                f"green-decoy-FP="
                f"{_ci(d['normalized_decoy_false_positive'], True)}, "
                f"recovery={_ci(d['recovery_slope'])}"
            )
    return "\n".join(lines)


def _mean_ci(values: List[float]) -> MeanCI:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return MeanCI(mean, 0.0)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return MeanCI(mean, 1.96 * standard_error)


def _ci(value: MeanCI, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{value.mean * scale:.3f}+/-{value.ci95 * scale:.3f}{suffix}"
