from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .simulation import Simulation, SimulationConfig


DEFAULT_STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)
SWEEP_METRICS = (
    "viability",
    "food_approach",
    "decoy_false_positive",
    "normalized_decoy_false_positive",
    "hazard_avoidance",
    "wall_collision",
    "destabilizations",
    "pre_change_destabilizations",
    "candidate_stability",
    "adaptation_latency",
    "stability_plasticity",
)


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float


@dataclass
class SweepCondition:
    strength: float
    runs: int
    metrics: Dict[str, MeanStd]


@dataclass
class SweepResult:
    conditions: List[SweepCondition]
    seeds: List[int]
    steps: int
    change_step: int


def run_reentry_sweep(
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
    runs: int = 20,
    seed_start: int = 0,
    steps: int = 350,
    change_step: int = 175,
) -> SweepResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = list(range(seed_start, seed_start + runs))
    conditions: List[SweepCondition] = []
    for strength in strengths:
        values = {name: [] for name in SWEEP_METRICS}
        for seed in seeds:
            simulation = Simulation(
                SimulationConfig(
                    steps=steps,
                    seed=seed,
                    mode="rd",
                    reentry_strength=float(strength),
                    environment_change_step=change_step,
                )
            )
            metrics = simulation.run()
            post_stability = metrics.candidate_stability[change_step:]
            values["viability"].append(simulation.world.viability)
            values["food_approach"].append(metrics.food_approach_rate)
            values["decoy_false_positive"].append(
                metrics.decoy_false_positive_rate
            )
            values["normalized_decoy_false_positive"].append(
                metrics.normalized_decoy_false_positive_rate
            )
            values["hazard_avoidance"].append(metrics.hazard_avoidance_rate)
            values["wall_collision"].append(metrics.wall_collision_rate)
            values["destabilizations"].append(
                float(metrics.destabilization_events[-1])
                if metrics.destabilization_events
                else 0.0
            )
            values["pre_change_destabilizations"].append(
                float(metrics.pre_change_destabilization_events)
            )
            values["candidate_stability"].append(
                sum(post_stability) / len(post_stability)
                if post_stability
                else 0.0
            )
            values["adaptation_latency"].append(float(metrics.adaptation_latency))
            values["stability_plasticity"].append(
                metrics.stability_plasticity_score
            )
        conditions.append(
            SweepCondition(
                strength=float(strength),
                runs=runs,
                metrics={
                    name: MeanStd(
                        mean=statistics.fmean(metric_values),
                        std=statistics.pstdev(metric_values),
                    )
                    for name, metric_values in values.items()
                },
            )
        )
    return SweepResult(conditions, seeds, steps, change_step)


def format_sweep_summary(result: SweepResult) -> str:
    lines = [
        f"re-entry strength sweep: {len(result.seeds)} seeds, "
        f"{result.steps} steps, change at {result.change_step}",
        (
            "strength | viability | food approach | decoy FP | normalized decoy FP "
            "| hazard avoid | wall collision | destabilizations | adapt latency "
            "| pre-change destabilizations | event stability | stability-plasticity"
        ),
    ]
    for condition in result.conditions:
        metric = condition.metrics
        lines.append(
            f"{condition.strength:8.2f} | "
            f"{_pm(metric['viability'])} | "
            f"{_pm(metric['food_approach'], percent=True)} | "
            f"{_pm(metric['decoy_false_positive'], percent=True)} | "
            f"{_pm(metric['normalized_decoy_false_positive'], percent=True)} | "
            f"{_pm(metric['hazard_avoidance'], percent=True)} | "
            f"{_pm(metric['wall_collision'], percent=True)} | "
            f"{_pm(metric['destabilizations'])} | "
            f"{_pm(metric['adaptation_latency'])} | "
            f"{_pm(metric['pre_change_destabilizations'])} | "
            f"{_pm(metric['candidate_stability'])} | "
            f"{_pm(metric['stability_plasticity'])}"
        )
    return "\n".join(lines)


def _pm(value: MeanStd, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{value.mean * scale:.2f}+/-{value.std * scale:.2f}{suffix}"
