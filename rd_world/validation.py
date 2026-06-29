from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Dict

from .metrics import CandidateGroupStats, group_candidates
from .simulation import MODES, Simulation, SimulationConfig
from .types import SensoryItem, StepResult
from .world import GridWorld


FORBIDDEN_LABELS = {"food", "wall", "hazard", "decoy"}
MODE_LABELS = {
    "rd": "RD re-entry",
    "trace": "trace memory / no re-entry",
    "fixed": "fixed-weight baseline",
    "random": "random",
}


@dataclass
class ValidationResult:
    simulations: Dict[str, Simulation]
    observation_contract_valid: bool

    @property
    def candidate_groups(self) -> Dict[str, Dict[str, CandidateGroupStats]]:
        return {
            mode: group_candidates(simulation.stabilizer.candidates)
            for mode, simulation in self.simulations.items()
        }


def verify_observation_contract() -> bool:
    world = GridWorld(seed=101)
    patch = world.observe()
    result = world.step((1, 0))
    public_names = {
        name.lower()
        for name in SensoryItem.__dataclass_fields__
        | StepResult.__dataclass_fields__
    }
    if public_names & FORBIDDEN_LABELS:
        return False
    if any(not isinstance(value, Real) for item in patch for value in item.features().values()):
        return False
    public_result = (
        result.moved,
        result.resistance,
        result.delta_viability,
        result.viability,
    )
    return all(isinstance(value, (bool, Real)) for value in public_result)


def run_validation(
    steps: int = 350,
    seed: int = 7,
    change_step: int = 175,
) -> ValidationResult:
    simulations: Dict[str, Simulation] = {}
    for mode in MODES:
        simulation = Simulation(
            SimulationConfig(
                steps=steps,
                seed=seed,
                mode=mode,
                environment_change_step=change_step,
            )
        )
        simulation.run()
        simulations[mode] = simulation
    return ValidationResult(
        simulations=simulations,
        observation_contract_valid=verify_observation_contract(),
    )


def format_validation_summary(result: ValidationResult) -> str:
    lines = [
        f"observation label contract: {'PASS' if result.observation_contract_valid else 'FAIL'}"
    ]
    for mode, simulation in result.simulations.items():
        metrics = simulation.metrics
        stability = (
            sum(metrics.candidate_stability[-50:])
            / max(1, len(metrics.candidate_stability[-50:]))
        )
        lines.append(
            f"{MODE_LABELS[mode]:28s} "
            f"V={simulation.world.viability:6.2f} "
            f"food={metrics.food_approach_rate:5.1%} "
            f"decoy-FP={metrics.decoy_false_positive_rate:5.1%} "
            f"green-decoy-FP={metrics.normalized_decoy_false_positive_rate:5.1%} "
            f"hazard-avoid={metrics.hazard_avoidance_rate:5.1%} "
            f"wall={metrics.wall_collision_rate:5.1%} "
            f"wall-reduction={metrics.wall_collision_reduction:+5.1%} "
            f"stability={stability:4.2f} "
            f"SP={metrics.stability_plasticity_score:4.2f} "
            f"split/merge={metrics.split_events[-1] if metrics.split_events else 0}/"
            f"{metrics.merge_events[-1] if metrics.merge_events else 0}"
        )

    rd = result.simulations["rd"]
    groups = group_candidates(rd.stabilizer.candidates)
    lines.append("RD candidate outcome groups:")
    for name in ("food_like", "decoy_like", "wall_like", "hazard_like"):
        group = groups[name]
        lines.append(
            f"  {name:12s} n={group.count:2d} obs={group.observations:3d} "
            f"conf={group.mean_confidence:.2f} "
            f"dV={group.mean_delta_viability:+.2f} R={group.mean_resistance:.2f}"
        )
    lines.append(
        "RD post-change adaptation: "
        f"new={rd.metrics.post_change_new_candidates}, "
        f"destabilized={rd.metrics.post_change_destabilization_events}, "
        f"splits={rd.metrics.post_change_split_events}, "
        f"merges={rd.metrics.post_change_merge_events}"
    )
    return "\n".join(lines)
