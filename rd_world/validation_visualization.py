from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .boundary import FEATURES
from .metrics import CANDIDATE_TYPES, candidate_type
from .validation import MODE_LABELS, ValidationResult


TYPE_COLORS = {
    "food_like": "forestgreen",
    "decoy_like": "yellowgreen",
    "wall_like": "dimgray",
    "hazard_like": "firebrick",
}


def plot_validation(
    result: ValidationResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    simulations = result.simulations

    for mode, simulation in simulations.items():
        axes[0, 0].plot(
            simulation.metrics.viability,
            label=MODE_LABELS[mode],
        )
    _change_line(axes[0, 0], simulations["rd"].metrics.environment_change_step)
    axes[0, 0].set_title("Ablation: viability")
    axes[0, 0].legend(fontsize=8)

    metric_names = (
        ("food approach", "food_approach_rate"),
        ("green decoy FP", "normalized_decoy_false_positive_rate"),
        ("hazard avoidance", "hazard_avoidance_rate"),
    )
    width = 0.18
    for index, mode in enumerate(simulations):
        metrics = simulations[mode].metrics
        axes[0, 1].bar(
            [position + index * width for position in range(len(metric_names))],
            [getattr(metrics, attribute) for _, attribute in metric_names],
            width,
            label=MODE_LABELS[mode],
        )
    axes[0, 1].set_xticks(
        [position + 1.5 * width for position in range(len(metric_names))],
        [name for name, _ in metric_names],
        rotation=15,
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Behavioral rates")
    axes[0, 1].legend(fontsize=7)

    for mode, simulation in simulations.items():
        cumulative = []
        total = 0.0
        for collision in simulation.metrics.wall_collision_history:
            total += collision
            cumulative.append(total)
        axes[0, 2].plot(cumulative, label=MODE_LABELS[mode])
    axes[0, 2].set_title("Cumulative wall collisions")
    axes[0, 2].legend(fontsize=8)

    rd = simulations["rd"]
    for feature in FEATURES:
        axes[1, 0].plot(
            [weights[feature] for weights in rd.metrics.boundary_weights],
            label=feature,
        )
    _change_line(axes[1, 0], rd.metrics.environment_change_step)
    axes[1, 0].set_title("RD feature weights")
    axes[1, 0].legend(fontsize=7, ncol=2)

    for name in CANDIDATE_TYPES:
        axes[1, 1].plot(
            rd.metrics.candidate_confidence_by_type[name],
            color=TYPE_COLORS[name],
            label=name,
        )
    _change_line(axes[1, 1], rd.metrics.environment_change_step)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("RD candidate confidence by outcome type")
    axes[1, 1].legend(fontsize=8)

    for mode in ("rd", "trace"):
        axes[1, 2].plot(
            simulations[mode].metrics.candidate_stability,
            label=MODE_LABELS[mode],
        )
    _change_line(axes[1, 2], rd.metrics.environment_change_step)
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].set_title("Objecthood candidate stability")
    axes[1, 2].legend(fontsize=8)

    event_steps = range(len(rd.metrics.split_events))
    axes[2, 0].step(
        event_steps, rd.metrics.split_events, label="splits", where="post"
    )
    axes[2, 0].step(
        event_steps, rd.metrics.merge_events, label="merges", where="post"
    )
    axes[2, 0].step(
        event_steps,
        rd.metrics.destabilization_events,
        label="destabilizations",
        where="post",
    )
    _change_line(axes[2, 0], rd.metrics.environment_change_step)
    axes[2, 0].set_title("RD candidate events")
    axes[2, 0].legend()

    _candidate_table(
        axes[2, 1],
        rd,
        ("food_like", "decoy_like"),
        "Green candidates: outcome-separated",
    )
    _candidate_table(
        axes[2, 2],
        rd,
        ("wall_like", "hazard_like"),
        "Blocking and negative-viability candidates",
    )

    fig.suptitle(
        "RD-World validation "
        f"(observation contract: {'PASS' if result.observation_contract_valid else 'FAIL'})"
    )
    fig.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _candidate_table(axis, simulation, types, title: str) -> None:
    axis.axis("off")
    lines = [title, "type        id obs obj   dV     R      RGB"]
    for name in types:
        candidates = sorted(
            (
                candidate
                for candidate in simulation.stabilizer.candidates
                if candidate_type(candidate) == name
            ),
            key=lambda candidate: candidate.objecthood_score,
            reverse=True,
        )[:4]
        if not candidates:
            lines.append(f"{name:11s} -- no candidate --")
            continue
        for candidate in candidates:
            rgb = tuple(
                candidate.centroid[channel]
                for channel in ("color_r", "color_g", "color_b")
            )
            lines.append(
                f"{name:11s} {candidate.candidate_id:2d} "
                f"{candidate.observations:3d} {candidate.objecthood_score:4.2f} "
                f"{candidate.predicted_delta_viability:+5.2f} "
                f"{candidate.predicted_resistance:4.2f} "
                f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})"
            )
    axis.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)


def _change_line(axis, step: int | None) -> None:
    if step is not None:
        axis.axvline(step, color="black", linestyle="--", linewidth=1, alpha=0.6)
