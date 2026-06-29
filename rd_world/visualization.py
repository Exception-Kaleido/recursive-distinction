from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .boundary import FEATURES
from .simulation import Simulation
from .world import COLORS


def plot_simulation(simulation: Simulation, output: str | None = None, show: bool = True) -> None:
    world = simulation.world
    metrics = simulation.metrics
    kind_to_number = {"empty": 0, "wall": 1, "food": 2, "hazard": 3, "decoy": 4}
    grid = [[0 for _ in range(world.width)] for _ in range(world.height)]
    for (x, y), entity in world.iter_entities():
        grid[y][x] = kind_to_number[entity.kind]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_list = [COLORS[name] for name in ("empty", "wall", "food", "hazard", "decoy")]
    axes[0, 0].imshow(grid, cmap=ListedColormap(color_list), vmin=0, vmax=4)
    ax, ay = world.agent_position
    axes[0, 0].scatter([ax], [ay], marker="*", s=220, c="royalblue", edgecolors="white")
    axes[0, 0].set_title("Hidden world (display only; agent sees numeric patches)")

    viability_axis = axes[0, 1]
    confidence_axis = viability_axis.twinx()
    viability_axis.plot(metrics.viability, color="tab:blue", label="viability")
    confidence_axis.plot(
        metrics.mean_confidence,
        color="tab:orange",
        label="mean candidate confidence",
    )
    confidence_axis.set_ylim(0.0, 1.0)
    confidence_axis.set_ylabel("objecthood confidence")
    axes[0, 1].set_title("Viability and objecthood confidence")
    lines = viability_axis.lines + confidence_axis.lines
    axes[0, 1].legend(lines, [line.get_label() for line in lines])

    for feature in FEATURES:
        axes[1, 0].plot(
            [weights[feature] for weights in metrics.boundary_weights],
            label=feature,
        )
    axes[1, 0].set_title("BoundaryGate weights over time")
    axes[1, 0].legend(fontsize=8, ncol=2)

    axes[1, 1].axis("off")
    candidates = sorted(
        simulation.stabilizer.candidates,
        key=lambda candidate: candidate.objecthood_score,
        reverse=True,
    )[:10]
    lines = [
        "Current objecthood candidates",
        "id  n  obj   predicted(dV, resistance)  RGB",
    ]
    for candidate in candidates:
        rgb = tuple(candidate.centroid[name] for name in ("color_r", "color_g", "color_b"))
        lines.append(
            f"{candidate.candidate_id:2d} {candidate.observations:2d} "
            f"{candidate.objecthood_score:4.2f} "
            f"({candidate.predicted_delta_viability:5.2f}, "
            f"{candidate.predicted_resistance:4.2f}) "
            f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})"
        )
    axes[1, 1].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace")
    fig.tight_layout()

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
