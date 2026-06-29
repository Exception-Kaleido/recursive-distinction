from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .sweep import SweepResult


def plot_sweep(
    result: SweepResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    strengths = [condition.strength for condition in result.conditions]
    panels = (
        ("viability", "Final viability", False),
        ("food_approach", "Food approach rate", True),
        (
            "normalized_decoy_false_positive",
            "Normalized decoy false positive",
            True,
        ),
        ("hazard_avoidance", "Hazard avoidance", True),
        ("wall_collision", "Wall collision rate", True),
        ("stability_plasticity", "Stability-plasticity score", False),
    )
    for axis, (metric_name, title, rate) in zip(axes.flat, panels):
        scale = 100.0 if rate else 1.0
        means = [
            condition.metrics[metric_name].mean * scale
            for condition in result.conditions
        ]
        errors = [
            condition.metrics[metric_name].std * scale
            for condition in result.conditions
        ]
        axis.errorbar(strengths, means, yerr=errors, marker="o", capsize=4)
        axis.set_title(title)
        axis.set_xlabel("re-entry strength")
        if rate:
            axis.set_ylabel("percent")
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"RD-World re-entry sweep: mean +/- std over {len(result.seeds)} seeds"
    )
    fig.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
