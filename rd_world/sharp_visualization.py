from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .sharp_validation import SharpSuiteResult


def plot_sharp_suite(
    result: SharpSuiteResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    strengths = [condition.strength for condition in result.conditions]
    panels = (
        ("adaptation_lag", "Adaptation lag", False),
        ("obsolete_persistence", "Obsolete candidate persistence", True),
        ("destabilization_precision", "Destabilization precision", True),
        ("false_stabilization_rate", "False stabilization rate", True),
        ("objecthood_f1", "Objecthood F1", False),
        (
            "normalized_decoy_false_positive",
            "Normalized decoy false positive",
            True,
        ),
        ("recovery_slope", "Post-change viability recovery slope", False),
    )
    for axis, (name, title, percent) in zip(axes.flat[:7], panels):
        scale = 100.0 if percent else 1.0
        means = [
            condition.metrics[name].mean * scale
            for condition in result.conditions
        ]
        ci = [
            condition.metrics[name].ci95 * scale
            for condition in result.conditions
        ]
        axis.errorbar(strengths, means, yerr=ci, marker="o", capsize=4)
        axis.axhline(
            result.conditions[0].metrics[name].mean * scale,
            color="gray",
            linestyle="--",
            linewidth=1,
        )
        axis.set_title(title)
        axis.set_xlabel("re-entry strength")
        axis.grid(alpha=0.25)
    for axis, metric, title, percent in (
        (
            axes.flat[7],
            "adaptation_lag",
            "Paired difference vs 0.0: adaptation lag",
            False,
        ),
        (
            axes.flat[8],
            "obsolete_persistence",
            "Paired difference vs 0.0: obsolete persistence",
            True,
        ),
    ):
        scale = 100.0 if percent else 1.0
        means = [
            condition.paired_differences[metric].mean * scale
            for condition in result.conditions
        ]
        ci = [
            condition.paired_differences[metric].ci95 * scale
            for condition in result.conditions
        ]
        axis.errorbar(strengths, means, yerr=ci, marker="o", capsize=4)
        axis.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("re-entry strength")
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Non-stationary ambiguous RD-World: mean +/- 95% CI, "
        f"{len(result.seeds)} paired seeds"
    )
    fig.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
