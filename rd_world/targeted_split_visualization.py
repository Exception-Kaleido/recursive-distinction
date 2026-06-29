from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .targeted_split_validation import TargetedSplitSuiteResult


def plot_targeted_split_suite(
    result: TargetedSplitSuiteResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 13))
    labels = [condition.label for condition in result.conditions]
    panels = (
        ("adaptation_lag", "Adaptation lag", False),
        ("obsolete_persistence", "Obsolete persistence", True),
        ("objecthood_f1", "Objecthood F1", False),
        ("false_stabilization_rate", "False stabilization", True),
        ("candidate_count", "Post-change candidate count", False),
        ("candidate_churn", "Post-change candidate churn", False),
        ("target_precision", "Conflict target precision", True),
        ("wrong_target_rate", "Wrong-target rate", True),
        (
            "useful_candidate_damage_rate",
            "Useful-candidate damage",
            True,
        ),
    )
    for axis, (name, title, percent) in zip(axes.flat, panels):
        scale = 100.0 if percent else 1.0
        axis.errorbar(
            range(len(labels)),
            [
                condition.metrics[name].mean * scale
                for condition in result.conditions
            ],
            yerr=[
                condition.metrics[name].ci95 * scale
                for condition in result.conditions
            ],
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Targeted split audit: mean +/- 95% CI, "
        f"{len(result.seeds)} paired seeds"
    )
    figure.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
