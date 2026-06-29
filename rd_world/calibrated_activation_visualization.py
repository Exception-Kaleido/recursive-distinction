from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .calibrated_activation_validation import CalibratedSuiteResult


def plot_calibrated_activation_suite(
    result: CalibratedSuiteResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(18, 19))
    labels = [condition.label for condition in result.conditions]
    short_labels = [
        "split_only",
        "provisional_select",
        "calibrated",
        "calibrated_budget",
    ]
    panels = (
        ("adaptation_lag", "Adaptation lag", False),
        ("objecthood_f1", "Objecthood F1", False),
        ("false_stabilization_rate", "False stabilization", True),
        ("candidate_count", "Candidate count", False),
        ("candidate_churn", "Candidate churn", False),
        ("useful_candidate_damage_rate", "Useful candidate damage", True),
        ("parent_survival_time", "Parent survival time", False),
        ("child_validation_rate", "Child validation rate", True),
        ("child_activation_rate", "Child activation rate", True),
        ("child_win_rate", "Child win rate", True),
        ("provisional_policy_influence", "Provisional influence", True),
        ("validated_policy_influence", "Validated influence", True),
        ("active_policy_influence", "Active influence", True),
        ("recovery_slope", "Post-change recovery slope", False),
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
        axis.set_xticks(
            range(len(labels)), short_labels, rotation=24, ha="right"
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(
        "Calibrated provisional activation: mean +/- 95% CI, "
        f"{len(result.seeds)} paired seeds",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
