from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .mechanism_validation import MechanismSuiteResult


def plot_mechanism_suite(
    result: MechanismSuiteResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    modes = [condition.mode for condition in result.conditions]
    panels = (
        ("feature_weight_change", "Feature weight change", False),
        ("distinction_change_rate", "Selected distinction change", True),
        ("candidate_revision_count", "Candidate revision count", False),
        ("policy_distribution_change", "Policy distribution change", True),
        ("adaptation_lag", "Adaptation lag", False),
        ("obsolete_persistence", "Obsolete candidate persistence", True),
        ("objecthood_f1", "Objecthood F1", False),
        ("recovery_slope", "Post-change viability recovery", False),
    )
    for axis, (name, title, percent) in zip(axes.flat[:8], panels):
        scale = 100.0 if percent else 1.0
        means = [
            condition.metrics[name].mean * scale
            for condition in result.conditions
        ]
        ci = [
            condition.metrics[name].ci95 * scale
            for condition in result.conditions
        ]
        axis.errorbar(range(len(modes)), means, yerr=ci, fmt="o", capsize=4)
        axis.set_xticks(range(len(modes)), modes, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)

    axis = axes.flat[8]
    paired_modes = result.conditions[1:]
    means = [
        condition.paired_differences["adaptation_lag"].mean
        for condition in paired_modes
    ]
    ci = [
        condition.paired_differences["adaptation_lag"].ci95
        for condition in paired_modes
    ]
    axis.errorbar(range(len(paired_modes)), means, yerr=ci, fmt="o", capsize=4)
    axis.axhline(0.0, color="gray", linestyle="--")
    axis.set_xticks(
        range(len(paired_modes)),
        [condition.mode for condition in paired_modes],
        rotation=25,
        ha="right",
    )
    axis.set_title("Paired adaptation lag vs no_reentry")
    axis.grid(alpha=0.25)
    fig.suptitle(
        f"Mechanism-level re-entry audit: mean +/- 95% CI, "
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
