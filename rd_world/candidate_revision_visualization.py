from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .candidate_revision_validation import CandidateRevisionSuiteResult


def plot_candidate_revision_suite(
    result: CandidateRevisionSuiteResult,
    output: str | None = None,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 13))
    modes = [condition.mode for condition in result.conditions]
    panels = (
        ("adaptation_lag", "Adaptation lag", False),
        ("obsolete_persistence", "Obsolete candidate persistence", True),
        ("objecthood_f1", "Objecthood F1", False),
        ("false_stabilization_rate", "False stabilization", True),
        ("destabilization_precision", "Destabilization precision", True),
        ("candidate_count", "Post-change candidate count", False),
        ("candidate_churn", "Post-change candidate churn", False),
        ("recovery_slope", "Post-change viability recovery", False),
    )
    for axis, (name, title, percent) in zip(axes.flat[:8], panels):
        scale = 100.0 if percent else 1.0
        means = [
            condition.metrics[name].mean * scale
            for condition in result.conditions
        ]
        intervals = [
            condition.metrics[name].ci95 * scale
            for condition in result.conditions
        ]
        axis.errorbar(
            range(len(modes)), means, yerr=intervals, fmt="o", capsize=4
        )
        axis.set_xticks(range(len(modes)), modes, rotation=28, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)

    paired = result.conditions[1:]
    axis = axes.flat[8]
    axis.errorbar(
        range(len(paired)),
        [
            condition.paired_differences["adaptation_lag"].mean
            for condition in paired
        ],
        yerr=[
            condition.paired_differences["adaptation_lag"].ci95
            for condition in paired
        ],
        fmt="o",
        capsize=4,
    )
    axis.axhline(0.0, color="gray", linestyle="--")
    axis.set_xticks(
        range(len(paired)),
        [condition.mode for condition in paired],
        rotation=28,
        ha="right",
    )
    axis.set_title("Paired adaptation lag vs formation-only")
    axis.grid(alpha=0.25)
    figure.suptitle(
        "Candidate revision decomposition: mean +/- 95% CI, "
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
