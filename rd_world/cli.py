from __future__ import annotations

import argparse

from .metrics import group_candidates
from .simulation import MODES, Simulation, SimulationConfig
from .validation import MODE_LABELS, format_validation_summary, run_validation


def run_one(
    steps: int,
    seed: int,
    mode: str,
    change_step: int | None = None,
) -> Simulation:
    simulation = Simulation(
        SimulationConfig(
            steps=steps,
            seed=seed,
            mode=mode,
            environment_change_step=change_step,
        )
    )
    metrics = simulation.run()
    groups = group_candidates(simulation.stabilizer.candidates)
    print(
        f"{MODE_LABELS[mode]:28s} | V={simulation.world.viability:6.2f} "
        f"| food={metrics.food_approach_rate:5.1%} "
        f"| decoy-FP={metrics.decoy_false_positive_rate:5.1%} "
        f"| hazard-avoid={metrics.hazard_avoidance_rate:5.1%} "
        f"| wall={metrics.wall_collision_rate:5.1%} "
        f"| food/decoy candidates={groups['food_like'].count}/"
        f"{groups['decoy_like'].count}"
    )
    return simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="RD-World v0.1")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=MODES
        + (
            "compare",
            "validation",
            "sweep",
            "sharp",
            "mechanism",
            "candidate-revision",
            "targeted-split",
            "targeted-split-sweep",
            "provisional-split",
            "calibrated-activation",
            "calibrated-activation-sweep",
        ),
        default="validation",
    )
    parser.add_argument("--change-step", type=int, default=175)
    parser.add_argument("--runs", type=int)
    parser.add_argument(
        "--strengths",
        default="0.0,0.1,0.25,0.5,1.0",
        help="comma-separated re-entry strengths for sweep mode",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save", help="save the selected plot to a PNG")
    parser.add_argument("--log-output", help="save conflict attribution JSONL")
    parser.add_argument(
        "--conflict-threshold",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--evidence-window",
        type=int,
        choices=(1, 3, 5),
        default=3,
    )
    parser.add_argument(
        "--parent-decay",
        choices=("none", "slow", "fast"),
        default="slow",
    )
    parser.add_argument(
        "--selection-window",
        type=int,
        choices=(3, 5, 10),
        default=5,
    )
    parser.add_argument(
        "--candidate-budget",
        choices=("none", "10", "15", "20"),
        default="15",
    )
    parser.add_argument(
        "--validation-threshold",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--activation-threshold",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--activation-delay",
        type=int,
        choices=(1, 3, 5),
        default=3,
    )
    args = parser.parse_args()

    if args.mode in (
        "calibrated-activation",
        "calibrated-activation-sweep",
    ):
        from .calibrated_activation_validation import (
            format_calibrated_activation_summary,
            run_calibrated_activation_suite,
            run_calibrated_activation_sweep,
        )

        if args.mode == "calibrated-activation-sweep":
            result = run_calibrated_activation_sweep(
                runs=args.runs or 20,
                seed_start=args.seed,
                steps=args.steps,
                change_step=args.change_step,
                selection_window=args.selection_window,
            )
        else:
            result = run_calibrated_activation_suite(
                runs=args.runs or 50,
                seed_start=args.seed,
                steps=args.steps,
                change_step=args.change_step,
                validation_threshold=args.validation_threshold,
                activation_threshold=args.activation_threshold,
                activation_delay=args.activation_delay,
                selection_window=args.selection_window,
                candidate_budget=(
                    None
                    if args.candidate_budget == "none"
                    else int(args.candidate_budget)
                ),
            )
        print(format_calibrated_activation_summary(result))
        if args.visualize or args.save:
            from .calibrated_activation_visualization import (
                plot_calibrated_activation_suite,
            )

            plot_calibrated_activation_suite(
                result, output=args.save, show=args.visualize
            )
        return

    if args.mode == "provisional-split":
        from .provisional_split_validation import (
            format_provisional_split_summary,
            run_provisional_split_suite,
        )

        result = run_provisional_split_suite(
            runs=args.runs or 50,
            seed_start=args.seed,
            steps=args.steps,
            change_step=args.change_step,
            selection_window=args.selection_window,
            candidate_budget=(
                None
                if args.candidate_budget == "none"
                else int(args.candidate_budget)
            ),
        )
        print(format_provisional_split_summary(result))
        if args.visualize or args.save:
            from .provisional_split_visualization import (
                plot_provisional_split_suite,
            )

            plot_provisional_split_suite(
                result, output=args.save, show=args.visualize
            )
        return

    if args.mode in ("targeted-split", "targeted-split-sweep"):
        from .targeted_split_validation import (
            format_targeted_split_summary,
            run_targeted_split_suite,
            run_targeted_split_sweep,
            write_conflict_attribution_logs,
        )

        if args.mode == "targeted-split-sweep":
            result = run_targeted_split_sweep(
                runs=args.runs or 20,
                seed_start=args.seed,
                steps=args.steps,
                change_step=args.change_step,
            )
        else:
            result = run_targeted_split_suite(
                runs=args.runs or 50,
                seed_start=args.seed,
                steps=args.steps,
                change_step=args.change_step,
                conflict_threshold=args.conflict_threshold,
                evidence_window=args.evidence_window,
                parent_decay=args.parent_decay,
            )
        print(format_targeted_split_summary(result))
        if args.log_output:
            write_conflict_attribution_logs(result, args.log_output)
        if args.visualize or args.save:
            from .targeted_split_visualization import (
                plot_targeted_split_suite,
            )

            plot_targeted_split_suite(
                result, output=args.save, show=args.visualize
            )
        return

    if args.mode == "candidate-revision":
        from .candidate_revision_validation import (
            format_candidate_revision_summary,
            run_candidate_revision_suite,
        )

        result = run_candidate_revision_suite(
            runs=args.runs or 50,
            seed_start=args.seed,
            steps=args.steps,
            change_step=args.change_step,
        )
        print(format_candidate_revision_summary(result))
        if args.visualize or args.save:
            from .candidate_revision_visualization import (
                plot_candidate_revision_suite,
            )

            plot_candidate_revision_suite(
                result, output=args.save, show=args.visualize
            )
        return

    if args.mode == "mechanism":
        from .mechanism_validation import (
            format_mechanism_summary,
            run_mechanism_suite,
        )

        result = run_mechanism_suite(
            runs=args.runs or 50,
            seed_start=args.seed,
            steps=args.steps,
            change_step=args.change_step,
        )
        print(format_mechanism_summary(result))
        if args.visualize or args.save:
            from .mechanism_visualization import plot_mechanism_suite

            plot_mechanism_suite(result, output=args.save, show=args.visualize)
        return

    if args.mode == "sharp":
        from .sharp_validation import format_sharp_summary, run_sharp_suite

        strengths = tuple(
            float(value.strip())
            for value in args.strengths.split(",")
            if value.strip()
        )
        result = run_sharp_suite(
            strengths=strengths,
            runs=args.runs or 50,
            seed_start=args.seed,
            steps=args.steps,
            change_step=args.change_step,
        )
        print(format_sharp_summary(result))
        if args.visualize or args.save:
            from .sharp_visualization import plot_sharp_suite

            plot_sharp_suite(result, output=args.save, show=args.visualize)
        return

    if args.mode == "sweep":
        from .sweep import format_sweep_summary, run_reentry_sweep

        strengths = tuple(
            float(value.strip())
            for value in args.strengths.split(",")
            if value.strip()
        )
        result = run_reentry_sweep(
            strengths=strengths,
            runs=args.runs or 20,
            seed_start=args.seed,
            steps=args.steps,
            change_step=args.change_step,
        )
        print(format_sweep_summary(result))
        if args.visualize or args.save:
            from .sweep_visualization import plot_sweep

            plot_sweep(result, output=args.save, show=args.visualize)
        return

    if args.mode == "validation":
        result = run_validation(args.steps, args.seed, args.change_step)
        print(format_validation_summary(result))
        if args.visualize or args.save:
            from .validation_visualization import plot_validation

            plot_validation(result, output=args.save, show=args.visualize)
        return

    modes = MODES if args.mode == "compare" else (args.mode,)
    simulations = [
        run_one(args.steps, args.seed, mode, None)
        for mode in modes
    ]
    if (args.visualize or args.save) and simulations:
        from .visualization import plot_simulation

        plot_simulation(simulations[0], output=args.save, show=args.visualize)


if __name__ == "__main__":
    main()
