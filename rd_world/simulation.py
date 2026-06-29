from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .boundary import BoundaryGate
from .metrics import HiddenEvaluator, Metrics
from .objecthood import Destabilizer, ObjecthoodStabilizer
from .policy import AgentPolicy, FixedWeightPolicy, RandomPolicy
from .trace_field import TemporalTraceField, TraceField
from .types import SensoryItem
from .world import GridWorld


MODES = ("rd", "trace", "fixed", "random")
MECHANISM_MODES = (
    "no_reentry",
    "feature_reentry",
    "candidate_reentry",
    "policy_reentry",
    "full_reentry",
)

MECHANISM_FLAGS = {
    "no_reentry": (False, True, False, False),
    "feature_reentry": (True, True, False, False),
    "candidate_reentry": (False, True, True, False),
    "policy_reentry": (False, True, False, True),
    "full_reentry": (True, True, True, True),
}

CANDIDATE_REVISION_SUBMODES = (
    "confidence_only",
    "split_only",
    "suppression_only",
    "split_plus_suppression",
    "full_candidate_revision",
    "targeted_split",
    "targeted_split_plus_suppression",
    "provisional_split",
    "provisional_split_plus_selection",
    "provisional_split_budgeted",
    "calibrated_provisional_split",
    "calibrated_provisional_split_budgeted",
)

CANDIDATE_REVISION_FLAGS = {
    "confidence_only": (True, False, False, False, False),
    "split_only": (False, True, False, False, False),
    "suppression_only": (False, False, True, False, False),
    "split_plus_suppression": (False, True, True, False, False),
    "full_candidate_revision": (True, True, True, True, True),
    "targeted_split": (False, True, False, False, False),
    "targeted_split_plus_suppression": (False, True, True, False, False),
    "provisional_split": (False, True, False, False, False),
    "provisional_split_plus_selection": (
        False,
        True,
        False,
        False,
        False,
    ),
    "provisional_split_budgeted": (
        False,
        True,
        False,
        False,
        False,
    ),
    "calibrated_provisional_split": (
        False,
        True,
        False,
        False,
        False,
    ),
    "calibrated_provisional_split_budgeted": (
        False,
        True,
        False,
        False,
        False,
    ),
}


@dataclass
class SimulationConfig:
    steps: int = 300
    seed: int = 7
    mode: str = "rd"
    width: int = 12
    height: int = 12
    environment_change_step: int | None = None
    food_to_decoy: int = 4
    change_hazard_color: bool = True
    reentry_strength: float = 1.0
    ambiguous_rgb: bool = False
    delayed_outcomes: bool = False
    role_reversal: bool = False
    counts: Dict[str, int] | None = None
    exploration: float = 0.12
    mechanism_mode: str | None = None
    feature_reentry_enabled: bool | None = None
    candidate_formation_enabled: bool | None = None
    candidate_revision_enabled: bool | None = None
    candidate_revision_submode: str | None = None
    conflict_threshold: str = "medium"
    conflict_evidence_window: int = 3
    parent_decay: str = "slow"
    post_split_selection_window: int = 5
    candidate_budget: int | None = None
    validation_threshold: str = "medium"
    activation_threshold: str = "medium"
    activation_delay: int = 3
    policy_reentry_enabled: bool | None = None
    reentry: bool | None = None  # Backward-compatible shortcut.

    def __post_init__(self) -> None:
        if self.reentry is not None:
            self.mode = "rd" if self.reentry else "trace"
            if not self.reentry:
                self.reentry_strength = 0.0
        if self.reentry_strength < 0.0:
            raise ValueError("reentry_strength must be non-negative")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mechanism_mode is not None and self.mechanism_mode not in MECHANISM_MODES:
            raise ValueError(f"mechanism_mode must be one of {MECHANISM_MODES}")
        if (
            self.candidate_revision_submode is not None
            and self.candidate_revision_submode not in CANDIDATE_REVISION_SUBMODES
        ):
            raise ValueError(
                "candidate_revision_submode must be one of "
                f"{CANDIDATE_REVISION_SUBMODES}"
            )
        if self.conflict_threshold not in ("low", "medium", "high"):
            raise ValueError("conflict_threshold must be low, medium, or high")
        if self.conflict_evidence_window not in (1, 3, 5):
            raise ValueError("conflict_evidence_window must be 1, 3, or 5")
        if self.parent_decay not in ("none", "slow", "fast"):
            raise ValueError("parent_decay must be none, slow, or fast")
        if self.post_split_selection_window not in (3, 5, 10):
            raise ValueError(
                "post_split_selection_window must be 3, 5, or 10"
            )
        if self.candidate_budget not in (None, 10, 15, 20):
            raise ValueError("candidate_budget must be None, 10, 15, or 20")
        if self.validation_threshold not in ("low", "medium", "high"):
            raise ValueError(
                "validation_threshold must be low, medium, or high"
            )
        if self.activation_threshold not in ("low", "medium", "high"):
            raise ValueError(
                "activation_threshold must be low, medium, or high"
            )
        if self.activation_delay not in (1, 3, 5):
            raise ValueError("activation_delay must be 1, 3, or 5")


class Simulation:
    def __init__(self, config: SimulationConfig, probe=None) -> None:
        self.config = config
        self.probe = probe
        self.world = GridWorld(
            config.width,
            config.height,
            config.seed,
            counts=config.counts,
            ambiguous_rgb=config.ambiguous_rgb,
            delayed_outcomes=config.delayed_outcomes,
        )
        if config.mechanism_mode:
            defaults = MECHANISM_FLAGS[config.mechanism_mode]
        else:
            defaults = (
                config.mode == "rd",
                config.mode in ("rd", "trace"),
                config.mode in ("rd", "trace"),
                False,
            )
        overrides = (
            config.feature_reentry_enabled,
            config.candidate_formation_enabled,
            config.candidate_revision_enabled,
            config.policy_reentry_enabled,
        )
        (
            self.feature_reentry_enabled,
            self.candidate_formation_enabled,
            self.candidate_revision_enabled,
            self.policy_reentry_enabled,
        ) = tuple(
            default if override is None else override
            for default, override in zip(defaults, overrides)
        )
        if self.candidate_revision_enabled and not self.candidate_formation_enabled:
            raise ValueError("candidate revision requires candidate formation")
        revision_submode = (
            config.candidate_revision_submode or "full_candidate_revision"
        )
        revision_flags = CANDIDATE_REVISION_FLAGS[revision_submode]
        if not self.candidate_revision_enabled:
            revision_flags = (False, False, False, False, False)
        (
            self.candidate_confidence_update_enabled,
            self.candidate_split_enabled,
            self.candidate_suppression_enabled,
            self.candidate_merge_enabled,
            self.candidate_obsolete_marking_enabled,
        ) = revision_flags
        self.candidate_revision_submode = (
            revision_submode if self.candidate_revision_enabled else None
        )
        self.candidate_targeted_split_enabled = revision_submode in (
            "targeted_split",
            "targeted_split_plus_suppression",
        )
        self.candidate_provisional_split_enabled = revision_submode in (
            "provisional_split",
            "provisional_split_plus_selection",
            "provisional_split_budgeted",
            "calibrated_provisional_split",
            "calibrated_provisional_split_budgeted",
        )
        self.candidate_selection_enabled = revision_submode in (
            "provisional_split_plus_selection",
            "provisional_split_budgeted",
            "calibrated_provisional_split",
            "calibrated_provisional_split_budgeted",
        )
        self.candidate_budget_enabled = (
            revision_submode == "provisional_split_budgeted"
            or revision_submode
            == "calibrated_provisional_split_budgeted"
        )
        self.candidate_calibrated_activation_enabled = (
            revision_submode
            in (
                "calibrated_provisional_split",
                "calibrated_provisional_split_budgeted",
            )
        )

        # Backward-compatible attributes for callers outside the mechanism audit.
        self.feature_reentry = self.feature_reentry_enabled
        self.candidate_reentry = self.candidate_revision_enabled
        self.policy_reentry = self.policy_reentry_enabled

        gate_strength = (
            config.reentry_strength if self.feature_reentry_enabled else 0.0
        )
        self.gate = BoundaryGate(
            reentry=self.feature_reentry_enabled,
            reentry_strength=gate_strength,
        )
        self.trace_field = (
            TemporalTraceField(self.gate)
            if config.delayed_outcomes
            else TraceField(self.gate)
        )
        self.stabilizer = ObjecthoodStabilizer(self.gate)
        self.destabilizer = Destabilizer()
        self.metrics = Metrics()
        self.evaluator = HiddenEvaluator(self.metrics)
        self.environment_change: dict[str, int] | None = None

        if config.mode in ("rd", "trace"):
            self.policy = AgentPolicy(
                self.stabilizer,
                config.seed,
                exploration=config.exploration,
                policy_reentry=self.policy_reentry_enabled,
            )
        elif config.mode == "fixed":
            self.policy = FixedWeightPolicy(config.seed)
        else:
            self.policy = RandomPolicy(config.seed)

    @property
    def learns_objecthood(self) -> bool:
        return self.candidate_formation_enabled

    def run(self) -> Metrics:
        for step in range(self.config.steps):
            if (
                self.config.environment_change_step is not None
                and step == self.config.environment_change_step
            ):
                if self.config.role_reversal:
                    self.environment_change = self.world.apply_role_reversal()
                else:
                    self.environment_change = self.world.apply_environment_change(
                        self.config.food_to_decoy,
                        self.config.change_hazard_color,
                    )
                self.evaluator.mark_environment_change(
                    step,
                    self.stabilizer,
                    self.destabilizer.total_splits,
                    self.stabilizer.total_merged,
                    self.stabilizer.total_created,
                    self.destabilizer.total_destabilizations,
                )
                if self.probe is not None:
                    self.probe.on_change(step, self)

            patch = self.world.observe()
            action = self.policy.choose(patch)
            if self.probe is not None:
                self.probe.before_step(step, self, patch, action)
            self.evaluator.record_decision(self.world, action)
            acted_on = self._adjacent_item(patch, action)
            result = self.world.step(action)

            destabilizations_before = self.destabilizer.total_destabilizations
            if isinstance(self.trace_field, TemporalTraceField):
                traces = self.trace_field.record_temporal(
                    acted_on, action, result
                )
            else:
                traces = [self.trace_field.record(acted_on, action, result)]
            for trace in traces:
                revised_candidate = None
                hold_parent = False
                if self.candidate_revision_enabled:
                    revised_candidate, hold_parent = self._revise_candidates(
                        trace, step
                    )
                if self.candidate_formation_enabled and not (
                    hold_parent and revised_candidate is None
                ):
                    self.stabilizer.formation_update(
                        trace,
                        preferred=revised_candidate,
                        record_evidence=not (
                            self.candidate_provisional_split_enabled
                            and hold_parent
                        ),
                    )
                if (
                    revised_candidate is not None
                    and self.candidate_targeted_split_enabled
                ):
                    self.destabilizer.mature_targeted_child(
                        revised_candidate,
                        self.stabilizer,
                        parent_decay=self.config.parent_decay,
                        suppression=self.candidate_suppression_enabled,
                    )
                if self.candidate_budget_enabled:
                    self.destabilizer.enforce_candidate_budget(
                        self.stabilizer,
                        self.config.candidate_budget,
                        step,
                    )
                if self.policy_reentry_enabled:
                    self.policy.update_from_trace(trace)
            if self.candidate_merge_enabled and (step + 1) % 25 == 0:
                self.stabilizer.merge_redundant()

            self.evaluator.record_step(
                result,
                self.stabilizer,
                self.gate.weights,
                self.destabilizer.total_splits,
                self.stabilizer.total_merged,
                self.destabilizer.total_destabilizations,
            )
            if self.probe is not None:
                self.probe.after_step(
                    step,
                    self,
                    result,
                    self.destabilizer.total_destabilizations
                    - destabilizations_before,
                )
        self.metrics._final_created = self.stabilizer.total_created
        return self.metrics

    def _revise_candidates(self, trace, step: int = 0):
        if self.candidate_provisional_split_enabled:
            return self.destabilizer.provisional_split(
                trace,
                self.stabilizer,
                step=step,
                selection_enabled=self.candidate_selection_enabled,
                selection_window=self.config.post_split_selection_window,
                calibrated_activation=(
                    self.candidate_calibrated_activation_enabled
                ),
                validation_threshold=self.config.validation_threshold,
                activation_threshold=self.config.activation_threshold,
                activation_delay=self.config.activation_delay,
            )

        if self.candidate_targeted_split_enabled:
            return self.destabilizer.targeted_split(
                trace,
                self.stabilizer,
                step=step,
                threshold=self.config.conflict_threshold,
                evidence_window=self.config.conflict_evidence_window,
            )

        prior, visual_distance = self.stabilizer.nearest(
            trace.features, trace.action
        )
        split_candidate = None
        conflict = False
        if prior is not None and visual_distance <= 0.20:
            conflict = self.destabilizer.is_conflict(prior, trace)
            split_candidate = self.destabilizer.revise(
                prior,
                trace,
                self.stabilizer,
                confidence_update=self.candidate_confidence_update_enabled,
                split=self.candidate_split_enabled,
                suppression=self.candidate_suppression_enabled,
                obsolete_marking=self.candidate_obsolete_marking_enabled,
            )
            if conflict:
                self.destabilizer.log_baseline_conflict(
                    prior,
                    trace,
                    self.stabilizer,
                    step,
                    split_triggered=split_candidate is not None,
                )
        if self.candidate_suppression_enabled:
            excluded = tuple(
                candidate
                for candidate in (prior, split_candidate)
                if candidate is not None
            )
            self.destabilizer.suppress_related(
                trace,
                self.stabilizer,
                excluded=excluded,
                obsolete_marking=self.candidate_obsolete_marking_enabled,
            )
        return split_candidate, False

    @staticmethod
    def _adjacent_item(
        patch: list[SensoryItem], action: tuple[int, int]
    ) -> SensoryItem:
        return next(
            item for item in patch if (item.rel_x, item.rel_y) == action
        )
