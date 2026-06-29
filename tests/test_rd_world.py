import unittest

from rd_world.boundary import BoundaryGate
from rd_world.calibrated_activation_validation import (
    run_calibrated_activation_suite,
    run_calibrated_activation_sweep,
)
from rd_world.candidate_revision_validation import run_candidate_revision_suite
from rd_world.objecthood import (
    ConflictAttribution,
    Destabilizer,
    ObjectCandidate,
    ObjecthoodStabilizer,
    ProvisionalFamily,
)
from rd_world.policy import AgentPolicy
from rd_world.provisional_split_validation import run_provisional_split_suite
from rd_world.simulation import Simulation, SimulationConfig
from rd_world.sweep import run_reentry_sweep
from rd_world.sharp_validation import run_sharp_suite
from rd_world.mechanism_validation import run_mechanism_suite
from rd_world.trace_field import TemporalTraceField
from rd_world.trace_field import TraceField
from rd_world.targeted_split_validation import (
    run_targeted_split_suite,
    run_targeted_split_sweep,
)
from rd_world.types import SensoryItem, StepResult, Trace
from rd_world.validation import run_validation, verify_observation_contract
from rd_world.world import COLORS, CHANGED_HAZARD_COLOR, Entity, GridWorld


class RDWorldTests(unittest.TestCase):
    def test_reentry_changes_weights_but_baseline_does_not(self) -> None:
        item_a = SensoryItem(0.2, 0.86, 0.22, 1, 0)
        item_b = SensoryItem(0.24, 0.82, 0.23, 1, 0)
        positive = StepResult(True, 0.0, 7.95, 58.0)
        neutral = StepResult(True, 0.0, -0.05, 50.0)

        learned_gate = BoundaryGate(reentry=True)
        learned = TraceField(learned_gate)
        fixed_gate = BoundaryGate(reentry=False)
        fixed = TraceField(fixed_gate)
        initial = fixed_gate.weights.copy()

        for _ in range(8):
            learned.record(item_a, (1, 0), positive)
            learned.record(item_b, (1, 0), neutral)
            fixed.record(item_a, (1, 0), positive)
            fixed.record(item_b, (1, 0), neutral)

        self.assertNotEqual(learned_gate.weights, initial)
        self.assertEqual(fixed_gate.weights, initial)
        self.assertAlmostEqual(sum(learned_gate.weights.values()), 1.0)
        self.assertGreater(
            max(learned_gate.weights.values()) - min(learned_gate.weights.values()),
            0.01,
        )
        self.assertTrue(
            any(
                learned_gate.weights[name] > initial[name]
                for name in learned_gate.weights
            )
        )
        self.assertTrue(
            any(
                learned_gate.weights[name] < initial[name]
                for name in learned_gate.weights
            )
        )

    def test_short_simulation_records_metrics(self) -> None:
        simulation = Simulation(SimulationConfig(steps=20, seed=3))
        metrics = simulation.run()
        self.assertEqual(len(metrics.viability), 20)
        self.assertEqual(len(metrics.boundary_weights), 20)
        self.assertGreater(len(simulation.trace_field.traces), 0)
        self.assertGreater(len(simulation.stabilizer.candidates), 0)

    def test_failed_prediction_creates_split_candidate(self) -> None:
        gate = BoundaryGate()
        stabilizer = ObjecthoodStabilizer(gate)
        destabilizer = Destabilizer()
        item = SensoryItem(0.2, 0.86, 0.22, 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)
        candidate = stabilizer.create_candidate(neutral)

        split = destabilizer.inspect(candidate, positive, stabilizer)

        self.assertTrue(split)
        self.assertEqual(len(stabilizer.candidates), 2)

    def test_background_frequency_does_not_create_objecthood(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["empty"], 1, 0)
        trace = Trace(item.features(), (1, 0), 0.0, -0.05)
        candidate = stabilizer.create_candidate(trace)
        for _ in range(20):
            candidate.update(trace)
        self.assertLess(candidate.objecthood_score, 0.05)

    def test_wall_trace_learns_resistance(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["wall"], 1, 0)
        trace = Trace(item.features(), (1, 0), 1.0, -0.05)
        candidate = stabilizer.create_candidate(trace)
        for _ in range(4):
            candidate.update(trace)
        delta, resistance, confidence = stabilizer.predict(item, (1, 0))
        self.assertAlmostEqual(delta, -0.05)
        self.assertGreater(resistance, 0.9)
        self.assertGreater(confidence, 0.5)

    def test_food_and_decoy_split_by_outcome_signature(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        food = SensoryItem(*COLORS["food"], 1, 0)
        decoy = SensoryItem(*COLORS["decoy"], 1, 0)
        for _ in range(4):
            stabilizer.process(Trace(food.features(), (1, 0), 0.0, 7.95))
            stabilizer.process(Trace(decoy.features(), (1, 0), 0.0, -0.05))

        food_delta, _, food_confidence = stabilizer.predict(food, (1, 0))
        decoy_delta, _, _ = stabilizer.predict(decoy, (1, 0))
        self.assertGreater(food_delta, 7.0)
        self.assertLess(decoy_delta, 0.5)
        self.assertGreater(food_confidence, 0.5)

    def test_policy_approaches_positive_and_avoids_hazard(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        food = SensoryItem(*COLORS["food"], 1, 0)
        hazard = SensoryItem(*COLORS["hazard"], 1, 0)
        for _ in range(5):
            stabilizer.process(Trace(food.features(), (1, 0), 0.0, 7.95))
            stabilizer.process(Trace(hazard.features(), (1, 0), 0.0, -7.05))
        policy = AgentPolicy(stabilizer, seed=1, exploration=0.0)

        empty = COLORS["empty"]
        approach_patch = [
            SensoryItem(*empty, 0, -1),
            SensoryItem(*empty, 1, 0),
            SensoryItem(*empty, 0, 1),
            SensoryItem(*empty, -1, 0),
            SensoryItem(*COLORS["food"], 2, 0),
        ]
        self.assertEqual(policy.choose(approach_patch), (1, 0))

        avoid_patch = [
            SensoryItem(*empty, 0, -1),
            SensoryItem(*COLORS["hazard"], 1, 0),
            SensoryItem(*empty, 0, 1),
            SensoryItem(*empty, -1, 0),
        ]
        self.assertNotEqual(policy.choose(avoid_patch), (1, 0))

    def test_public_observation_contract_has_no_hidden_labels(self) -> None:
        self.assertTrue(verify_observation_contract())
        forbidden = {"food", "wall", "hazard", "decoy", "kind", "label"}
        for data_class in (SensoryItem, StepResult, Trace, ObjectCandidate):
            self.assertFalse(
                forbidden
                & {name.lower() for name in data_class.__dataclass_fields__}
            )
        self.assertFalse(
            forbidden
            & {
                name.lower()
                for name in ConflictAttribution.__dataclass_fields__
            }
        )

        policy = AgentPolicy(ObjecthoodStabilizer(BoundaryGate()), seed=1)
        self.assertFalse(forbidden & {name.lower() for name in vars(policy)})

    def test_environment_change_preserves_food_rgb_for_converted_decoys(self) -> None:
        world = GridWorld(seed=7)
        changed = world.apply_environment_change(food_to_decoy=3)
        converted = [
            entity
            for entity in world.entities.values()
            if entity.kind == "decoy" and entity.color_override == COLORS["food"]
        ]
        self.assertEqual(len(converted), changed["food_to_decoy"])
        self.assertEqual(world.colors["hazard"], CHANGED_HAZARD_COLOR)

    def test_validation_runs_all_four_ablations(self) -> None:
        result = run_validation(steps=30, seed=5, change_step=15)
        self.assertEqual(set(result.simulations), {"rd", "trace", "fixed", "random"})
        self.assertTrue(result.observation_contract_valid)
        for simulation in result.simulations.values():
            self.assertEqual(len(simulation.metrics.viability), 30)
        self.assertGreaterEqual(
            result.simulations["rd"].metrics.post_change_new_candidates, 0
        )

    def test_conflicting_same_rgb_trace_weakens_old_candidate(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        destabilizer = Destabilizer()
        item = SensoryItem(*COLORS["food"], 1, 0)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)
        neutral = Trace(item.features(), (-1, 0), 0.0, -0.05)
        candidate = stabilizer.create_candidate(positive)
        before = candidate.objecthood_score

        weakened = destabilizer.weaken_related(neutral, stabilizer)

        self.assertEqual(weakened, 1)
        self.assertLess(candidate.objecthood_score, before)

    def test_redundant_candidates_can_merge(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["wall"], 1, 0)
        trace = Trace(item.features(), (1, 0), 1.0, -0.05)
        stabilizer.create_candidate(trace)
        stabilizer.create_candidate(trace)
        merges = stabilizer.merge_redundant()
        self.assertEqual(merges, 1)
        self.assertEqual(len(stabilizer.candidates), 1)

    def test_zero_reentry_strength_keeps_gate_fixed(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=40,
                seed=7,
                mode="rd",
                reentry_strength=0.0,
            )
        )
        initial = simulation.gate.weights.copy()
        simulation.run()
        self.assertEqual(simulation.gate.weights, initial)

    def test_normalized_decoy_false_positive_uses_green_approaches(self) -> None:
        simulation = Simulation(
            SimulationConfig(steps=100, seed=7, mode="random")
        )
        metrics = simulation.run()
        denominator = (
            metrics.green_food_approaches + metrics.green_decoy_approaches
        )
        expected = (
            metrics.green_decoy_approaches / denominator if denominator else 0.0
        )
        self.assertEqual(metrics.normalized_decoy_false_positive_rate, expected)

    def test_event_based_stability_drops_on_destabilization(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=220,
                seed=7,
                mode="rd",
                environment_change_step=150,
            )
        )
        metrics = simulation.run()
        event_steps = [
            index
            for index in range(1, len(metrics.destabilization_events))
            if metrics.destabilization_events[index]
            > metrics.destabilization_events[index - 1]
        ]
        self.assertTrue(event_steps)
        self.assertTrue(
            any(metrics.candidate_stability[index] < 0.95 for index in event_steps)
        )

    def test_small_reentry_sweep_reports_mean_and_std(self) -> None:
        result = run_reentry_sweep(
            strengths=(0.0, 0.25),
            runs=2,
            seed_start=3,
            steps=40,
            change_step=20,
        )
        self.assertEqual(len(result.conditions), 2)
        self.assertEqual(result.conditions[0].runs, 2)
        self.assertIn("stability_plasticity", result.conditions[0].metrics)

    def test_role_reversal_swaps_food_and_decoy_effects(self) -> None:
        world = GridWorld(seed=2)
        self.assertGreater(world.current_effect("food"), 0.0)
        self.assertEqual(world.current_effect("decoy"), 0.0)
        world.apply_role_reversal()
        self.assertEqual(world.current_effect("food"), 0.0)
        self.assertGreater(world.current_effect("decoy"), 0.0)

    def test_ambiguous_rgb_distributions_overlap(self) -> None:
        world = GridWorld(seed=4, ambiguous_rgb=True)
        colors = {
            kind: [
                entity.color_override
                for entity in world.entities.values()
                if entity.kind == kind
            ]
            for kind in ("food", "decoy", "hazard")
        }
        means = {
            kind: tuple(
                sum(color[channel] for color in values) / len(values)
                for channel in range(3)
            )
            for kind, values in colors.items()
        }
        distance = sum(
            abs(a - b) for a, b in zip(means["food"], means["decoy"])
        ) / 3.0
        self.assertLess(distance, 0.15)

    def test_temporal_trace_binds_delayed_outcome_to_prior_context(self) -> None:
        gate = BoundaryGate(reentry=False)
        field = TemporalTraceField(gate)
        salient = SensoryItem(0.3, 0.7, 0.3, 1, 0)
        empty = SensoryItem(*COLORS["empty"], 1, 0)
        field.record_temporal(
            salient, (1, 0), StepResult(True, 0.0, -0.05, 50.0)
        )
        for _ in range(2):
            field.record_temporal(
                empty, (1, 0), StepResult(True, 0.0, -0.05, 50.0)
            )
        traces = field.record_temporal(
            empty, (1, 0), StepResult(True, 0.0, 7.95, 58.0)
        )
        credited = max(traces, key=lambda trace: trace.delta_viability)
        self.assertEqual(credited.features, salient.features())
        self.assertGreater(credited.delta_viability, 7.0)

    def test_small_sharp_suite_reports_paired_ci(self) -> None:
        result = run_sharp_suite(
            strengths=(0.0, 0.5),
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
        )
        self.assertEqual(len(result.conditions), 2)
        self.assertIn(
            "adaptation_lag", result.conditions[1].paired_differences
        )

    def test_paired_seed_worlds_share_initial_layout_and_colors(self) -> None:
        configs = [
            SimulationConfig(
                steps=1,
                seed=9,
                mode="rd",
                reentry_strength=strength,
                ambiguous_rgb=True,
                delayed_outcomes=True,
                role_reversal=True,
            )
            for strength in (0.0, 0.5, 1.0)
        ]
        signatures = []
        for config in configs:
            simulation = Simulation(config)
            signatures.append(
                sorted(
                    (
                        position,
                        entity.kind,
                        entity.color_override,
                        entity.entity_id,
                    )
                    for position, entity in simulation.world.entities.items()
                )
            )
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_mechanism_modes_enable_only_named_reentry_targets(self) -> None:
        expected = {
            "no_reentry": (False, True, False, False),
            "feature_reentry": (True, True, False, False),
            "candidate_reentry": (False, True, True, False),
            "policy_reentry": (False, True, False, True),
            "full_reentry": (True, True, True, True),
        }
        for mode, flags in expected.items():
            simulation = Simulation(
                SimulationConfig(
                    steps=1,
                    seed=2,
                    mode="rd",
                    mechanism_mode=mode,
                )
            )
            self.assertEqual(
                (
                    simulation.feature_reentry_enabled,
                    simulation.candidate_formation_enabled,
                    simulation.candidate_revision_enabled,
                    simulation.policy_reentry_enabled,
                ),
                flags,
            )

    def test_non_candidate_reentry_modes_still_form_candidates(self) -> None:
        for mode in ("no_reentry", "feature_reentry", "policy_reentry"):
            simulation = Simulation(
                SimulationConfig(
                    steps=80,
                    seed=3,
                    mode="rd",
                    mechanism_mode=mode,
                    width=8,
                    height=8,
                    counts={"wall": 4, "food": 15, "decoy": 15, "hazard": 8},
                    exploration=0.3,
                )
            )
            simulation.run()
            self.assertGreater(
                len(simulation.stabilizer.candidates),
                0,
                msg=f"{mode} did not form candidates",
            )
            if mode == "policy_reentry":
                self.assertGreater(len(simulation.policy.preference_values), 0)

    def test_candidate_reentry_adds_revision_without_gating_formation(self) -> None:
        simulations = {}
        for mode in ("no_reentry", "candidate_reentry"):
            simulation = Simulation(
                SimulationConfig(
                    steps=120,
                    seed=8,
                    mode="rd",
                    mechanism_mode=mode,
                    environment_change_step=60,
                    role_reversal=True,
                    ambiguous_rgb=True,
                    width=8,
                    height=8,
                    counts={"wall": 4, "food": 15, "decoy": 15, "hazard": 8},
                    exploration=0.3,
                )
            )
            simulation.run()
            simulations[mode] = simulation

        no_reentry = simulations["no_reentry"]
        candidate_reentry = simulations["candidate_reentry"]
        self.assertGreater(len(no_reentry.stabilizer.candidates), 0)
        self.assertGreater(len(candidate_reentry.stabilizer.candidates), 0)
        self.assertEqual(no_reentry.destabilizer.total_destabilizations, 0)
        self.assertEqual(no_reentry.destabilizer.total_splits, 0)
        self.assertEqual(no_reentry.stabilizer.total_merged, 0)
        self.assertGreater(
            candidate_reentry.destabilizer.total_destabilizations
            + candidate_reentry.destabilizer.total_splits
            + candidate_reentry.stabilizer.total_merged,
            0,
        )

    def test_small_mechanism_suite_reports_paired_audits(self) -> None:
        result = run_mechanism_suite(
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
        )
        self.assertEqual(
            [condition.mode for condition in result.conditions],
            [
                "no_reentry",
                "feature_reentry",
                "candidate_reentry",
                "policy_reentry",
                "full_reentry",
            ],
        )
        self.assertIn(
            "obsolete_persistence",
            result.conditions[-1].paired_differences,
        )

    def test_candidate_revision_submode_flags_are_isolated(self) -> None:
        expected = {
            "confidence_only": (True, False, False, False, False),
            "split_only": (False, True, False, False, False),
            "suppression_only": (False, False, True, False, False),
            "split_plus_suppression": (False, True, True, False, False),
            "full_candidate_revision": (True, True, True, True, True),
            "targeted_split": (False, True, False, False, False),
            "targeted_split_plus_suppression": (
                False,
                True,
                True,
                False,
                False,
            ),
            "provisional_split": (
                False,
                True,
                False,
                False,
                False,
            ),
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
        }
        for submode, flags in expected.items():
            simulation = Simulation(
                SimulationConfig(
                    steps=1,
                    candidate_formation_enabled=True,
                    candidate_revision_enabled=True,
                    candidate_revision_submode=submode,
                    feature_reentry_enabled=False,
                    policy_reentry_enabled=False,
                )
            )
            self.assertEqual(
                (
                    simulation.candidate_confidence_update_enabled,
                    simulation.candidate_split_enabled,
                    simulation.candidate_suppression_enabled,
                    simulation.candidate_merge_enabled,
                    simulation.candidate_obsolete_marking_enabled,
                ),
                flags,
            )

    def test_candidate_revision_submodes_perform_only_allowed_operations(self) -> None:
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)

        simulations = {}
        for submode in (
            "confidence_only",
            "split_only",
            "suppression_only",
            "split_plus_suppression",
            "full_candidate_revision",
        ):
            simulation = Simulation(
                SimulationConfig(
                    steps=0,
                    candidate_formation_enabled=True,
                    candidate_revision_enabled=True,
                    candidate_revision_submode=submode,
                    feature_reentry_enabled=False,
                    policy_reentry_enabled=False,
                )
            )
            candidate = simulation.stabilizer.create_candidate(neutral)
            simulation._revise_candidates(positive)
            simulations[submode] = (simulation, candidate)

        confidence, confidence_candidate = simulations["confidence_only"]
        self.assertLess(confidence_candidate.inhibition, 1.0)
        self.assertEqual(confidence.destabilizer.total_splits, 0)
        self.assertEqual(confidence.destabilizer.total_suppressions, 0)

        split, split_candidate = simulations["split_only"]
        self.assertEqual(split_candidate.inhibition, 1.0)
        self.assertEqual(split.destabilizer.total_splits, 1)
        self.assertEqual(split.destabilizer.total_suppressions, 0)

        suppression, suppression_candidate = simulations["suppression_only"]
        self.assertLess(suppression_candidate.inhibition, 1.0)
        self.assertEqual(suppression.destabilizer.total_splits, 0)
        self.assertEqual(suppression.destabilizer.total_suppressions, 1)
        self.assertEqual(suppression.destabilizer.total_obsolete_marked, 0)

        combined, _ = simulations["split_plus_suppression"]
        self.assertEqual(combined.destabilizer.total_splits, 1)
        self.assertEqual(combined.destabilizer.total_suppressions, 1)
        self.assertEqual(combined.destabilizer.total_obsolete_marked, 0)

        full, _ = simulations["full_candidate_revision"]
        full._revise_candidates(positive)
        self.assertGreaterEqual(full.destabilizer.total_splits, 1)
        self.assertGreaterEqual(full.destabilizer.total_suppressions, 1)
        self.assertGreaterEqual(full.destabilizer.total_obsolete_marked, 1)
        self.assertGreaterEqual(full.stabilizer.total_pruned, 1)

    def test_small_candidate_revision_suite_uses_formation_baseline(self) -> None:
        result = run_candidate_revision_suite(
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
        )
        self.assertEqual(
            [condition.mode for condition in result.conditions],
            [
                "no_candidate_revision",
                "confidence_only",
                "split_only",
                "suppression_only",
                "split_plus_suppression",
                "full_candidate_revision",
            ],
        )
        self.assertIn(
            "candidate_churn",
            result.conditions[-1].paired_differences,
        )
        self.assertIn("suppression_added_to_split", result.contrasts)

    def test_targeted_split_requires_compatible_evidence_window(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_formation_enabled=True,
                candidate_revision_enabled=True,
                candidate_revision_submode="targeted_split",
                conflict_threshold="low",
                conflict_evidence_window=3,
                parent_decay="slow",
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(
            item.features(),
            (1, 0),
            0.0,
            7.95,
            source_step=1,
            commit_step=4,
            temporal_delay=3,
            temporal_eligibility=1.0,
        )
        parent = simulation.stabilizer.create_candidate(neutral)

        for step in range(2):
            child, targeted = simulation._revise_candidates(positive, step)
            self.assertTrue(targeted)
            self.assertIsNone(child)
        child, targeted = simulation._revise_candidates(positive, 2)

        self.assertTrue(targeted)
        self.assertIsNotNone(child)
        self.assertEqual(simulation.destabilizer.total_splits, 1)
        self.assertEqual(
            [
                log.evidence_count
                for log in simulation.destabilizer.conflict_logs
            ],
            [1, 2, 3],
        )
        self.assertEqual(parent.inhibition, 1.0)

        incompatible = Trace(
            item.features(),
            (-1, 0),
            0.0,
            7.95,
            source_step=2,
            commit_step=5,
            temporal_delay=3,
            temporal_eligibility=1.0,
        )
        _, targeted = simulation._revise_candidates(incompatible, 3)
        self.assertFalse(targeted)

    def test_targeted_split_separates_viability_and_resistance_conflicts(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="targeted_split",
                conflict_threshold="low",
                conflict_evidence_window=1,
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        simulation.stabilizer.create_candidate(neutral)
        positive = Trace(
            item.features(),
            (1, 0),
            0.0,
            7.95,
            temporal_eligibility=1.0,
        )
        simulation._revise_candidates(positive, 1)
        viability_log = simulation.destabilizer.conflict_logs[-1]
        self.assertTrue(viability_log.viability_conflict)
        self.assertFalse(viability_log.resistance_conflict)

        wall_simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="targeted_split",
                conflict_threshold="low",
                conflict_evidence_window=1,
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        wall_simulation.stabilizer.create_candidate(neutral)
        blocked = Trace(
            item.features(),
            (1, 0),
            1.0,
            -0.05,
            temporal_eligibility=1.0,
        )
        wall_simulation._revise_candidates(blocked, 1)
        resistance_log = wall_simulation.destabilizer.conflict_logs[-1]
        self.assertFalse(resistance_log.viability_conflict)
        self.assertTrue(resistance_log.resistance_conflict)

    def test_targeted_child_matures_before_parent_decay(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="targeted_split",
                conflict_threshold="low",
                conflict_evidence_window=1,
                parent_decay="slow",
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(
            item.features(),
            (1, 0),
            0.0,
            7.95,
            temporal_eligibility=1.0,
        )
        parent = simulation.stabilizer.create_candidate(neutral)
        child, _ = simulation._revise_candidates(positive, 1)
        self.assertIsNotNone(child)
        simulation.stabilizer.formation_update(positive, preferred=child)
        simulation.destabilizer.mature_targeted_child(
            child,
            simulation.stabilizer,
            parent_decay="slow",
            suppression=False,
        )
        self.assertEqual(parent.inhibition, 1.0)

        child, _ = simulation._revise_candidates(positive, 2)
        simulation.stabilizer.formation_update(positive, preferred=child)
        simulation.destabilizer.mature_targeted_child(
            child,
            simulation.stabilizer,
            parent_decay="slow",
            suppression=False,
        )
        self.assertAlmostEqual(parent.inhibition, 0.85)
        log = simulation.destabilizer.conflict_logs[-1]
        self.assertTrue(log.parent_decay_applied)
        self.assertEqual(log.action_distance, 0.0)
        self.assertGreaterEqual(log.temporal_eligibility, 0.0)

    def test_small_targeted_split_audits_report_targeting_metrics(self) -> None:
        result = run_targeted_split_suite(
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
            conflict_threshold="low",
            evidence_window=1,
            parent_decay="none",
        )
        self.assertEqual(
            [condition.label for condition in result.conditions],
            list(("split_only", "targeted_split", "targeted_split_plus_suppression")),
        )
        self.assertIn("target_precision", result.conditions[1].metrics)
        self.assertTrue(result.evaluated_logs)

        sweep = run_targeted_split_sweep(
            runs=1,
            seed_start=1,
            steps=30,
            change_step=15,
            thresholds=("low",),
            evidence_windows=(1,),
            parent_decays=("none", "slow"),
        )
        self.assertEqual(len(sweep.conditions), 3)

    def test_candidate_status_and_evidence_buffers_update(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)
        candidate = stabilizer.create_candidate(neutral)

        self.assertEqual(candidate.status, "active")
        candidate.record_evidence(positive)

        self.assertEqual(candidate.conflict_count, 1)
        self.assertTrue(candidate.prediction_error)
        self.assertTrue(candidate.action_outcome_consistency)
        self.assertTrue(candidate.viability_prediction_accuracy)
        self.assertTrue(candidate.resistance_prediction_accuracy)

        candidate.provisional = True
        self.assertEqual(candidate.status, "provisional")
        candidate.obsolete = True
        self.assertEqual(candidate.status, "obsolete")

    def test_candidate_status_calibrates_policy_influence(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["food"], 1, 0)
        candidate = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, 7.95)
        )

        candidate.status = "provisional"
        provisional = candidate.policy_influence
        candidate.status = "validated"
        validated = candidate.policy_influence
        candidate.status = "active"
        active = candidate.policy_influence
        candidate.status = "obsolete"

        self.assertLess(provisional, validated)
        self.assertLess(validated, active)
        self.assertEqual(candidate.policy_influence, 0.0)

    def test_policy_influence_is_separate_from_candidate_existence(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, -0.05)
        )
        positive = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, 7.95)
        )
        neutral.status = "active"
        positive.status = "provisional"

        evaluator_delta, _, _ = stabilizer.predict(item, (1, 0))
        policy_delta, _, _ = stabilizer.predict(
            item, (1, 0), record_policy_influence=True
        )

        self.assertGreater(evaluator_delta, policy_delta)
        self.assertGreater(evaluator_delta, 3.0)

        stable_delta, _, _ = stabilizer.predict_stabilized(item, (1, 0))
        self.assertAlmostEqual(stable_delta, neutral.predicted_delta_viability)

    def test_calibrated_child_lifecycle_delays_parent_obsolescence(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="calibrated_provisional_split",
                post_split_selection_window=3,
                validation_threshold="low",
                activation_threshold="low",
                activation_delay=1,
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)

        def trace(delta: float, step: int) -> Trace:
            return Trace(
                item.features(),
                (1, 0),
                0.0,
                delta,
                source_step=step,
                commit_step=step,
            )

        parent = simulation.stabilizer.create_candidate(trace(-0.05, 0))
        child, handled = simulation._revise_candidates(trace(7.95, 1), 1)
        self.assertTrue(handled)
        self.assertIsNotNone(child)
        simulation.stabilizer.formation_update(
            trace(7.95, 1), preferred=child, record_evidence=False
        )
        self.assertEqual(child.status, "provisional")
        self.assertEqual(parent.inhibition, 1.0)

        for step in (2, 3, 4):
            preferred, _ = simulation._revise_candidates(
                trace(7.95, step), step
            )
            simulation.stabilizer.formation_update(
                trace(7.95, step),
                preferred=preferred,
                record_evidence=False,
            )

        self.assertEqual(child.status, "validated")
        self.assertTrue(child.ever_validated)
        self.assertLess(parent.inhibition, 1.0)
        self.assertIn(parent, simulation.stabilizer.candidates)

        preferred, _ = simulation._revise_candidates(trace(7.95, 5), 5)
        simulation.stabilizer.formation_update(
            trace(7.95, 5), preferred=preferred, record_evidence=False
        )
        self.assertEqual(child.status, "active")
        self.assertTrue(child.ever_activated)
        self.assertIn(parent, simulation.stabilizer.candidates)

        for step in (6, 7):
            preferred, _ = simulation._revise_candidates(
                trace(7.95, step), step
            )
            simulation.stabilizer.formation_update(
                trace(7.95, step),
                preferred=preferred,
                record_evidence=False,
            )

        self.assertNotIn(parent, simulation.stabilizer.candidates)
        self.assertEqual(parent.status, "obsolete")
        self.assertTrue(child.won_parent_competition)
        self.assertEqual(simulation.destabilizer.total_child_validated, 1)
        self.assertEqual(simulation.destabilizer.total_child_activated, 1)

    def test_provisional_parent_stays_active_until_child_wins(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="provisional_split_plus_selection",
                post_split_selection_window=3,
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)
        parent = simulation.stabilizer.create_candidate(neutral)

        child, handled = simulation._revise_candidates(positive, 1)
        self.assertTrue(handled)
        self.assertIsNotNone(child)
        simulation.stabilizer.formation_update(
            positive, preferred=child, record_evidence=False
        )
        self.assertEqual(parent.status, "active")
        self.assertEqual(parent.inhibition, 1.0)
        self.assertEqual(child.status, "provisional")

        for step in (2, 3):
            preferred, handled = simulation._revise_candidates(
                positive, step
            )
            self.assertTrue(handled)
            simulation.stabilizer.formation_update(
                positive,
                preferred=preferred,
                record_evidence=False,
            )
            self.assertEqual(parent.inhibition, 1.0)

        preferred, handled = simulation._revise_candidates(positive, 4)
        self.assertTrue(handled)
        simulation.stabilizer.formation_update(
            positive, preferred=preferred, record_evidence=False
        )

        self.assertEqual(child.status, "active")
        self.assertTrue(child.won_parent_competition)
        self.assertLess(parent.inhibition, 1.0)
        self.assertEqual(simulation.destabilizer.total_child_wins, 1)

    def test_selection_prunes_weak_child_without_parent_decay(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                steps=0,
                candidate_revision_enabled=True,
                candidate_revision_submode="provisional_split_plus_selection",
                post_split_selection_window=3,
                feature_reentry_enabled=False,
                policy_reentry_enabled=False,
            )
        )
        item = SensoryItem(*COLORS["food"], 1, 0)
        neutral = Trace(item.features(), (1, 0), 0.0, -0.05)
        positive = Trace(item.features(), (1, 0), 0.0, 7.95)
        parent = simulation.stabilizer.create_candidate(neutral)
        child, _ = simulation._revise_candidates(positive, 1)
        simulation.stabilizer.formation_update(
            positive, preferred=child, record_evidence=False
        )

        for step in (2, 3, 4):
            preferred, _ = simulation._revise_candidates(neutral, step)
            simulation.stabilizer.formation_update(
                neutral,
                preferred=preferred,
                record_evidence=False,
            )

        self.assertNotIn(child, simulation.stabilizer.candidates)
        self.assertEqual(child.status, "obsolete")
        self.assertEqual(parent.inhibition, 1.0)

    def test_candidate_budget_prunes_lowest_usefulness_not_age(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        destabilizer = Destabilizer()
        food = SensoryItem(*COLORS["food"], 1, 0)
        hazard = SensoryItem(*COLORS["hazard"], -1, 0)
        empty = SensoryItem(*COLORS["empty"], 0, 1)
        positive = stabilizer.create_candidate(
            Trace(food.features(), (1, 0), 0.0, 7.95)
        )
        negative = stabilizer.create_candidate(
            Trace(hazard.features(), (-1, 0), 0.0, -7.05)
        )
        low_usefulness = stabilizer.create_candidate(
            Trace(empty.features(), (0, 1), 0.0, -0.05)
        )

        destabilizer.enforce_candidate_budget(stabilizer, 2, step=10)

        self.assertIn(positive, stabilizer.candidates)
        self.assertIn(negative, stabilizer.candidates)
        self.assertNotIn(low_usefulness, stabilizer.candidates)

    def test_redundant_provisional_children_merge(self) -> None:
        stabilizer = ObjecthoodStabilizer(BoundaryGate())
        destabilizer = Destabilizer()
        item = SensoryItem(*COLORS["food"], 1, 0)
        parent = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, -0.05)
        )
        first = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, 7.95)
        )
        second = stabilizer.create_candidate(
            Trace(item.features(), (1, 0), 0.0, 7.80)
        )
        for child in (first, second):
            child.parent_candidate_id = parent.candidate_id
            child.status = "provisional"
        family = ProvisionalFamily(
            parent_id=parent.candidate_id,
            child_ids=[first.candidate_id, second.candidate_id],
            split_step=1,
        )

        merges = destabilizer.merge_redundant_children(
            family, stabilizer
        )

        self.assertEqual(merges, 1)
        self.assertEqual(len(family.child_ids), 1)
        self.assertEqual(stabilizer.total_merged, 1)

    def test_small_provisional_split_suite_reports_lifecycle_metrics(self) -> None:
        result = run_provisional_split_suite(
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
            selection_window=3,
            candidate_budget=10,
        )
        self.assertEqual(
            [condition.mode for condition in result.conditions],
            [
                "split_only",
                "targeted_split",
                "provisional_split",
                "provisional_split_plus_selection",
                "provisional_split_budgeted",
            ],
        )
        self.assertIn(
            "parent_survival_time", result.conditions[-1].metrics
        )
        self.assertIn(
            "child_win_rate",
            result.conditions[-1].paired_differences,
        )
        self.assertIn("selection_vs_targeted", result.contrasts)

    def test_small_calibrated_activation_suite_reports_status_metrics(self) -> None:
        result = run_calibrated_activation_suite(
            runs=2,
            seed_start=1,
            steps=60,
            change_step=30,
            validation_threshold="low",
            activation_threshold="low",
            activation_delay=1,
            selection_window=3,
            candidate_budget=10,
        )
        self.assertEqual(
            [condition.label for condition in result.conditions],
            [
                "split_only",
                "provisional_split_plus_selection",
                "calibrated_provisional_split",
                "calibrated_provisional_split_budgeted",
            ],
        )
        calibrated = result.conditions[2]
        self.assertIn("child_validation_rate", calibrated.metrics)
        self.assertIn("child_activation_rate", calibrated.metrics)
        self.assertIn("validated_policy_influence", calibrated.metrics)
        self.assertIn(
            "false_stabilization_rate",
            calibrated.paired_vs_split_only,
        )
        self.assertIn(
            "false_stabilization_rate",
            calibrated.paired_vs_provisional_selection,
        )

        sweep = run_calibrated_activation_sweep(
            runs=1,
            seed_start=1,
            steps=30,
            change_step=15,
            validation_thresholds=("low",),
            activation_thresholds=("medium",),
            activation_delays=(1, 3, 5),
            selection_window=3,
        )
        self.assertEqual(len(sweep.conditions), 5)


if __name__ == "__main__":
    unittest.main()
