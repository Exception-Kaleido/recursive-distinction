# Changelog

## v0.1 - 2026-06-16

Initial RD-World research snapshot.

### Added

- A label-hidden 2D grid world with walls, food, hazards, and visually similar
  decoys.
- Rule-based `BoundaryGate`, `TraceField`, `ObjecthoodStabilizer`,
  `Destabilizer`, and candidate-guided policy components.
- Feature-, candidate-, and policy-level re-entry ablations with paired seeds.
- Ambiguous RGB, delayed outcomes, non-stationary role reversal, and
  evaluator-only hidden-label metrics.
- Candidate formation/revision separation and candidate revision decomposition.
- Conflict attribution, precision-first targeted split, generate-then-select
  provisional branching, evidence-window selection, and calibrated activation.
- Candidate lifecycle states: `active`, `provisional`, `validated`, and
  `obsolete`.
- Validation dashboards, confidence intervals, threshold sweeps, and research
  documentation.

### Frozen v0.1 finding

The strongest supported claim is:

> Conflict-driven objecthood splitting improves adaptation in a non-stationary,
> ambiguous role-reversal world.

The refined v0.1 mechanism is:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

Feature re-entry alone, policy re-entry alone, and precision-first targeting
without an adaptation cost are not supported by the v0.1 audits. Reliable
post-change viability recovery and broader generalization remain unresolved.

### Scope

v0.1 is a transparent rule-based objecthood experiment. It does not establish
consciousness, selfhood, or worldhood.
