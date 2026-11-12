# Recursive Distinction (RD) · RD-World v0.1

**An exploratory attempt to translate a philosophical intuition about distinction, trace, and re-entry into an executable computational experiment.**

**Project status:** June 2026 research prototype · historical results preserved · November 2026 audit identified implementation and measurement limitations. [Read the audit notice](docs/AUDIT_STATUS.md).

## The question that started this project

What if a distinction does not merely identify something already given, but leaves a trace that changes how subsequent distinctions are made?

The starting intuition can be expressed as:

```text
distinction -> trace -> re-entry into future distinction
       ^                          |
       |__________________________|
```

If distinctions repeatedly alter the conditions of subsequent distinctions, could relatively stable, revisable organizations emerge from that process?

**Recursive Distinction (RD)** names this proposed generative process. It is a *research hypothesis*, not an established general principle. The purpose of RD-World v0.1 was to take this abstract intuition out of philosophical discussion and make some of its consequences computationally explicit, observable, and open to failure.

The first experimental target was **objecthood**: not a metaphysical claim about what an object ultimately is, but an operational question about how an agent might develop persistent, action-relevant predictive candidates from its own interactions.

> **Central research question:** Can traces of distinctions, actions, and outcomes re-enter an agent's subsequent processing so that object-like predictive structures form, persist, and change when their predictions fail?

## From intuition to operational hypothesis

The project makes a deliberate translation from conceptual terms to specific computational operations:

| Conceptual idea | RD-World v0.1 operationalization |
|---|---|
| Distinction | Selected numerical differences in a local sensory context |
| Action | A choice that probes the environment |
| Trace | A stored relation among sensory context, action, resistance, and viability change |
| Re-entry | Stored evidence changes later feature weighting or candidate organization |
| Stabilization | Repeated predictive support increases a candidate's persistence and influence |
| Revision | Prediction conflict can weaken, branch, select, merge, or obsolete candidates |
| Objecthood | A revisable, re-identifiable sensorimotor prediction prototype |

These correspondences are **design decisions**, not proofs that the philosophical concepts have been fully captured. In particular, the agent receives engineered features, actions, and viability feedback. Semantic labels such as `food` and `decoy` are hidden from the agent, but the environment and evaluation criteria are handcrafted.

The experiment asks whether the proposed feedback structure has useful observable consequences *within these constraints*.

## The computational experiment

RD-World is a small, rule-based 2D grid world containing walls, hazards, food, and visually similar decoys. The agent observes numerical RGB values and relative positions, acts, and experiences outcomes such as movement resistance and viability change.

Some experiments make outcomes arrive after a delay. In role-reversal experiments, visually similar entities exchange their viability effects while retaining their appearance. An old predictive distinction may therefore become misleading even though the sensory input appears familiar.

Each encounter can leave a trace:

```text
local sensory distinction + action -> resistance + viability change
```

The implementation contains:

1. **Trace memory** (`rd_world/trace_field.py`): records interaction evidence, including temporal associations.
2. **Feature-level re-entry** (`rd_world/boundary.py`): updates which sensory differences receive weight.
3. **Objecthood candidates** (`rd_world/objecthood.py`): maintains predictive sensorimotor prototypes.
4. **Candidate-level re-entry**: changes the organization of candidates when evidence conflicts with their predictions.
5. **Branching, selection, and activation**: separates the existence of an alternative candidate from the evidence required for it to influence behavior.
6. **Evaluation and ablations**: compares mechanisms under matched simulated conditions.

This is a testbed for a proposed mechanism, not a system that autonomously invents an unrestricted ontology.

## How the research question changed during experimentation

The project did not begin with the final candidate lifecycle as a predetermined answer.

**Initial emphasis — feature-level re-entry.** The early computational interpretation focused on whether traces could change the salience of sensory features through `BoundaryGate`. In the historical mechanism comparisons, feature-weight changes alone did not show a clear adaptation-latency advantage.

**Revised emphasis — candidate-level re-entry.** Attention shifted to whether the *organization of existing predictive candidates* could be revised after contradictory outcomes. Historical experiments showed larger adaptation differences in candidate-revision conditions, but a later audit identified a duplicate-evidence defect affecting the interpretation of those differences.

**Subsequent design exploration — generation, selection, commitment.** Candidate revision was decomposed into splitting, provisional branching, evidence-window selection, and calibrated activation:

```text
conflict -> provisional alternatives -> evidence-based selection -> graded activation
```

This sequence reflects the explored design trajectory, **not** a universally validated mechanism. The experiments exposed trade-offs among rapid revision, candidate proliferation, premature stabilization, and delayed commitment.

The main research artifact is therefore not only a particular performance number. It is the attempt to turn a generative intuition into an explicit mechanism, distinguish competing implementations, and revise the hypothesis in response to experimental outcomes.

## What was observed — and what remains uncertain

The original implementation reported the following **historical paired comparisons** (generally 50 seeds):

| Historical comparison | Reported paired difference | Current interpretation |
|---|---:|---|
| Candidate re-entry vs no re-entry: adaptation lag | −30.160 ± 15.451 steps | Historical implementation; isolated causal contribution requires revalidation |
| Split-only vs formation-only: adaptation lag | −42.840 ± 17.176 steps | Strongly affected by the founding-trace update issue |
| Split-only vs formation-only: objecthood F1 | +0.378 ± 0.085 | Historical metric and implementation; corrected validation pending |
| Calibrated activation vs provisional selection: false stabilization | −10.382 ± 4.544 percentage points | Reduction persisted in a limited sensitivity check, not a complete corrected study |

The `±` values are historical paired 95% confidence-interval half-widths, not standard deviations. Results from different code and metric versions should not be treated as one uniform experiment.

**Important November 2026 audit finding:** newly split candidates can count their founding trace twice. In a limited single-update sensitivity test, the split-only adaptation-lag contrast against formation-only changed from **−42.840 ± 17.176** to **−10.300 ± 10.350 steps**. This substantially weakens the original causal interpretation, without establishing that the actual splitting effect is zero.

Other audit findings include a rolling-mean versus consecutive-step adaptation-criterion mismatch, phase-inappropriate false-positive counters after role reversal, censored non-adaptation latencies, and incomplete denominator/coverage handling. **The source code and historical output artifacts have not yet been corrected.**

See [November audit status](docs/AUDIT_STATUS.md), [historical results](docs/RESULTS.md), [historical hypotheses](docs/HYPOTHESES.md), and [original June README](docs/HISTORICAL_README_2026_06.md). Historical conclusions are preserved as a record of what was believed at the time, not endorsed as corrected causal findings.

## What this work does not establish

- It does not demonstrate that Recursive Distinction is a universal generative principle.
- It does not establish unconstrained object discovery: features, actions, viability, thresholds, and evaluation targets were engineered.
- It does not demonstrate consciousness, selfhood, worldhood, AGI, or biological plausibility.
- It does not establish generalization beyond the tested grid-world tasks or superiority over all reinforcement-learning, clustering, or latent-cause approaches.
- It does not establish equivalence merely because a confidence interval includes zero.

The distinction between **conceptual motivation**, **implemented mechanism**, and **empirically supported conclusion** is essential to interpreting this project.

## Run the historical implementation

Python 3.10+ is recommended. From the repository root:

```bash
python -m unittest discover -s tests
python -m rd_world.cli --mode compare --steps 350
python -m rd_world.cli --mode validation --steps 350 --change-step 175
python -m rd_world.cli --mode mechanism --steps 300 --change-step 150 --runs 50
python -m rd_world.cli --mode candidate-revision --steps 300 --change-step 150 --runs 50
python -m rd_world.cli --mode calibrated-activation --steps 300 --change-step 150 --runs 50 --selection-window 5 --validation-threshold high --activation-threshold high --activation-delay 5 --candidate-budget 15
```

Optional visualization dependency:

```bash
python -m pip install 'matplotlib>=3.7'
```

These commands run the historical, uncorrected implementation. Archived numerical results may depend on experiment-specific seed ranges. Existing tests passing would not resolve the audit findings.

## Documentation and code map

- [Original research framing / AI context](AI_CONTEXT.md) — includes the initial distinction–trace–re-entry formulation.
- [Glossary](docs/GLOSSARY.md) — operational meanings of the concepts.
- [Historical hypotheses](docs/HYPOTHESES.md) and [results](docs/RESULTS.md) — experimental progression and recorded evidence.
- [Historical research summary](docs/V0_1_SUMMARY.md) — original interpretation and proposed extensions.
- [Audit status](docs/AUDIT_STATUS.md) — limitations identified after the historical experiments.
- [Archived June README](docs/HISTORICAL_README_2026_06.md) — preserved original project documentation.
- `rd_world/world.py` — environment and observations.
- `rd_world/boundary.py`, `trace_field.py`, `objecthood.py`, `policy.py` — mechanisms.
- `rd_world/metrics.py`, `*validation.py`, `cli.py` — evaluation and execution.
- `output/` — historical experimental artifacts.

## Next research steps

1. Define and test a consistent founding-trace update contract.
2. Correct adaptation and phase-sensitive metrics; report censoring, valid denominators, and coverage.
3. Rerun matched ablations with a fixed commit, explicit seed manifests, and per-seed outputs.
4. Test whether re-entry offers benefits beyond simpler predictive-memory and latent-cause baselines.
5. Examine whether the proposed distinction–trace–re-entry structure scales to richer environments and more persistent relations, without assuming that it must.

## Authorship and usage

RD-World was developed as an independent research project with Codex-assisted programming. The precise division of conceptual, implementation, and verification contributions is not established by this statement alone.

**License:** No open-source license is granted by this repository. Public visibility does not by itself authorize general modification, redistribution, or commercial reuse beyond applicable law and GitHub's Terms of Service. Third-party dependencies retain their own licenses.
