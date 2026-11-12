# Recursive Distinction (RD-World) · v0.1

**Status:** Exploratory research prototype · historical experiments frozen in June 2026 · **results under re-evaluation following a November 2026 audit**

RD-World investigates whether an agent can form and revise *objecthood candidates* from recurring sensorimotor interactions **without receiving semantic object-class labels**. This is a small, rule-based grid-world experiment, not a general-intelligence system or a demonstration of consciousness.

> **Research question:** Can recurring distinction–action–outcome traces support revisable predictive object representations when visually ambiguous entities change their causal roles?

**Read first:** [Audit status](docs/AUDIT_STATUS.md) · [Historical research summary](docs/V0_1_SUMMARY.md) · [Historical results](docs/RESULTS.md) · [Hypotheses](docs/HYPOTHESES.md) · [Glossary](docs/GLOSSARY.md) · [AI context](AI_CONTEXT.md) · [Original June README](docs/HISTORICAL_README_2026_06.md)

## Motivation

Providing an agent with labels such as `food`, `wall`, or `decoy` would presuppose the categories under investigation. RD-World instead exposes numerical local observations and the consequences of actions. The environment and evaluator retain hidden entity identities, but the agent does not receive their names.

This is **not** an experiment in unconstrained ontology discovery: features, actions, viability signals, candidate structures, and update rules are engineered in advance. The question is whether predictive prototypes can be maintained and revised under conflicting evidence.

## Environment and mechanism

The environment is a discrete 2D grid containing walls, hazards, food, and visually similar decoys. Observations include RGB and relative position. Interactions may produce movement resistance and changes in viability. Some evaluations introduce delayed outcomes and reverse the consequences of food-like and decoy-like entities without changing their visual distributions.

An interaction generates a trace:

```text
local distinctions + action -> movement resistance + viability change
```

The implementation combines:

1. **Trace memory:** store observation–action–outcome records.
2. **Boundary re-entry:** adjust feature weighting from predictive contrasts in traces.
3. **Objecthood candidates:** represent provisional sensorimotor prototypes linking observations, actions, and expected outcomes.
4. **Conflict-driven revision:** weaken, split, merge, or obsolete candidates when predictions fail.
5. **Provisional branching and selection:** collect evidence before committing to a new candidate.
6. **Calibrated activation:** distinguish candidate existence from its influence on policy.

These are handcrafted computational mechanisms. *Objecthood* here denotes revisable predictive organization, not proof of semantic object discovery.

## Experiments and historical observations

The repository includes fixed, random, trace-based, feature-re-entry, candidate-revision, and policy-re-entry comparisons. Subsequent ablations examine confidence updates, splitting, suppression, evidence-window selection, and calibrated activation.

The original experiments reported the following **historical paired comparisons**, generally over 50 seeds:

| Historical contrast | Originally reported paired difference | Interpretation after audit |
|---|---:|---|
| Candidate re-entry vs no re-entry: adaptation latency | −30.160 ± 15.451 steps | Original-implementation result; corrected causal attribution pending |
| Split-only vs formation-only: adaptation latency | −42.840 ± 17.176 steps | Sensitive to duplicate founding-trace updates |
| Split-only vs formation-only: objecthood F1 | +0.378 ± 0.085 | Original metric and implementation; corrected validation pending |
| Calibrated activation vs provisional selection: false stabilization | −10.382 ± 4.544 percentage points | A reduction persisted in a limited sensitivity test, not a complete corrected suite |

The `±` values are historical paired 95% confidence-interval half-widths, not standard deviations. Comparisons from different experiment versions must not be combined as one controlled study. See [the historical results](docs/RESULTS.md) for complete tables.

## November 2026 audit — known limitations

**The historical source code and output artifacts remain uncorrected.** A subsequent audit identified:

- **Duplicate founding-trace evidence:** after a candidate split, a newly initialized child may incorporate its founding observation a second time. In a limited single-update sensitivity test, the split-only adaptation-latency contrast changed from **−42.840 ± 17.176** to **−10.300 ± 10.350 steps** against formation-only. The true isolated effect is not established; this does not prove the effect is zero.
- **Adaptation criterion mismatch:** documentation describes five consecutive acceptable steps, but implementation evaluates a five-step rolling mean.
- **Post-reversal metric semantics:** some decoy false-positive counters use original entity kind even after causal roles reverse.
- **Censoring:** non-adapting runs are assigned the maximum latency; success and censoring rates must accompany latency comparisons.
- **Coverage:** zero-denominator rates may be reported as zero, and some candidate-damage measurements cover restricted logged events.
- **Version/configuration drift:** historical experiments differ in code, metric definitions, and seed ranges; precise reproduction requires explicit manifests and per-seed records.

**The original strong claim that splitting itself accounts for the full measured adaptation improvement is not established by the current evidence.** Corrected ablations and independent evaluation are needed. The [audit status](docs/AUDIT_STATUS.md) provides a concise record.

## Scope and non-claims

- Semantic class labels are hidden from the agent, but engineered numerical features, action choices, and viability feedback are provided.
- The simulation uses rule-based prototypes and policies, not deep learning.
- Hidden evaluator labels and objecthood F1 are operational proxies, not general tests of object discovery.
- Generalization beyond the tested grid, role reversals, and delayed-outcome settings has not been established.
- No claims are made about consciousness, selfhood, AGI, biological plausibility, or superiority to all RL or latent-cause baselines.
- A confidence interval containing zero does not demonstrate equivalence.

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

For visualization:

```bash
python -m pip install 'matplotlib>=3.7'
```

These commands execute the **historical, uncorrected implementation**. Reproducing specific archived numbers may require an experiment-specific seed range. Passing existing tests does not resolve the audit findings.

## Repository map

| Path | Role |
|---|---|
| `rd_world/world.py` | Grid dynamics and numeric observations |
| `rd_world/trace_field.py` | Trace storage and temporal association |
| `rd_world/boundary.py` | Feature weighting and re-entry |
| `rd_world/objecthood.py` | Candidate formation, revision, and lifecycle |
| `rd_world/policy.py` | Candidate-informed action selection |
| `rd_world/metrics.py` | Operational evaluation |
| `rd_world/*validation.py` | Experiments and ablations |
| `rd_world/cli.py` | Command-line entry point |
| `output/` | Historical figures and experiment artifacts |
| `docs/` | Research documents and audit notice |

## Next steps

1. Specify and test whether a founding trace contributes once or receives a documented pseudo-count.
2. Align adaptation criteria and post-reversal metrics with their definitions.
3. Report adaptation success, censoring, valid denominators, and uncertainty.
4. Rerun matched ablations from one fixed commit with seed manifests and per-seed outputs.
5. Test independent environments and stronger external baselines.
6. Publish corrected, versioned results separately from the historical record.

## Authorship and usage

RD-World was developed as an independent research project with Codex-assisted programming. This statement does not establish a precise allocation of design, implementation, and verification contributions.

**License:** No open-source license is granted in this repository. Public visibility alone does not authorize general redistribution, modification, or commercial reuse beyond rights provided by applicable law and GitHub's Terms of Service. Third-party dependencies retain their own licenses.
