# RD-World v0.1 Research Summary

## Core conclusion

RD-World v0.1 supports one strong, narrow claim:

> Conflict-driven objecthood splitting improves adaptation in a non-stationary,
> ambiguous role-reversal world.

The evidence does not establish consciousness, selfhood, or worldhood. It
shows that a rule-based agent can form and revise useful sensorimotor
objecthood candidates without receiving object labels.

## Research question

Can recurring perturbation-action-outcome traces support label-free
objecthood, and can re-entry revise that organization when visually ambiguous
entities reverse causal roles?

The environment hides `food`, `decoy`, `hazard`, and `wall` labels from the
agent. The agent receives local numeric features, action outcomes, movement
resistance, and viability change. Hidden labels are used only for evaluation.

## What v0.1 established

### Supported

- Candidate revision is the only isolated re-entry target that clearly reduces
  adaptation lag.
- Candidate splitting is the main internal revision mechanism responsible for
  that reduction.
- Evidence-window selection improves liberal provisional branching by pruning
  weak branches and reducing candidate count.
- Calibrated activation reduces false stabilization relative to immediate
  provisional selection.

### Partially supported

- Generate-then-select appears to preserve fast-split adaptation and F1 while
  controlling proliferation, but the experiment did not define formal
  equivalence margins.
- Calibrated activation preserves lag and F1 relative to provisional selection
  only in the limited sense that paired intervals include zero. It is slower
  than immediate `split_only`.
- Candidate budgets improve parsimony, but can increase churn and reduce F1.
- Candidate revision sometimes reduces obsolete persistence, but the effect is
  not robust across all later mechanisms.

### Unsupported

- Feature re-entry alone is sufficient for faster objecthood revision.
- Policy re-entry alone is sufficient for faster objecthood revision.
- Precision-first split targeting can simultaneously improve precision and
  preserve adaptation speed.
- Suppression is the mechanism responsible for adaptation.
- The current model demonstrates consciousness, selfhood, or worldhood.

### Unresolved

- Whether the revised candidates reliably improve post-change viability
  recovery.
- Whether obsolete persistence can be reduced robustly without slowing
  adaptation.
- Whether the same mechanism generalizes beyond the current grid, action set,
  viability dynamics, and role-reversal schedule.

## Mechanism refinement

The original narrow interpretation of Recursive Distinction emphasized
feature-level re-entry through `BoundaryGate`. The audits weakened that account:
feature re-entry alone did not clearly improve adaptation, while
candidate-level objecthood revision retained a clear adaptation effect. The
refined v0.1 interpretation therefore locates the main recursive mechanism in
the revision of objecthood organization rather than feature weighting alone.

The experiments refined candidate revision in four stages:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

### 1. Fast split

An outcome conflict immediately creates a distinct candidate. This provides
plasticity and produced the largest adaptation-lag reduction, but also caused
candidate proliferation and false stabilization.

### 2. Provisional branching

The child exists and collects evidence while the parent remains active. This
avoids destructive early replacement, but liberal branching alone raises false
stabilization.

### 3. Evidence-window selection

Parent and children compete using short-window predictive evidence. Weak
children are pruned and redundant children can merge. This substantially
reduces candidate count without evaluator-confirmed useful-candidate damage.

### 4. Calibrated activation

A winning child does not become fully active immediately:

```text
active parent
  -> provisional child
  -> validated child
  -> active child
  -> obsolete parent
```

Policy influence is graded by status. Parent decay begins after validation, and
parent obsolescence requires an active child to outperform it over an evidence
window. This lowers false stabilization but delays full adaptation relative to
immediate splitting.

## Main trade-offs

### Split speed vs false stabilization

Fast branching adapts quickly because it preserves conflicting causal
signatures immediately. The same plasticity lets weak or temporally
misattributed traces become active candidates.

### Precision-first targeting vs adaptation

Requiring strong evidence before branching improves target precision and
reduces proliferation. It also reduces recall and delays the creation of the
candidate needed for adaptation.

### Calibration vs adaptation lag

Delayed activation prevents provisional evidence from receiving full
behavioral authority. This reduces false stabilization, but active replacement
occurs later.

### Budget vs churn

A hard candidate budget limits count but can repeatedly prune and recreate
borderline candidates. The calibrated budgeted audit showed this as increased
churn.

## Strongest quantitative results

- Candidate-only re-entry vs no re-entry:
  adaptation lag `-30.160 +/- 15.451` steps.
- Split-only vs formation-only:
  adaptation lag `-42.840 +/- 17.176` steps and F1
  `+0.378 +/- 0.085`.
- Provisional selection vs split-only:
  candidate count `-14.852 +/- 4.016`, lag
  `+4.600 +/- 17.259`, F1 `-0.004 +/- 0.031`.
- Calibrated activation vs provisional selection:
  false stabilization `-10.382 +/- 4.544` percentage points, lag
  `+13.480 +/- 18.279`, F1 `-0.023 +/- 0.024`.

All values are paired mean differences with 95% confidence intervals. Results
from different audit versions should not be compared as if they came from one
unchanged implementation.

## Interpretation

RD-World's useful result is not that it discovers semantic objects. It learns
graded predictive organizations that behave like objecthood candidates because
they are:

- re-identifiable from local distinctions,
- linked to action and outcome signatures,
- relevant to viability or blockage,
- revisable when their predictions fail.

The lifecycle implements **graded ontological commitment**: a candidate may
exist before it is allowed to guide policy fully. This language describes a
computational control structure, not a philosophical proof of ontology.

## Limits

- The simulator is small, discrete, and hand-designed.
- Candidate scores, thresholds, and policy influence are rule-based.
- Temporal outcomes use only a short eligibility window.
- Hidden labels are excluded from learning but still define evaluation.
- Non-significant differences are not evidence of formal equivalence.
- No representation of a persistent self, body schema, or global causal world
  model exists.
- No result supports claims about phenomenal consciousness.

## Next candidate research directions

### Temporal credit assignment

Replace the short eligibility heuristic with explicit competing temporal
hypotheses. Measure whether better causal credit reduces wrong-target splits
without reintroducing precision-first delay.

### Richer evidence binding

Bind encounter identity, action sequence, spatial trajectory, temporal context,
and multi-step outcome structure. Test whether richer evidence improves child
validation and obsolete-parent identification.

### Selfhood extension

Add an explicitly operational self-model: persistent body location, endogenous
action source, controllability, and viability-regulation boundaries. Keep
selfhood hypotheses separate from objecthood results.

### Worldhood/causal partition extension

Move from local prototypes to persistent relations among candidates. Test
whether the agent can learn causal partitions, stable places, and
object-object interactions across longer horizons without receiving world
labels.
