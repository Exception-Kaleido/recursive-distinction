# Recursive Distinction v0.1

RD-World is a minimal grid-world experiment in which an agent develops
**objecthood candidates from recurring perturbation-action-outcome traces**.
The environment contains walls, food, hazards, and food-like decoys, but those
names exist only inside the world and evaluator. They are never supplied to the
agent.

## v0.1 frozen claims

**Strongest supported claim:** conflict-driven objecthood splitting improves
adaptation in a non-stationary, ambiguous role-reversal world.

The refined v0.1 mechanism is:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

**Unsupported claims:**

- feature-level re-entry alone is sufficient for faster objecthood revision,
- policy-level re-entry alone is sufficient for faster objecthood revision,
- precision-first split targeting improves precision without an adaptation
  cost,
- suppression is the mechanism responsible for adaptation,
- v0.1 establishes consciousness, selfhood, or worldhood.

**Unresolved questions:**

- whether revised candidates reliably improve post-change viability recovery,
- whether obsolete persistence can be reduced without slowing adaptation,
- whether the mechanism generalizes beyond the tested grid, actions, viability
  dynamics, and role-reversal schedule.

See [docs/README.md](docs/README.md) for the frozen glossary, hypotheses,
results, and research summary.

## What counts as objecthood?

An objecthood candidate is an online sensorimotor prototype, not a predefined
class. It contains:

- a centroid of recurring local distinctions (RGB and relative position),
- an action and an outcome signature associated with those distinctions,
- predicted movement resistance and viability change,
- re-identifiability, action relevance, and an objecthood score.

The objecthood score prioritizes viability effects, movement resistance,
outcome consistency, and re-identifiability. Recurrence supplies only a small
evidence term. Consequently, a frequently traversed empty patch remains near
zero objecthood, while a repeatably blocking or viability-changing pattern can
become a strong candidate.

Candidates bind RGB, action, and outcome signature. Thus visually similar food
and decoys can form different candidates after touch: one predicts positive
viability change while the other predicts only the metabolic cost.

## How re-entry works

Every interaction creates a trace:

```text
selected local distinctions + action -> resistance + delta viability
```

`TraceField` stores these traces. In the RD condition, each new trace re-enters
`BoundaryGate`. Pairwise feature differences are compared with differences in
viability and blockage outcomes. Predictively useful features gain weight and
below-average features lose weight.

The weights are produced by a softmax and always sum to one. Features therefore
compete for a fixed attention budget instead of all saturating at one. A moving
average retains learned usefulness when a later trace window contains no new
outcome contrast.

This closes the loop:

```text
boundary -> trace -> candidate/outcome -> updated boundary
```

Validation uses four explicit conditions:

- `rd`: trace memory, objecthood stabilization/destabilization, and gate re-entry,
- `trace`: the same trace/objecthood machinery with a fixed BoundaryGate,
- `fixed`: a memoryless fixed-weight baseline using numeric color/brightness,
- `random`: a uniform random-action baseline.

## Policy

The policy strongly penalizes candidates that predict negative viability or
high resistance. It also scans the local sensory radius for re-identified
positive-viability candidates and moves toward them. Exploration remains
stochastic, but actions predicted to be dangerous are excluded from the
ordinary exploration set when safer alternatives exist.

## Why labels are hidden

Giving the agent labels such as `food`, `wall`, or `decoy` would assume the
object partition the experiment is meant to study. The agent receives only
numeric local sensory patches and post-action viability/resistance changes.
Hidden entity types are used solely by the environment to produce consequences
and by a separate evaluator to calculate validation metrics. `SensoryItem` and
`StepResult` contain no kind, label, or entity-name field. A unit test and the
validation runner inspect this public contract.

Food and decoys have deliberately similar RGB values. Food raises viability;
decoys do not. A successful agent must use repeated outcome-bearing traces to
make the subtle sensory distinction matter.

## Modules

- `rd_world.world`: hidden grid dynamics and local numeric observations
- `rd_world.boundary`: `BoundaryGate` and re-entry weight updates
- `rd_world.trace_field`: trace storage
- `rd_world.objecthood`: stabilizer, candidates, and destabilizer
- `rd_world.policy`: candidate-guided action selection
- `rd_world.metrics`: viability, confidence, weights, and decoy accuracy
- `rd_world.visualization`: matplotlib summary
- `rd_world.validation`: four-condition validation experiment
- `rd_world.validation_visualization`: validation dashboard
- `rd_world.cli`: command-line runner and ablation comparison

## Validation experiment

At the configured change step, the evaluator changes the hidden environment:

- some remaining food entities become decoys while retaining the exact food RGB,
- hazards change to a new color.

If too little food remains at the change step, the evaluator inserts enough
food-colored transition decoys and a small amount of comparison food to ensure
that the second phase is present in every run.

The unchanged food appearance makes RGB alone insufficient. A converted entity
can only be revised after a touch produces neutral rather than positive
viability. Conflicting traces weaken old candidates, create outcome-separated
candidates, and may later merge redundant prototypes.

The validation metrics are:

- viability over time,
- food approach rate when food is visible,
- decoy false-positive rate when a decoy is adjacent,
- hazard avoidance rate when a hazard is adjacent,
- wall collision rate and early-to-late collision reduction,
- candidate stability from prototype score/prediction change and candidate churn,
- cumulative destabilization, split, merge, and new-candidate events,
- confidence histories for food-like, decoy-like, wall-like, and hazard-like
  outcome groups.

Candidate type names are evaluator-only descriptions. The agent never receives
them.

## Re-entry strength sweep

`reentry_strength` scales only the BoundaryGate feedback:

- `0.0` is trace memory with no gate re-entry,
- `1.0` is the full v0.1 re-entry update,
- intermediate values linearly scale the competitive gate update.

The default sweep runs strengths `0.0, 0.1, 0.25, 0.5, 1.0` over 20 paired
seeds and reports population mean +/- standard deviation.

Normalized decoy false positive is:

```text
selected adjacent decoys / selected adjacent (food + decoy)
```

This prevents a policy that rarely approaches any green entity from appearing
to discriminate decoys well.

Candidate stability now combines prototype prediction change, candidate churn,
and per-step destabilization/split/merge penalties. The stability-plasticity
score multiplies post-change adaptation responsiveness by a penalty for
pre-change destabilization:

```text
responsiveness = 1 - adaptation_latency / post_change_horizon
stability_penalty = 1 / (1 + 5 * pre_change_destabilizations / change_step)
score = responsiveness * stability_penalty
```

Adaptation is detected when a post-change candidate with the original food RGB
recurrently predicts the neutral decoy outcome.

In the role-reversal mechanism audit, adaptation requires the balanced
food/decoy probe set to remain at or below 40% obsolete predictions for five
consecutive steps. A candidate-free system remains at 50% because it cannot
recognize the newly positive role, so abstention cannot count as adaptation.

## Sharper non-stationary validation

The sharper suite is designed around revision rather than final viability:

- every strength uses the same paired seed, initial layout, entity RGB samples,
  deterministic entity-specific delay, and role-reversal step,
- food, decoy, and hazard RGB values overlap,
- outcomes arrive 3-5 steps after interaction,
- an eligibility window binds delayed viability changes to recent
  action-context traces,
- food and decoy causal roles reverse while their visual distributions remain
  unchanged.

The suite reports mean +/- 95% confidence intervals over 50 paired seeds for:

- adaptation lag,
- obsolete candidate persistence,
- destabilization precision,
- false stabilization rate,
- hidden-label evaluator objecthood F1,
- normalized decoy false positive,
- post-change viability recovery slope.

It also reports paired differences from `reentry_strength=0.0`. Hidden labels
are used only by the evaluator and never enter observations, traces, candidates,
or policy decisions.

## Run

Python 3.10+ is recommended.

```powershell
python -m unittest discover -s tests
python -m rd_world.cli --mode compare --steps 350
python -m rd_world.cli --mode validation --steps 350 --change-step 175
python -m rd_world.cli --mode validation --steps 350 --change-step 175 --visualize
python -m rd_world.cli --mode validation --save output/validation.png
python -m rd_world.cli --mode sweep --runs 20 --save output/reentry_sweep.png
python -m rd_world.cli --mode sharp --steps 300 --change-step 150 `
  --runs 50 --save output/sharp_validation.png
```

In the current 50-seed run, strength `0.5` reduced mean adaptation lag by
11.46 steps relative to `0.0`, but its paired 95% confidence interval included
zero. All nonzero strengths also had paired obsolete-persistence intervals that
included zero. The current sharper suite therefore does not yet provide
statistically clear evidence that re-entry reduces either primary metric.

## Mechanism-level re-entry audit

The mechanism audit separates the target of feedback:

- `no_reentry`: traces form and update basic candidates, but do not revise
  them or alter the gate or policy preferences,
- `feature_reentry`: basic candidate formation plus BoundaryGate updates,
- `candidate_reentry`: basic candidate formation plus confidence weakening,
  splitting, merging, and obsolete suppression,
- `policy_reentry`: basic candidate formation plus a rule-based RGB-action
  preference table,
- `full_reentry`: all three paths are active.

Candidate formation is held constant across all five modes. Outcome conflicts
are averaged into a basic prototype unless candidate revision is enabled; only
the revision path may split, merge, weaken, suppress, or obsolete candidates.

The audit reports feature-weight L1 change, selected-distinction change rate,
candidate revision count, pre/post policy action-distribution total variation,
adaptation lag, obsolete persistence, objecthood F1, and viability recovery.
All modes use paired layouts and role-reversal schedules.

```powershell
python -m rd_world.cli --mode mechanism --steps 300 --change-step 150 `
  --runs 50 --save output/mechanism_audit.png
```

Current 50-seed mechanism result:

- all modes formed candidates; `no_reentry` objecthood F1 was
  `0.365 +/- 0.077`,
- candidate revision reduced paired adaptation lag by
  `30.160 +/- 15.451` steps,
- full re-entry reduced paired adaptation lag by
  `25.320 +/- 15.947` steps,
- candidate-only objecthood F1 was `0.726 +/- 0.048`,
- feature-only objecthood F1 was `0.630 +/- 0.046`,
- candidate-only obsolete persistence changed by
  `-2.058 +/- 1.645` percentage points versus `no_reentry`.

With formation controlled, candidate revision remains the only isolated target
with a statistically clear adaptation-lag reduction in this run. Candidate-only
revision also shows a modest obsolete-persistence reduction, while full
re-entry does not preserve that secondary effect.

## Candidate revision decomposition

The candidate-level audit holds basic candidate formation constant and disables
feature and policy re-entry. It includes a formation-only causal baseline and
five revision submodes:

- `confidence_only`: persistent confidence inhibition on conflicts,
- `split_only`: create outcome-separated prototypes on conflicts,
- `suppression_only`: inhibit and prune contradicted prototypes,
- `split_plus_suppression`: split new outcomes and suppress contradicted
  prototypes,
- `full_candidate_revision`: confidence updates, split, suppression, periodic
  merge, and explicit obsolete marking before pruning.

Candidate count is the mean number of active prototypes after role reversal.
Candidate churn counts post-change prototype creation, pruning, and merging.
The report includes paired incremental contrasts, especially suppression added
to an otherwise identical split mechanism.

```powershell
python -m rd_world.cli --mode candidate-revision --steps 300 `
  --change-step 150 --runs 50 `
  --save output/candidate_revision_audit_50seeds.png
```

Current 50-seed result:

- `split_only` reduced adaptation lag by `42.840 +/- 17.176` steps versus
  formation-only,
- `confidence_only` and `suppression_only` did not clearly reduce adaptation
  lag,
- adding suppression to split changed lag by `+14.580 +/- 13.971` steps and
  obsolete persistence by `-0.322 +/- 2.183` percentage points,
- split alone changed obsolete persistence by
  `-2.090 +/- 2.064` percentage points,
- adding suppression to split reduced false stabilization by
  `6.772 +/- 6.304` percentage points and active candidate count by
  `10.754 +/- 3.696`,
- no revision mode produced a clear post-change viability recovery-slope
  improvement.

In this implementation, splitting is the mechanism responsible for faster
objecthood revision. Suppression is not required for the measured obsolete
persistence reduction and does not add a clear reduction of that metric.
Instead, it controls prototype proliferation and false stabilization, at the
cost of slower adaptation. Destabilization precision remains low, so conflict
targeting is still a major limitation.

## Targeted split audit

Targeted splitting separates conflict attribution from split triggering.
Every attributed conflict records:

- target candidate ID and child candidate ID,
- RGB, relative-position, and action distance,
- candidate predicted viability change and resistance,
- observed viability change and resistance,
- temporal delay and eligibility,
- viability-conflict and resistance-conflict flags,
- compatible evidence count, threshold result, split event, parent decay, and
  suppression.

These fields contain no hidden entity labels. A separate evaluator maps each
trace source step to the hidden expected numeric outcome and later annotates
the log with `trace_valid`, `actual_wrong`, and `actually_obsolete`.

Targeted split requires an exact action match. Viability conflict is accepted
only when resistance remains compatible; resistance conflict is accepted only
when viability remains compatible. Compatible evidence is accumulated by
candidate and conflict type. A provisional child is created only after the
configured evidence window, and its parent remains active until the child has
enough observations.

The targeting metrics are:

- target precision: correct triggered split targets / triggered split targets,
- target recall: wrong candidate IDs hit / wrong candidate IDs encountered,
- wrong-target rate: incorrect targets / triggered split targets,
- useful-candidate damage: decay or suppression applied to evaluator-correct
  candidates,
- obsolete-candidate hit rate: obsolete pre-change candidate IDs hit / obsolete
  candidate IDs encountered.

The threshold sweep covers:

- `conflict_threshold`: `low`, `medium`, `high`,
- `evidence_window`: `1`, `3`, `5`,
- `parent_decay`: `none`, `slow`, `fast`.

```powershell
python -m rd_world.cli --mode targeted-split-sweep --steps 300 `
  --change-step 150 --runs 20 `
  --save output/targeted_split_threshold_sweep.png

python -m rd_world.cli --mode targeted-split --steps 300 `
  --change-step 150 --runs 50 --conflict-threshold low `
  --evidence-window 1 --parent-decay none `
  --save output/targeted_split_audit_50seeds.png `
  --log-output output/targeted_split_conflicts_50seeds.jsonl
```

The sweep exposed a trade-off rather than a dominating setting:

- `low/w1/none` preserved adaptation lag relative to `split_only`
  (`+10.880 +/- 19.618` steps) and reduced active candidate count by
  `12.780 +/- 3.834`, but increased wrong-target rate by
  `10.414 +/- 5.745` percentage points,
- `medium/w3/none` reduced wrong-target rate by
  `15.810 +/- 8.458` percentage points, false stabilization by
  `18.993 +/- 7.794` percentage points, and candidate count by
  `18.230 +/- 4.356`, but slowed adaptation by
  `26.920 +/- 21.239` steps and reduced objecthood F1 by
  `0.214 +/- 0.063`,
- suppression further reduced candidate count but slowed adaptation and
  produced measurable useful-candidate damage.

Targeted splitting therefore does not yet satisfy the full primary objective.
It can preserve adaptation while controlling proliferation, or improve target
precision and false stabilization, but the current rule cannot do both at
once. Temporal credit errors remain a major source of wrong conflict evidence.

## Generate-then-select splitting

Generate-then-select moves precision control after branching:

- `provisional_split` creates an outcome-compatible child immediately and
  keeps the parent active,
- `provisional_split_plus_selection` evaluates parent and children over a
  post-split evidence window, promotes winning children, prunes weak children,
  and decays the parent only after a child wins by a score margin,
- `provisional_split_budgeted` adds a global candidate budget and removes the
  lowest predictive-usefulness candidate while protecting unresolved parents.

`ObjectCandidate.status` is one of `active`, `provisional`, or `obsolete`.
Each candidate keeps rolling evidence:

- support and conflict counts,
- normalized prediction errors,
- action-outcome consistency,
- viability prediction accuracy,
- resistance prediction accuracy.

Sibling children with converged visual/action/outcome signatures merge before
selection. Suppression is never applied at split time. The configurable
selection windows are `3`, `5`, and `10`; budgets are `none`, `10`, `15`, and
`20`.

```powershell
python -m rd_world.cli --mode provisional-split --steps 300 `
  --change-step 150 --runs 50 --selection-window 5 `
  --candidate-budget 15 `
  --save output/provisional_split_audit_50seeds.png
```

Current 50-seed result with window `5` and budget `15`:

- `provisional_split_plus_selection` preserved adaptation lag relative to
  `split_only` (`+4.600 +/- 17.259` steps) and objecthood F1
  (`-0.004 +/- 0.031`),
- it reduced active candidate count by `14.852 +/- 4.016`,
- false stabilization changed by `+2.104 +/- 4.432` percentage points and
  candidate churn by `-0.700 +/- 3.050`, so neither improved clearly,
- useful-candidate damage was zero after restricting that evaluator metric to
  valid traces,
- selection reduced false stabilization by `5.811 +/- 3.378` percentage
  points relative to provisional branching without selection,
- the budgeted mode also preserved adaptation and F1, reduced candidate count
  by `15.195 +/- 4.076`, and produced no useful-candidate damage,
- compared with precision-first targeted split, selection was
  `22.320 +/- 18.354` steps faster and had F1 higher by
  `0.209 +/- 0.058`, but false stabilization was
  `21.097 +/- 6.785` percentage points higher.

Generate-then-select therefore resolves the adaptation-versus-proliferation
trade-off better than precision-first targeting: it keeps fast revision and
controls candidate count without damaging useful candidates. It does not yet
reduce false stabilization below the fast split baseline, so the full primary
objective remains only partially satisfied.

## Calibrated provisional activation

Calibrated activation inserts two evidence-gated states between branching and
parent replacement:

```text
active parent -> provisional child -> validated child -> active child
              -> obsolete parent
```

Candidate existence and policy influence are separate. Provisional children
collect evidence with weak policy influence, validated children have moderate
influence, and active children have full influence. The activation score uses
support, prediction error, action-outcome consistency, viability and resistance
accuracy, and encounter diversity.

Parent decay starts only after child validation. Parent obsolescence requires
an active child to outperform it over the evidence window. Thresholds are
`low`, `medium`, or `high`; activation delay is `1`, `3`, or `5` consecutive
compatible traces. Objecthood F1 evaluates candidate existence independently
of policy influence. False stabilization counts only predictions from fully
active candidates, so provisional branches are not mislabeled as stabilized.

```powershell
python -m rd_world.cli --mode calibrated-activation --steps 300 `
  --change-step 150 --runs 50 --selection-window 5 `
  --validation-threshold high --activation-threshold high `
  --activation-delay 5 --candidate-budget 15 `
  --save output/calibrated_activation_audit_50seeds.png

python -m rd_world.cli --mode calibrated-activation-sweep --steps 300 `
  --change-step 150 --runs 20 --selection-window 5
```

Current 50-seed paired result:

- non-budgeted calibrated activation reduced false stabilization by
  `10.382 +/- 4.544` percentage points versus provisional selection,
- adaptation lag changed by `+13.480 +/- 18.279` steps and objecthood F1 by
  `-0.023 +/- 0.024`; both intervals include zero,
- versus `split_only`, false stabilization changed by
  `-22.640 +/- 9.000` percentage points, lag by `+17.200 +/- 14.645`, and F1 by
  `-0.011 +/- 0.032`,
- useful-candidate damage remained zero and mean candidate count fell below
  both baselines,
- the budgeted calibrated mode reduced false stabilization by
  `9.327 +/- 4.897` percentage points versus provisional selection, but its F1
  reduction was larger (`-0.030 +/- 0.025`).

Against provisional selection, calibrated activation gives a statistically
clear false-stabilization reduction while preserving adaptation lag and F1
within their paired 95% confidence intervals. It does not preserve the faster
`split_only` adaptation lag: that paired interval is entirely above zero.

Matplotlib is needed only for visualization:

```powershell
python -m pip install matplotlib
```

The validation visualization contains ablation viability curves, behavioral
rates, wall collisions, RD feature weights, confidence by candidate type,
candidate stability, event histories, and separate top-candidate tables for
food-like, decoy-like, wall-like, and hazard-like groups.

## Scope of v0.1

This is intentionally a transparent toy model. It uses online prototypes and
simple relevance updates, not deep learning. Objecthood here is operational:
stable predictive organization in sensorimotor traces. It is not claimed to be
a complete cognitive or biological model.
