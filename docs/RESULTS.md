# RD-World v0.1 Results

> **Historical results (June 2026): values and causal interpretations below predate the November 2026 audit. Duplicate founding-trace updates and metric-definition mismatches were identified; the original splitting effect cannot be treated as an isolated corrected causal estimate. See [audit status](AUDIT_STATUS.md) and [current README](../README.md).**

## Experimental scope

The consolidated audits use a rule-based agent in a non-stationary,
ambiguous-RGB grid world. Food-like and decoy-like causal roles reverse at step
150. Outcomes may be delayed by 3-5 steps. Hidden entity labels are available
only to the evaluator.

Unless stated otherwise, results are means `+/-` paired 95% confidence
intervals over 50 seeds.

## Comparison rule

Use paired differences within the same audit for causal interpretation.
Absolute values across audits are not directly interchangeable because the
implementation and some metric definitions evolved. In particular, the final
calibrated audit separates:

- candidate existence, used for objecthood F1,
- status-weighted policy influence,
- fully active candidates, used for false stabilization.

## 1. Mechanism-level re-entry audit

Candidate formation was enabled in every mode.

| Mode | Paired adaptation lag vs `no_reentry` | Paired obsolete persistence | Objecthood F1 | Interpretation |
|---|---:|---:|---:|---|
| `no_reentry` | baseline: `146.440 +/- 4.884` | baseline: `49.703% +/- 0.473%` | `0.365 +/- 0.077` | Formation alone adapts slowly. |
| `feature_reentry` | `-4.180 +/- 7.657` | `+0.165% +/- 0.475%` | `0.630 +/- 0.046` | Gate weights change, but lag does not clearly improve. |
| `candidate_reentry` | `-30.160 +/- 15.451` | `-2.058% +/- 1.645%` | `0.726 +/- 0.048` | Strongest isolated re-entry target. |
| `policy_reentry` | `+0.640 +/- 7.631` | `+0.436% +/- 0.532%` | `0.326 +/- 0.085` | No clear revision benefit. |
| `full_reentry` | `-25.320 +/- 15.947` | `-0.198% +/- 1.845%` | `0.763 +/- 0.053` | Improves lag, but does not preserve the obsolete-persistence effect. |

**Conclusion:** candidate revision, rather than feature or policy feedback
alone, is the main isolated source of faster revision.

## 2. Candidate revision decomposition

All conditions retained basic candidate formation. Feature and policy re-entry
were disabled.

| Revision mode | Paired lag vs formation-only | Paired F1 | Paired false stabilization | Main finding |
|---|---:|---:|---:|---|
| `confidence_only` | `+1.760 +/- 3.450` | `-0.175 +/- 0.059` | `-8.073% +/- 4.719%` | Conservative but damages objecthood recall/precision balance. |
| `split_only` | `-42.840 +/- 17.176` | `+0.378 +/- 0.085` | `+36.550% +/- 9.154%` | Fastest demonstrated adaptation; severe proliferation and false stabilization. |
| `suppression_only` | `-8.380 +/- 9.893` | `+0.170 +/- 0.084` | `+8.875% +/- 6.448%` | Not sufficient for fast adaptation. |
| `split_plus_suppression` | `-28.260 +/- 13.721` | `+0.377 +/- 0.082` | `+29.778% +/- 8.561%` | Retains some split benefit with more control. |
| `full_candidate_revision` | `-30.160 +/- 15.451` | `+0.361 +/- 0.092` | `+23.134% +/- 7.970%` | Balanced but slower than split-only. |

Adding suppression to split:

- slowed adaptation by `14.580 +/- 13.971` steps,
- reduced false stabilization by `6.772 +/- 6.304` percentage points,
- reduced candidate count by `10.754 +/- 3.696`,
- did not clearly improve obsolete persistence.

**Conclusion:** split is the causal adaptation mechanism. Suppression primarily
controls commitment and proliferation.

## 3. Targeted split audit

Precision-first targeting required compatible action/outcome evidence before
creating a child.

### Adaptation-preserving setting: `low/w1/none`

Relative to `split_only`:

- adaptation lag: `+10.880 +/- 19.618`,
- objecthood F1: `-0.089 +/- 0.051`,
- false stabilization: `-4.774% +/- 5.878%`,
- candidate count: `-12.780 +/- 3.834`,
- wrong-target rate: `+10.414% +/- 5.745`.

### Precision-oriented setting: `medium/w3/none`

Relative to `split_only`:

- adaptation lag: `+26.920 +/- 21.239`,
- objecthood F1: `-0.214 +/- 0.063`,
- false stabilization: `-18.993% +/- 7.794`,
- candidate count: `-18.230 +/- 4.356`,
- wrong-target rate: `-15.810% +/- 8.458`,
- target recall: `30.490% +/- 7.182`.

Adding suppression caused useful-candidate damage of
`21.982% +/- 5.707%`.

**Conclusion:** precision-first gating exposes a precision-plasticity frontier.
Higher pre-split precision reduces false commitments but rejects useful
revision opportunities and slows adaptation.

## 4. Generate-then-select provisional split audit

This audit creates children liberally, keeps the parent active, and selects
after an evidence window of five traces.

### `provisional_split_plus_selection` vs `split_only`

- adaptation lag: `+4.600 +/- 17.259`,
- objecthood F1: `-0.004 +/- 0.031`,
- false stabilization: `+2.104% +/- 4.432`,
- candidate count: `-14.852 +/- 4.016`,
- candidate churn: `-0.700 +/- 3.050`,
- useful-candidate damage: `0.000%`.

### Selection vs provisional branching without selection

- adaptation lag: `-0.500 +/- 18.822`,
- false stabilization: `-5.811% +/- 3.378`,
- candidate count: `-2.467 +/- 0.599`,
- objecthood F1: `+0.010 +/- 0.029`.

### Candidate budget of 15 vs selection

- adaptation lag: `+3.480 +/- 10.680`,
- false stabilization: `-0.258% +/- 1.056`,
- candidate count: `-0.342 +/- 0.234`,
- objecthood F1: `+0.012 +/- 0.020`.

**Conclusion:** generate-then-select controls candidate count and avoids useful
candidate damage while retaining the split adaptation signal. Selection
corrects provisional branching, but false stabilization does not fall below
the fast split baseline.

## 5. Calibrated provisional activation audit

The selected final configuration used:

- validation threshold: `high`,
- activation threshold: `high`,
- activation delay: `5`,
- selection window: `5`,
- candidate budget: `15` in the budgeted condition.

False stabilization in this audit is measured from fully active candidates.
Provisional and validated candidates still exist for F1 evaluation but do not
count as fully stabilized.

### Non-budgeted calibration vs provisional selection

- adaptation lag: `+13.480 +/- 18.279`,
- objecthood F1: `-0.023 +/- 0.024`,
- false stabilization: `-10.382% +/- 4.544`,
- candidate count: `-0.293 +/- 0.351`,
- useful-candidate damage: `0.000%`.

### Non-budgeted calibration vs `split_only`

- adaptation lag: `+17.200 +/- 14.645`,
- objecthood F1: `-0.011 +/- 0.032`,
- false stabilization: `-22.640% +/- 9.000`,
- candidate count: `-12.624 +/- 3.692`.

### Budgeted calibration vs provisional selection

- adaptation lag: `+7.400 +/- 18.015`,
- objecthood F1: `-0.030 +/- 0.025`,
- false stabilization: `-9.327% +/- 4.897`,
- candidate count: `-0.687 +/- 0.405`,
- candidate churn: absolute `15.680 +/- 10.344`.

**Conclusion:** calibrated activation clearly lowers false stabilization
relative to provisional selection. The lag and F1 intervals versus provisional
selection include zero, but no equivalence margin was specified, so strict
preservation is not proven. Calibration is slower than `split_only`.

## Consolidated trade-offs

The refined v0.1 mechanism is:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

### Split speed vs false stabilization

Fast splitting gives the strongest adaptation-lag reduction, but creates many
high-confidence wrong candidates. Delaying commitment reduces false
stabilization at the cost of slower activation.

### Precision-first targeting vs adaptation

Demanding precision before branching improves target precision and parsimony,
but lowers recall and delays revision. Liberal branching followed by selection
performs better on adaptation and F1.

### Calibration vs adaptation lag

Validated and delayed-active states reduce premature ontological commitment.
The final calibrated mode lowers false stabilization but is slower than
immediate `split_only`.

### Budget vs churn

A moderate budget reduces candidate count only slightly after evidence-window
selection. In the calibrated condition it increases churn substantially and
can reduce F1.

## Strongest current result

The most defensible v0.1 result is:

> Conflict-driven objecthood splitting improves adaptation in a non-stationary,
> ambiguous role-reversal world.

The later mechanisms refine how that split is controlled. They do not overturn
the finding that split supplies the main adaptation benefit.
