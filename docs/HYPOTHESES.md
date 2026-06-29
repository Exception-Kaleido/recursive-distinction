# RD-World v0.1 Hypotheses

## Status vocabulary

- **Supported**: the relevant paired 95% confidence interval excludes zero in
  the predicted direction, or multiple audits give convergent evidence.
- **Partially supported**: one part of a composite prediction is supported, but
  another part is inconclusive or shows a cost.
- **Unsupported**: the predicted effect is absent, reversed, or contradicted by
  the tested conditions.
- **Unresolved**: the current experiment or metric cannot distinguish the
  alternatives.

These are claims about this rule-based toy implementation and its tested
non-stationary role-reversal environment. They are not claims about
consciousness, selfhood, or worldhood.

## Hypothesis table

| ID | Hypothesis | Status | Supporting result | Caveat |
|---|---|---|---|---|
| H1 | Candidate-level re-entry improves objecthood revision after role reversal. | **Supported** | In the mechanism audit, `candidate_reentry` reduced adaptation lag by `30.160 +/- 15.451` steps versus `no_reentry`. `full_reentry` reduced it by `25.320 +/- 15.947`. | Candidate formation was enabled in all modes. The result isolates revision better than earlier audits, but it remains environment-specific. |
| H2 | Feature re-entry alone is sufficient to improve adaptation. | **Unsupported** | `feature_reentry` changed feature weights by `0.392 +/- 0.098` and selected distinctions by `49.205% +/- 12.242%`, but adaptation lag changed only `-4.180 +/- 7.657` steps. | Feature adaptation may still matter in tasks where the sensory basis itself changes more strongly. |
| H3 | Policy re-entry alone is sufficient to improve objecthood revision. | **Unsupported** | `policy_reentry` changed adaptation lag by `+0.640 +/- 7.631` and produced objecthood F1 `0.326 +/- 0.085`. | The policy feedback mechanism is deliberately simple and may be underpowered. |
| H4 | Conflict-driven splitting is the main candidate-revision mechanism responsible for faster adaptation. | **Supported** | `split_only` reduced adaptation lag by `42.840 +/- 17.176` steps versus formation-only and increased objecthood F1 by `0.378 +/- 0.085`. Confidence-only and suppression-only did not show comparable lag reductions. | Fast splitting increased false stabilization by `36.550 +/- 9.154` percentage points and candidate count by `19.604 +/- 4.406`. |
| H5 | Suppression is necessary for adaptation and obsolete-candidate removal. | **Unsupported** | Adding suppression to split slowed adaptation by `14.580 +/- 13.971` steps and changed obsolete persistence by only `-0.322 +/- 2.183` percentage points. | Suppression did reduce false stabilization and candidate count; it is useful for control, not the demonstrated source of adaptation. |
| H6 | Precision-first conflict targeting can improve targeting precision without sacrificing adaptation. | **Unsupported** | The precision-oriented `medium/w3/none` setting improved target precision and reduced false stabilization by `18.993 +/- 7.794` percentage points, but slowed adaptation by `26.920 +/- 21.239` steps and reduced F1 by `0.214 +/- 0.063`. | The liberal `low/w1/none` setting preserved more speed but worsened wrong-target rate by `10.414 +/- 5.745` percentage points. |
| H7 | Generate-then-select branching preserves fast adaptation while controlling candidate proliferation. | **Partially supported** | `provisional_split_plus_selection` differed from `split_only` by `+4.600 +/- 17.259` lag steps and `-0.004 +/- 0.031` F1, while reducing candidate count by `14.852 +/- 4.016`. It caused no evaluator-confirmed useful-candidate damage. | Candidate-count control is supported, but non-significant lag/F1 differences do not prove formal equivalence. False stabilization remained `+2.104 +/- 4.432` percentage points above `split_only`. |
| H8 | Evidence-window selection improves liberal provisional branching. | **Supported** | Selection versus provisional branching reduced false stabilization by `5.811 +/- 3.378` percentage points and candidate count by `2.467 +/- 0.599`, without a clear lag or F1 cost. | It still did not beat the fast split baseline on false stabilization in that audit. |
| H9 | Calibrated activation reduces false stabilization without a clear loss relative to provisional selection. | **Partially supported** | In the calibrated audit, non-budgeted calibration reduced active-candidate false stabilization by `10.382 +/- 4.544` percentage points versus provisional selection. Lag changed `+13.480 +/- 18.279`; F1 changed `-0.023 +/- 0.024`. | The experiment did not define equivalence margins, so CI overlap with zero is not proof of preserved lag or F1. Against `split_only`, lag worsened by `17.200 +/- 14.645` steps. |
| H10 | A candidate budget improves parsimony without secondary costs. | **Partially supported** | In the provisional audit, budget versus selection reduced candidate count by `0.342 +/- 0.234` with no clear lag or F1 change. | In the calibrated audit, the budgeted mode had high churn (`15.680 +/- 10.344`) and a larger F1 reduction versus provisional selection (`-0.030 +/- 0.025`). |
| H11 | Candidate revision reliably reduces obsolete-candidate persistence. | **Partially supported** | Candidate-only re-entry reduced obsolete persistence by `2.058 +/- 1.645` percentage points versus no re-entry. Split-only changed it by `-2.090 +/- 2.064`. | Later targeted and provisional mechanisms did not show clear additional reductions. The metric remains sensitive to probe and candidate matching rules. |
| H12 | Improved objecthood revision produces reliable post-change viability recovery. | **Unresolved** | Recovery-slope intervals were inconclusive across mechanism, decomposition, provisional, and calibrated audits. | Policy behavior, exploration, resource depletion, and delayed credit confound viability recovery. |
| H13 | The current results establish selfhood, consciousness, or worldhood. | **Unsupported** | No experiment represents a self-model, phenomenal state, or learned global causal partition. | RD-World v0.1 only tests operational sensorimotor objecthood candidates. |

## Strongest current claim

> Conflict-driven objecthood splitting improves adaptation in a non-stationary,
> ambiguous role-reversal world.

This claim is supported by the formation-controlled candidate revision
decomposition. It should not be generalized beyond the tested rule-based
environment without further validation.

## Refined mechanism hypothesis

The best current design hypothesis is:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

Each stage addresses a different failure:

- conflict-driven objecthood split supplies plasticity,
- provisional branching avoids premature parent deletion,
- evidence-window selection controls weak branches,
- calibrated activation delays full policy commitment.

The sequence improves control, but it does not remove all trade-offs.
