# RD-World v0.1 Glossary

These terms are operational definitions for the current simulator. They should
not be read as claims about consciousness or metaphysical ontology.

## Recursive Distinction

A distinction-making process in which action-outcome consequences feed back
into how later distinctions are selected, organized, revised, and allowed to
guide action. The recursion need not occur only at the level of sensory feature
weights. It can also revise the objecthood candidates formed from recurring
trace structure.

```text
distinction structure
  -> action/outcome trace
  -> revised future distinction structure
```

### Feature-level re-entry

Trace outcomes update `BoundaryGate` feature weights and therefore change which
sensory features are salient in later processing. This was the original narrow
v0.1 interpretation of Recursive Distinction.

### Candidate-level re-entry

Trace conflicts revise objecthood candidates after formation through
confidence change, splitting, selection, suppression, merging, or obsolete
marking. The mechanism audits found stronger adaptation evidence for this level
than for feature-level re-entry alone.

### Refined v0.1 Recursive Distinction mechanism

The operational v0.1 mechanism is the candidate-level recursive sequence:

```text
conflict-driven objecthood split → provisional branching → evidence-window selection → calibrated activation
```

An established distinction participates in prediction, conflicting outcomes
create alternative candidate organization, accumulated evidence selects among
those alternatives, and activation controls how strongly the revised
distinction influences future action.

## Objecthood

Stable, re-identifiable predictive organization in sensorimotor traces. In
v0.1, objecthood is represented by an online prototype binding local sensory
features, relative position, action, predicted viability change, and predicted
resistance. It is not a hidden entity label.

## Trace

A recorded distinction-action-outcome relation:

```text
selected sensory context + action
  -> movement resistance + delta viability
```

Delayed traces also contain source step, commit step, temporal delay, and
eligibility.

## Re-entry

Feedback from stored traces or candidates into future processing. The audits
separate three targets:

- feature-level re-entry changes `BoundaryGate` weights,
- candidate-level re-entry revises object candidates,
- policy re-entry changes action preferences.

## Candidate formation

The basic process that creates or updates an `ObjectCandidate` from a trace.
Formation is enabled across fair mechanism-ablation conditions.

## Candidate revision

Operations that respond to prediction conflict after formation: confidence
weakening, split, merge, suppression, pruning, and obsolete marking.

## Split

Creation of a distinct candidate when a trace conflicts with an existing
candidate's action-outcome prediction. A split preserves incompatible causal
signatures instead of averaging them into one prototype.

## Provisional child

A newly split candidate that can accumulate evidence but has weak policy
influence. Its existence is acknowledged without full commitment.

## Validated child

A provisional child that passes an evidence-window validation threshold. It can
moderately influence policy, and its validation may begin parent decay.

## Active child

A validated child that passes the activation threshold for the required number
of consecutive compatible traces. It receives full policy influence.

## Obsolete parent

A former active candidate whose predictions have been consistently
outperformed by an active child over the evidence window. Obsolescence is
stronger than temporary inhibition.

## False stabilization

A fully active objecthood prediction for a pattern that the hidden evaluator
judges action-irrelevant in the current phase. In the calibrated audit,
provisional and validated candidates do not count as fully stabilized.

## Obsolete persistence

The fraction of post-change evaluator probes still governed by pre-change or
missing role predictions. It measures how long outdated object organization
continues after role reversal.

## Adaptation lag

The number of steps after role reversal until obsolete predictions remain
below the audit threshold for the required consecutive evaluation window.
Lower is better.

## Objecthood F1

Evaluator-only F1 for detecting currently action-relevant patterns from
candidate predictions. Hidden labels determine evaluation targets but never
enter agent observations, traces, candidates, or policy.

## Candidate churn

Post-change candidate creation, pruning, and merging. Churn measures structural
turnover, not merely candidate count.

## Useful-candidate damage

Decay or suppression applied to a candidate that the evaluator later judges
predictively correct for its trace context.

## Graded ontological commitment

The design principle that candidate existence and behavioral authority are
different variables. RD-World represents commitment through lifecycle states:

```text
provisional < validated < active
```

This is an engineering metaphor for graded predictive commitment, not a claim
that the agent possesses an ontology in the human or philosophical sense.

## Selfhood

A future extension in which the system would distinguish persistent
self-caused, body-bound, or viability-regulating processes from external
candidate dynamics. RD-World v0.1 does not implement this.

## Worldhood

A future extension in which object candidates participate in a learned,
persistent causal partition larger than local action-outcome prototypes.
RD-World v0.1 does not implement this.
