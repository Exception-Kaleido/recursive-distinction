# November 2026 audit status — RD-World v0.1

**Scope:** historical June 2026 code and results, commit `215b1cddcfa570ea34cf7bfae0a7a9f2394eb537`.

The original June reports are retained as historical evidence, **not** as corrected causal estimates. A November audit found:

1. A newly split candidate may incorporate its founding trace twice. A limited single-update sensitivity test changed the split-only paired latency difference from −42.840 ± 17.176 to −10.300 ± 10.350 steps against formation-only (50 paired seeds, capped latency). This is not a fully corrected replication and does not prove the true effect is zero.
2. The documented five consecutive acceptable steps are implemented as a five-step rolling mean.
3. Some post-reversal decoy false-positive counters use the original entity kind instead of its current viability role.
4. Non-adapting runs are assigned the evaluation horizon as latency, and some zero-denominator metrics return zero.
5. Historical result tables span differing code/metric configurations; paired differences must be interpreted within their own audits.

**Interpretation:** the system implements label-free (in the semantic-class sense) predictive candidate formation, revision and activation in a handcrafted grid world. The magnitude and isolated causal contribution of splitting require corrected tests. No claim of AGI, consciousness, selfhood or general ontology discovery is made.

**Next:** fix the evidence update contract and metric definitions, add regression tests, rerun paired seeds with per-seed outputs, then issue a versioned corrected results document. The source and archived figures remain uncorrected as of this notice.
