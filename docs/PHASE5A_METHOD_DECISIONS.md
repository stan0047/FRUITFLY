# Phase 5A methodology decisions

Status: finalised methodology decision record. No experiment was rerun for this
document, no frozen artifact was modified, and no parameter was tuned after
seeing the corrected run.

Frozen reference run:
`outputs/mcns_phase5a_run_corrected/`

## G4 definition

Currently implemented, `evaluate_gates` gates G4 on:

1. the drive-matching invariants returned by `verify_drive_matching`;
2. the swap-participant fraction reaching/exceeding 0.5;
3. `matched_verification.self_loops == 0`.

The selected G4 quantity is swap-participant involvement, not final displaced
edge-target relations.

`matched_verification.edge_change_fraction` is **final displaced
source→target relations**:

```text
1 - |measured_edges ∩ matched_edges| / num_edges
```

`matched_control_verification.metadata_edge_change_fraction` is **swap-participant
involvement**:

```text
number_of_unique_edge_indices_used_in_at_least_one_swap / num_edges
```

These are different because a swap may be later undone or overlap an original edge
in the final graph. Therefore the metadata value can exceed 0.5 while the verifier
edge-change fraction remains below 0.5.

Current explicit state in source:

```text
G4_DEFINITION = SWAP_PARTICIPATION
```

`evaluate_gates` now uses `matched_verification.metadata_edge_change_fraction`
when present, falling back to the verifier `edge_change_fraction` only if the
metadata is unavailable. The lower-level verifier still reports the final
displaced fraction for inspection as `final_displaced_edge_change_fraction`.

## G5 calibration

`calibration_shift_null` constructs the observed statistic and a seeded count-
preserving scramble null using the same representation: lateral profile for the
group, bin-centroid at positions `0 .. 1`, observed centroid shift
`c_left - c_right`, and null samples of the same shift after within-trial
count-preserving permutation.

The declared gate test is:

```text
G5_CALIBRATION = UPPER_TAIL
finite_observed_max_abs_shift > finite_null_abs_p95
```

with finite NaN inputs removed only for the summary array and the choice explicit
in the decision record. Equality of observed and null
**maxima** no longer causes a false pass; it is the 95th percentile of the null
absolute-shift distribution that becomes the pass boundary.

Scientific semantics:

- **MAX_NULL** asks whether the largest observed magnitude exceeds the largest
  magnitude the count-preserving null can produce.
- **UPPER_TAIL** asks whether the observed magnitude sits in the upper tail of the
  null distribution (for example p95). This would be a per-small-sample empirical
  tail comparison, not a maximum-null comparison.

`calibration_shift_null` now returns:

```text
observed_exceeds_calibration
null_max_abs_shift
null_abs_p95
```

where `observed_exceeds_calibration` is true only when the finite observed
maximum strictly exceeds `null_abs_p95`. `null_max_abs_shift` remains reported as
an auxiliary maximum, not as the comparator.

## G3 liveness requirement

`DRIVE_MATCHED_CONTROL` currently preserves:

- neuron count and body ids;
- cell-type labels;
- soma side and soma location;
- per-neuron in-degree and out-degree;
- per-neuron incoming and outgoing synaptic weight;
- every per-edge synapse count attached to the same source;
- total synapse count and edge-weight multiset.

It does **not** preserve target composition and is not required to preserve
activity, liveness, motif structure, or downstream layer ordering.

G3 therefore measures the liveness of the measured/matched/shuffled arms. A
readout group is informative for the declared T4/T5 comparison only when the
declared zero-activity and saturation thresholds are met in both primary arms.

Current explicit state in source:

```text
G3_LIVENESS_REQUIREMENT = REQUIRED_FOR_PRIMARY_CONTROL
```

That is already the corrected implementation: `evaluate_gates` computes G3 with
per-arm liveness records and marks a T4/T5 group informative only when both
`MEASURED` and `DRIVE_MATCHED_CONTROL` satisfy the declared thresholds. A silent
matched control blocks the paired comparison rather than being reinterpreted.

## NaN handling

`_phase5a_paired_margin` deliberately preserves NaN per-fold margins and reports:

- `nan_folds_present`;
- `insufficient_finite_folds`;
- `effect_supported=False` when appropriate.

No NaN is imputed to zero, dropped silently, or overridden by finite folds.

## File status

This document is the authoritative decision record for the three fields above.
It does not permit a new real-data run by itself: no rerun, sweep, or parameter
change may be started from this file alone.
