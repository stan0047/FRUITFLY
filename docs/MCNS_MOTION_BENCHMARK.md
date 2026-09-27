# MCNS motion representation benchmark

What this phase measures, and — more importantly — what it cannot.

## Status: read this before any number below

| | |
| --- | --- |
| **Phase 4B preliminary recurrent-vs-shuffled signal** | **RETRACTED.** The 8-trial margin of +0.079 did not replicate and must not be cited, quoted or reused. |
| **Phase 4C** | The reproducible result. A pre-declared replication: `synapse_scale` fixed at 0.25, 20 trials, 500 permutations, seed 0. |
| **Phase 4C outcome** | **A null for one contrast, in one simulation, at one gain.** `MCNS_RECURRENT` − `SHUFFLED_CONTROL` = **+0.0000** (95% CI [−0.1040, +0.1040], paired t p = 1.0, paired sign-flip null p = 1.0). |
| **What that does NOT establish** | It does **not** show that the MCNS circuit computes nothing. It does **not** show direction selectivity. It does **not** license any biological claim. |
| **Artifacts** | `outputs/mcns_motion_benchmark_phase4c/` |

> **Provenance note.** The frozen Phase 4C artifact on disk
> (`phase4c_run.log`, `phase4c_report.json`) was written *before* the retraction
> decision and its verdict paragraph reads "NOT retracted: it is superseded".
> The report source now emits retraction language. The artifact has **not** been
> edited — it is a record of what that run printed, and rewriting it would
> destroy the provenance of the run. All statistics in it are unaffected; only
> that one sentence differs. This document and the report source are
> authoritative; the artifact is the historical record.

> **Correction note — `constant_features`.** artifact and code now differ by design. The frozen Phase 4C artifacts store constant_features under the old definition (counted on fold 0's training block only): 13 / 24 / 12 of 32 for recurrent / randomised / feedforward. The code now reports a whole-design count — columns constant in at least one fold's training block — plus a stricter constant_features_every_fold companion and an explicit constant_features_definition string. A re-run would therefore report different constant_features values than the frozen artifact contains. This is a reporting correction only: the field is never an input to the decoder fit, and the frozen accuracies, separation ratios, liveness and the margin are unaffected. Note also that the single-split decode_population path still uses a third, per-split definition and is inconsistent with _standardise_batch; it is not used by this benchmark.

```bash
python scripts/run_motion_benchmark.py --quick      # bounded subset, ~4 min
python scripts/run_motion_benchmark.py              # whole population, hours

# Phase 4C: a pre-declared replication. Every scientific parameter is pinned;
# a flag that would change any of them is refused by name.
python scripts/run_motion_benchmark.py --phase-4c --estimate   # timing estimate
python scripts/run_motion_benchmark.py --phase-4c             # the declared run
```

## The question, stated precisely

> **Does measured MCNS recurrent wiring transform spatiotemporal visual input
> differently from a feedforward version of the same wiring, and from a
> degree-preserving randomisation of it?**

Three things follow from that phrasing, and all three are load-bearing.

1. It is a question about **transformation**, not about relay. A circuit that
   passed its input through unchanged would score the *relay bounds* in the table
   below. Anything above those bounds is something the wiring did.
2. It is a question about **the wiring**, not about the stimulus. The
   `SHUFFLED_CONTROL` has the same node count, the same edge count, both degree
   sequences and the same synapse-count multiset, and different wiring. If it
   scores the same, the measured wiring is not what produced the number.
   **Caveat, quantified in Phase 4C:** preserving the degree *multisets* does
   **not** preserve per-cell-type drive, so this control is not matched on the
   quantity the readout depends on. See [the measured mismatch](#the-shuffled-control-is-not-drive-matched).
3. It is **not** a question about direction selectivity. See the claim ladder.

## Biological facts this rests on

Read from the MCNS v1.0 annotation and connectivity tables, and from prior
literature. None of it is inferred by this project.

| Fact | Source |
| --- | --- |
| `L1 -> Mi1 -> T4a/b/c/d` is the ON motion chain | Takemura 2013 / Behnia 2014 / Pinto-Teixeira 2015, via `docs/VISUAL_CIRCUIT_CANDIDATES.md` |
| `L2 -> Tm1`/`Tm2 -> T5a/b/c/d` is the OFF motion chain | same |
| L1 is the ON contrast channel, L2 the OFF contrast channel | standard fly literature |
| T4a-d and T5a-d are direction-selective in the fly | prior physiology |
| All 13 cell types are present in MCNS with L/R-balanced populations | measured: L1 1,776 · L2 1,779 · Mi1 1,773 · Tm1 1,777 · Tm2 1,766 · T4a-d 6,861 · T5a-d 6,719 |
| L1, L2, Mi1 and T4/T5 are columnar, so the circuit tiles | Hoeller et al. 2026 |

## MEASURED

Read out of the reconstruction, unmodified by this project:

- body ids, exact cell-type labels, `superclass`, `somaSide`, soma coordinates
- directed body-to-body connectivity
- **integer biological synapse counts**, stored `int64` and never normalised,
  rescaled or reinterpreted. A synapse count is a count of chemical synapses. It
  is not efficacy, not a conductance and not a strength.
- `somaSide`, which is what makes a hemisphere-balanced neuron selection possible

## SIMULATED

Computed here, and a property of this model rather than a recording:

- every membrane potential, spike and firing rate
- every population representation, distance and decoder accuracy
- the propagation of drive from L1/L2 through Mi1/Tm1/Tm2 to T4/T5

## ASSUMED

Chosen by this project because no data supports them:

- **`synapse_scale` and the coupling law.** `coupling = count * synapse_scale`.
  Mapping a count to a driving current is a modelling decision, and it is the
  single largest assumption in the model. The model is extremely sensitive to it:
  at 0.05 nothing propagates past the input layer at all. Every gain in the sweep
  is reported; none is selected after the fact.
- **all LIF parameters.** `tau_membrane`, `v_threshold`, `v_reset`,
  `refractory_periods` are generic. Not measured for any of these neurons.
- **the synthetic stimulus.** Energy-matched moving spot, not photoreceptor
  activity, not luminance, not a contrast model, not derived from any retina.
- **the position map.** Real `somaLocation` where present, a deterministic rank
  fallback otherwise. The fallback is needed because only ~17 percent of L1/L2
  bodies carry coordinates. The layout is *not* an exact mirror between
  hemispheres, so a per-hemisphere drive level carries a small residual
  asymmetry; the direction cue is the trajectory.
- **the readout resolution.** 8 exact subtypes x 4 lateral bins = 32 columns.
  The number of lateral bins is a free parameter.
- **the decoder.** Multinomial logistic regression, its optimiser settings, and
  the leave-one-trial-out fold structure.
- **the `SHUFFLED_CONTROL` construction.**
- **the bounded neuron subset**, when the run is in subset mode.

## The stimulus, and why it does not hand over the answer

Four properties, each of which is asserted by a test:

1. **Direction is a trajectory, never a label.** `LEFTWARD` and `RIGHTWARD`
   differ only in the sign of the sweep velocity, `x(t) = 1 - 2t` against
   `x(t) = -1 + 2t`. The ON/OFF balance is identical to within floating point
   of the total drive: `ON/(ON+OFF) = 0.5` exactly, for every condition.
2. **Energy is matched exactly.** At every timestep each channel drives exactly
   `n_units` neurons at exactly `current`, so total drive magnitude
   `2 * n_units * current * n_steps` is identical for every condition, in every
   polarity, in every trial.
3. **The sweep start is randomised per trial**, drawn from `(seed, trial)` and
   **not** from the condition, so the two directions within a trial traverse the
   same stretch of field and the comparison stays paired. A sweep that always
   began at the same edge would let a decoder key on that edge.
4. **The noise is common random numbers**, shared across conditions within a
   trial and step. Conditions then differ only in drive pattern, so any
   difference a decoder finds is stimulus structure rather than an independent
   noise draw.

`STATIC` is the matched-energy control: the same spot, the same drive, the same
duration, held at the field centre. It differs from the moving conditions in
trajectory and in nothing else.

`LEFTWARD_POLARITY_REVERSED` inverts the contrast sign on the OFF channel,
leaving the ON channel, the trajectory and the energy untouched. It is a
**channel-assignment** control, not a discrimination task: the prediction is that
the OFF-driven group (L2 -> Tm1/Tm2 -> T5) is suppressed and the ON-driven group
(L1 -> Mi1 -> T4) relatively spared. Expect it to separate trivially, which is
why it is reported next to `MAGNITUDE_ONLY`.

## The controls

| Control | Question it answers | What a passing result looks like |
| --- | --- | --- |
| `INPUT_INSTANT` | Is the answer already in two numbers? | **must not separate.** This is the acceptance gate. |
| `INPUT_SPATIAL` | How much is in the drive pattern, at the readout's spatial resolution? | the spatial **relay bound** |
| `INPUT_TEMPORAL` | How much is in the *ordering*, ignoring magnitude entirely? | the temporal **relay bound** |
| `MAGNITUDE_ONLY` | Is it just drive level? | must not separate on its own |
| `STATIC` | Is it motion at all? | must not separate from moving |
| `SHUFFLED_CONTROL` | Does the **wiring** matter? | the decisive wiring control |
| `MCNS_FEEDFORWARD` | Do the measured feedback edges matter? | a wiring ablation against `RECURRENT` |
| label-permutation null | Is the decoder overfitting? | observed above the null's 95th percentile |

### The acceptance gate

`INPUT_INSTANT` is two numbers: the whole-run mean drive of the ON channel and of
the OFF channel. If those already separate the conditions above the 95th
percentile of their own label-permutation null, then the conditions differ in
something other than their spatiotemporal structure, and **no downstream number
may be attributed to MCNS processing**. The run is flagged
`interpretation_allowed: false`, not quietly reported.

Because the energy match is exact, the gate features are *identical* across
conditions within a trial — the summary reports
`gate_feature_max_spread: 0.0`. The gate therefore cannot fail for this stimulus.
That is a stronger statement than a decoder failing to find signal, and both are
reported so the distinction is visible.

The gate reports four states, and `undecidable` is deliberately distinct from
`pass`: with `--permutations 0` nothing was tested, and reporting that as a pass
would be a false positive.

## The decoder protocol

**Leave-one-trial-out cross-validation.** Each fold holds out one trial index
across *all* conditions, so a held-out fold always contains every class and no
fold can be advantaged by which condition happened to be excluded. Reported per
group: mean accuracy, standard deviation across folds, number of folds, the
pooled confusion matrix over every held-out sample, and the sample count, which
is **asserted** against the expected value rather than trusted.

Three properties make the previous failure impossible to repeat:

- records are indexed by `(stimulus, trial)`, never by trial alone. Trial
  indices run 0..N-1 *within each stimulus*, so keying on trial alone
  silently discarded every stimulus but the last, and every group then reported
  accuracy 1.000 with a confusion matrix of `[[0,0],[0,n]]`. That was read as
  "the benchmark is saturated" and it cost a whole experimental phase.
- a problem that does not contain both classes is **refused**, not scored.
- the expected test-sample count is computed and compared, and a mismatch is
  reported in the payload.

**The permutation null** relabels whole trials and rescores with the identical
code path. Because a permutation changes only the labels, the leave-one-out
design and its standardisation are built once and every permutation is scored in
a single batched fit; the batched fold solver is asserted bit-identical to the
reference single-fold solver.

## The readout

The primary representation is **8 exact subtypes x 4 lateral bins = 32 columns**
for `T4+T5`. T4a/T4b/T4c/T4d and T5a/T5b/T5c/T5d stay distinct at every level.

A direction cue for a moving stimulus lives in *which part of the field*
responded, so a population *mean* discards exactly the information the question is
about. `cell_type` (8 columns) and `per_neuron` (one column per neuron) are
reported as secondary comparisons so the difference is visible rather than
asserted. `per_neuron` is overparameterised at this trial count by construction.

Lateral bins use **quantile** edges, not equal-width cuts. The position map
concentrates neurons near the two hemisphere lines, so equal-width bins over the
real subset come out as roughly 562 / 88 / 102 / 548 neurons: two fat outer bins
and two starved inner ones.

## What this does not show

- **Not direction selectivity.** See below.
- **Not that the fly detects motion.** The stimulus is synthetic drive.
- **Not that the circuit computes a motion signal in the fly.** It is a sparse
  LIF model with one generic parameter set over a bounded slice of a measured
  graph.
- **Not that a T4/T5 score above the relay bounds is biologically meaningful.** It
  is a property of this model at this gain.
- **Not a whole-brain result.** In subset mode the report says
  `SUBSET: X of Y neurons` and states the omitted edge count.

## The four-level claim ladder

Every statement this project makes belongs to exactly one of these levels. A
result at one level never licenses a claim at a higher one.

**Level 1 — Anatomical connectivity. MEASURED.**
The extract contains edges `L1 -> Mi1 -> T4a-d` and `L2 -> Tm1/Tm2 -> T5a-d` with
integer synapse counts, read from MCNS v1.0.

**Level 2 — Simulated spike propagation. SIMULATED.**
A sparse LIF model over that graph propagates drive from L1/L2 to Mi1/Tm1/Tm2
and, at sufficient coupling gain, to T4/T5. *A decoder score on a silent
population is noise, so the report prints a liveness table and the score is
meaningless without it.*

**Level 3 — Representation of a synthetic motion trajectory. SIMULATED, with an
assumed stimulus and readout.**
T4/T5 activity may differ between two energy-matched sweeps more than its own
trial-to-trial spread, and more than a degree-preserving randomisation of the
same graph does. *This is the level this benchmark operates at.*

**Level 4 — Biological direction selectivity. NOT CLAIMED, AND NOT TESTABLE WITH
THIS DATA.**
Reaching it would need signed connectivity and recorded responses to a controlled
stimulus sweep. Neither exists here.

### Why level 4 is not reachable

The MCNS connectivity table has exactly three columns: `body_pre`, `body_post`,
`weight`. There is **no synaptic sign**. Measured on the benchmark subset: zero
negative coupling entries out of 1,709, minimum 0.25.

Every coupling in this model is therefore non-negative. The network can sum and
never subtract. A Reichardt-type motion detector is built from ON-minus-OFF
subtraction between spatially offset channels; it cannot be constructed from
unsigned weights. `docs/VISUAL_CIRCUIT_CANDIDATES.md` already states this: *"A
pure feedforward LIF chain will not reproduce direction tuning. This is the single
biggest scientific risk in the project."*

The measured `feedforward` result is consistent with that: the documented chains
are a pure excitatory relay, so any left/right asymmetry it shows is the input
pattern arriving, not a computation performed on it.

## Reporting rules

- `MEASURED`, `SIMULATED` and `ASSUMED` are separate top-level keys in
  `benchmark_summary.json` and the run prints them.
- The claim ladder is written into the summary, so a reader who only has the JSON
  still gets the limitation.
- `identical_across_conditions` lists what the three circuit conditions share,
  including the selected neuron set, the input energy, the noise realisation, the
  sweep start offset and the decoder settings.
- Every gain in the sweep is reported with its liveness. No gain is selected
  after the fact.
- A saturated or undecidable result is stated, not inferred by the reader.

---

# Phase 4C: a pre-declared replication

## What it is, and what it is not

Phase 4B produced **one** observation at `synapse_scale = 0.25` on the bounded
quick subset with 8 trials: `MCNS_RECURRENT` scored above `SHUFFLED_CONTROL` on
the primary `T4+T5` readout. Eight trials is a very small number and one gain is
one draw, so that observation cannot on its own distinguish signal from
trial-to-trial noise.

Phase 4C asks exactly one question about it:

> **Is the Phase 4B `MCNS_RECURRENT` − `SHUFFLED_CONTROL` difference reproducible
> at substantially higher trial count, under the same declared conditions?**

This is a **replication and precision experiment**. It is **not** a new model,
not a redesigned question, and not a search for a better gain. Nothing about the
circuit, the encoder, the readout, the stimulus or the biological assumptions was
changed to run it.

## The pre-declaration, and why each number was chosen before the run

The whole declared configuration lives in
`src/flybrain/benchmark/phase4c.py::PREDECLARED` and is written into every
Phase 4C artifact under `phase4c.predeclaration`. `enforce_predeclared` refuses
any value that differs, and the script refuses a conflicting flag **by name**,
so no declared parameter can be moved from the command line after the fact.

### Why `synapse_scale = 0.25`, and only that one

0.25 is the **lowest** gain in the Phase 4B sweep, and it is the gain at which the
Phase 4B recurrent-versus-shuffled observation was made. Re-running an
observation at the condition it was made under is the only replication of it.

It was chosen **before** the run, from the Phase 4B design. There is no sweep in
Phase 4C. `--scales 0.5` is refused. If 0.25 turns out to be silent, saturated or
otherwise uninformative, **that is the finding**: it is reported as such, the gain
is *not* changed, and the run is *not* repeated. The population budget, the
feature bins, the stimulus, the trial protocol and the permutation count are not
changed either. `predeclared_block()["not_done"]` lists every one of those
refusals and is copied into the output.

### Why 20 trials

Leave-one-trial-out accuracy is an average over trials, so its standard error
falls roughly as `1/sqrt(trials)`. Going from the Phase 4B 8 trials to 20 shrinks
the standard error of a single condition's mean accuracy by about
`sqrt(8/20) = 0.63`, i.e. roughly 1.6×, and 20 is what makes the paired
recurrent-minus-shuffled fold differences wide enough to carry an uncertainty
statement at all.

20 is also the smallest round number above the 12 the full-mode preset used, and
it leaves the **train/test structure untouched**: still leave-one-trial-out, still
20 folds, each holding out one trial index across *both* sweep directions so no
fold can be advantaged by which direction was excluded. Trial seeding stays
deterministic, the noise stays common random numbers within a trial, and the
randomised per-trial sweep start stays on. The stimulus energy is unchanged and no
condition-dependent noise was introduced.

### Why 500 permutations

The Phase 4B quick preset used 100. A permutation null needs enough draws for a
95th percentile to mean anything: from 100 draws the p95 sits on the 95th ordered
draw, from 500 it sits on the 475th, and the smallest attainable p-value drops
from 0.01 to 0.002. The cost is linear in this number. It is **fixed here and
recorded in the output** (`predeclared_permutations`, `permutations_run`,
`n_permutations`, and `n_permutations` inside every null payload), and it is never
reduced to make a run faster — `--permutations 100` is refused under
`--phase-4c`.

## What was held fixed from Phase 4B

Everything except the three numbers above:

| Held fixed | Value |
| --- | --- |
| stimulus conditions and polarities | `LEFTWARD`, `RIGHTWARD`, `STATIC`, `LEFTWARD_POLARITY_REVERSED` |
| contrasts decoded | all three, as in Phase 4B quick |
| energy match | exact, and re-asserted every run (`gate_feature_max_spread == 0.0`) |
| noise | common random numbers within a trial, SD 1.0 |
| sweep start | randomised per trial, drawn from `(seed, trial)` only |
| primary readout | `subtype_lateral`, 4 lateral bins, 8 exact subtypes → 32 `T4+T5` columns |
| secondary readout | `subtype_lateral_raw`, magnitude kept, reported and never promoted |
| controls | `INPUT_INSTANT`, `INPUT_SPATIAL`, `INPUT_TEMPORAL`, `MAGNITUDE_ONLY` |
| protocol | leave-one-trial-out, 20 folds, decoder settings unchanged |
| null machinery | the same `cv_permutation_null` and `separation_ratio_null` |
| `n_steps` / `bin_steps` / `skip_bins` / `dt` / `seed` | 150 / 25 / 2 / 0.001 / 0 |
| node budget | 100 per exact cell type |
| circuit conditions | `MCNS_FEEDFORWARD`, `MCNS_RECURRENT`, `SHUFFLED_CONTROL`, on the **same** population and edge subset |

The population identity is now **asserted** rather than described: the run checks
`body_ids` and cell-type labels pairwise across all three conditions, prints the
result, and raises if they differ.

## Liveness before decoding

A linear decoder returns an accuracy for any set of numbers, including an all-zero
set. Phase 4C therefore reports liveness **first**, per condition, per trial, for
`T4`, `T5` and `T4+T5`: mean, median, min, max spikes per trial, the fraction of
zero-activity trials, and the number of non-zero feature columns out of the 32.

`classify_informative` then marks each condition `informative`,
`uninformative_silent` or `uninformative_saturated`, with the thresholds echoed in
the payload. **An uninformative condition's decoder accuracy is reported and must
not be interpreted.** That is stated in the output rather than left for the reader
to infer from a feature width.

## The primary quantity, and its uncertainty

```
MCNS_RECURRENT accuracy - SHUFFLED_CONTROL accuracy
```

on `subtype_lateral` / `T4+T5` / `LEFTWARD_vs_RIGHTWARD`, plus each condition's
separation ratio. Both conditions are evaluated on the same trials with the same
noise realisation, the same sweep start and the same fold structure, so the margin
is a **paired** quantity and is analysed as one: per-fold differences, mean, sd,
a paired t interval, and a paired sign-flip permutation null at the declared
count. `effect_supported` is true only when both tests agree, and the word
"effect" is not used otherwise.

Each condition's separation ratio is additionally judged against its own
trial-permutation null, because the decoder permutation null is uninformative when
the two classes form tight clusters: it can equal the observed accuracy while the
geometry is clean.

A ratio **below 1** means repeated presentations of the *same* stimulus are further
apart than the two directions are from each other. A ratio at or below its own 95th
percentile means the readout is not separating the conditions at all, whatever the
decoder reports.

## Informative versus uninformative — the three verdicts

`_replication_verdict` decides one thing, by a rule stated in the output:

| Outcome | When |
| --- | --- |
| `replicated` | the paired t and the paired sign-flip null both reject at α = 0.05 |
| `uninformative` | either condition is silent or saturated at the pre-declared gain — the run did not test the question, and says so instead of reporting the number it produced |
| `not_reproduced` | both conditions were live, and the margin's interval contains zero |

The separation-ratio verdicts are reported alongside but do **not** decide this,
because a ratio sitting at its own null is a statement about the readout at this
gain rather than about the wiring, and conflating the two is the error this
experiment exists to avoid.

Every verdict carries the scope statement: *one comparison, in one simulation, at
`synapse_scale` 0.25, on a bounded subset*. It is not a statement about the MCNS
circuit and not a statement about a fly.

## Reporting the run

Per condition: per-fold accuracy, mean, sd, chance level, null p95, z-score,
separation ratio and its null, number of trials, number of folds, number of
permutations, held-out sample count asserted against the expected count, and the
pooled confusion matrix. Then `MEASURED` / `SIMULATED` / `ASSUMED` /
`UNRESOLVED` as separate keys, with `UNRESOLVED` as a category in its own right —
a replication has to say what it failed to settle, and a negative or uninformative
result is information that belongs in the report rather than an absent number.

Conditions are never ranked and none is called a winner. A positive margin and a
negative margin are reported with identical statistics.

## The limits, restated

Unchanged from Phase 4B and repeated in every Phase 4C artifact:

- **A synapse count is not synaptic efficacy.** It is a count of chemical
  synapses: not a conductance, not a strength.
- **The connectivity graph has no synaptic sign.** Every coupling is non-negative,
  the network can only sum and never subtract, and a Reichardt-type ON-minus-OFF
  detector is not constructible from it.
- **This experiment cannot establish direction selectivity.** The stimulus is
  synthetic drive whose direction is specified by the encoder.
- **The encoder contains the externally specified directional structure.** The spot
  is driven left or right by construction, so any left/right asymmetry at T4/T5 is
  that supplied structure arriving, possibly transformed.
- **T4/T5 asymmetry alone is therefore not evidence of MCNS computation.** Only the
  circuit-versus-shuffled comparison, and only with its uncertainty, speaks to
  that.
- Downstream decoding above the input relay is not evidence of MCNS computation
  unless that comparison supports it. `INPUT_SPATIAL` and `INPUT_TEMPORAL` are
  reported for exactly this reason: they establish how much direction information
  exists *before* the circuit.

## The timing estimate, and why it takes two measurements

Before the run, `--phase-4c --estimate` times one circuit condition and projects
the declared run. The permutation nulls dominate, and their cost is not a simple
function of the declared numbers, so two points are taken:

1. the required cheap one — one circuit condition, a small number of trials;
2. the same thing at the **declared** trial count, with the same reduced
   permutation count.

Point 2 is what the launch decision uses. The batched leave-one-trial-out design
is cache-resident at a small trial count and not at 20, so extrapolating in the
trial count mis-projects the run by a factor that only measurement settles. Each
point also times its nulls a second time at twice the permutation count, to
separate the fixed cost of building the design from the marginal per-permutation
cost. Both projections and their ratio are reported, so the correction is visible
rather than buried.

The rule for launching is stated, not improvised: a run projected above four hours
is not launched as a single job.

**How accurate it was, measured.** The estimate projected **15.7 min**; the
declared run took **35.2 min** (2109 s), so the projection was optimistic by about
2.2×. The two-point measurement removed a trial-count extrapolation error of only
4%, so the residual gap is not the model: it is that the single-circuit, isolated
null timing runs with a warm allocator and no interleaved work, while the real run
interleaves three circuits' decodes and nulls and their memory traffic competes.
The **bounding decision was still correct** — 35 min against a four-hour rule is
comfortably bounded — but the projection should be read as a lower estimate, and
the run's own `timings.total_wall_clock_seconds` is the number to trust. This is
why the estimate is reported *before* the run and why the run prints staged,
flushed progress: a projection that is wrong by a factor of two must not be able
to look like a frozen process.

## The outcome of the declared run

Recorded here so the replication has a result and not only a procedure. Every
number below is `SIMULATED` and belongs to `synapse_scale = 0.25` on the bounded
100-per-type subset with 20 trials and 500 permutations, seed 0.

**Replication verdict: `not_reproduced`.**

| Quantity | Value |
| --- | --- |
| `MCNS_RECURRENT` accuracy | 0.5687 (sd 0.1217 over 20 folds) |
| `SHUFFLED_CONTROL` accuracy | 0.5687 (sd 0.1699 over 20 folds) |
| `MCNS_FEEDFORWARD` accuracy | 0.5500 (sd 0.1500) |
| **recurrent − shuffled margin** | **+0.0000** (sd 0.2221) |
| paired t (19 df) | 0.000, p = 1.0000 |
| paired sign-flip null, 500 permutations | p = 1.0000 |
| 95% interval on the margin | [−0.1040, +0.1040], contains zero |
| recurrent separation ratio | 0.147 (own null p95 0.156) |
| shuffled separation ratio | 0.153 (own null p95 0.169) |

Both conditions were live — the gate passed, no condition was silent or saturated,
and the zero-activity fraction was 0.0 for every `T4+T5` group — so this is a
**measurement of a null**, not an uninformative void. Two further facts belong with
it:

- The separation ratios are **below 1**, and both the recurrent and the shuffled
  ratio sit at or below their own 95th-percentile null. At this gain and this trial
  count the `T4+T5` readout is not separating the two sweep directions above
  trial-to-trial noise, for the measured circuit *and* for the randomised control
  alike. That is a statement about the readout at this gain, not about the wiring,
  which is why it does not decide the verdict.
- The `INPUT_TEMPORAL` relay bound reaches 0.775 and `INPUT_SPATIAL` 0.562, both
  computed from the drive and identical across the three circuit conditions by
  construction. Direction information is present *before* the circuit; the
  downstream readout does not exceed that story.

The Phase 4B 8-trial margin of +0.079 is **RETRACTED**. It is withdrawn, not
merely superseded: it was a small-sample fluctuation at this gain, and it must not
be cited, quoted or reused as a result. No gain, node budget, bin, stimulus, trial
protocol or permutation count was changed in response, and none will be changed to
obtain a positive result. `phase4c.predeclaration` in the artifact lists every one
of those refusals, and the test suite asserts the refusals are enforced.

### What the null does and does not mean

- It **does** mean: at `synapse_scale` 0.25, on this bounded population, with 20
  trials, the measured and the randomised graph produced the same `T4+T5` decode
  accuracy, and the readout's separation ratio sat at or below its own null for
  both. The 8-trial difference is gone.
- It does **not** mean the MCNS circuit computes nothing. The test had no power
  against a small effect: the 95% interval bounds |Δ| at 0.104 and the minimum
  detectable |Δ| at 80% power is 0.147, so any true wiring effect below ~0.10
  accuracy units would be invisible here.
- It does **not** mean the readout carried no information. Both the decoder null
  and the separation-ratio null sit close to their observed values, and the two
  disagree on which side — neither is decisive at n = 20.
- It does **not** mean the wiring is irrelevant, because the control is not
  matched on drive (below).
- It is **not** a direction-selectivity result, and the direction cue is supplied
  by the encoder.

### The `SHUFFLED_CONTROL` is not drive-matched

A read-only diagnostic on the stored graph (no simulation) shows the shuffle
preserves the multiset invariants exactly — 1300/1300 neurons with identical
in- and out-degree, 5605 edges, 50824 synapses — but redistributes drive
substantially:

| cell type | mean incoming drive, measured | randomised | ratio |
| --- | ---: | ---: | ---: |
| L1 | 4.67 | 1.82 | 0.39 |
| Mi1 | 19.02 | 3.78 | 0.20 |
| Tm1 | 31.24 | 4.88 | 0.16 |
| Tm2 | 25.53 | 6.35 | 0.25 |
| T4a–d | 4.6–5.7 | 8.7–13.7 | 1.63–2.50 |
| T5a–d | 5.5–6.6 | 14.9–17.0 | 2.53–2.75 |

Overall per-neuron incoming-drive correlation is **−0.0055**. The global mean is
preserved (9.77 both), but drive moves *out of* the intermediate layers and
*into* the readout layer. The control's own metadata already records
`not_preserved: ["cell-type composition of edges"]`, so this is documented rather
than hidden, but it means a recurrent-versus-shuffled difference here could never
be attributed to edge arrangement alone. Consistent with that, the randomised
readout is 2.5× sparser (8/32 non-zero features vs 20/32).

What this does **not** say: that the MCNS circuit computes nothing, that the fly
has no direction-selective mechanism, or that direction selectivity is absent from
T4/T5. The connectivity table still carries no synaptic sign, so this experiment
still cannot address that question at any trial count or gain. The only thing
resolved here is that the Phase 4B recurrent-versus-shuffled number was **not a
reproducible signal**, which is the question this phase was run to settle.

### The readout instrument is not currently measuring structure

`MAGNITUDE_ONLY` is a **one-feature** control: total T4+T5 activity. On the primary
contrast it scored **0.5750** for `MCNS_RECURRENT` against **0.5687** for the
32-column subtype × lateral readout. A single number matches — and slightly
exceeds — the structural representation.

That comparison is now printed explicitly in the report (`section 3b`) and stored
under `magnitude_only_versus_structural_readout`, because it is the most
consequential omitted comparison in the first Phase 4C report. It means the
32-column readout is **not resolving spatial or cell-type structure at this
operating point**, and any accuracy it produces is a restatement of drive level.
Neither side carries a permutation null (`null_groups` covers `INPUT_INSTANT`,
`T4`, `T5`, `T4+T5` only), so this is a point-estimate comparison and not a
significance statement.

### Two null families, and they are not interchangeable

| | family 1 | family 2 |
| --- | --- | --- |
| statistic | leave-one-trial-out accuracy | between/within centroid-distance ratio (euclidean) |
| null of | the trial-to-condition mapping | the trial-to-condition mapping |
| asks | can a linear decoder beat relabelled trials **on accuracy**? | is the observed distance ratio bigger than relabelled trials give? |
| failure mode | saturates when the two classes form tight clusters, because permuting labels destroys the mapping but not the geometry | continuous, so it does not saturate that way |
| on the primary contrast | recurrent **above** p95, shuffled **above** p95 | recurrent **below** p95, shuffled **below** p95 |

They disagree, and **neither is "the" null and neither supersedes the other**:
they measure different quantities with different power. Both are reported, both
nulls are recorded in the payload under `null_families`, and the report prints
which family every `null p95` and `z` column came from.

## Evidence classification

Every claim this project makes belongs to exactly one row. Nothing in the report
may cross a row boundary.

### MEASURED

Read out of the reconstruction, or computed by this project's own code from it,
and reported without reinterpretation.

- MCNS body ids, exact cell-type labels, `superclass`, `somaSide`, soma coordinates
- directed connectivity between candidate bodies, and the **integer biological
  synapse counts**, stored `int64` and never normalised or rescaled
- the bounded population actually simulated: 1300 of 22,451 bodies, 5605 of 340,310
  measured edges, 50,824 measured synapses
- **observed spike activity** in the simulated populations, as counts: T4+T5
  11.34 / 10.41 / 8.11 mean spikes per trial for recurrent / randomised /
  feedforward, 0 of 40 zero-activity trials for T4+T5 on the primary contrast in
  every condition
- **the 20-trial benchmark outputs**: 240 simulations, 20 leave-one-trial-out
  folds per condition, 160 held-out samples per condition, sample counts asserted
  against the expected value
- **recurrent / randomised separation statistics**: accuracy 0.56875 vs 0.56875
  (both 91/160), separation ratio 0.146537 vs 0.152869 against their own
  500-permutation nulls, paired margin +0.0000 with its paired t and sign-flip null
- **liveness and relay measurements**, reported separately from any circuit effect:
  `INPUT_INSTANT` 0.5000, `INPUT_SPATIAL` 0.5625, `INPUT_TEMPORAL` 0.7750,
  acceptance gate `pass` with a gate-feature spread of exactly 0.0
- the injected drive is provably identical across the three conditions, and
  `INPUT_INSTANT` sits exactly at chance, so the input does not hand over the answer

### SIMULATED

Computed by this project's model. Properties of the model, not of a fly.

- every membrane potential, spike, firing rate and refractory event
- the propagation of drive from L1/L2 through Mi1/Tm1/Tm2 to T4/T5
- every population representation, separation ratio, distance and decoder accuracy
- the degree-preserving randomisation used as `SHUFFLED_CONTROL` — a computational
  control, not biology

### ASSUMED

Chosen by this project because no data supports them. The largest is first.

- **the coupling law and gain**: `coupling = count * synapse_scale`, with
  `synapse_scale` pre-declared at 0.25. Mapping a count to a driving current is a
  modelling decision and the single largest assumption in the model; at 0.05
  nothing propagates past the input layer
- **the unsigned treatment of synapse counts.** A synapse count is a count of
  chemical synapses. It is **not** synaptic efficacy, not a conductance, not a
  strength, and it carries no excitatory/inhibitory identity
- all LIF parameters (`tau_membrane` 0.02, `v_threshold` 1.0, `v_reset` 0.0,
  `refractory_periods` 2) — generic, not measured for any of these neurons
- the synthetic stimulus: the energy-matched moving spot, the position-map
  fallback, the ±5% trial amplitude jitter, the noise SD, the randomised sweep
  start, and the fact that the two sweep directions differ only in the sign of the
  sweep velocity
- the readout resolution: 8 exact subtypes × 4 lateral bins = 32 columns, and the
  4-bin quantile lateral partition
- the decoder: multinomial logistic regression, 400 iterations, lr 0.5, l2 0.01,
  and the leave-one-trial-out fold structure
- the bounded population selection rule, and the node budget of 100 per exact type
- the `SHUFFLED_CONTROL` construction, including its documented failure to preserve
  per-cell-type drive

### UNKNOWN / NOT TESTED

Not measured, not modelled, and not addressable with the data this project has.

- **synaptic sign.** The connectivity table has `body_pre`, `body_post`, `weight`
  and nothing else. Every coupling here is non-negative, the network sums and never
  subtracts, and a Reichardt-type ON-minus-OFF detector is **not constructible**
  from it
- **inhibitory versus excitatory identity**, and therefore any sign-structured
  computation
- **true biological synaptic efficacy.** A synapse count is not a conductance, and
  nothing here measures what a synapse actually contributes to a postsynaptic
  membrane
- **direction-selective computation.** Never tested, and not testable here: the
  direction cue is supplied by the encoder, so any left/right asymmetry at T4/T5
  is that supplied structure arriving. **Encoder-injected directional information is
  not evidence of circuit computation**, and a T4/T5 asymmetry alone is not
  evidence of MCNS computation
- **whether the real circuit computes the tested visual variable outside this
  model.** The Phase 4C null bounds one contrast in one simulation at one gain; it
  is not a statement about the biological circuit
- the behaviour of the full 22,451-body circuit, and of the omitted 334,705 edges
- any electrophysiological quantity: no membrane potential, no conductance, no
  synaptic current was ever recorded

## Artifacts

```
outputs/mcns_motion_benchmark_phase4c/          FROZEN record of the real run
    phase4c_timing_estimate.json   pre-run estimate and both projections
    phase4c_run.log                full console report, in reading order
    benchmark_summary.json         everything, with a top-level `phase4c` key
    phase4c_report.json            the Phase 4C block plus per-condition activity
    decoder_results.json           every decode, every null
    activity.json                  per-trial spike counts per group
    extraction_cache.npz           the cached measured edge list the run used

outputs/phase4c_timing_estimates/               estimates, deliberately separate
    phase4c_timing_estimate.json   reduced-configuration timing measurements
```

The two directories are kept apart **by construction**: the estimate path ignores
`--output-dir` and always writes to `PHASE4C_ESTIMATE_DIR`, so a timing estimate
can never be dropped beside the frozen artifacts of a completed run. The names are
also not prefix-extensions of each other, because a glob of one would otherwise
sweep in the other. The estimate payload carries `is_not_a_result` and points at
the frozen directory it must not be confused with.

Reproducibility: the run is deterministic given the local MCNS files, the code, and
the declared seed. Two runs of the declaration produce identical feature digests
(pinned by a test). The measured components of the cost are simulation **1.525 s**
for all 240 trials, graph construction 1.129 s, tensor warm 2.577 s, total wall
clock 2109.294 s — the remaining ~35 minutes are the twelve 500-permutation decoder
nulls, which are optional to reproduce a *number* and required only to reproduce
the *null bands*.
