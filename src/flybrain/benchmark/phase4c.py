"""Phase 4C: a **pre-declared replication** of the Phase 4B recurrent-vs-shuffled number.

What this module is for
-----------------------
Phase 4B produced one observation at ``synapse_scale = 0.25`` on a bounded subset
with 8 trials: ``MCNS_RECURRENT`` scored above ``SHUFFLED_CONTROL`` on the primary
``T4+T5`` readout. Eight trials is a very small number and a single gain is a
single draw, so that observation cannot on its own distinguish signal from
trial-to-trial noise. Phase 4C asks exactly one question about it:

    **Is the Phase 4B recurrent-minus-shuffled difference reproducible at
    substantially higher trial count, under the same declared conditions?**

That is a *replication and precision* experiment. It is **not** a new model, not
a redesigned question, and not a search for a better gain. The pre-declaration
below was fixed before the run and the run refuses to change it.

The pre-declaration
-------------------
:data:`PREDECLARED` is the whole declared configuration. Three choices deserve a
stated reason:

``synapse_scale = 0.25``, and only that one
    0.25 is the lowest gain in the Phase 4B sweep, and it is the gain at which the
    Phase 4B recurrent-minus-shuffled observation was made. Re-running the
    observation at the gain where it was made is the only replication of it. It
    was chosen **before** the run, from the Phase 4B design, and there is no
    sweep: :func:`enforce_predeclared` refuses any other value, so no gain can be
    selected after seeing a result. If 0.25 turns out to be silent, that is the
    finding, and it is reported as such rather than fixed by rerunning at 2.0.

``20 trials``
    Leave-one-trial-out accuracy is an average over trials, so its uncertainty
    falls roughly as ``1/sqrt(trials)``. Going from 8 to 20 trials shrinks the
    standard error of a single condition's mean accuracy by roughly ``sqrt(8/20)``
    = 0.63, i.e. about 1.6x, and it is what makes the paired fold differences
    wide enough to carry an uncertainty statement at all. 20 is also the smallest
    round number above the 12 the full-mode preset used, which keeps the
    train/test structure of the existing leave-one-trial-out protocol untouched:
    still 20 folds, each holding out one trial index across *both* directions.

``n_permutations = 500``
    The Phase 4B quick preset used 100. A permutation null needs enough draws
    for a 95th percentile to mean anything: 100 puts the p95 on the 95th
    ordered draw, 500 on the 475th, and it also drops the granularity of the
    smallest attainable p-value from 0.01 to 0.002. The cost is linear in this
    number, and it is fixed here and recorded in the output rather than reduced
    to make a run faster.

Everything else is held at its Phase 4B value: the stimulus conditions and their
polarities, the exact energy match, the common-random-number noise, the
randomised-per-trial sweep start, the ``subtype_lateral`` primary readout, the
``INPUT_SPATIAL`` / ``INPUT_TEMPORAL`` / ``MAGNITUDE_ONLY`` controls, the
leave-one-trial-out protocol, the permutation/null machinery, the node budget
and the exact population and edge subset shared by all three circuit conditions.

The primary quantity
--------------------
``MCNS_RECURRENT`` accuracy minus ``SHUFFLED_CONTROL`` accuracy, on the primary
readout and the primary contrast, plus each condition's separation ratio.

A positive difference is **not** an effect on its own. The difference of two
accuracies measured on the *same* trials is a paired quantity, so
:func:`margin_statistics` treats it as one: it reports the per-fold differences,
their mean and spread, a paired t interval, and a paired sign-flip permutation
null. Only if the uncertainty supports it may the difference be called an effect
in this model's terms, and even then it is a statement about simulated spike
counts, not about a fly.

Liveness before decoding
------------------------
A linear decoder will return an accuracy for any set of numbers, including an
all-zero set. :func:`liveness` and :func:`classify_informative` exist so a run
can state "T4/T5 were silent at 0.25" instead of leaving the reader to infer it
from a feature width. A silent or saturated condition is marked
**scientifically uninformative** and its decoder accuracy is reported but not
interpreted.

What this module cannot do
--------------------------
Nothing here, at any sample size, can establish direction selectivity. The MCNS
connectivity table carries ``body_pre``, ``body_post`` and ``weight`` and **no
synaptic sign**, so every coupling is non-negative, the network can only sum and
never subtract, and a Reichardt-type ON-minus-OFF detector is not constructible
from it. Direction is not simulated here: it is *specified* by the encoder, which
sweeps a spot left or right. Any left/right asymmetry the readout shows is that
externally supplied structure arriving, possibly transformed, and a T4/T5
asymmetry on its own is therefore **not** evidence that the MCNS circuit
performed a motion computation. :data:`CLAIM_BLOCK` carries all of this into the
run output.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "ALPHA",
    "CLAIM_BLOCK",
    "PREDECLARED",
    "PREDECLARED_CONTRAST",
    "PREDECLARED_GROUP",
    "PREDECLARED_MAX_BODIES_PER_TYPE",
    "PREDECLARED_N_STEPS",
    "PREDECLARED_NULL_GROUPS",
    "PREDECLARED_PERMUTATIONS",
    "PREDECLARED_PHASE",
    "PREDECLARED_SEED",
    "PREDECLARED_SYNAPSE_SCALE",
    "PREDECLARED_TRIALS",
    "PREDECLARED_WHY",
    "REPLICATION_CIRCUITS",
    "SATURATION_MAX_SPIKE_FRACTION",
    "classify_informative",
    "claim_block",
    "enforce_predeclared",
    "feature_report",
    "fingerprint",
    "liveness",
    "margin_statistics",
    "predeclared_block",
    "timing_projection",
]

#: The phase tag written into every Phase 4C artifact.
PREDECLARED_PHASE = "4C"

#: The pre-declared gain. Fixed before the run. Not swept, not retuned.
PREDECLARED_SYNAPSE_SCALE = 0.25

#: Pre-declared trial count per stimulus condition.
PREDECLARED_TRIALS = 20

#: Pre-declared permutation count for every label-permutation null.
PREDECLARED_PERMUTATIONS = 500

#: Pre-declared node budget per exact cell type, i.e. the Phase 4B quick budget.
PREDECLARED_MAX_BODIES_PER_TYPE = 100

#: Pre-declared simulated duration and readout binning, all Phase 4B quick values.
PREDECLARED_N_STEPS = 150
PREDECLARED_BIN_STEPS = 25
PREDECLARED_SKIP_BINS = 2
PREDECLARED_DT = 0.001

#: Pre-declared seed. Deterministic trial seeding is a requirement, not a default.
PREDECLARED_SEED = 0

#: Pre-declared coupling law. ``linear`` maps a synapse count to a driving current
#: as ``count * synapse_scale``; ``log1p`` would compress the observed ~100:1
#: dynamic range. Both are modelling choices, and the pre-declaration holds the
#: one Phase 4B used, so a Phase 4C run cannot quietly switch it.
PREDECLARED_COUPLING = "linear"

#: Pre-declared stimulus protocol. These four, together with ``n_steps``, fix the
#: injected drive completely: ``n_units`` per channel per step, the per-neuron
#: current, the noise SD, and the trial-to-trial amplitude jitter. The energy
#: match ``2 * n_units * current * n_steps`` is a consequence, not a setting.
PREDECLARED_SPOT_FRACTION = 0.15
PREDECLARED_STIMULUS_CURRENT = 2.0
PREDECLARED_INPUT_NOISE = 1.0
PREDECLARED_AMPLITUDE_PERTURBATION = 0.05

#: Pre-declared decoder settings. Identical for every condition, so a
#: recurrent-versus-shuffled comparison is not a comparison of two different
#: classifiers.
PREDECLARED_DECODER_ITERATIONS = 400
PREDECLARED_DECODER_LEARNING_RATE = 0.5
PREDECLARED_DECODER_L2 = 1e-2

#: The contrast the Phase 4B observation was made on, and the one replicated.
PREDECLARED_CONTRAST = "LEFTWARD_vs_RIGHTWARD"

#: The readout group the observation was made on, and the one replicated.
PREDECLARED_GROUP = "T4+T5"

#: Every readout variant, in reporting order. The primary is first and is the one
#: the primary comparison is computed on; the rest are reported, never promoted.
PREDECLARED_VARIANTS: tuple[str, ...] = ("subtype_lateral", "subtype_lateral_raw")

#: The three circuit conditions, all on the same population and edge subset.
REPLICATION_CIRCUITS: tuple[str, ...] = (
    "MCNS_RECURRENT",
    "SHUFFLED_CONTROL",
    "MCNS_FEEDFORWARD",
)

#: The condition whose accuracy the primary comparison subtracts, and the one it
#: subtracts from, named explicitly so a reader never has to infer the direction.
PRIMARY_CIRCUIT = "MCNS_RECURRENT"
CONTROL_CIRCUIT = "SHUFFLED_CONTROL"

#: The stimulus-side relay controls, reported as in Phase 4B. Their purpose is to
#: establish how much direction information exists *before* the MCNS circuit.
RELAY_CONTROLS: tuple[str, ...] = ("INPUT_SPATIAL", "INPUT_TEMPORAL")

#: Groups that receive a label-permutation null, as in Phase 4B quick.
PREDECLARED_NULL_GROUPS: tuple[str, ...] = (
    "INPUT_INSTANT",
    "T4",
    "T5",
    "T4+T5",
)

#: Every stimulus condition presented, identical to Phase 4B quick.
PREDECLARED_CONDITIONS: tuple[str, ...] = (
    "LEFTWARD",
    "RIGHTWARD",
    "STATIC",
    "LEFTWARD_POLARITY_REVERSED",
)

#: Every contrast decoded, identical to Phase 4B quick.
PREDECLARED_CONTRASTS: tuple[str, ...] = (
    "LEFTWARD_vs_RIGHTWARD",
    "LEFTWARD_vs_STATIC",
    "LEFTWARD_vs_POLARITY_REVERSED",
)

#: Only the primary contrast gets a permutation null, as in Phase 4B quick.
PREDECLARED_NULL_CONTRASTS: tuple[str, ...] = (PREDECLARED_CONTRAST,)

#: Two-sided significance level used for every uncertainty statement.
ALPHA = 0.05

#: A condition at or above this fraction of its maximum possible spike count is
#: saturated: the readout is a rate code at its ceiling and the accuracy is a
#: property of the ceiling, not of the wiring.
SATURATION_MAX_SPIKE_FRACTION = 0.5

#: A condition at or above this mean accuracy whose own null p95 is also at or
#: above the same level is a saturated benchmark rather than a real effect.
SATURATION_ACCURACY = 0.999

#: A condition whose null p95 is at or above this level cannot be separated from
#: its own null, whatever the observed accuracy is.
SATURATION_NULL_P95 = 0.99

#: The whole declared configuration, in one auditable place.
PREDECLARED: dict[str, Any] = {
    "phase": PREDECLARED_PHASE,
    "kind": "replication / precision experiment, not a new model",
    "primary_question": (
        "Is the Phase 4B MCNS_RECURRENT minus SHUFFLED_CONTROL difference on the T4+T5 "
        "subtype_lateral readout reproducible at substantially higher trial count, under the "
        "same declared conditions?"
    ),
    "synapse_scale": PREDECLARED_SYNAPSE_SCALE,
    "gain_sweep": False,
    "trials_per_condition": PREDECLARED_TRIALS,
    "n_permutations": PREDECLARED_PERMUTATIONS,
    "max_bodies_per_type": PREDECLARED_MAX_BODIES_PER_TYPE,
    "n_steps": PREDECLARED_N_STEPS,
    "bin_steps": PREDECLARED_BIN_STEPS,
    "skip_bins": PREDECLARED_SKIP_BINS,
    "dt": PREDECLARED_DT,
    "seed": PREDECLARED_SEED,
    "coupling": PREDECLARED_COUPLING,
    "spot_fraction": PREDECLARED_SPOT_FRACTION,
    "stimulus_current": PREDECLARED_STIMULUS_CURRENT,
    "input_noise": PREDECLARED_INPUT_NOISE,
    "amplitude_perturbation": PREDECLARED_AMPLITUDE_PERTURBATION,
    "decoder_iterations": PREDECLARED_DECODER_ITERATIONS,
    "decoder_learning_rate": PREDECLARED_DECODER_LEARNING_RATE,
    "decoder_l2": PREDECLARED_DECODER_L2,
    "lateral_bins": 4,
    "readout": "subtype_lateral",
    "primary_contrast": PREDECLARED_CONTRAST,
    "primary_group": PREDECLARED_GROUP,
    "primary_readout": "subtype_lateral",
    "variants": list(PREDECLARED_VARIANTS),
    "circuits": list(REPLICATION_CIRCUITS),
    "conditions_presented": list(PREDECLARED_CONDITIONS),
    "contrasts": list(PREDECLARED_CONTRASTS),
    "null_groups": list(PREDECLARED_NULL_GROUPS),
    "null_contrasts": list(PREDECLARED_NULL_CONTRASTS),
    "relay_controls": list(RELAY_CONTROLS),
    "alpha": ALPHA,
    "saturation_max_spike_fraction": SATURATION_MAX_SPIKE_FRACTION,
    "saturation_accuracy": SATURATION_ACCURACY,
    "saturation_null_p95": SATURATION_NULL_P95,
}

#: Why each of the three changed quantities was chosen, and why before the run.
PREDECLARED_WHY: dict[str, str] = {
    "synapse_scale": (
        "0.25 is the lowest gain in the Phase 4B sweep and the gain at which the Phase 4B "
        "recurrent-minus-shuffled observation was made. Re-running an observation at the "
        "condition it was made under is the only replication of it. Chosen from the Phase 4B "
        "design before this run; there is no sweep, and the run refuses any other value, so no "
        "gain can be chosen after seeing a result. If 0.25 is silent, saturated or otherwise "
        "uninformative, that is the finding and it is reported as such; the gain is NOT changed "
        "and the run is NOT repeated."
    ),
    "trials_per_condition": (
        "Leave-one-trial-out accuracy averages over trials, so its standard error falls roughly "
        "as 1/sqrt(trials). Going from the Phase 4B 8 trials to 20 shrinks the standard error of "
        "one condition's mean accuracy by about sqrt(8/20) = 0.63, i.e. roughly 1.6x, and 20 is "
        "what makes the paired recurrent-minus-shuffled fold differences wide enough to carry an "
        "uncertainty statement. The leave-one-trial-out structure is unchanged: 20 folds, each "
        "holding out one trial index across BOTH sweep directions, and trial seeding stays "
        "deterministic."
    ),
    "n_permutations": (
        "The Phase 4B quick preset used 100. A 95th percentile from 100 draws sits on the 95th "
        "ordered draw; from 500 it sits on the 475th, and the smallest attainable p-value drops "
        "from 0.01 to 0.002. The cost is linear in this number. The count is fixed here, recorded "
        "in the output, and never reduced to make a run faster."
    ),
    "not_changed": (
        "Stimulus conditions and polarities, the exact energy match, the common-random-number "
        "noise, the randomised-per-trial sweep start, the subtype_lateral primary readout and its "
        "4 lateral bins, the INPUT_SPATIAL / INPUT_TEMPORAL / MAGNITUDE_ONLY controls, the "
        "leave-one-trial-out protocol, the permutation/null machinery, the decoder settings, and "
        "the node budget are all held at their Phase 4B values. No condition-dependent noise was "
        "added, no stimulus energy was altered, and the readout was not switched to per-neuron."
    ),
}

#: What was deliberately NOT done, so a reader does not have to ask.
NOT_DONE: tuple[str, ...] = (
    "no gain sweep and no gain selected after seeing a result",
    "no change to the population budget",
    "no change to the feature bins or the readout resolution",
    "no change to the stimulus, its energy, its polarity handling or its noise realisation",
    "no change to the trial protocol, the seeding, or the sweep-start mechanism",
    "no independent per-condition noise",
    "no switch to per-neuron as the primary readout",
    "no change to the decoder, its optimiser settings, or the cross-validation folds",
    "no reduction of the permutation count to shorten the run",
    "no change to NeuralSimulation biology, no new cell model, no new architecture",
    "no biological claim of any kind",
)


def predeclared_block() -> dict[str, Any]:
    """The declaration, as written into the run output before any number exists."""
    return {
        "predeclared": True,
        "predeclared_before_run": True,
        "may_be_tuned_after_seeing_results": False,
        "configuration": dict(PREDECLARED),
        "why": dict(PREDECLARED_WHY),
        "not_done": list(NOT_DONE),
        "statement": (
            "synapse_scale = 0.25 was pre-declared for this Phase 4C replication, before the "
            "run, from the Phase 4B design. It was not swept, not selected after seeing results, "
            "and will not be changed if the result is uninformative."
        ),
    }


def enforce_predeclared(
    synapse_scale: float,
    trials: int,
    n_permutations: int,
    max_bodies_per_type: int,
    n_steps: int,
    bin_steps: int,
    skip_bins: int,
    seed: int,
) -> None:
    """Refuse any Phase 4C parameter that differs from the pre-declaration.

    Every mismatch is reported at once rather than the first, so a caller who
    passed a contradictory configuration sees the whole list. This is the
    mechanism behind the "no post-hoc tuning" commitment: the run cannot be
    pointed at a different gain or a different trial count by a flag.
    """
    requested = {
        "synapse_scale": (float(synapse_scale), PREDECLARED_SYNAPSE_SCALE),
        "trials": (int(trials), PREDECLARED_TRIALS),
        "permutations": (int(n_permutations), PREDECLARED_PERMUTATIONS),
        "max_bodies_per_type": (int(max_bodies_per_type), PREDECLARED_MAX_BODIES_PER_TYPE),
        "n_steps": (int(n_steps), PREDECLARED_N_STEPS),
        "bin_steps": (int(bin_steps), PREDECLARED_BIN_STEPS),
        "skip_bins": (int(skip_bins), PREDECLARED_SKIP_BINS),
        "seed": (int(seed), PREDECLARED_SEED),
    }
    mismatches = [
        f"{name}={got!r} (pre-declared {want!r})"
        for name, (got, want) in requested.items()
        if got != want
    ]
    if mismatches:
        raise SystemExit(
            "error: Phase 4C is a pre-declared replication and its declared parameters cannot be "
            "changed. Mismatches: " + "; ".join(mismatches) + ". Remove the overriding flag."
        )


# ----------------------------------------------------------------- liveness


def liveness(spikes_per_trial: Sequence[int], n_neurons: int = 0, n_steps: int = 0) -> dict[str, Any]:
    """Spike-count liveness of one readout group in one condition.

    A decoder will happily report an accuracy on an all-zero feature set, and a
    reader cannot tell a silent condition from a live one by looking at an
    accuracy. So the count of spikes in the readout population is summarised
    here, per trial, with the median and the extremes as well as the mean: a
    mean driven by a handful of trials is not liveness, and a median of zero is
    not either.

    ``n_neurons`` and ``n_steps`` are the ceiling on a per-trial spike count,
    used to report the saturation fraction. Pass them when known; the fraction
    is then ``nan`` rather than a guess.
    """
    counts = [int(v) for v in spikes_per_trial]
    total = int(sum(counts))
    n_trials = int(len(counts))
    if n_trials == 0:
        return {
            "n_trials": 0,
            "total_spikes": 0,
            "mean_spikes_per_trial": float("nan"),
            "median_spikes_per_trial": float("nan"),
            "min_spikes_per_trial": None,
            "max_spikes_per_trial": None,
            "zero_activity_trials": 0,
            "fraction_zero_activity_trials": float("nan"),
            "active": False,
            "spike_saturation_fraction": float("nan"),
            "spikes_per_trial": [],
        }
    mean = total / n_trials
    zeros = int(sum(1 for v in counts if v == 0))
    ceiling = int(n_neurons) * int(n_steps)
    out: dict[str, Any] = {
        "n_trials": n_trials,
        "total_spikes": total,
        "mean_spikes_per_trial": float(mean),
        "median_spikes_per_trial": float(np.median(np.asarray(counts, dtype=np.float64))),
        "min_spikes_per_trial": int(min(counts)),
        "max_spikes_per_trial": int(max(counts)),
        "zero_activity_trials": zeros,
        "fraction_zero_activity_trials": float(zeros / n_trials),
        "active": bool(total > 0),
        "spikes_per_trial": counts,
    }
    out["spike_saturation_fraction"] = (
        float(max(counts) / ceiling) if ceiling > 0 else float("nan")
    )
    return out


def feature_report(records: Sequence[Any]) -> dict[str, Any]:
    """Feature dimensionality and how many of those features are ever non-zero.

    ``subtype_lateral`` gives 32 columns for ``T4+T5`` (8 exact subtypes x 4
    lateral bins) whether or not anything responds. A column that is zero in
    every trial carries no information and inflates the apparent feature count,
    so the two numbers are reported together and never conflated.
    """
    if not records:
        return {
            "n_features": 0,
            "n_nonzero_features": 0,
            "zero_features": 0,
            "fraction_nonzero": float("nan"),
            "n_records": 0,
        }
    width = int(records[0].n_columns)
    nonzero = np.zeros(width, dtype=bool)
    for record in records:
        block = np.asarray(record.features, dtype=np.float64)
        if block.ndim == 2 and block.shape[1] == width:
            nonzero |= np.any(block != 0.0, axis=0)
    count = int(nonzero.sum())
    return {
        "n_features": width,
        "n_nonzero_features": count,
        "zero_features": int(width - count),
        "fraction_nonzero": float(count / width) if width else float("nan"),
        "n_records": int(len(records)),
        "readout": str(getattr(records[0], "readout", "")),
        "lateral_bins": int(getattr(records[0], "lateral_bins", 0)),
    }


def fingerprint(records: Sequence[Any]) -> str:
    """A short content hash of a set of feature records, for reproducibility checks.

    A replication is only a replication if the same declared configuration
    produces the same numbers twice, so the run output carries a digest of the
    actual feature arrays rather than only of the accuracies derived from them.
    The arrays are rounded to 12 decimals first, because the cached stimulus
    tensors are float32 and a last-bit difference in an intermediate would
    otherwise make two identical runs look different.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: (str(r.stimulus), int(r.trial))):
        block = np.asarray(record.features, dtype=np.float64)
        digest.update(str(record.stimulus).encode("utf-8"))
        digest.update(str(int(record.trial)).encode("utf-8"))
        digest.update(str(block.shape).encode("utf-8"))
        digest.update(np.round(block, 12).tobytes())
    return digest.hexdigest()


def classify_informative(
    live: Mapping[str, Any],
    features: Mapping[str, Any] | None = None,
    accuracy: float | None = None,
    null_p95: float | None = None,
) -> dict[str, Any]:
    """Is this condition's decoding scientifically informative, or not?

    Three failure modes, each stated rather than left for the reader to notice:

    ``uninformative_silent``
        The readout population barely fired, so a decoder accuracy on it is
        measuring noise on an almost-empty feature set.
    ``uninformative_saturated``
        The readout population is at its spike ceiling, or the accuracy sits on
        the 95th percentile of its own label-permutation null. Either way the
        number is a property of the ceiling, not of the wiring.
    ``informative``
        Neither. The accuracy may then be compared, against the relay bounds
        and against the other circuit conditions.

    Thresholds are module constants and are echoed in the returned payload, so a
    verdict can always be traced back to the number that produced it.
    """
    reasons: list[str] = []
    silent = False
    saturated = False

    mean_spikes = live.get("mean_spikes_per_trial", float("nan"))
    zero_fraction = live.get("fraction_zero_activity_trials", float("nan"))
    median_spikes = live.get("median_spikes_per_trial", float("nan"))
    saturation = live.get("spike_saturation_fraction", float("nan"))
    nonzero = (features or {}).get("n_nonzero_features")

    if float(mean_spikes) == 0.0:
        silent = True
        reasons.append("mean spikes per trial is exactly zero")
    elif np.isfinite(float(zero_fraction)) and float(zero_fraction) >= 0.5:
        silent = True
        reasons.append(
            f"{100 * float(zero_fraction):.0f}% of trials had zero activity in this group"
        )
    elif np.isfinite(float(median_spikes)) and float(median_spikes) == 0.0:
        silent = True
        reasons.append("the median trial had zero activity in this group")
    if nonzero is not None and int(nonzero) == 0:
        silent = True
        reasons.append("every feature column was zero in every trial")

    if np.isfinite(float(saturation)) and float(saturation) >= SATURATION_MAX_SPIKE_FRACTION:
        saturated = True
        reasons.append(
            f"the busiest trial reached {100 * float(saturation):.0f}% of the group's maximum "
            f"possible spike count, i.e. the readout is at its ceiling"
        )
    if accuracy is not None and np.isfinite(float(accuracy)):
        if np.isfinite(float(null_p95)) and float(null_p95) >= SATURATION_NULL_P95:
            saturated = True
            reasons.append(
                f"the label-permutation null p95 is {float(null_p95):.3f}, so the observed "
                f"{float(accuracy):.3f} cannot be separated from its own null"
            )
        elif float(accuracy) >= SATURATION_ACCURACY:
            saturated = True
            reasons.append(
                f"mean accuracy {float(accuracy):.3f} is at the saturation ceiling"
            )

    if silent:
        status, verdict = "uninformative_silent", (
            "SCIENTIFICALLY UNINFORMATIVE: the readout population is silent at this gain, so the "
            "decoder accuracy below is noise on an empty feature set and must not be interpreted"
        )
    elif saturated:
        status, verdict = "uninformative_saturated", (
            "SCIENTIFICALLY UNINFORMATIVE: the readout is saturated at this gain, so the decoder "
            "accuracy below is a property of the ceiling and must not be interpreted"
        )
    else:
        status, verdict = "informative", (
            "informative: the readout population is neither silent nor saturated, so its decoder "
            "accuracy may be compared against the relay controls and the other circuit conditions"
        )
    return {
        "status": status,
        "verdict": verdict,
        "may_interpret_accuracy": bool(status == "informative"),
        "reasons": reasons,
        "thresholds": {
            "zero_activity_trial_fraction": 0.5,
            "max_spike_fraction": SATURATION_MAX_SPIKE_FRACTION,
            "accuracy_ceiling": SATURATION_ACCURACY,
            "null_p95_ceiling": SATURATION_NULL_P95,
        },
    }


# ------------------------------------------------------------- the margin


def margin_statistics(
    folds_primary: Sequence[float],
    folds_control: Sequence[float],
    n_permutations: int = PREDECLARED_PERMUTATIONS,
    seed: int = PREDECLARED_SEED,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Uncertainty on ``primary - control`` accuracy, over shared trials.

    The two conditions are evaluated on the **same** trials with the same
    common-random-number noise, the same randomised sweep start and the same
    leave-one-trial-out fold structure, so fold ``i`` of one condition and fold
    ``i`` of the other describe the same held-out trial. The difference is
    therefore a paired quantity and is analysed as one, rather than as two
    independent means with their variances added.

    Two uncertainty statements are reported, deliberately:

    * a paired t interval over the fold differences, which assumes the
      differences are roughly normal and independent across trials;
    * a paired sign-flip permutation null, which assumes only that the sign of a
      fold difference is exchangeable, at the pre-declared permutation count.

    Neither licenses calling a difference an effect on its own. The
    ``effect_supported`` field is what decides whether this module is willing to
    use the word, and it is ``True`` only when both statements agree.
    """
    a = np.asarray(list(folds_primary), dtype=np.float64)
    b = np.asarray(list(folds_control), dtype=np.float64)
    if a.size != b.size:
        return {
            "status": "incomparable_folds",
            "n_folds_primary": int(a.size),
            "n_folds_control": int(b.size),
            "reason": (
                "the two conditions do not have the same number of folds, so their accuracies "
                "cannot be differenced fold by fold"
            ),
        }
    if a.size < 2:
        return {
            "status": "insufficient_folds",
            "n_folds": int(a.size),
            "reason": "at least two shared folds are needed to estimate a spread",
        }

    difference = a - b
    n = int(difference.size)
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    se = sd / math.sqrt(n)

    t_statistic = float(mean / se) if se > 0 else float("nan")
    if se > 0:
        from scipy.stats import t as student_t

        p_two_sided = float(2.0 * student_t.sf(abs(t_statistic), n - 1))
        critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
        ci_low, ci_high = mean - critical * se, mean + critical * se
    else:
        p_two_sided = float("nan")
        ci_low = ci_high = float("nan")

    # Paired sign-flip null: the sign of a fold difference carries the information
    # under the null of no systematic difference, and flipping it preserves the
    # magnitudes, so this is a within-pair permutation rather than a relabelling.
    rng = np.random.default_rng([int(seed), 0x4D41])
    n_permutations = int(n_permutations)
    if n_permutations > 0:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
        null_means = (signs * difference[None, :]).mean(axis=1)
        null_p95 = float(np.percentile(np.abs(null_means), 100.0 * (1.0 - alpha)))
        null_sd = float(null_means.std())
        p_permuted = float((np.count_nonzero(np.abs(null_means) >= abs(mean)) + 1) / (n_permutations + 1))
        z = float(mean / null_sd) if null_sd > 0 else float("nan")
    else:
        null_p95 = null_sd = z = p_permuted = float("nan")

    supported = bool(
        np.isfinite(p_two_sided) and np.isfinite(p_permuted)
        and p_two_sided < alpha and p_permuted < alpha
    )
    return {
        "status": "ok",
        "n_folds": n,
        "n_trials": n,
        "mean_margin": mean,
        "sd_margin": sd,
        "se_margin": float(se),
        "per_fold_margin": [float(v) for v in difference],
        "paired_t_statistic": t_statistic,
        "paired_t_df": int(n - 1),
        "paired_t_p_two_sided": p_two_sided,
        "confidence_interval": [float(ci_low), float(ci_high)],
        "confidence_level": float(1.0 - alpha),
        "margin_permutation_null": {
            "null_of": "sign of each shared-fold difference",
            "n_permutations": n_permutations,
            "seed": int(seed),
            "null_mean": 0.0,
            "null_sd": null_sd,
            "null_p95": null_p95,
            "z_score": z,
            "p_two_sided": p_permuted,
            "p_value_is_add_one_corrected": True,
        },
        "alpha": float(alpha),
        "effect_supported": supported,
        "statement": (
            f"The recurrent-minus-shuffled margin is {mean:+.4f} with a paired t interval of "
            f"[{ci_low:+.4f}, {ci_high:+.4f}] and a paired sign-flip permutation p of "
            f"{p_permuted:.4f} over {n_permutations} permutations. The uncertainty "
            + ("supports" if supported else "does NOT support")
            + " calling this a difference rather than an observation."
        ),
        "caveat": (
            "This is a difference between two simulated accuracy values on the same trials. It is "
            "not evidence of a biological computation, and it is not evidence of direction "
            "selectivity. A negative margin is reported with the same statistics as a positive "
            "one; conditions are not ranked."
        ),
    }


# ------------------------------------------------------------ the projection


def timing_projection(
    measured_simulation_seconds: float,
    measured_decode_seconds: float,
    measured_null_seconds: float,
    measured_null_marginal_per_permutation: float,
    measured_trials: int,
    measured_permutations: int,
    measured_circuits: int = 1,
    circuits: int = len(REPLICATION_CIRCUITS),
    conditions: int = len(PREDECLARED_CONDITIONS),
    contrasts: int = len(PREDECLARED_CONTRASTS),
    groups: int = len(PREDECLARED_NULL_GROUPS) + len(RELAY_CONTROLS) + 3,
    n_permutations: int = PREDECLARED_PERMUTATIONS,
    trials: int = PREDECLARED_TRIALS,
) -> dict[str, Any]:
    """Project a small measured run onto the declared Phase 4C run.

    Reported **before** the full run, so a long job is a decision rather than a
    surprise. Three cost centres are scaled separately, because they scale
    differently and scaling only the dominant one is how a projection ends up
    wrong by an order of magnitude:

    * **simulation** — linear in trials, in stimulus conditions and in circuits.
    * **plain decoding** — linear in trials, in contrasts, in groups and in
      circuits. A leave-one-trial-out fit on ``T`` trials holds out ``T`` rows and
      trains on ``~2(T-1)``, so the work per fit is first-order in ``T``.
    * **permutation nulls** — split into a fixed part and a marginal part. The
      fixed part (building the leave-one-out design, fitting and scoring the
      observed run) is one-off per null; the marginal part is the per-permutation
      batched fit, whose design is ``(n_folds, n_train_rows, width)`` with
      ``n_folds = T`` and ``n_train_rows = c(T-1)``, so the work is exactly
      proportional to ``T * (T-1)`` per permutation.

    The split is not decoration. A single small measurement of the null cost is
    dominated by whichever part happens to be large at that size, and scaling the
    whole number as if it were entirely marginal is wrong by a large factor in
    one direction or the other. The caller therefore times the same nulls at two
    permutation counts and passes the difference as
    ``measured_null_marginal_per_permutation``. Both the corrected and the naive
    projections are reported, together with their ratio, so the correction is
    visible rather than buried.
    """
    circuit_ratio = float(circuits) / max(int(measured_circuits), 1)
    trial_ratio = float(trials) / max(int(measured_trials), 1)
    rows_ratio = float(trials - 1) / max(int(measured_trials) - 1, 1)
    null_ratio = trial_ratio * rows_ratio
    perm_ratio = float(n_permutations) / max(int(measured_permutations), 1)
    group_ratio = float(groups) / max(len(PREDECLARED_NULL_GROUPS) + len(RELAY_CONTROLS) + 3, 1)

    marginal = float(measured_null_marginal_per_permutation)
    fixed = float(measured_null_seconds) - marginal * int(measured_permutations)

    simulation = float(measured_simulation_seconds) * circuit_ratio * trial_ratio
    decode = float(measured_decode_seconds) * circuit_ratio * trial_ratio * group_ratio
    nulls = circuit_ratio * (
        max(fixed, 0.0) * trial_ratio + marginal * null_ratio * float(n_permutations)
    )
    total = simulation + decode + nulls
    naive = float(measured_null_seconds) * circuit_ratio * null_ratio * perm_ratio
    return {
        "measured": {
            "simulation_seconds": float(measured_simulation_seconds),
            "decode_seconds": float(measured_decode_seconds),
            "null_seconds": float(measured_null_seconds),
            "null_fixed_seconds": float(fixed),
            "null_marginal_per_permutation": marginal,
            "total_seconds": float(
                measured_simulation_seconds + measured_decode_seconds + measured_null_seconds
            ),
            "trials": int(measured_trials),
            "permutations": int(measured_permutations),
            "circuits": int(measured_circuits),
        },
        "declared": {
            "trials": int(trials),
            "permutations": int(n_permutations),
            "circuits": int(circuits),
            "stimulus_conditions": int(conditions),
            "contrasts": int(contrasts),
            "simulations": int(circuits * conditions * trials),
            "label_permutation_nulls": int(circuits * len(PREDECLARED_NULL_GROUPS)),
        },
        "scale_factors": {
            "circuits": circuit_ratio,
            "trials": trial_ratio,
            "nulls_folds_times_rows": null_ratio,
            "permutations": perm_ratio,
        },
        "projected_simulation_seconds": simulation,
        "projected_decode_seconds": decode,
        "projected_null_seconds": nulls,
        "projected_total_seconds": total,
        "projected_total_minutes": total / 60.0,
        "projected_total_hours": total / 3600.0,
        "naive_linear_projection_seconds": naive,
        "fixed_correction_factor": float(total / naive) if naive > 0 else float("nan"),
        "model": (
            "simulation and plain decoding scale linearly in trials. Each permutation null is "
            "split into a fixed part (design build plus the observed fit, scaled linearly in "
            "trials) and a marginal per-permutation part (scaled as folds x train_rows = "
            "trials x (trials-1)), measured by timing the same nulls at two permutation counts. "
            "Scaling the whole small measurement as if it were marginal is wrong by the reported "
            "fixed_correction_factor in one direction or the other."
        ),
        "is_reasonably_bounded": bool(total <= 4 * 3600.0),
        "treat_as_a_lower_estimate": True,
        "lower_estimate_note": (
            "Measured on the real data: this projection said 15.7 min and the declared run took "
            "35.2 min, about 2.2x longer. The projection is a lower estimate, not a promise. The "
            "isolated single-circuit null timing runs with a warm allocator, while the real run "
            "interleaves three circuits' decodes and nulls and their memory traffic competes. The "
            "bounding decision (under four hours) was still correct, but the run's own reported "
            "wall clock is the number to trust, which is why the run prints staged flushed "
            "progress and cannot look frozen."
        ),
        "boundedness_rule": (
            "a full run projected above 4 hours is not launched as a single job; the projection is "
            "reported and the run is split or the bound is revised with the declaration amended, "
            "never by quietly changing a declared parameter"
        ),
    }


# ------------------------------------------------------------------ claims


def claim_block() -> dict[str, Any]:
    """MEASURED / SIMULATED / ASSUMED / UNRESOLVED, with the limits kept explicit.

    ``UNRESOLVED`` is a category Phase 4B did not need, because Phase 4B was the
    first run of the question. A replication has to say what it failed to settle,
    and that is the whole point of running it: a negative or uninformative
    result is information, and it belongs in the report under its own heading
    rather than being left as an absent number.
    """
    return {
        "MEASURED": [
            "MCNS body ids, exact cell-type labels, superclass, somaSide, soma coordinates",
            "directed connectivity between the selected candidate bodies",
            "integer biological synapse counts, stored unmodified and never rescaled",
            "the neuron identities and the measured edge subset the three circuit conditions share",
        ],
        "SIMULATED": [
            "every membrane potential, spike and firing rate",
            "every population representation, separation ratio and decoder accuracy",
            "the propagation of drive from L1/L2 through Mi1/Tm1/Tm2 to T4/T5",
            "the degree-preserving randomisation used as SHUFFLED_CONTROL (a computational control, "
            "not biology)",
        ],
        "ASSUMED": [
            "synapse_scale = 0.25 and the coupling law mapping a synapse count to a driving "
            "current. A synapse count is a count of chemical synapses: it is NOT synaptic "
            "efficacy, not a conductance, and not a strength",
            "generic LIF parameters, not measured for any of these neurons",
            "the synthetic stimulus, including the position-map fallback, the energy match and "
            "the randomised sweep start",
            "the linear decoder, its optimiser settings and the leave-one-trial-out folds",
            "the readout resolution: 8 exact subtypes x 4 lateral bins",
            "the node budget on the bounded population",
            "the SHUFFLED_CONTROL construction",
            "that the direction label is externally specified. THE ENCODER CONTAINS THE "
            "DIRECTIONAL STRUCTURE: the spot is driven left or right by construction, so any "
            "left/right asymmetry at T4/T5 is that supplied structure arriving, possibly "
            "transformed",
        ],
        "UNRESOLVED": [
            "whether the Phase 4B recurrent-minus-shuffled difference is a reproducible property "
            "of this model or a small-sample fluctuation; this run is what decides it, and the "
            "verdict is reported whatever it is",
            "whether the circuit computes anything at all. The connectivity graph has NO SYNAPTIC "
            "SIGN, so every coupling is non-negative, the network can only sum and never "
            "subtract, and a Reichardt-type ON-minus-OFF detector is not constructible from it",
            "whether any T4/T5 asymmetry reflects MCNS computation rather than the encoder. T4/T5 "
            "ASYMMETRY ALONE IS NOT EVIDENCE OF MCNS COMPUTATION; only the circuit-versus-shuffled "
            "comparison, and only with its uncertainty, speaks to that",
            "whether the bound population behaves like the full 22,451-body circuit",
            "the true synaptic strengths, which would require electrophysiology this project does "
            "not have",
        ],
        "CANNOT_BE_ESTABLISHED": (
            "Direction selectivity. THIS EXPERIMENT CANNOT ESTABLISH DIRECTION SELECTIVITY. The "
            "MCNS connectivity table carries body_pre, body_post and weight and no synaptic sign, "
            "so every coupling here is non-negative and the network can only sum. A Reichardt-type "
            "motion detector is built from ON-minus-OFF subtraction between spatially offset "
            "channels and cannot be built from unsigned weights. The stimulus is synthetic drive "
            "whose direction is specified by the encoder. No number in this report may be "
            "presented as biological direction selectivity, as reproduction of T4/T5 directional "
            "tuning, as Reichardt computation, or as a discovery about the fly."
        ),
    }
