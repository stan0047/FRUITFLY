"""Phase 5A: pre-declaration, acceptance gates, power, and the claim block.

Scope, and what it deliberately is not
--------------------------------------
Phase 4C established a **null** for one contrast, in one simulation, at one gain:
``MCNS_RECURRENT`` minus ``SHUFFLED_CONTROL`` was exactly 0 at 20 trials, and the
run is underpowered below roughly ``|margin| = 0.104``. That result says the Phase 4B
preliminary signal is retracted, and it says nothing about whether the MCNS circuit
computes anything.

Phase 5A does not extend that benchmark. It builds the infrastructure for a
**different and smaller question**, chosen because it is the one the unsigned
MCNS data can actually answer:

> **Does the measured wiring deliver spatially offset, subtype-differentiated
> drive to T4/T5?**

That is a *necessary precondition* for the cardinal direction selectivity of
T4a-d. It is identifiable without synaptic sign, because it asks **where** drive
arrives rather than ON-minus-OFF, and lateral position is a measured coordinate
where subtraction is not. A positive result bounds a substrate claim; a negative
result is equally informative, because it would place the spatial precondition
outside the simulated subset.

Direction selectivity is **out of scope** and is not testable here
-------------------------------------------------------------------
The MCNS connectivity table carries ``body_pre``, ``body_post`` and ``weight``, and
``weight`` is a positive integer synapse count with **no sign column**. Every
coupling this project derives is non-negative, the network sums and never
subtracts, and a Reichardt-type ON-minus-OFF detector is not constructible from
unsigned weights. No gain, trial count, population, readout or control in this
module changes that.

Two further limits are load-bearing and are repeated in every payload:

* **The encoder supplies the direction cue.** A moving spot is driven left or right
  by construction. Any left/right asymmetry at T4/T5 is that supplied structure
  arriving. A lateral *offset between subtypes* is a different claim from a
  direction preference, and only the first is tested here.
* **A lateral offset is not direction selectivity.** T4a-d being spatially offset
  is necessary for cardinal direction selectivity and is nowhere near sufficient.
  T4/T5 asymmetry alone is not evidence of MCNS computation.

The five gates
--------------
Gates are evaluated **before** any interpretation, and a failed gate stops the
affected claim rather than qualifying it in a footnote. Two of the five did not
exist in Phase 4C and are the ones whose absence mattered most:

``G1`` input locality
    The injected drive must be spatially non-uniform across the lateral bins.
    Without this there is no spatial structure to detect and the pilot is void
    *by construction*. Phase 4C never checked it, which is part of why a
    near-chance readout could not be told apart from a floor effect.
``G5`` statistic calibration
    The count-preserving lateral scramble must fail to reproduce the observed
    separation. If it succeeds, the statistic is measuring activity, not structure.

The power calculation
---------------------
The declared trial count is not a round number chosen for taste. It is computed
from a **predeclared variance-estimation stage** whose output is a variance and
nothing else: no effect is inspected, no condition is compared, and the result is
frozen into the configuration before the main run. The calculation is
``n = ((t_{1-alpha/2} + t_{power}) * sd / delta)** 2`` for a declared minimum
detectable effect ``delta``, using the same pairing structure as the Phase 4C
margin, so the machinery and the vocabulary are shared.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "ALPHA",
    "DIRECTION_SELECTIVITY_OUT_OF_SCOPE",
    "GATE_ORDER",
    "GATE_SPECS",
    "PILOT_POWER",
    "PREDECLARED",
    "PREDECLARED_PHASE",
    "PREDECLARED_WHY",
    "arms",
    "claim_block",
    "evaluate_gates",
    "power_requirement",
    "predeclared_block",
]

PREDECLARED_PHASE = "5A"

#: Two-sided significance level for every inferential statement.
ALPHA = 0.05

#: The four arms. The existing Phase 4C shuffle is **retained**, because it is a
#: different null and dropping it would replace a control rather than add one.
ARMS: tuple[str, ...] = (
    "MEASURED",
    "DRIVE_MATCHED_CONTROL",
    "EXISTING_SHUFFLE",
)

#: Which arms enter the primary comparison, and which are secondary. Declared, so
#: the primary contrast cannot be chosen after seeing the numbers.
PRIMARY_ARM = "MEASURED"
PRIMARY_CONTROL = "DRIVE_MATCHED_CONTROL"
SECONDARY_CONTROL = "EXISTING_SHUFFLE"

#: The scientific question, stated once.
PRIMARY_QUESTION = (
    "Does the measured, unsigned MCNS wiring deliver spatially offset, subtype-differentiated "
    "drive to T4/T5, relative to a strength-preserving rewiring of the same graph?"
)

#: Minimum edge-change fraction requested from the drive-matched control. A
#: control that barely moves is not a control, so this is high and the achieved
#: value is reported and gated rather than assumed.
TARGET_EDGE_CHANGE_FRACTION = 0.5

#: Gate G5 needs the count-preserving calibration to beat the observed
#: separation. How many scrambles per trial. Fixed, recorded, never reduced to
#: shorten a run.
CALIBRATION_SCRAMBLES = 200

#: Number of trials in the predeclared variance-estimation stage. It reports a
#: variance and nothing else.
VARIANCE_ESTIMATION_TRIALS = 5

#: Significance and power for the power calculation.
POWER_TARGET = 0.80

#: The whole declared configuration, in one auditable place.
PREDECLARED: dict[str, Any] = {
    "phase": PREDECLARED_PHASE,
    "kind": "infrastructure for a pilot; a different and smaller question from Phase 4C",
    "primary_question": PRIMARY_QUESTION,
    "question_is_not": (
        "direction selectivity, which is out of scope and not testable on unsigned connectivity"
    ),
    "arms": list(ARMS),
    "primary_arm": PRIMARY_ARM,
    "primary_control": PRIMARY_CONTROL,
    "secondary_control": SECONDARY_CONTROL,
    "existing_shuffle_retained": True,
    "synapse_scale": 0.25,
    "gain_swept": False,
    "stimulus_conditions": ["LEFTWARD", "RIGHTWARD", "STATIC"],
    "primary_contrast": "LEFTWARD_vs_RIGHTWARD",
    "moving_vs_static_control": "LEFTWARD_vs_STATIC",
    "readout": "lateral activity profile and centroid, per cell group",
    "readout_decoder_accuracy_is_primary": False,
    "lateral_bins": 4,
    "calibration_scrambles": CALIBRATION_SCRAMBLES,
    "target_edge_change_fraction": TARGET_EDGE_CHANGE_FRACTION,
    "variance_estimation_trials": VARIANCE_ESTIMATION_TRIALS,
    "alpha": ALPHA,
    "power_target": POWER_TARGET,
    "gates": ["G1", "G2", "G3", "G4", "G5"],
    "statistic": (
        "paired per-trial difference in lateral centroid, on the primary contrast, with a paired "
        "t statistic and a paired sign-flip permutation null; both reuse the Phase 4C machinery "
        "unchanged"
    ),
    "statistic_is_not": (
        "leave-one-trial-out decoder accuracy, which is reported as secondary only; at 8 held-out "
        "rows per fold it moves in steps of 1/8 and cannot resolve a small effect"
    ),
}

#: Why each changed or new choice was made before the run.
PREDECLARED_WHY: dict[str, str] = {
    "synapse_scale": (
        "0.25 is the already-declared and already-audited Phase 4C gain, at which T4 was measured "
        "live (mean 4.33 spikes per trial, 2.5 percent zero-activity, saturation fraction 1.5e-4). "
        "Re-declaring a different gain would re-open the tuning question this project closed. It is "
        "fixed and not swept, and it is NOT changed if the pilot comes out uninformative."
    ),
    "arms": (
        "The Phase 4C comparison failed because its control was not matched on drive: per-cell-type "
        "incoming drive moved by 0.16x to 2.75x with a per-neuron correlation of -0.0055, and the "
        "drive that moved went into the readout population. A strength-preserving rewiring fixes "
        "exactly that. The existing shuffle is retained rather than replaced, because it is a "
        "different null -- it preserves the weight multiset and destroys the strength profile -- "
        "and reporting only one would be a choice of convenience."
    ),
    "readout": (
        "The 32-column subtype x lateral decoder was not resolving structure: a ONE-feature "
        "MAGNITUDE_ONLY control scored 0.5750 against its 0.5687. The replacement works on "
        "per-neuron counts, which have no 1/8 quantisation, and reduces them to a lateral profile "
        "and a continuous centroid. Type identity is held out of the primary statistic so a "
        "lateral effect cannot be expressed as a subtype effect."
    ),
    "calibration": (
        "A centroid can be moved by more spikes or by the same spikes landing elsewhere. Those are "
        "different claims, so the statistic is calibrated before it is interpreted: a "
        "count-preserving within-trial scramble must fail to reproduce the observed separation. "
        "This is a property of the statistic, so it is a gate (G5) and not a p-value."
    ),
    "gates": (
        "G1 and G5 did not exist in Phase 4C and are the two whose absence mattered most. Without "
        "G1 there is no way to distinguish a real null from a floor effect; without G5 there is no "
        "way to distinguish structure from activity."
    ),
    "power": (
        "The trial count is computed from a predeclared variance-estimation stage that reports a "
        "variance and nothing else, so the required n is fixed before the main run rather than "
        "chosen after seeing an effect. Phase 4C's 20 trials were powered only for |margin| >= "
        "0.147, which is not enough for a small spatial effect."
    ),
    "not_changed": (
        "The predeclared gain, the population selection rule, the stimuli and their energy and "
        "polarity handling, the common-random-number noise, the randomised sweep start, the "
        "node budget, and the leave-one-trial-out machinery are all held at their Phase 4C values. "
        "The statistical question is different by design and is stated above; nothing else is "
        "retuned."
    ),
}

#: What was deliberately NOT done.
NOT_DONE: tuple[str, ...] = (
    "no gain sweep and no gain changed after seeing a result",
    "no change to the population selection rule, the node budget, or the cell types",
    "no change to the stimuli, their energy, their polarity handling or their noise realisation",
    "no change to the trial protocol, the seeding, or the randomised sweep start",
    "no silent replacement of the existing shuffle; it is retained as a secondary control",
    "no new biological assumption and no change to NeuralSimulation biology",
    "no direction-selectivity claim, and no attempt to make one identifiable",
    "no reduction of the calibration scramble count to shorten a run",
    "no real-data pilot run before the trial count is fixed by the variance-estimation stage",
)


#: Direction selectivity, kept out of scope in one quotable place.
DIRECTION_SELECTIVITY_OUT_OF_SCOPE = (
    "DIRECTION SELECTIVITY IS OUT OF SCOPE FOR PHASE 5A AND CANNOT BE DEMONSTRATED BY IT. The MCNS "
    "connectivity table carries body_pre, body_post and weight, and weight is a positive integer "
    "synapse count with no sign column, so every coupling is non-negative, the network sums and "
    "never subtracts, and a Reichardt-type ON-minus-OFF detector is not constructible from it. "
    "The direction cue is additionally supplied by the encoder: a moving spot is driven left or "
    "right by construction, so any left/right asymmetry at T4/T5 is that supplied structure "
    "arriving. A spatially offset lateral profile between T4a-d is a NECESSARY PRECONDITION for "
    "cardinal direction selectivity and is nowhere near sufficient; no result in this module may "
    "be presented as direction selectivity, as reproduction of T4/T5 directional tuning, or as "
    "evidence that the MCNS circuit performs a motion computation."
)

#: The five gates, in evaluation order, with what each one decides.
GATE_SPECS: dict[str, dict[str, str]] = {
    "G1": {
        "name": "input locality",
        "requirement": (
            "the injected drive is spatially non-uniform across the lateral bins: the variance of "
            "the noise-free drive profile across bins exceeds a declared minimum, and the profile "
            "is not flat"
        ),
        "on_failure": (
            "STOP. No spatial structure exists in the input to be transformed, so the pilot is void "
            "by construction and no downstream number may be interpreted. This is a floor effect, "
            "not a result."
        ),
    },
    "G2": {
        "name": "input identity across arms",
        "requirement": (
            "the injected drive is identical across arms within a trial (maximum spread exactly "
            "0), and the two sweep directions carry equal total drive"
        ),
        "on_failure": (
            "STOP. The arms were not given the same input, so no comparison between them is a "
            "contrast about wiring."
        ),
    },
    "G3": {
        "name": "liveness of the readout population",
        "requirement": (
            "every readout group is active in a declared fraction of trials: the zero-activity "
            "fraction is below a declared threshold, and the spike saturation fraction is below "
            "its declared threshold"
        ),
        "on_failure": (
            "Mark the affected arm SCIENTIFICALLY UNINFORMATIVE. Its statistic is reported for "
            "completeness and must not be interpreted. A centroid computed on a silent population "
            "is not a measurement of space."
        ),
    },
    "G4": {
        "name": "control validity",
        "requirement": (
            "the drive-matched control preserves per-neuron in-degree, out-degree, incoming weight "
            "and outgoing weight exactly, and actually changes at least the declared minimum "
            "fraction of edges"
        ),
        "on_failure": (
            "VOID the control arm. If the control cannot move, or does not match drive, the "
            "measured-versus-drive-matched comparison is not available. The measured-versus-"
            "existing-shuffle comparison may still be reported, labelled drive-mismatched."
        ),
    },
    "G5": {
        "name": "statistic calibration",
        "requirement": (
            "the count-preserving lateral scramble fails to reproduce the observed centroid "
            "separation and the observed profile-shape separation"
        ),
        "on_failure": (
            "STOP. A scramble that preserves activity exactly and reproduces the separation means "
            "the statistic is measuring activity, not spatial structure. No structure claim may "
            "be made from it."
        ),
    },
}

#: Evaluation order. G1 and G2 are cheap input checks and come first; G3 is liveness;
#: G4 validates the control; G5 calibrates the statistic.
GATE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")

#: Thresholds referenced by the gates, declared once and echoed in every payload.
PILOT_POWER: dict[str, float] = {
    "g1_drive_profile_min_variance_fraction": 0.01,
    "g2_max_drive_spread": 0.0,
    "g3_max_zero_activity_fraction": 0.25,
    "g3_max_saturation_fraction": 0.01,
    "g4_min_edge_change_fraction": TARGET_EDGE_CHANGE_FRACTION,
    "alpha": ALPHA,
    "power_target": POWER_TARGET,
}


def predeclared_block() -> dict[str, Any]:
    """The declaration, as written into the output before any number exists."""
    return {
        "predeclared": True,
        "predeclared_before_run": True,
        "may_be_tuned_after_seeing_results": False,
        "configuration": dict(PREDECLARED),
        "why": dict(PREDECLARED_WHY),
        "not_done": list(NOT_DONE),
        "gates": {key: dict(spec) for key, spec in GATE_SPECS.items()},
        "gate_order": list(GATE_ORDER),
        "thresholds": dict(PILOT_POWER),
        "direction_selectivity_out_of_scope": DIRECTION_SELECTIVITY_OUT_OF_SCOPE,
        "statement": (
            "Phase 5A tests whether the measured unsigned wiring delivers spatially offset, "
            "subtype-differentiated drive to T4/T5, relative to a strength-preserving rewiring. "
            "Direction selectivity is out of scope and is not testable on unsigned connectivity. "
            "A positive result bounds a substrate precondition; a negative result places the "
            "spatial precondition outside the simulated subset. Neither is a statement about the "
            "biological circuit."
        ),
    }


def arms() -> dict[str, str]:
    """What each arm is, in one line, for the report header."""
    return {
        "MEASURED": (
            "the measured MCNS recurrent graph on the declared population, with measured synapse "
            "counts used unmodified"
        ),
        "DRIVE_MATCHED_CONTROL": (
            "a strength-preserving rewiring of the same graph: per-neuron in-degree, out-degree, "
            "incoming weight and outgoing weight all preserved exactly, only edge targets "
            "re-randomised. This is the PRIMARY control."
        ),
        "EXISTING_SHUFFLE": (
            "the Phase 4C degree-preserving configuration-model randomisation, RETAINED and "
            "reported as a SECONDARY control. It preserves the weight multiset and does not "
            "preserve per-cell-type drive, so it is not drive-matched and is labelled as such."
        ),
    }


def evaluate_gates(
    drive_profile_variance_fraction: float | None,
    input_identity: dict[str, Any] | None,
    liveness: dict[str, dict[str, Any]] | None,
    control_verification: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate all five gates and state, per gate, what it permits.

    Every input is optional so a caller can pass what it has; a gate whose input is
    ``None`` is reported ``not_evaluated`` rather than silently passed. A gate that
    was not run must never read as a gate that passed.

    ``input_identity`` is a dict, not a single spread, because the declared G2
    requirement has two clauses: the drive must be identical across arms within a
    trial, *and* the two sweep directions must carry equal energy. Passing a bare
    float here checked only the first clause while reporting the gate as though
    both had been tested.

    ``liveness`` must be the arm-first liveness record written by the pilot:
    ``{arm_name: {group_name: liveness_record}}``. Each ``liveness_record`` must
    be keyed by exact readout group (T4/T5) and derived from every simulated
    condition in that arm, pooled as per-trial counts. The on-failure path also
    carries a group-level roll-up so a readout group that lives in only one of
    the two primary arms is reported as uninformative, not interpreted. A
    malformed arm/group shape is deliberately a blocking gate failure, not a
    verdict.
    """
    results: dict[str, Any] = {}

    # G1 -- input locality
    if drive_profile_variance_fraction is None:
        results["G1"] = {"status": "not_evaluated",
                          "permits": "nothing: the input's spatial structure was never checked"}
    else:
        ok = float(drive_profile_variance_fraction) >= PILOT_POWER[
            "g1_drive_profile_min_variance_fraction"
        ]
        results["G1"] = {
            "status": "pass" if ok else "FAIL",
            "name": GATE_SPECS["G1"]["name"],
            "observed_drive_profile_variance_fraction": float(drive_profile_variance_fraction),
            "threshold": PILOT_POWER["g1_drive_profile_min_variance_fraction"],
            "permits": "proceed to interpretation" if ok else GATE_SPECS["G1"]["on_failure"],
        }

    # G2 -- input identity across arms
    if not input_identity:
        results["G2"] = {"status": "not_evaluated",
                          "permits": "nothing: input identity was never checked"}
    else:
        spread = input_identity.get("max_spread_across_arms")
        energy_equal = bool(input_identity.get("direction_energy_equal", False))
        order_identical = bool(input_identity.get("order_identical", False))
        lateral_identical = bool(input_identity.get("lateral_identical", False))
        if spread is None:
            results["G2"] = {"status": "not_evaluated",
                              "permits": "nothing: input identity was never measured"}
        else:
            ok = (
                float(spread) <= PILOT_POWER["g2_max_drive_spread"]
                and energy_equal
                and order_identical
                and lateral_identical
            )
            results["G2"] = {
                "status": "pass" if ok else "FAIL",
                "name": GATE_SPECS["G2"]["name"],
                "observed_max_spread": float(spread),
                "threshold": PILOT_POWER["g2_max_drive_spread"],
                "arms_compared": int(input_identity.get("arms_compared", 0)),
                "direction_energy_equal": energy_equal,
                "order_identical": order_identical,
                "lateral_identical": lateral_identical,
                "permits": "proceed to interpretation" if ok else GATE_SPECS["G2"]["on_failure"],
            }

    # G3 -- liveness, per group
    if not liveness:
        results["G3"] = {"status": "not_evaluated",
                          "permits": "nothing: liveness was never checked"}
    else:
        malformed = False
        for arm_name, groups in liveness.items():
            if arm_name not in ARMS or not isinstance(groups, dict):
                malformed = True
                break
            for group_name, record in groups.items():
                if not isinstance(record, dict) or not {
                    "fraction_zero_activity_trials", "spike_saturation_fraction"
                }.issubset(record):
                    malformed = True
                    break
            if malformed:
                break
        if malformed:
            results["G3"] = {
                "status": "FAIL",
                "name": GATE_SPECS["G3"]["name"],
                "reason": "malformed_liveness_payload",
                "permits": (
                    "STOP. The liveness payload was not the declared "
                    "{arm: {group: record}} structure, so no scientific verdict may be read."
                ),
            }
        else:
            required_arms = (PRIMARY_ARM, PRIMARY_CONTROL)
            per_arm: dict[str, dict[str, Any]] = {}
            for arm_name, groups in liveness.items():
                per_group_inner: dict[str, Any] = {}
                for group_name, record in groups.items():
                    zero = float(record.get("fraction_zero_activity_trials", 1.0))
                    sat = float(record.get("spike_saturation_fraction", 1.0))
                    live = (
                        zero <= PILOT_POWER["g3_max_zero_activity_fraction"]
                        and sat <= PILOT_POWER["g3_max_saturation_fraction"]
                    )
                    per_group_inner[group_name] = {
                        "informative": bool(live),
                        "fraction_zero_activity_trials": zero,
                        "spike_saturation_fraction": sat,
                        "mean_spikes_per_trial": float(record.get("mean_spikes_per_trial", float("nan"))),
                    }
                per_arm[arm_name] = per_group_inner

            per_group: dict[str, Any] = {}
            for group_name in {
                g for groups in liveness.values() for g in groups.keys()
            }:
                primary_truth = {
                    arm_name: per_arm.get(arm_name, {}).get(group_name, {}).get("informative", False)
                    for arm_name in required_arms
                }
                informative = all(primary_truth.values())
                per_group[group_name] = {
                    "informative": bool(informative),
                    "primary_arms": primary_truth,
                }
            any_informative = any(g["informative"] for g in per_group.values())
            results["G3"] = {
                "status": "pass" if any_informative else "FAIL",
                "name": GATE_SPECS["G3"]["name"],
                "per_arm": per_arm,
                "per_group": per_group,
                "thresholds": {
                    "max_zero_activity_fraction": PILOT_POWER["g3_max_zero_activity_fraction"],
                    "max_saturation_fraction": PILOT_POWER["g3_max_saturation_fraction"],
                },
                "permits": (
                    "live groups may be interpreted; uninformative groups are reported and not "
                    "interpreted" if any_informative else GATE_SPECS["G3"]["on_failure"]
                ),
            }

    # G4 -- control validity
    if not control_verification:
        results["G4"] = {"status": "not_evaluated",
                          "permits": "nothing: the control was never verified"}
    else:
        v = control_verification
        matched = all(
            bool(v.get(key))
            for key in (
                "in_degree_per_neuron_equal",
                "out_degree_per_neuron_equal",
                "incoming_weight_per_neuron_equal",
                "outgoing_weight_per_neuron_equal",
                "edge_weight_multiset_equal",
                "total_synapses_equal",
                "soma_location_equal",
            )
        )
        swap_fraction = float(
            v.get("metadata_edge_change_fraction", v.get("edge_change_fraction", 0.0))
        )
        moved = swap_fraction >= PILOT_POWER["g4_min_edge_change_fraction"]
        ok = bool(matched and moved and int(v.get("self_loops", 1)) == 0)
        results["G4"] = {
            "status": "pass" if ok else "FAIL",
            "name": GATE_SPECS["G4"]["name"],
            "drive_matched_invariants_hold": bool(matched),
            "edge_change_definition": "SWAP_PARTICIPATION",
            "edge_change_fraction": swap_fraction,
            "final_displaced_edge_change_fraction": float(v.get("edge_change_fraction", 0.0)),
            "min_edge_change_fraction": PILOT_POWER["g4_min_edge_change_fraction"],
            "self_loops": int(v.get("self_loops", 0)),
            "verification": v,
            "permits": (
                "the measured-versus-drive-matched comparison may be interpreted"
                if ok else GATE_SPECS["G4"]["on_failure"]
            ),
        }

    # G5 -- statistic calibration
    if not calibration or calibration.get("status") != "ok":
        results["G5"] = {
            "status": "not_evaluated",
            "permits": "nothing: the statistic was never calibrated against a count-preserving scramble",
        }
    else:
        ok = bool(calibration.get("observed_exceeds_calibration"))
        results["G5"] = {
            "status": "pass" if ok else "FAIL",
            "name": GATE_SPECS["G5"]["name"],
            "calibration_definition": "UPPER_TAIL",
            "observed_max_abs_shift": calibration.get("observed_max_abs_shift"),
            "null_max_abs_shift": calibration.get("null_max_abs_shift"),
            "null_abs_p95": calibration.get("null_abs_p95"),
            "n_scrambles": calibration.get("n_scrambles"),
            "permits": (
                "the centroid may be read as carrying spatial structure" if ok
                else GATE_SPECS["G5"]["on_failure"]
            ),
        }

    blocking = [key for key in GATE_ORDER if results[key]["status"] != "pass"]
    hard_stop = [key for key in blocking if key in {"G1", "G2", "G5"}]
    return {
        "per_gate": results,
        "gate_order": list(GATE_ORDER),
        "all_pass": not blocking,
        "blocking_gates": blocking,
        "hard_stop_gates": hard_stop,
        "interpretation_allowed": not hard_stop,
        "statement": (
            "ALL GATES PASS. The pilot may be interpreted, with the G3 and G4 conditions above "
            "noted."
            if not blocking else
            f"GATES NOT PASSED: {blocking}. "
            + ("HARD STOP: no structural claim may be made. " if hard_stop else "")
            + "Gates G1, G2 and G5 are hard stops; G3 and G4 void the affected arm only."
        ),
        "specs": {key: dict(spec) for key, spec in GATE_SPECS.items()},
    }


def power_requirement(
    sd: float,
    delta: float,
    n_trials: int | None = None,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> dict[str, Any]:
    """Trials required to detect a centroid difference of ``delta`` at ``power``.

    The same pairing structure as the Phase 4C margin, so the vocabulary and the
    machinery are shared rather than reinvented: the statistic is a mean of
    per-trial paired differences with standard deviation ``sd``, and

    ``n = ceil(((t_{1-alpha/2}(n-1) + t_{power}(n-1)) * sd / delta) ** 2)``

    solved for the smallest ``n`` that is at least ``n_trials`` and at least 3.

    ``sd`` comes from the predeclared variance-estimation stage, which reports a
    variance and nothing else. The result is frozen into the configuration before
    the main run; it is never revised after an effect is seen.
    """
    sd = float(sd)
    delta = float(delta)
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"sd must be a positive finite number, got {sd!r}")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be a positive finite number, got {delta!r}")

    from scipy.stats import t as student_t

    n = max(int(n_trials or 0), 3)
    required = n
    for _ in range(1000):
        critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
        power_crit = float(student_t.ppf(power, n - 1))
        needed = int(math.ceil(((critical + power_crit) * sd / delta) ** 2))
        if needed <= n:
            required = n
            break
        n = max(needed, n + 1)
    critical = float(student_t.ppf(1.0 - alpha / 2.0, required - 1))
    half_width = critical * sd / math.sqrt(required)
    return {
        "sd_of_paired_differences": sd,
        "delta_minimum_detectable": delta,
        "n_trials_required": int(required),
        "n_trials_minimum": 3,
        "alpha": float(alpha),
        "power": float(power),
        "ci_half_width_at_required_n": float(half_width),
        "sd_source": (
            "predeclared variance-estimation stage, which reports a variance and no effect; frozen "
            "into the configuration before the main run"
        ),
        "protocol": (
            "paired per-trial differences, matching the Phase 4C margin structure, so the "
            "existing margin machinery is reused unchanged"
        ),
    }


def claim_block() -> dict[str, Any]:
    """MEASURED / SIMULATED / ASSUMED / UNKNOWN, for the pilot output."""
    return {
        "MEASURED": [
            "MCNS directed connectivity; weight is a positive integer synapse count",
            "body identity, exact cell-type label (curated by the dataset, not measured here), "
            "superclass, somaSide, soma coordinates",
            "the candidate chains and their synapse counts",
            "per-neuron in-degree, out-degree, incoming weight and outgoing weight of the measured "
            "graph, and the fact that the drive-matched control preserves all four",
            "the Phase 4C result, which is a measurement of the model: recurrent 0.56875, shuffled "
            "0.56875, margin 0.0000, paired t p=1.0, sign-flip p=1.0, 20 trials, 240 simulations",
        ],
        "SIMULATED": [
            "every membrane potential, spike and firing rate",
            "every lateral activity profile, centroid and profile-shape separation",
            "every statistic carrying a permutation null",
            "the count-preserving scramble null, which is a constructed object",
            "the drive-matched control graph, whose preserved set is a scientific choice",
        ],
        "ASSUMED": [
            "synapse count to coupling: count * synapse_scale, with synapse_scale fixed at 0.25. A "
            "synapse count is a count of chemical synapses; it is NOT synaptic efficacy, not a "
            "conductance, and not a strength",
            "EVERY COUPLING IS EXCITATORY, because the table has no sign column. This is the "
            "assumption that makes direction selectivity untestable here",
            "generic LIF parameters: tau_membrane 0.02, v_threshold 1.0, v_reset 0.0, "
            "refractory_periods 2",
            "the synthetic energy-matched moving spot, the position-map rank fallback for the "
            "83 percent of L1/L2 without coordinates, the noise SD, and the randomised sweep start",
            "that the strength-preserving rewiring is the right null",
            "the node budget, the bin count, and which cell types enter the readout",
            "the variance used for the power calculation, until the predeclared estimation stage "
            "supplies it",
            "that the simulated chain is the fly's chain",
        ],
        "UNKNOWN": [
            "SYNAPTIC SIGN, and therefore excitatory versus inhibitory identity of every edge. The "
            "table carries body_pre, body_post and weight and nothing else, so every coupling "
            "here is non-negative and the network can only sum",
            "true biological synaptic efficacy of any connection",
            "the direction preference of T4a-d and T5a-d, which are curated labels rather than "
            "MCNS measurements",
            "whether the fly's real T4/T5 subtractive microcircuit lies inside the simulated "
            "subset at all",
            "whether lamina connectivity, which the source paper excludes as less reliable, carries "
            "the spatial offset if the pilot returns a null",
            "what the omitted edges and the full 22,451-body circuit do",
            "whether the fly computes direction by any mechanism resembling this model",
        ],
        "OUT_OF_SCOPE": DIRECTION_SELECTIVITY_OUT_OF_SCOPE,
        "PILOT_VERDICT_MEANING": {
            "positive": (
                "the measured wiring delivers subtype-specific lateral offset at the declared gain, "
                "with a strength-matched control. This is a bounded claim about a substrate "
                "precondition for direction selectivity, and nothing more"
            ),
            "null": (
                "this wiring provides no detectable lateral offset at the declared gain, so the "
                "spatial precondition is unlocated within the simulated subset. That is a real "
                "result and it is NOT evidence that the MCNS circuit computes nothing"
            ),
            "uninformative": (
                "a gate failed, so the pilot did not test the question. That is reported as the "
                "outcome, and no parameter is changed in response"
            ),
        },
    }
