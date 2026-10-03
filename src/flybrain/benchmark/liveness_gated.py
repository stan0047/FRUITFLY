"""Liveness-gated structural control for a future Phase 5A experiment.

PROSPECTIVE DESIGN -- NOT RETROACTIVE. This protocol was specified after the
frozen Phase 5A run (``outputs/mcns_phase5a_run_locked/``) and makes no claim
that these choices were predeclared before it. It does not relax, re-run, or
reinterpret that artifact.

Contract
--------
``CONTROL_NAME = DRIVE_MATCHED_LIVENESS_GATED``

``STRUCTURAL_CONSTRAINTS = UNCONSTRAINED_DRIVE_MATCHED_2SWITCH``: exactly the
invariants :data:`matched.DRIVE_MATCHED_PRESERVED` preserves, nothing more. No
additional cell-type, pathway, layer, motif, reachability, activity, centroid,
or spatial-profile constraints.

``TARGET_RANDOMIZATION_REQUIRED = YES``: the candidate must genuinely re-point
edge targets (at least one edge changed, topology differs from the measured
graph) to qualify.

``MAX_ATTEMPTS = 32``

``ATTEMPT_SEED_POLICY``: ``candidate_seed = base_seed + attempt_index`` with
``base_seed = 0`` and ``attempt_index`` in ``0..31``.

``CANDIDATE_SEARCH_STOP_RULE``: generate candidates sequentially in the
deterministic attempt order; stop at the FIRST candidate that satisfies all of

1. every ``verify_drive_matching`` structural invariant,
2. genuine target randomization,
3. the locked G3 liveness requirements (pooled zero-activity fraction <= 0.25
   and spike saturation fraction <= 0.01, T4/T5, in BOTH primary arms) for the
   measured arm and for the candidate.

Candidates are NEVER ranked. No scientific outcome -- centroid, lateral
profile, LEFTWARD/RIGHTWARD response, paired margin, G4, G5, observed effect,
or measured-vs-control similarity -- is inspected or consumed. Liveness is an
acceptance gate, never an optimization target.

``FAILURE_BEHAVIOR``: if no candidate qualifies within the 32-attempt frozen
budget, a BLOCKED primary comparison is returned with the complete failure
diagnostics. Thresholds are not relaxed, attempts are not extended, seeds are
not altered, constraints are not modified, and the old ``DRIVE_MATCHED_CONTROL``
is not silently substituted.

The accepted control is a **LIVENESS-CONDITIONED STRUCTURAL NULL**: conditioning
on a liveness gate means it is NOT an unbiased sample from the unconditional
drive-matched 2-switch rewiring ensemble. It is reported so, every time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from flybrain.benchmark.matched import (
    DEFAULT_TARGET_CHANGE_FRACTION,
    DRIVE_MATCHED_PRESERVED,
    MCNSCircuit,
    drive_matched_control,
    verify_drive_matching,
)
from flybrain.benchmark.phase5a import PILOT_POWER

__all__ = [
    "CONTROL_NAME",
    "STRUCTURAL_CONSTRAINTS",
    "TARGET_RANDOMIZATION_REQUIRED",
    "MAX_ATTEMPTS",
    "BASE_SEED",
    "ATTEMPT_SEED_POLICY",
    "CANDIDATE_SEARCH_STOP_RULE",
    "LIVENESS_IS_OPTIMIZATION_TARGET",
    "NULL_INTERPRETATION",
    "candidate_seed",
    "genuine_target_randomization",
    "g3_record_passes",
    "g3_satisfied",
    "select_liveness_gated_control",
]

CONTROL_NAME = "DRIVE_MATCHED_LIVENESS_GATED"

STRUCTURAL_CONSTRAINTS = "UNCONSTRAINED_DRIVE_MATCHED_2SWITCH"

TARGET_RANDOMIZATION_REQUIRED = True

MAX_ATTEMPTS = 32

BASE_SEED = 0

ATTEMPT_SEED_POLICY = "candidate_seed = base_seed + attempt_index; base_seed = 0; attempt_index = 0..31"

CANDIDATE_SEARCH_STOP_RULE = (
    "first candidate satisfying (1) verify_drive_matching invariants, (2) genuine target "
    "randomization, (3) locked G3 liveness for both primary arms; no ranking"
)

LIVENESS_IS_OPTIMIZATION_TARGET = False

NULL_INTERPRETATION = (
    "DRIVE_MATCHED_LIVENESS_GATED is a LIVENESS-CONDITIONED STRUCTURAL NULL. Conditioning on a "
    "liveness acceptance gate means it is NOT an unbiased sample from the unconditional "
    "drive-matched 2-switch rewiring ensemble."
)

_G3_ZERO_ACTIVITY_KEY = "fraction_zero_activity_trials"
_G3_SATURATION_KEY = "spike_saturation_fraction"


def candidate_seed(attempt_index: int, base_seed: int = BASE_SEED) -> int:
    """Deterministic per-attempt seed: ``base_seed + attempt_index``."""
    if not 0 <= int(attempt_index) < MAX_ATTEMPTS:
        raise ValueError(f"attempt_index must be in 0..{MAX_ATTEMPTS - 1}, got {attempt_index}")
    return int(base_seed) + int(attempt_index)


def genuine_target_randomization(verification: dict[str, Any]) -> dict[str, Any]:
    """The candidate must have re-pointed at least one edge target."""
    changed = float(verification.get("edge_change_fraction", 0.0)) > 0.0
    topology = bool(verification.get("topology_changed", False))
    distinct = int(verification.get("self_loops", 0)) == 0
    return {
        "required": TARGET_RANDOMIZATION_REQUIRED,
        "passed": bool(changed and topology and distinct),
        "edge_change_fraction": float(verification.get("edge_change_fraction", 0.0)),
        "topology_changed": topology,
        "self_loops": int(verification.get("self_loops", 0)),
    }


def g3_record_passes(record: dict[str, Any]) -> bool:
    """One group record passes the locked G3 thresholds exactly as declared."""
    zero = float(record.get(_G3_ZERO_ACTIVITY_KEY, float("inf")))
    sat = float(record.get(_G3_SATURATION_KEY, float("inf")))
    return bool(
        zero <= PILOT_POWER["g3_max_zero_activity_fraction"]
        and sat <= PILOT_POWER["g3_max_saturation_fraction"]
    )


def g3_satisfied(
    measured_liveness: dict[str, dict[str, Any]],
    control_liveness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Locked G3: every group in the union must pass in BOTH primary arms.

    Uses the exact declared thresholds from ``phase5a.PILOT_POWER``
    (pooled zero-activity <= 0.25, saturation <= 0.01). There is no code path
    to pass a looser threshold: the only thresholds this function consults are
    the locked constants.
    """
    groups = sorted(set(measured_liveness) | set(control_liveness))
    per_group: dict[str, dict[str, Any]] = {}
    all_pass = True
    for group in groups:
        m = measured_liveness.get(group)
        c = control_liveness.get(group)
        m_ok = m is not None and g3_record_passes(m)
        c_ok = c is not None and g3_record_passes(c)
        per_group[group] = {
            "measured": m_ok,
            "candidate_control": c_ok,
            "measured_zero_activity": None if m is None else m.get(_G3_ZERO_ACTIVITY_KEY),
            "candidate_zero_activity": None if c is None else c.get(_G3_ZERO_ACTIVITY_KEY),
            "measured_saturation": None if m is None else m.get(_G3_SATURATION_KEY),
            "candidate_saturation": None if c is None else c.get(_G3_SATURATION_KEY),
        }
        all_pass = all_pass and m_ok and c_ok
    if not groups:
        all_pass = False
    return {
        "passed": bool(all_pass),
        "thresholds": {
            "max_zero_activity_fraction": PILOT_POWER["g3_max_zero_activity_fraction"],
            "max_saturation_fraction": PILOT_POWER["g3_max_saturation_fraction"],
        },
        "per_group": per_group,
    }


def select_liveness_gated_control(
    measured: MCNSCircuit,
    liveness_probe: Callable[[str, MCNSCircuit], dict[str, dict[str, Any]]],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_seed: int = BASE_SEED,
    target_change_fraction: float = DEFAULT_TARGET_CHANGE_FRACTION,
    max_attempts_factor: int = 200,
) -> dict[str, Any]:
    """Run the predeclared liveness-gated candidate search.

    ``liveness_probe(arm_label, circuit)`` must return the per-group liveness
    record ``{group: {fraction_zero_activity_trials, spike_saturation_fraction,
    ...}}`` for that arm. It is an acceptance input only; its return value is
    never compared, ranked, or optimised across candidates, and no scientific
    outcome is ever passed to this function.

    Returns a result dict. ``status`` is ``"accepted"`` (with the qualifying
    circuit under ``control_circuit``) or ``"blocked"`` (with
    ``control_circuit`` None). ``attempts`` holds the full per-candidate
    diagnostics; no candidate after the first qualifier is generated or
    evaluated.
    """
    if int(max_attempts) != MAX_ATTEMPTS:
        raise ValueError(
            f"max_attempts is frozen at {MAX_ATTEMPTS}; got {max_attempts}. "
            "The attempt budget cannot be extended by the search."
        )
    if int(base_seed) != BASE_SEED:
        raise ValueError(f"base_seed is frozen at {BASE_SEED}; got {base_seed}")

    measured_liveness = liveness_probe("MEASURED", measured)
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for attempt_index in range(MAX_ATTEMPTS):
        seed = candidate_seed(attempt_index, base_seed)
        candidate = drive_matched_control(
            measured,
            seed=seed,
            target_change_fraction=target_change_fraction,
            max_attempts_factor=max_attempts_factor,
        )
        verification = verify_drive_matching(measured, candidate)
        structure_ok = bool(
            all(
                bool(verification.get(key))
                for key in (
                    "compatible",
                    "neurons_equal",
                    "body_ids_equal",
                    "cell_types_equal",
                    "soma_side_equal",
                    "soma_location_equal",
                    "edges_equal",
                    "in_degree_per_neuron_equal",
                    "out_degree_per_neuron_equal",
                    "incoming_weight_per_neuron_equal",
                    "outgoing_weight_per_neuron_equal",
                    "edge_weight_multiset_equal",
                    "total_synapses_equal",
                )
            )
            and int(verification.get("self_loops", 1)) == 0
        )
        randomization = genuine_target_randomization(verification)
        control_liveness = liveness_probe(CONTROL_NAME, candidate)
        measured_candidate_liveness = g3_satisfied(measured_liveness, control_liveness)
        qualifies = (
            structure_ok
            and randomization["passed"]
            and measured_candidate_liveness["passed"]
        )
        record = {
            "attempt_index": attempt_index,
            "candidate_seed": seed,
            "structure_ok": structure_ok,
            "structural_verification": verification,
            "target_randomization": randomization,
            "liveness": {
                "measured": measured_liveness,
                "candidate": control_liveness,
                "g3": measured_candidate_liveness,
            },
            "qualifies": bool(qualifies),
        }
        attempts.append(record)
        if qualifies:
            accepted = {
                "attempt_index": attempt_index,
                "candidate_seed": seed,
                "control_circuit": candidate,
                "structural_verification": verification,
                "target_randomization": randomization,
                "liveness": {
                    "measured": measured_liveness,
                    "candidate": control_liveness,
                    "g3": measured_candidate_liveness,
                },
            }
            break

    if accepted is None:
        return {
            "status": "blocked",
            "control_name": CONTROL_NAME,
            "structural_constraints": STRUCTURAL_CONSTRAINTS,
            "max_attempts": MAX_ATTEMPTS,
            "base_seed": base_seed,
            "attempt_seed_policy": ATTEMPT_SEED_POLICY,
            "target_randomization_required": TARGET_RANDOMIZATION_REQUIRED,
            "liveness_is_optimization_target": LIVENESS_IS_OPTIMIZATION_TARGET,
            "null_interpretation": NULL_INTERPRETATION,
            "reason": "no_candidate_qualified_within_budget",
            "control_circuit": None,
            "accepted": None,
            "attempts": attempts,
        }

    return {
        "status": "accepted",
        "control_name": CONTROL_NAME,
        "structural_constraints": STRUCTURAL_CONSTRAINTS,
        "max_attempts": MAX_ATTEMPTS,
        "base_seed": base_seed,
        "attempt_seed_policy": ATTEMPT_SEED_POLICY,
        "target_randomization_required": TARGET_RANDOMIZATION_REQUIRED,
        "liveness_is_optimization_target": LIVENESS_IS_OPTIMIZATION_TARGET,
        "null_interpretation": NULL_INTERPRETATION,
        "reason": None,
        "control_circuit": accepted["control_circuit"],
        "accepted": {
            key: accepted[key]
            for key in (
                "attempt_index",
                "candidate_seed",
                "structural_verification",
                "target_randomization",
                "liveness",
            )
        },
        "attempts": attempts,
    }
