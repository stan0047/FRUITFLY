"""Contract tests for the liveness-gated structural control (DRIVE_MATCHED_LIVENESS_GATED).

All tests are synthetic/unit-level. No real-data Phase 5A run, no variance
estimation, no expensive benchmark.
"""

from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp

from flybrain.benchmark.liveness_gated import (
    ATTEMPT_SEED_POLICY,
    BASE_SEED,
    CONTROL_NAME,
    LIVENESS_IS_OPTIMIZATION_TARGET,
    MAX_ATTEMPTS,
    NULL_INTERPRETATION,
    STRUCTURAL_CONSTRAINTS,
    TARGET_RANDOMIZATION_REQUIRED,
    candidate_seed,
    g3_record_passes,
    g3_satisfied,
    genuine_target_randomization,
    select_liveness_gated_control,
)
from flybrain.benchmark.matched import (
    DRIVE_MATCHED_PRESERVED,
    drive_matched_control,
    verify_drive_matching,
)
from flybrain.brain.mcns_circuit import MCNSCircuit


def _healthy() -> dict:
    return {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}


def _ring_circuit(n: int = 8, weight: int = 1) -> MCNSCircuit:
    rows = np.arange(n, dtype=np.int64)
    cols = (np.arange(n, dtype=np.int64) + 1) % n
    data = np.full(n, weight, dtype=np.int64)
    return MCNSCircuit(
        body_ids=np.arange(n, dtype=np.int64),
        cell_type=np.array(["T4a"] * n, dtype=object),
        csr=sp.csr_matrix((data, (rows, cols)), shape=(n, n)),
    )


def _degenerate_circuit(n: int = 6) -> MCNSCircuit:
    rows = np.arange(n - 1, dtype=np.int64)
    cols = np.arange(1, n, dtype=np.int64)
    data = np.arange(1, n, dtype=np.int64)
    return MCNSCircuit(
        body_ids=np.arange(n, dtype=np.int64),
        cell_type=np.array(["T4a"] * n, dtype=object),
        csr=sp.csr_matrix((data, (rows, cols)), shape=(n, n)),
    )


def _probe_accepting_after(attempt_index: int, calls: list | None = None):
    """Probe that qualifies MEASURED and the candidate only after a given attempt."""

    def probe(arm_label: str, circuit: MCNSCircuit) -> dict:
        if calls is not None and arm_label == CONTROL_NAME:
            calls.append(int(circuit.metadata.get("rewire_seed")))
        if arm_label == "MEASURED":
            return {"T4": _healthy(), "T5": _healthy()}
        seed = int(circuit.metadata.get("rewire_seed"))
        if seed - BASE_SEED >= attempt_index:
            return {"T4": _healthy(), "T5": _healthy()}
        return {"T4": {"fraction_zero_activity_trials": 1.0, "spike_saturation_fraction": 0.5}}

    return probe


def test_candidate_seed_sequence_is_deterministic() -> None:
    seeds = [candidate_seed(i) for i in range(MAX_ATTEMPTS)]
    assert seeds == list(range(0, MAX_ATTEMPTS))
    assert seeds == [candidate_seed(i) for i in range(MAX_ATTEMPTS)]
    try:
        candidate_seed(MAX_ATTEMPTS)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("attempt_index out of range must raise")


def test_max_attempts_is_frozen_and_search_cannot_extend_it() -> None:
    circuit = _ring_circuit()
    result = select_liveness_gated_control(
        circuit, lambda arm, c: {"T4": _healthy(), "T5": _healthy()}
    )
    assert result["max_attempts"] == 32
    for bad in (0, 16, 64):
        try:
            select_liveness_gated_control(
                circuit, lambda arm, c: {"T4": _healthy()}, max_attempts=bad
            )
        except ValueError:
            continue
        except Exception:
            continue
        raise AssertionError("max_attempts other than 32 must raise")
    try:
        select_liveness_gated_control(circuit, lambda arm, c: {}, base_seed=7)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("base_seed other than 0 must raise")


def test_first_qualifying_candidate_is_selected() -> None:
    circuit = _ring_circuit()
    result = select_liveness_gated_control(circuit, _probe_accepting_after(5))
    assert result["status"] == "accepted"
    assert result["accepted"]["attempt_index"] == 5
    assert result["accepted"]["candidate_seed"] == 5


def test_no_candidate_after_the_first_qualifier_is_evaluated() -> None:
    circuit = _ring_circuit()
    calls: list[int] = []
    result = select_liveness_gated_control(circuit, _probe_accepting_after(5, calls=calls))
    assert result["status"] == "accepted"
    assert calls == [0, 1, 2, 3, 4, 5]
    assert len(result["attempts"]) == 6


def test_structural_invariants_are_enforced_on_every_attempt() -> None:
    circuit = _ring_circuit()
    result = select_liveness_gated_control(circuit, _probe_accepting_after(0))
    for record in result["attempts"]:
        verification = record["structural_verification"]
        for key in (
            "neurons_equal",
            "body_ids_equal",
            "cell_types_equal",
            "soma_side_equal",
            "soma_location_equal",
            "in_degree_per_neuron_equal",
            "out_degree_per_neuron_equal",
            "incoming_weight_per_neuron_equal",
            "outgoing_weight_per_neuron_equal",
            "edge_weight_multiset_equal",
            "total_synapses_equal",
        ):
            assert verification[key] is True, key
    accepted = result["accepted"]
    check = verify_drive_matching(circuit, result["control_circuit"])
    assert check["incoming_weight_per_neuron_equal"] is True
    assert accepted["structural_verification"]["self_loops"] == 0


def test_genuine_target_randomization_is_required() -> None:
    circuit = _degenerate_circuit()
    result = select_liveness_gated_control(
        circuit, lambda arm, c: {"T4": _healthy(), "T5": _healthy()}
    )
    assert result["status"] == "blocked"
    assert all(
        record["target_randomization"]["passed"] is False for record in result["attempts"]
    )


def test_g3_requires_both_primary_arms() -> None:
    circuit = _ring_circuit()

    def measured_dead(arm_label: str, circuit_arg) -> dict:
        if arm_label == "MEASURED":
            return {"T4": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0}}
        return {"T4": _healthy(), "T5": _healthy()}

    result = select_liveness_gated_control(circuit, measured_dead)
    assert result["status"] == "blocked"
    assert all(not r["liveness"]["g3"]["passed"] for r in result["attempts"])

    def candidate_dead(arm_label: str, circuit_arg) -> dict:
        if arm_label == "MEASURED":
            return {"T4": _healthy(), "T5": _healthy()}
        return {"T5": {"fraction_zero_activity_trials": 1.0, "spike_saturation_fraction": 0.0}}

    result = select_liveness_gated_control(circuit, candidate_dead)
    assert result["status"] == "blocked"
    assert all(
        r["liveness"]["g3"]["per_group"]["T5"]["candidate_control"] is False
        for r in result["attempts"]
    )


def test_all_fail_returns_blocked_with_full_diagnostics() -> None:
    circuit = _ring_circuit()
    result = select_liveness_gated_control(
        circuit,
        lambda arm, c: {"T4": {"fraction_zero_activity_trials": 0.99, "spike_saturation_fraction": 0.5}},
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "no_candidate_qualified_within_budget"
    assert result["control_circuit"] is None
    assert result["accepted"] is None
    assert len(result["attempts"]) == 32
    for record in result["attempts"]:
        assert "candidate_seed" in record
        assert "structural_verification" in record
        assert "liveness" in record


def test_thresholds_cannot_be_relaxed_by_the_search() -> None:
    borderline = {"fraction_zero_activity_trials": 0.25, "spike_saturation_fraction": 0.0101}
    assert g3_record_passes(borderline) is False
    at_threshold = {"fraction_zero_activity_trials": 0.25, "spike_saturation_fraction": 0.01}
    assert g3_record_passes(at_threshold) is True
    signature = inspect.signature(select_liveness_gated_control)
    forbidden = {"g3_max_zero_activity_fraction", "g3_max_saturation_fraction", "zero_activity_threshold", "saturation_threshold", "threshold_override", "relax_thresholds"}
    assert forbidden.isdisjoint(signature.parameters)
    assert "threshold" not in str(signature).lower() or "target_change_fraction" in signature.parameters


def test_candidate_selection_does_not_consume_scientific_outcomes() -> None:
    signature = inspect.signature(select_liveness_gated_control)
    forbidden = {
        "centroid",
        "lateral_profile",
        "lateral",
        "left",
        "right",
        "margin",
        "g4",
        "g5",
        "effect",
        "outcome",
        "similarity",
        "profile",
    }
    names = {name.lower() for name in signature.parameters}
    assert forbidden.isdisjoint(names)
    # Liveness is a gate, never an optimization target: there is no key that
    # ranks candidates by any quantity.
    assert LIVENESS_IS_OPTIMIZATION_TARGET is False


def test_old_drive_matched_control_behavior_remains_separately_testable() -> None:
    circuit = _ring_circuit()
    old = drive_matched_control(circuit, seed=0)
    check = verify_drive_matching(circuit, old)
    assert check["incoming_weight_per_neuron_equal"] is True
    assert old.metadata["condition"] == "DRIVE_MATCHED_CONTROL"

    # The new control, when accepted, is a distinct object with its own name,
    # and a blocked search never silently substitutes the old control.
    result = select_liveness_gated_control(
        circuit,
        lambda arm, c: {"T4": {"fraction_zero_activity_trials": 0.99, "spike_saturation_fraction": 0.5}},
    )
    assert result["status"] == "blocked"
    assert result["control_circuit"] is None
    assert result["control_name"] == CONTROL_NAME


def test_contract_metadata_is_recorded() -> None:
    circuit = _ring_circuit()
    result = select_liveness_gated_control(circuit, _probe_accepting_after(0))
    assert result["control_name"] == "DRIVE_MATCHED_LIVENESS_GATED"
    assert result["structural_constraints"] == STRUCTURAL_CONSTRAINTS
    assert result["target_randomization_required"] is True
    assert result["attempt_seed_policy"] == ATTEMPT_SEED_POLICY
    assert "NOT an unbiased sample" in NULL_INTERPRETATION
    assert result["max_attempts"] == MAX_ATTEMPTS
    assert TARGET_RANDOMIZATION_REQUIRED is True
    assert BASE_SEED == 0
    assert DRIVE_MATCHED_PRESERVED
