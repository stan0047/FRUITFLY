"""Contract tests that prevent accidental reinterpretation of Phase 5A quantities.

These tests do not encode the frozen real-data result. They pin down the code's
current scientific contract:

* G4 gates the verifier's ``edge_change_fraction`` field, not the matched-control
  metadata's ``metadata_edge_change_fraction``.
* G5's calibration flag means "observed finite max-abs centroid shift exceeds the
  finite max-abs null shift"; equality does not exceed it.
* G3 requires a group to satisfy the declared liveness record in both primary
  arms before the comparison is allowed to interpret it.
"""

from __future__ import annotations

import numpy as np

from flybrain.benchmark.lateral import calibration_shift_null
from flybrain.benchmark.phase5a import (
    PRIMARY_ARM,
    PRIMARY_CONTROL,
    SECONDARY_CONTROL,
    evaluate_gates,
)


def _healthy_gate_inputs(**overrides):
    ok = {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}
    inputs = {
        "drive_profile_variance_fraction": 0.5,
        "input_identity": {
            "max_spread_across_arms": 0.0,
            "arms_compared": 1,
            "order_identical": True,
            "lateral_identical": True,
            "direction_energy_equal": True,
            "direction_energy_spread": 0.0,
        },
        "liveness": {
            PRIMARY_ARM: {"T4": dict(ok)},
            PRIMARY_CONTROL: {"T4": dict(ok)},
            SECONDARY_CONTROL: {"T4": dict(ok)},
        },
        "control_verification": {
            "in_degree_per_neuron_equal": True,
            "out_degree_per_neuron_equal": True,
            "incoming_weight_per_neuron_equal": True,
            "outgoing_weight_per_neuron_equal": True,
            "edge_weight_multiset_equal": True,
            "total_synapses_equal": True,
            "soma_location_equal": True,
            "edge_change_fraction": 0.6,
            "self_loops": 0,
            "topology_changed": True,
        },
        "calibration": {
            "status": "ok",
            "observed_exceeds_calibration": True,
            "observed_max_abs_shift": 0.4,
            "null_max_abs_shift": 0.1,
            "n_scrambles": 5,
        },
    }
    inputs.update(overrides)
    return inputs


def test_g4_uses_swap_participation_not_final_displacement() -> None:
    inputs = _healthy_gate_inputs()
    inputs["control_verification"]["edge_change_fraction"] = 0.49
    inputs["control_verification"]["metadata_edge_change_fraction"] = 0.51
    gates = evaluate_gates(**inputs)
    assert gates["per_gate"]["G4"]["status"] == "pass"
    assert gates["per_gate"]["G4"]["edge_change_definition"] == "SWAP_PARTICIPATION"
    assert gates["per_gate"]["G4"]["final_displaced_edge_change_fraction"] == 0.49

    inputs = _healthy_gate_inputs()
    inputs["control_verification"]["edge_change_fraction"] = 0.51
    inputs["control_verification"]["metadata_edge_change_fraction"] = 0.49
    gates = evaluate_gates(**inputs)
    assert gates["per_gate"]["G4"]["status"] == "FAIL"
    assert gates["per_gate"]["G4"]["edge_change_definition"] == "SWAP_PARTICIPATION"
    assert gates["per_gate"]["G4"]["edge_change_fraction"] == 0.49


def test_g5_treats_observed_equal_to_null_maxabs_as_not_calibrating() -> None:
    calibration = {
        "status": "ok",
        "observed_exceeds_calibration": False,
        "observed_max_abs_shift": 1.0,
        "null_max_abs_shift": 1.0,
        "n_scrambles": 5,
    }
    gates = evaluate_gates(**_healthy_gate_inputs(calibration=calibration))
    assert gates["per_gate"]["G5"]["status"] == "FAIL"

    calibration["observed_exceeds_calibration"] = True
    calibration["observed_max_abs_shift"] = 0.8
    calibration["null_max_abs_shift"] = 0.5
    gates = evaluate_gates(**_healthy_gate_inputs(calibration=calibration))
    assert gates["per_gate"]["G5"]["status"] == "pass"


def test_calibration_shift_null_uses_upper_tail_for_observed_exceedance() -> None:
    counts_by_key = {
        ("LEFTWARD", 0): np.array([2, 0, 0, 0], dtype=np.int64),
        ("RIGHTWARD", 0): np.array([0, 0, 0, 2], dtype=np.int64),
    }
    lateral_bin = np.array([0, 1, 2, 3], dtype=np.int64)
    cell_type_index = {"T4a": np.arange(4, dtype=np.int64)}
    result = calibration_shift_null(
        counts_by_key,
        lateral_bin,
        cell_type_index,
        group=("T4a",),
        left="LEFTWARD",
        right="RIGHTWARD",
        trials=1,
        n_scrambles=5,
        seed=0,
    )
    assert result["status"] == "ok"
    assert result["observed_max_abs_shift"] == 1.0
    assert "null_abs_p95" in result
    # This is the current implemented contract: the gate looks at the finite
    # observed maximum relative to the null absolute-shift upper tail, never at
    # whether the observed value merely ties the null maximum.
    assert result["observed_exceeds_calibration"] == (
        result["observed_max_abs_shift"] > result["null_abs_p95"]
    )


def test_primary_arm_requires_both_arms_for_a_group_to_be_live() -> None:
    inputs = _healthy_gate_inputs()
    inputs["liveness"] = {
        PRIMARY_ARM: {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
        PRIMARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0}},
        SECONDARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
    }
    gates = evaluate_gates(**inputs)
    assert gates["per_gate"]["G3"]["per_group"]["T4"]["informative"] is False

    inputs["liveness"][PRIMARY_CONTROL]["T4"]["fraction_zero_activity_trials"] = 0.0
    gates = evaluate_gates(**inputs)
    assert gates["per_gate"]["G3"]["per_group"]["T4"]["informative"] is True
