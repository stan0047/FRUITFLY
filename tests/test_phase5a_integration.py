"""Integration tests: the Phase 5A pilot's PRIMARY_CONTROL is now the frozen
liveness-gated structural null.

These tests stub ``select_liveness_gated_control`` so no real 32-candidate
search (or real-data run) happens; the stub returns either an accepted fake
selection or the contract's BLOCKED result, and the pilot's handling of each
path is verified against the summary payload and exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_phase5a_pilot.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from test_phase5a_pilot import (  # noqa: E402
    _load_script,
    _write_pathway_fixtures,
)

from flybrain.benchmark.matched import drive_matched_control  # noqa: E402
from flybrain.benchmark.phase5a import PILOT_POWER, PREDECLARED  # noqa: E402


@pytest.fixture(scope="module")
def pathway_data(tmp_path_factory):
    return _write_pathway_fixtures(tmp_path_factory.mktemp("p5a_int"))


def _arms_argv(tmp_path, pathway_data):
    return [
        "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
        "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
        "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
        "--output-dir", str(tmp_path / "out"), "--cache", str(tmp_path / "c.npz"), "--quiet",
    ]


def _accepted_stub(captured):
    def stub(measured, liveness_probe, **kwargs):
        captured["kwargs"] = kwargs
        captured["probe_results"] = {
            "MEASURED": liveness_probe("MEASURED", measured),
        }
        return {
            "status": "accepted",
            "control_circuit": drive_matched_control(measured, seed=0),
            "accepted": {"attempt_index": 0, "candidate_seed": 0,
                         "structural_verification": {}, "target_randomization": {},
                         "liveness": {}},
            "attempts": [{"attempt_index": 0, "candidate_seed": 0}],
        }

    return stub


def _blocked_stub(captured):
    def stub(measured, liveness_probe, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "status": "blocked",
            "reason": "no_candidate_qualified_within_budget",
            "control_name": "DRIVE_MATCHED_LIVENESS_GATED",
            "control_circuit": None,
            "accepted": None,
            "attempts": [
                {"attempt_index": i, "candidate_seed": i} for i in range(32)
            ],
        }

    return stub


def _read_summary(tmp_path):
    return json.loads((tmp_path / "out" / "phase5a_pilot.json").read_text(encoding="utf-8"))


def test_pilot_primary_control_is_the_liveness_gated_null(tmp_path, pathway_data, monkeypatch) -> None:
    module = _load_script()
    captured: dict = {}
    monkeypatch.setattr(module, "select_liveness_gated_control", _accepted_stub(captured))
    code = module.main(_arms_argv(tmp_path, pathway_data))
    assert code == 0
    summary = _read_summary(tmp_path)
    assert summary["primary_control_implementation"] == "DRIVE_MATCHED_LIVENESS_GATED"
    sel = summary["control_selection"]
    assert sel["control_name"] == "DRIVE_MATCHED_LIVENESS_GATED"
    assert sel["structural_constraints"] == "UNCONSTRAINED_DRIVE_MATCHED_2SWITCH"
    assert sel["max_attempts"] == 32
    assert sel["attempt_seed_policy"]
    assert sel["accepted_attempt_index"] == 0
    assert "matched_control_verification" in summary


def test_blocked_search_propagates_blocked_and_preserves_diagnostics(tmp_path, pathway_data, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "select_liveness_gated_control", _blocked_stub({}))
    code = module.main(_arms_argv(tmp_path, pathway_data))
    assert code == 3
    summary = _read_summary(tmp_path)
    assert summary["status"] == "BLOCKED"
    assert summary["reason"] == "no_candidate_qualified_within_budget"
    assert len(summary["control_selection"]["attempts"]) == 32
    # No legal primary comparison is emitted, and no silent fallback happened.
    assert "circuits" not in summary
    assert "matched_control_verification" not in summary
    assert "liveness" not in summary


def test_no_old_control_fallback_in_blocked_path(tmp_path, pathway_data, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "select_liveness_gated_control", _blocked_stub({}))
    code = module.main(_arms_argv(tmp_path, pathway_data))
    assert code == 3
    summary = _read_summary(tmp_path)
    assert summary["control_selection"]["control_name"] == "DRIVE_MATCHED_LIVENESS_GATED"
    # The old control name must not appear as a selected implementation.
    assert summary.get("primary_control_implementation") in (None, "DRIVE_MATCHED_LIVENESS_GATED")


def test_liveness_probe_exposes_only_g3_compatible_information(tmp_path, pathway_data, monkeypatch) -> None:
    module = _load_script()
    captured: dict = {}
    monkeypatch.setattr(module, "select_liveness_gated_control", _accepted_stub(captured))
    code = module.main(_arms_argv(tmp_path, pathway_data))
    assert code == 0
    probe_output = captured["probe_results"]["MEASURED"]
    assert set(probe_output) <= {"T4", "T5"}
    for group, record in probe_output.items():
        assert "fraction_zero_activity_trials" in record
        assert "spike_saturation_fraction" in record
    serialized = json.dumps(probe_output, default=str)
    for forbidden in ("centroid", "lateral", "profile", "margin", "G4", "G5", "effect", "similarity"):
        assert forbidden not in serialized


def test_locked_parameters_unchanged(tmp_path, pathway_data, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "select_liveness_gated_control", _accepted_stub({}))
    code = module.main(_arms_argv(tmp_path, pathway_data))
    assert code == 0
    summary = _read_summary(tmp_path)
    config = summary["predeclaration"]["configuration"]
    assert config["synapse_scale"] == 0.25
    assert config["stimulus_conditions"] == ["LEFTWARD", "RIGHTWARD", "STATIC"]
    assert config["primary_contrast"] == "LEFTWARD_vs_RIGHTWARD"
    assert config["lateral_bins"] == 4
    assert config["calibration_scrambles"] == 200
    assert config["alpha"] == 0.05
    assert config["power_target"] == 0.8


def test_g3_g4_g5_definitions_unchanged() -> None:
    assert PILOT_POWER["g3_max_zero_activity_fraction"] == 0.25
    assert PILOT_POWER["g3_max_saturation_fraction"] == 0.01
    from flybrain.benchmark.phase5a import evaluate_gates
    inputs = {
        "drive_profile_variance_fraction": 0.5,
        "input_identity": {"max_spread_across_arms": 0.0, "arms_compared": 1,
                           "order_identical": True, "lateral_identical": True,
                           "direction_energy_equal": True},
        "liveness": {
            "MEASURED": {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
            "DRIVE_MATCHED_CONTROL": {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
        },
        "control_verification": {
            "in_degree_per_neuron_equal": True, "out_degree_per_neuron_equal": True,
            "incoming_weight_per_neuron_equal": True, "outgoing_weight_per_neuron_equal": True,
            "edge_weight_multiset_equal": True, "total_synapses_equal": True,
            "soma_location_equal": True, "edge_change_fraction": 0.49,
            "metadata_edge_change_fraction": 0.51, "self_loops": 0,
        },
        "calibration": {"status": "ok", "observed_exceeds_calibration": True,
                        "observed_max_abs_shift": 0.4, "null_max_abs_shift": 0.1, "n_scrambles": 5},
    }
    gates = evaluate_gates(**inputs)
    assert gates["per_gate"]["G4"]["edge_change_definition"] == "SWAP_PARTICIPATION"
    assert gates["per_gate"]["G5"]["calibration_definition"] == "UPPER_TAIL"
