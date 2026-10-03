"""Tests for the Phase 5A pilot infrastructure.

Runs entirely on a tiny purpose-built Feather fixture, like
``tests/test_motion_benchmark.py``. No network access, no neuPrint token, no
dependency on the real 1 GB MaleCNS file, and **no real-data pilot run**.

What is pinned down here
------------------------
The Phase 4C audit found three things that made the previous comparison unable to
answer its own question, and this module's contracts are the fixes:

* the control was not drive-matched, so no difference could be attributed to
  wiring (:class:`TestDriveMatchedControl`);
* the readout was not resolving structure, and was quantised below the effect
  (:class:`TestLateralReadout`);
* nothing distinguished spatial structure from activity (:class:`TestCalibration`).

Plus the two gates that did not exist in Phase 4C and could not be reconstructed
after the fact (:class:`TestGates`), and the pre-declared power calculation
(:class:`TestPower`).

Nothing here asserts a biological result. Every test that touches pilot output
also asserts that direction selectivity stays out of scope.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.benchmark.lateral import (
    LATERAL_READOUT_DISCLAIMER,
    calibration_shift_null,
    lateral_profile,
    profile_centroid,
    profile_shape_distance,
    scramble_counts_preserving_total,
)
from flybrain.benchmark.matched import (
    DRIVE_MATCHED_DISCLAIMER,
    DRIVE_MATCHED_NOT_PRESERVED,
    DRIVE_MATCHED_PRESERVED,
    drive_matched_control,
    verify_drive_matching,
)
from flybrain.benchmark.phase5a import (
    ALPHA,
    ARMS,
    DIRECTION_SELECTIVITY_OUT_OF_SCOPE,
    GATE_ORDER,
    GATE_SPECS,
    PREDECLARED,
    PREDECLARED_PHASE,
    PREDECLARED_WHY,
    PRIMARY_ARM,
    PRIMARY_CONTROL,
    SECONDARY_CONTROL,
    arms,
    claim_block,
    evaluate_gates,
    power_requirement,
    predeclared_block,
)
from flybrain.benchmark.phase4c import liveness
from flybrain.benchmark.shuffled import shuffled_control, verify_degree_preservation
from flybrain.benchmark.stimulus import lateral_coordinates
from flybrain.brain.mcns_circuit import MCNSCircuit, MODE_RECURRENT
from flybrain.brain.mcns_simulation import run_simulation
from flybrain.brain.neuron import NeuronParams
from flybrain.benchmark.stimulus import MotionStimulusGenerator
import flybrain.data.motion_candidate as motion_candidate

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_phase5a_pilot.py"

T4_SUBTYPES = ("T4a", "T4b", "T4c", "T4d")
T5_SUBTYPES = ("T5a", "T5b", "T5c", "T5d")
ALL_SUBTYPES = T4_SUBTYPES + T5_SUBTYPES

#: Fixture edge weights run from this to this+4. Inflated well above real MCNS
#: counts so the chain actually propagates on a fixture this small; see
#: ``_write_pathway_fixtures``.
WEIGHT_BASE = 50


def _load_script():
    """Import the pilot script as a module, so ``main`` runs in-process."""
    spec = importlib.util.spec_from_file_location("run_phase5a_pilot_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stub_liveness_gated_selection(module) -> None:
    """Intentionally stub the frozen candidate search in payload-plumbing tests.

    These tests assert what the pilot EMITS (circuits, liveness, centroids,
    matched-control verification, predeclaration), not that a candidate is
    successfully discovered. They cannot discover one: the synthetic
    4-body-per-type pathway fixture has measured T4 pooled zero-activity
    fraction 1.0, so every candidate is correctly BLOCKED by the locked G3
    contract (pooled zero-activity <= 0.25). G3 is therefore not loosened and
    the fixture is not altered to manufacture a qualifying candidate.

    The real search is covered unstubbed in tests/test_liveness_gated.py, and
    the BLOCKED path with its 32 per-candidate diagnostics is covered in
    tests/test_phase5a_integration.py.
    """

    def stub(measured, liveness_probe, **kwargs):
        return {
            "status": "accepted",
            "control_circuit": drive_matched_control(measured, seed=0),
            "accepted": {
                "attempt_index": 0,
                "candidate_seed": 0,
                "structural_verification": {},
                "target_randomization": {},
                "liveness": {},
            },
            "attempts": [{"attempt_index": 0, "candidate_seed": 0}],
        }

    module.select_liveness_gated_control = stub


def _write_pathway_fixtures(directory: Path) -> tuple[Path, Path]:
    """A tiny annotation + connectivity pair containing the real chains.

    Synapse counts are deliberately **not** all equal, so the weight-class
    restriction in the 2-switch has something to work with and the drive-matching
    invariants are actually tested rather than trivially satisfied.
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    directory.mkdir(parents=True, exist_ok=True)
    layout = (
        ("L1", ("Mi1",)),
        ("Mi1", T4_SUBTYPES),
        ("L2", ("Tm1", "Tm2")),
        ("Tm1", T5_SUBTYPES),
        ("Tm2", T5_SUBTYPES),
    )
    bodies: dict[str, list[int]] = {}
    next_id = 900_000
    names: list[str] = []
    for name, targets in layout:
        for entry in (name, *targets):
            if entry not in names:
                names.append(entry)
    per_type = 8
    for name in names:
        bodies[name] = list(range(next_id, next_id + per_type))
        next_id += per_type

    rows = []
    for name, ids in bodies.items():
        for position, body in enumerate(ids):
            # Real lateral coordinates, in the bands MaleCNS actually uses: left
            # hemisphere x in the 77k-93k nm band, right in 3k-21k. Without these
            # the position map falls back to a rank spread that lands every neuron
            # in two tight clusters, the quantile edges collapse, a lateral bin
            # comes out empty, and the script correctly refuses to report a
            # centroid it cannot define.
            #
            # Note the per-type x values are necessarily identical: the position
            # map rescales x within each side, so any per-type offset is removed
            # again. A subset of k bodies per type therefore retains at most k
            # distinct lateral coordinates, and the script tests must keep
            # k >= --lateral-bins for every bin to be reachable.
            left_hemisphere = position % 2 == 0
            if left_hemisphere:
                side, x = "L", 77_000.0 + 1_000.0 * (position // 2)
            else:
                side, x = "R", 21_000.0 - 1_000.0 * (position // 2)
            rows.append(
                {
                    "bodyId": int(body), "type": name, "superclass": "ol_intrinsic",
                    "somaSide": side,
                    "somaLocation": [x, 20_000.0 + 1_000.0 * position, 2_000.0],
                }
            )
    annotations_path = directory / "p5a_annotations.feather"
    feather.write_feather(pa.Table.from_pylist(rows), annotations_path)

    edges = []
    for source, targets in layout:
        for target in targets:
            for index, source_body in enumerate(bodies[source]):
                for offset in (0, 1, 2):
                    target_body = bodies[target][(index + offset) % per_type]
                    if target_body == source_body:
                        continue
                    # Weights are inflated (50-54) on purpose. With realistic
                    # single-digit counts and only 8 bodies per type, nothing
                    # downstream of L1/L2 reaches threshold, every readout
                    # population is silent, and all three arms produce a
                    # bitwise identical spike log -- so a test that rewires the
                    # graph and runs the LIF loop cannot detect the rewiring at
                    # all. The five distinct values are what the equal-weight
                    # 2-switch needs, and the absolute scale is irrelevant to
                    # every invariant it is checked against.
                    edges.append(
                        {
                            "body_pre": int(source_body), "body_post": int(target_body),
                            "weight": int(WEIGHT_BASE + ((index + offset) % 5)),
                        }
                    )
    edges_path = directory / "p5a_edges.feather"
    feather.write_feather(pa.Table.from_pylist(edges), edges_path)
    return annotations_path, edges_path


@pytest.fixture(scope="module")
def pathway_data(tmp_path_factory):
    return _write_pathway_fixtures(tmp_path_factory.mktemp("p5a"))


@pytest.fixture(scope="module")
def circuit(pathway_data) -> MCNSCircuit:
    annotations = motion_candidate.load_annotations(pathway_data[0])
    populations = motion_candidate.build_populations(
        annotations, motion_candidate.MOTION_CELL_TYPES, require_superclass=True
    )
    extraction = motion_candidate.extract_candidate_edges(
        pathway_data[1], populations, direction="both"
    )
    built = MCNSCircuit.from_extraction(
        populations, extraction, annotations, mode=MODE_RECURRENT
    )
    # No soma coordinates, so the deterministic rank fallback drives the layout.
    built.soma_location = np.full((built.num_neurons, 3), -1, dtype=np.int64)
    return built


@pytest.fixture(scope="module")
def generator(circuit) -> MotionStimulusGenerator:
    return MotionStimulusGenerator(
        circuit, spot_fraction=0.3, current=2.0, steps_per_crossing=40,
        amplitude_perturbation=0.05, input_noise=1.0, seed=0,
    )


# ======================================================================
# The control: strength-preserving rewiring.
# ======================================================================


class TestDriveMatchedControl:
    def test_every_preserved_invariant_holds_per_neuron(self, circuit) -> None:
        """The exact property the Phase 4C control did not have.

        Checked per neuron, not on aggregate totals, because a control can look
        matched on the mean and be unmatched on every individual neuron.
        """
        matched = drive_matched_control(circuit, seed=0)
        check = verify_drive_matching(circuit, matched)
        assert check["compatible"] is True
        for key in (
            "in_degree_per_neuron_equal", "out_degree_per_neuron_equal",
            "incoming_weight_per_neuron_equal", "outgoing_weight_per_neuron_equal",
            "edge_weight_multiset_equal", "total_synapses_equal",
            "body_ids_equal", "cell_types_equal", "soma_side_equal", "soma_location_equal",
        ):
            assert check[key] is True, key
        assert check["incoming_weight_max_abs_diff"] == 0
        assert check["outgoing_weight_max_abs_diff"] == 0
        assert check["in_degree_mismatches"] == 0
        assert check["out_degree_mismatches"] == 0
        assert check["edges_equal"] is True
        assert check["self_loops"] == 0

    def test_the_control_actually_moves(self, circuit) -> None:
        """A control that barely re-wires is a copy, and must not read as a control."""
        matched = drive_matched_control(circuit, seed=0)
        check = verify_drive_matching(circuit, matched)
        assert check["topology_changed"] is True
        assert check["edge_change_fraction"] > 0.2, check["edge_change_fraction"]
        assert matched.metadata["distinct_from_original"] is True
        # Same edge count means the overlap arithmetic is meaningful.
        assert check["edges_equal"] is True

    def test_no_synapse_count_is_created_or_destroyed(self, circuit) -> None:
        matched = drive_matched_control(circuit, seed=0)
        before = np.sort(circuit.csr.data)
        after = np.sort(matched.csr.data)
        assert before.size == after.size
        assert np.array_equal(before, after)
        assert circuit.total_synapse_count == matched.total_synapse_count

    def test_no_multi_edges_and_no_self_loops(self, circuit) -> None:
        matched = drive_matched_control(circuit, seed=0)
        coo = matched.csr.tocoo()
        keys = coo.row.astype(np.int64) * np.int64(matched.num_neurons) + coo.col
        assert np.unique(keys).size == keys.size, "a multi-edge was created"
        assert int(matched.self_loop_count) == 0

    def test_it_is_reproducible_from_its_seed_and_seed_sensitive(self, circuit) -> None:
        a = drive_matched_control(circuit, seed=3)
        b = drive_matched_control(circuit, seed=3)
        assert np.array_equal(a.csr.indices, b.csr.indices)
        assert np.array_equal(a.csr.data, b.csr.data)
        c = drive_matched_control(circuit, seed=4)
        assert not np.array_equal(a.csr.indptr, c.csr.indptr) or not np.array_equal(
            a.csr.indices, c.csr.indices
        )

    def test_it_differs_from_the_measured_graph_in_dynamics(self, circuit, generator) -> None:
        """The gap the Phase 4C test suite left open, now closed on the fixture.

        Graph-level invariants are necessary but do not prove the simulator reads
        a different matrix, so the LIF loop is actually run on both.
        """
        matched = drive_matched_control(circuit, seed=0)
        params = NeuronParams()
        kwargs = dict(n_steps=40, params=params, synapse_scale=0.25, record_input=False)
        real = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        control = run_simulation(matched, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        assert not np.array_equal(real.spikes, control.spikes)
        real_counts = real.spikes.sum(axis=0)
        control_counts = control.spikes.sum(axis=0)
        differing = int(np.count_nonzero(real_counts != control_counts))
        assert differing > 1, "only one neuron differed, which is not a wiring difference"

    def test_it_is_unlike_the_existing_shuffle(self, circuit) -> None:
        """Two different nulls, both retained. The Phase 4C control is not replaced.

        The existing shuffle is verified to still satisfy its own invariants, and
        the drive-matched control is verified to satisfy a strictly larger set,
        so neither can quietly stand in for the other.
        """
        existing = shuffled_control(circuit, seed=0)
        old = verify_degree_preservation(circuit, existing)
        for key in ("neurons_equal", "edges_equal", "in_degree_sequence_equal",
                    "out_degree_sequence_equal", "synapse_count_distribution_equal",
                    "topology_changed"):
            assert old[key] is True, key
        matched = drive_matched_control(circuit, seed=0)
        new = verify_drive_matching(circuit, matched)
        # The new control additionally guarantees per-neuron weight equality,
        # which the old one's contract does not even claim.
        assert new["incoming_weight_per_neuron_equal"] is True
        assert "incoming synaptic weight" in " ".join(DRIVE_MATCHED_PRESERVED)
        # The declared concessions are the reason the two nulls are not the same
        # null, and they are stated rather than discovered by the reader.
        assert "cell-type composition of edges" in " ".join(DRIVE_MATCHED_NOT_PRESERVED)
        assert "soma-side composition of edges" in " ".join(DRIVE_MATCHED_NOT_PRESERVED)

    def test_a_degenerate_graph_reports_its_achieved_fraction(self) -> None:
        """When no edge can move, the achieved fraction is reported, not hidden."""
        n = 6
        # Every edge a different weight: no equal-weight pair exists.
        rows = np.arange(n - 1)
        cols = np.arange(1, n)
        data = np.arange(1, n, dtype=np.int64)
        built = MCNSCircuit(
            body_ids=np.arange(n, dtype=np.int64),
            cell_type=np.array(["T4a"] * n, dtype=object),
            csr=sp.csr_matrix((data, (rows, cols)), shape=(n, n)),
        )
        matched = drive_matched_control(built, seed=0)
        assert matched.metadata["edge_change_fraction"] == 0.0
        assert verify_drive_matching(built, matched)["edge_change_fraction"] == 0.0
        assert verify_drive_matching(built, matched)["incoming_weight_per_neuron_equal"] is True

    def test_the_disclaimer_states_it_is_not_biology(self) -> None:
        assert "not a biological model" in DRIVE_MATCHED_DISCLAIMER
        assert "computational control" in DRIVE_MATCHED_DISCLAIMER


# ======================================================================
# The readout: lateral profile and centroid.
# ======================================================================


class TestLateralReadout:
    def test_profile_shape_is_one_value_per_lateral_bin(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        counts = np.arange(circuit.num_neurons, dtype=np.int64)
        profile = lateral_profile(counts, bins, circuit.cell_type_index(), T4_SUBTYPES)
        assert profile.shape == (4,)
        assert np.isfinite(profile).all()

    def test_a_subtype_only_group_reads_its_own_neurons(self, circuit) -> None:
        """A T4a-only group must not pick up T5a spikes, or bins shift silently."""
        bins = lateral_coordinates(circuit, 4)
        index_map = circuit.cell_type_index()
        counts = np.zeros(circuit.num_neurons, dtype=np.int64)
        counts[index_map["T5a"]] = 1000
        t4a = lateral_profile(counts, bins, index_map, ("T4a",))
        assert np.allclose(t4a, 0.0), "T5a activity leaked into a T4a-only profile"
        counts[index_map["T4a"]] = 5
        assert np.allclose(
            lateral_profile(counts, bins, index_map, ("T4a",)), 5.0
        )

    def test_a_missing_group_returns_zeros_not_a_shifted_profile(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        profile = lateral_profile(
            np.zeros(circuit.num_neurons, dtype=np.int64), bins,
            circuit.cell_type_index(), ("NotAType",),
        )
        assert np.allclose(profile, 0.0)
        assert profile.shape == (int(bins.max()) + 1,)

    def test_profile_length_equals_the_geometry_not_the_group(self, circuit) -> None:
        """All 4 bins stay in the vector even if a group occupies only some."""
        bins = lateral_coordinates(circuit, 4)
        profile = lateral_profile(
            np.zeros(circuit.num_neurons, dtype=np.int64), bins,
            circuit.cell_type_index(), T4_SUBTYPES,
        )
        assert profile.size == 4

    def test_centroid_is_normalised_and_direction_sensitive(self) -> None:
        """0 in the first bin, 1 in the last, so thresholds are in lateral-extent units."""
        left = np.array([1.0, 0.0, 0.0, 0.0])
        right = np.array([0.0, 0.0, 0.0, 1.0])
        assert profile_centroid(left) == pytest.approx(0.0)
        assert profile_centroid(right) == pytest.approx(1.0)
        assert profile_centroid(left) < profile_centroid(right)

    def test_a_symmetric_profile_centroid_is_in_the_middle(self) -> None:
        assert profile_centroid(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)
        assert profile_centroid(np.array([0.0, 2.0, 2.0, 0.0])) == pytest.approx(0.5)

    def test_an_empty_profile_is_nan_not_a_position(self) -> None:
        value = profile_centroid(np.zeros(4))
        assert np.isnan(value), "a zero profile must not average into a real position"

    def test_shape_distance_is_symmetric_and_zero_on_identity(self) -> None:
        a = np.array([1.0, 2.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 3.0, 1.0])
        assert profile_shape_distance(a, a) == pytest.approx(0.0, abs=1e-12)
        assert profile_shape_distance(a, b) == pytest.approx(profile_shape_distance(b, a))
        assert np.isnan(profile_shape_distance(np.zeros(4), b))

    def test_mismatched_inputs_are_refused(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        with pytest.raises(ValueError, match="disagree"):
            lateral_profile(np.zeros(3, dtype=np.int64), bins,
                            circuit.cell_type_index(), T4_SUBTYPES)
        with pytest.raises(ValueError, match="same length"):
            profile_centroid(np.zeros(4), np.arange(3, dtype=np.float64))

    def test_the_disclaimer_excludes_direction_selectivity(self) -> None:
        assert "not how much" in LATERAL_READOUT_DISCLAIMER
        assert "not which direction" in LATERAL_READOUT_DISCLAIMER
        assert "out of scope" in LATERAL_READOUT_DISCLAIMER


# ======================================================================
# Calibration: the count-preserving scramble.
# ======================================================================


class TestCalibration:
    def test_the_scramble_preserves_total_and_multiset_exactly(self) -> None:
        counts = np.array([0, 1, 3, 3, 7, 7, 7, 12], dtype=np.int64)
        out = scramble_counts_preserving_total(counts, seed=1)
        assert out.size == counts.size
        assert int(out.sum()) == int(counts.sum())
        assert np.array_equal(np.sort(out), np.sort(counts))
        assert sorted(out.tolist()) == sorted(counts.tolist())

    def test_the_scramble_can_move_counts(self) -> None:
        counts = np.array([0, 5, 5, 5, 5, 9], dtype=np.int64)
        moved = any(
            not np.array_equal(
                scramble_counts_preserving_total(counts, seed=s), counts
            )
            for s in range(12)
        )
        assert moved, "the scramble never moved anything, so it calibrates nothing"

    def test_the_scramble_is_reproducible(self) -> None:
        counts = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        a = scramble_counts_preserving_total(counts, seed=7)
        b = scramble_counts_preserving_total(counts, seed=7)
        assert np.array_equal(a, b)

    def test_group_restriction_keeps_other_groups_in_place(self) -> None:
        counts = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        out = scramble_counts_preserving_total(counts, seed=0, group=[0, 2, 4])
        assert out[1] == counts[1] and out[3] == counts[3] and out[5] == counts[5]
        assert np.array_equal(np.sort(out[[0, 2, 4]]), np.sort(counts[[0, 2, 4]]))

    def test_calibration_reports_both_sides(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        index_map = circuit.cell_type_index()
        rng = np.random.default_rng(0)
        counts = {}
        for condition, offset in (("LEFTWARD", 0.0), ("RIGHTWARD", 2.0)):
            for trial in range(6):
                vector = rng.integers(0, 4, size=circuit.num_neurons)
                vector[index_map["T4a"]] += int(offset) + 1
                counts[(condition, trial)] = vector.astype(np.int64)
        result = calibration_shift_null(
            counts, bins, index_map, T4_SUBTYPES, "LEFTWARD", "RIGHTWARD", 6,
            n_scrambles=20, seed=0,
        )
        assert result["status"] == "ok"
        assert result["n_trials"] == 6
        assert result["n_scrambles"] == 20
        assert len(result["observed_centroid_shift"]) == 6
        assert "preserving the total" in result["null_of"]
        assert "measuring activity" in result["reading"] or "carrying spatial" in result["reading"]

    def test_calibration_fails_on_a_purely_uniform_profile(self, circuit) -> None:
        """A profile with no lateral gradient must not pass the calibration.

        This is the property that stops a rate-only effect being read as spatial
        structure, and it is the gate Phase 4C could not evaluate.
        """
        bins = lateral_coordinates(circuit, 4)
        index_map = circuit.cell_type_index()
        counts = {}
        for condition in ("LEFTWARD", "RIGHTWARD"):
            for trial in range(6):
                # Every neuron in a bin gets the SAME count, so a scramble cannot
                # change any profile at all. Nothing is added on top: a single
                # extra spike on a random neuron would itself be lateral structure
                # and the observed shift would no longer be zero.
                vector = np.zeros(circuit.num_neurons, dtype=np.int64)
                for bin_index in range(4):
                    chosen = np.flatnonzero(bins == bin_index)
                    vector[chosen] = 3
                counts[(condition, trial)] = vector
        result = calibration_shift_null(
            counts, bins, index_map, T4_SUBTYPES, "LEFTWARD", "RIGHTWARD", 6,
            n_scrambles=10, seed=0,
        )
        assert result["observed_max_abs_shift"] == pytest.approx(0.0, abs=1e-12)
        assert result["null_max_abs_shift"] == pytest.approx(0.0, abs=1e-12)
        assert result["observed_exceeds_calibration"] is False
        assert "measuring activity" in result["reading"]

    def test_calibration_handles_absent_and_missing(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        index_map = circuit.cell_type_index()
        absent = calibration_shift_null(
            {}, bins, index_map, ("NotAType",), "LEFTWARD", "RIGHTWARD", 3
        )
        assert absent["status"] == "group_absent"
        no_trials = calibration_shift_null(
            {}, bins, index_map, T4_SUBTYPES, "LEFTWARD", "RIGHTWARD", 3
        )
        assert no_trials["status"] == "no_paired_trials"


# ======================================================================
# The five gates.
# ======================================================================


def _healthy_inputs():
    live_T4 = {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}
    return {
        "drive_profile_variance_fraction": 0.4,
        "input_identity": {
            "max_spread_across_arms": 0.0, "arms_compared": 30,
            "order_identical": True, "lateral_identical": True,
            "direction_energy_equal": True, "direction_energy_spread": 0.0,
        },
        "liveness": {
            PRIMARY_ARM: {"T4": dict(live_T4)},
            PRIMARY_CONTROL: {"T4": dict(live_T4)},
            SECONDARY_CONTROL: {"T4": dict(live_T4)},
        },
        "control_verification": {
            "in_degree_per_neuron_equal": True, "out_degree_per_neuron_equal": True,
            "incoming_weight_per_neuron_equal": True,
            "outgoing_weight_per_neuron_equal": True,
            "edge_weight_multiset_equal": True, "total_synapses_equal": True,
            "soma_location_equal": True, "edge_change_fraction": 0.5, "self_loops": 0,
        },
        "calibration": {"status": "ok", "observed_exceeds_calibration": True,
                        "observed_max_abs_shift": 0.2, "null_max_abs_shift": 0.05,
                        "n_scrambles": 200},
    }


class TestGates:
    def test_all_five_gates_exist_and_are_ordered(self) -> None:
        assert GATE_ORDER == ("G1", "G2", "G3", "G4", "G5")
        assert set(GATE_ORDER) == set(GATE_SPECS) == set(PREDECLARED["gates"])
        for spec in GATE_SPECS.values():
            assert spec["requirement"] and spec["on_failure"] and spec["name"]

    def test_healthy_inputs_pass_every_gate(self) -> None:
        gates = evaluate_gates(**_healthy_inputs())
        assert gates["all_pass"] is True
        assert gates["blocking_gates"] == []
        assert gates["interpretation_allowed"] is True
        assert "ALL GATES PASS" in gates["statement"]

    def test_g1_input_locality_is_a_hard_stop(self) -> None:
        """A flat drive profile means there is nothing to transform. Not a result."""
        inputs = _healthy_inputs()
        inputs["drive_profile_variance_fraction"] = 0.0
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G1"]["status"] == "FAIL"
        assert "G1" in gates["hard_stop_gates"]
        assert gates["interpretation_allowed"] is False
        assert "floor effect" in gates["per_gate"]["G1"]["permits"]

    def test_g2_input_identity_is_a_hard_stop(self) -> None:
        inputs = _healthy_inputs()
        inputs["input_identity"]["max_spread_across_arms"] = 1e-9
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G2"]["status"] == "FAIL"
        assert gates["interpretation_allowed"] is False

    def test_g2_also_requires_equal_direction_energy(self) -> None:
        """The declared G2 requirement has two clauses; both are gated.

        Identical drive across arms says nothing about whether the two sweep
        directions carry equal energy, so a run where the stimulus is unbalanced
        must not read as a passing G2.
        """
        inputs = _healthy_inputs()
        inputs["input_identity"]["direction_energy_equal"] = False
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G2"]["status"] == "FAIL"
        assert "G2" in gates["hard_stop_gates"]

    def test_g2_fails_when_the_arms_disagree_on_neuron_order(self) -> None:
        inputs = _healthy_inputs()
        inputs["input_identity"]["order_identical"] = False
        assert evaluate_gates(**inputs)["per_gate"]["G2"]["status"] == "FAIL"

    def test_g2_fails_when_the_arms_disagree_on_lateral_bins(self) -> None:
        """Identical drive rows, different bin assignment, is still not a wiring contrast."""
        inputs = _healthy_inputs()
        inputs["input_identity"]["lateral_identical"] = False
        assert evaluate_gates(**inputs)["per_gate"]["G2"]["status"] == "FAIL"

    def test_g2_is_not_evaluated_when_nothing_was_measured(self) -> None:
        gates = evaluate_gates(
            drive_profile_variance_fraction=None, input_identity=None,
            liveness=None, control_verification=None, calibration=None,
        )
        assert gates["per_gate"]["G2"]["status"] == "not_evaluated"
        assert "G2" in gates["blocking_gates"]

    def test_g3_voids_only_the_silent_group(self) -> None:
        inputs = _healthy_inputs()
        inputs["liveness"] = {
            PRIMARY_ARM: {
                "T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4},
                "T5": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0},
            },
            PRIMARY_CONTROL: {
                "T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4},
                "T5": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0},
            },
            SECONDARY_CONTROL: {
                "T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4},
                "T5": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0},
            },
        }
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G3"]["per_group"]["T4"]["informative"] is True
        assert gates["per_gate"]["G3"]["per_group"]["T5"]["informative"] is False
        assert "not interpreted" in gates["per_gate"]["G3"]["permits"]
        # A silent group is not a hard stop for the live ones.
        assert "G3" not in gates["hard_stop_gates"]

    def test_g3_rejects_a_group_alive_in_only_one_arm(self) -> None:
        inputs = _healthy_inputs()
        inputs["liveness"] = {
            PRIMARY_ARM: {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
            PRIMARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.9, "spike_saturation_fraction": 0.0}},
            SECONDARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4}},
        }
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G3"]["per_group"]["T4"]["informative"] is False
        assert gates["per_gate"]["G3"]["per_arm"][PRIMARY_ARM]["T4"]["informative"] is True
        assert gates["per_gate"]["G3"]["per_arm"][PRIMARY_CONTROL]["T4"]["informative"] is False
        assert gates["per_gate"]["G3"]["status"] == "FAIL"

    def test_g3_detects_saturation(self) -> None:
        inputs = _healthy_inputs()
        inputs["liveness"] = {
            PRIMARY_ARM: {"T4": {"fraction_zero_activity_trials": 0.0,
                                  "spike_saturation_fraction": 0.5}},
            PRIMARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.0,
                                      "spike_saturation_fraction": 0.5}},
            SECONDARY_CONTROL: {"T4": {"fraction_zero_activity_trials": 0.0,
                                       "spike_saturation_fraction": 0.5}},
        }
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G3"]["per_group"]["T4"]["informative"] is False
        assert gates["per_gate"]["G3"]["status"] == "FAIL"

    def test_g4_voids_a_control_that_did_not_move(self) -> None:
        inputs = _healthy_inputs()
        inputs["control_verification"]["edge_change_fraction"] = 0.01
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G4"]["status"] == "FAIL"
        assert "VOID" in gates["per_gate"]["G4"]["permits"]
        assert "drive-mismatched" in gates["per_gate"]["G4"]["permits"]

    def test_g4_voids_a_control_that_did_not_match_drive(self) -> None:
        inputs = _healthy_inputs()
        inputs["control_verification"]["incoming_weight_per_neuron_equal"] = False
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G4"]["status"] == "FAIL"
        assert "VOID" in gates["per_gate"]["G4"]["permits"]

    def test_g4_requires_no_self_loops(self) -> None:
        inputs = _healthy_inputs()
        inputs["control_verification"]["self_loops"] = 2
        assert evaluate_gates(**inputs)["per_gate"]["G4"]["status"] == "FAIL"

    def test_g5_statistic_calibration_is_a_hard_stop(self) -> None:
        inputs = _healthy_inputs()
        inputs["calibration"]["observed_exceeds_calibration"] = False
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G5"]["status"] == "FAIL"
        assert gates["interpretation_allowed"] is False
        assert "activity, not spatial structure" in gates["per_gate"]["G5"]["permits"]

    def test_a_gate_with_no_input_is_not_a_pass(self) -> None:
        """The failure mode this whole check exists to prevent."""
        gates = evaluate_gates(
            drive_profile_variance_fraction=None, input_identity=None,
            liveness=None, control_verification=None, calibration=None,
        )
        for key in GATE_ORDER:
            assert gates["per_gate"][key]["status"] == "not_evaluated", key
            assert gates["per_gate"][key]["permits"].startswith("nothing")
        assert gates["all_pass"] is False
        assert gates["interpretation_allowed"] is False

    def test_a_partially_built_calibration_is_not_evaluated(self) -> None:
        inputs = _healthy_inputs()
        inputs["calibration"] = {"status": "group_absent"}
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G5"]["status"] == "not_evaluated"

    def test_thresholds_are_declared_not_buried(self) -> None:
        block = predeclared_block()
        assert set(block["thresholds"]) >= {
            "g1_drive_profile_min_variance_fraction", "g2_max_drive_spread",
            "g3_max_zero_activity_fraction", "g3_max_saturation_fraction",
            "g4_min_edge_change_fraction",
        }
        assert block["thresholds"]["g2_max_drive_spread"] == 0.0
        assert block["thresholds"]["alpha"] == ALPHA


# ======================================================================
# Power, pre-declaration, and scope.
# ======================================================================


class TestPower:
    def test_more_trials_are_never_required_for_a_larger_n(self) -> None:
        small = power_requirement(0.1, 0.05)
        large = power_requirement(0.1, 0.05, n_trials=500)
        assert large["n_trials_required"] >= small["n_trials_required"]
        assert large["n_trials_required"] == 500

    def test_a_smaller_effect_needs_more_trials(self) -> None:
        coarse = power_requirement(0.2, 0.10)
        fine = power_requirement(0.2, 0.05)
        assert fine["n_trials_required"] > coarse["n_trials_required"]

    def test_a_larger_sd_needs_more_trials(self) -> None:
        assert (power_requirement(0.4, 0.05)["n_trials_required"]
                > power_requirement(0.1, 0.05)["n_trials_required"])

    def test_the_ci_half_width_shrinks_with_n(self) -> None:
        narrow = power_requirement(0.2, 0.05)
        wide = power_requirement(0.2, 0.05, n_trials=2 * narrow["n_trials_required"])
        assert wide["ci_half_width_at_required_n"] < narrow["ci_half_width_at_required_n"]

    def test_a_degenerate_variance_is_refused(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                power_requirement(bad, 0.05)
        with pytest.raises(ValueError, match="delta"):
            power_requirement(0.2, 0.0)

    def test_the_protocol_reuses_the_phase4c_pairing(self) -> None:
        result = power_requirement(0.15, 0.05)
        assert "paired per-trial differences" in result["protocol"]
        assert "Phase 4C margin" in result["protocol"]
        assert "variance and no effect" in result["sd_source"]
        assert result["alpha"] == ALPHA
        assert result["power"] == 0.80

    def test_the_estimate_stage_cannot_choose_n_after_an_effect(self) -> None:
        """The variance stage must report a spread and nothing else."""
        block = predeclared_block()
        assert block["may_be_tuned_after_seeing_results"] is False
        assert "nothing else" in block["why"]["power"].lower()
        assert any("variance-estimation" in item for item in block["not_done"])


class TestPreDeclaration:
    def test_the_gain_is_held_at_the_audited_phase4c_value(self) -> None:
        assert PREDECLARED["synapse_scale"] == 0.25
        assert PREDECLARED["gain_swept"] is False
        assert "0.25" in PREDECLARED_WHY["synapse_scale"]
        assert "NOT changed if the pilot comes out uninformative" in (
            PREDECLARED_WHY["synapse_scale"]
        )

    def test_the_existing_shuffle_is_retained_not_replaced(self) -> None:
        assert PREDECLARED["existing_shuffle_retained"] is True
        assert PREDECLARED["primary_control"] == PRIMARY_CONTROL == "DRIVE_MATCHED_CONTROL"
        assert PREDECLARED["secondary_control"] == SECONDARY_CONTROL == "EXISTING_SHUFFLE"
        assert ARMS == ("MEASURED", "DRIVE_MATCHED_CONTROL", "EXISTING_SHUFFLE")
        assert "retained rather than replaced" in PREDECLARED_WHY["arms"]
        described = arms()
        assert "RETAINED" in described["EXISTING_SHUFFLE"]
        assert "SECONDARY" in described["EXISTING_SHUFFLE"]
        assert "not drive-matched" in described["EXISTING_SHUFFLE"].lower() or (
            "preserves the weight multiset" in described["EXISTING_SHUFFLE"]
        )

    def test_the_readout_change_is_justified_by_measurement(self) -> None:
        assert "0.5750" in PREDECLARED_WHY["readout"]
        assert "1/8" in PREDECLARED_WHY["readout"]
        assert PREDECLARED["readout_decoder_accuracy_is_primary"] is False
        assert "1/8" in PREDECLARED["statistic_is_not"]

    def test_the_population_stimuli_and_noise_are_held(self) -> None:
        fixed = PREDECLARED_WHY["not_changed"]
        for item in ("population selection rule", "stimuli", "node budget",
                     "common-random-number noise", "randomised sweep start"):
            assert item in fixed, item
        assert PREDECLARED["stimulus_conditions"] == ["LEFTWARD", "RIGHTWARD", "STATIC"]
        assert PREDECLARED["primary_contrast"] == "LEFTWARD_vs_RIGHTWARD"

    def test_the_arms_and_gates_are_declared(self) -> None:
        block = predeclared_block()
        assert block["predeclared"] is True
        assert block["configuration"]["arms"] == list(ARMS)
        assert set(block["gates"]) == set(GATE_ORDER)
        assert "decoder accuracy" in block["configuration"]["statistic_is_not"]

    def test_nothing_declared_is_retuned_after_a_result(self) -> None:
        block = predeclared_block()
        assert block["may_be_tuned_after_seeing_results"] is False
        assert any("no gain changed after seeing a result" in i for i in block["not_done"])
        assert any("no direction-selectivity claim" in i for i in block["not_done"])


class TestScopeAndClaims:
    def test_direction_selectivity_is_explicitly_out_of_scope(self) -> None:
        text = DIRECTION_SELECTIVITY_OUT_OF_SCOPE
        assert "OUT OF SCOPE" in text
        assert "no sign column" in text
        assert "sums and never subtracts" in text
        assert "supplied by the encoder" in text
        assert "NECESSARY PRECONDITION" in text
        assert "nowhere near sufficient" in text
        assert "no result in this module may be presented as direction selectivity" in (
            text.lower()
        )

    def test_the_question_is_a_precondition_not_direction_selectivity(self) -> None:
        assert PREDECLARED["question_is_not"].startswith("direction selectivity")
        assert "lateral offset" not in PREDECLARED["question_is_not"]
        assert "spatially offset" in PREDECLARED["primary_question"]
        assert "unsigned" in PREDECLARED["primary_question"]

    def test_measured_simulated_assumed_unknown_are_all_present(self) -> None:
        claims = claim_block()
        for heading in ("MEASURED", "SIMULATED", "ASSUMED", "UNKNOWN"):
            assert claims[heading], heading
        assert claims["OUT_OF_SCOPE"] == DIRECTION_SELECTIVITY_OUT_OF_SCOPE
        for key in ("positive", "null", "uninformative"):
            assert claims["PILOT_VERDICT_MEANING"][key], key

    def test_measured_names_the_real_connectivity_facts(self) -> None:
        measured = " ".join(claim_block()["MEASURED"]).lower()
        assert "positive integer synapse count" in measured
        assert "curated" in measured
        assert "0.56875" in measured, "the Phase 4C result is measured and must be carried"

    def test_assumed_carries_the_unsigned_consequence(self) -> None:
        assumed = " ".join(claim_block()["ASSUMED"]).lower()
        assert "not synaptic efficacy" in assumed
        assert "every coupling is excitatory" in assumed
        assert "0.25" in assumed

    def test_unknown_carries_the_permanent_limits(self) -> None:
        unknown = " ".join(claim_block()["UNKNOWN"]).lower()
        assert "synaptic sign" in unknown
        assert "excitatory versus inhibitory" in unknown
        assert "true biological synaptic efficacy" in unknown
        assert "direction preference" in unknown
        assert "whether the fly computes direction" in unknown

    def test_a_null_verdict_is_not_read_as_no_computation(self) -> None:
        meaning = claim_block()["PILOT_VERDICT_MEANING"]
        assert "NOT evidence that the MCNS circuit computes nothing" in meaning["null"]
        assert "precondition" in meaning["positive"]
        assert "nothing more" in meaning["positive"]
        assert "no parameter is changed" in meaning["uninformative"]

    def test_the_script_source_keeps_direction_selectivity_out_of_scope(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        assert "DIRECTION_SELECTIVITY_OUT_OF_SCOPE" in source
        assert "PREDECLARED[\"synapse_scale\"]" in source
        for forbidden in ("direction_selectivity", "DirectionSelective"):
            assert f'"{forbidden}"' not in source


class TestScriptContract:
    def test_the_script_imports_and_declares_its_contract(self) -> None:
        module = _load_script()
        assert module.PREDECLARED_PHASE == PREDECLARED_PHASE
        assert module.READOUT_GROUPS == (("T4", T4_SUBTYPES), ("T5", T5_SUBTYPES))
        assert module.SUBTYPE_PRIMARY == "T4"
        assert module.PRIMARY_CONTRAST == ("LEFTWARD", "RIGHTWARD")

    def test_the_pilot_refuses_to_start_without_a_declared_sd(self, tmp_path, pathway_data,
                                                              capsys) -> None:
        module = _load_script()
        code = module.main([
            "--trials", "4", "--n-steps", "40",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(tmp_path / "out"), "--cache", str(tmp_path / "c.npz"),
            "--quiet",
        ])
        assert code == 2
        # readouterr() drains the buffer, so the two assertions share one capture.
        err = capsys.readouterr().err
        assert "requires --sd" in err
        assert "variance-estimation" in err

    def test_the_variance_stage_reports_sd_and_no_effect(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        _stub_liveness_gated_selection(module)
        code = module.main([
            "--variance-estimation", "--quiet", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(tmp_path / "var"), "--cache", str(tmp_path / "c.npz"),
        ])
        assert code == 0
        payload = json.loads(
            (tmp_path / "var" / "phase5a_variance.json").read_text(encoding="utf-8")
        )
        assert payload["stage"] == "variance estimation"
        assert np.isfinite(payload["sd"]) or np.isnan(payload["sd"])
        assert "NOTHING else" in payload["purpose"]
        # It must not contain any result.
        for forbidden in ("primary_margins", "gates", "verdict"):
            assert forbidden not in payload, forbidden
        assert payload["predeclaration"]["predeclared"] is True

    def test_the_pilot_runs_end_to_end_on_the_fixture(self, tmp_path, pathway_data) -> None:
        """The whole pipeline, on synthetic data, in seconds. No real-data run."""
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot"
        code = module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "4", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "10",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        assert code == 0
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        assert summary["phase"] == PREDECLARED_PHASE
        assert summary["synapse_scale"] == 0.25
        assert summary["readout"]["decoder_accuracy_used"] is False
        assert summary["existing_shuffle_retained"] is True
        assert summary["existing_shuffle_is_drive_matched"] is False
        assert set(summary["circuits"]) == set(ARMS)
        assert set(summary["gates"]["per_gate"]) == set(GATE_ORDER)
        assert summary["power"]["n_trials_required"] >= 3
        assert summary["claims"]["OUT_OF_SCOPE"] == DIRECTION_SELECTIVITY_OUT_OF_SCOPE

    def test_all_three_arms_share_one_neuron_set(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot2"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        neurons = {name: c["neurons"] for name, c in summary["circuits"].items()}
        assert len(set(neurons.values())) == 1
        synapses = {name: c["biological_synapse_count"] for name, c in summary["circuits"].items()}
        assert len(set(synapses.values())) == 1
        assert summary["circuits"][PRIMARY_ARM]["is_measured"] is True
        assert summary["circuits"][PRIMARY_CONTROL]["is_measured"] is False
        assert summary["circuits"][SECONDARY_CONTROL]["is_measured"] is False

    def test_the_matched_control_verification_is_in_the_payload(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot3"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        v = summary["matched_control_verification"]
        assert v["incoming_weight_per_neuron_equal"] is True
        assert v["outgoing_weight_per_neuron_equal"] is True
        assert v["edge_change_fraction"] > 0.0
        assert summary["matched_control_disclaimer"] == DRIVE_MATCHED_DISCLAIMER
        # The existing shuffle is reported too, and labelled as not drive-matched.
        assert summary["existing_shuffle_is_drive_matched"] is False
        assert "0.16x" in summary["secondary_control_note"]

    def test_liveness_is_reported_per_arm_and_group(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot4"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        for arm in ARMS:
            assert set(summary["liveness"][arm]) == {"T4", "T5"}
            for group, entry in summary["liveness"][arm].items():
                for field in ("fraction_zero_activity_trials", "spike_saturation_fraction",
                              "mean_spikes_per_trial"):
                    assert field in entry, f"{arm}/{group}/{field}"

    def test_no_artifact_claims_a_biological_discovery(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot5"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        text = (out / "phase5a_pilot.json").read_text(encoding="utf-8").lower()
        for forbidden in (
            "demonstrates direction selectivity",
            "reproduces direction selectivity",
            "mcns computes motion",
            "biological discovery",
            "we have discovered",
        ):
            assert forbidden not in text, forbidden

    def test_the_four_arms_run_with_unaveraged_counts(self, tmp_path, pathway_data) -> None:
        """Counts, not a 1/8-quantised fold accuracy: the readout's whole point."""
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot6"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        for arm in ARMS:
            for group in ("T4", "T5"):
                per_condition = summary["centroids"][arm][group]
                assert "LEFTWARD" in per_condition and "RIGHTWARD" in per_condition
                for condition, values in per_condition.items():
                    assert len(values) == summary["n_trials"], f"{arm}/{group}/{condition}"

    def test_liveness_preserves_per_condition_records_for_G3(self, tmp_path, pathway_data) -> None:
        """The persisted payload keeps the pooled value *and* its condition sources.

        Without this, G3 and the JSON could be derived from different folds, which
        is exactly the silent producer/consumer drift the frozen real-data artifact
        exposed for G3.
        """
        module = _load_script()
        _stub_liveness_gated_selection(module)
        out = tmp_path / "pilot7"
        module.main([
            "--sd", "0.1", "--delta", "0.2", "--trials", "3", "--n-steps", "40",
            "--max-bodies-per-type", "4", "--scrambles", "5", "--permutations", "5",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(out), "--cache", str(tmp_path / "c.npz"), "--quiet",
        ])
        summary = json.loads((out / "phase5a_pilot.json").read_text(encoding="utf-8"))
        conditions = set(summary["predeclaration"]["configuration"]["stimulus_conditions"])
        for arm in ARMS:
            for group in ("T4", "T5"):
                entry = summary["liveness"][arm][group]
                for field in ("fraction_zero_activity_trials", "spike_saturation_fraction",
                              "mean_spikes_per_trial"):
                    assert field in entry, f"{arm}/{group}/{field}"
                assert set(entry["_per_condition"]) == conditions, f"{arm}/{group}"
                for _cond, rec in entry["_per_condition"].items():
                    assert "fraction_zero_activity_trials" in rec
                    assert "spike_saturation_fraction" in rec


# ======================================================================
# Real-data gate-shape regression: the frozen artifact as diagnostic input,
# plus small hand-built records matching its contract. No experiment run.
# ======================================================================


def _tiny_g1_circuit(n_per_type: int = 8) -> MCNSCircuit:
    types = ["L1"] * n_per_type + ["L2"] * n_per_type
    body_ids = np.arange(len(types), dtype=np.int64)
    cell_type = np.asarray(types, dtype=object)
    csr = sp.csr_matrix(np.zeros((len(types), len(types)), dtype=np.int64))
    return MCNSCircuit(
        body_ids=body_ids, cell_type=cell_type, csr=csr,
        superclass=np.asarray(["visual"] * len(types), dtype=object),
        soma_side=np.asarray(["left", "right"] * (len(types) // 2), dtype=object),
        soma_location=np.zeros((len(types), 3), dtype=np.int64),
        cell_type_names=["L1", "L2"],
    )


class _FakeInputGenerator:
    """Feed ``_drive_profile_variance_fraction`` a controlled input-pathway drive."""

    def __init__(self, circuit: MCNSCircuit, drive: np.ndarray):
        self.circuit = circuit
        self._drive = np.asarray(drive, dtype=np.float64)
        self.on_types = ("L1",)
        self.off_types = ("L2",)

    def drive_tensor(self, condition, trial: int = 0, n_steps: int | None = None) -> np.ndarray:
        return self._drive


class TestGateInputCorrectness:
    def test_g1_uses_declared_input_drive_not_a_downstream_proxy(self) -> None:
        circuit = _tiny_g1_circuit(8)
        lateral = np.arange(circuit.num_neurons) % 4
        varying = np.zeros((4, circuit.num_neurons), dtype=np.float64)
        varying[:, 0] = 4.0
        varying[:, 2] = 1.0
        flat = np.ones((4, circuit.num_neurons), dtype=np.float64)
        drive_group = ("L1", "L2")

        varying_metrics = _load_script()._drive_profile_variance_fraction(
            _FakeInputGenerator(circuit, varying), lateral, drive_group, n_steps=4
        )
        flat_metrics = _load_script()._drive_profile_variance_fraction(
            _FakeInputGenerator(circuit, flat), lateral, drive_group, n_steps=4
        )

        assert varying_metrics["variance_fraction"] > 0.5
        assert flat_metrics["variance_fraction"] == pytest.approx(0.0)
        assert "input population" in varying_metrics["note"]

    def test_g3_rejects_malformed_arm_and_group_shape(self) -> None:
        inputs = _healthy_inputs()
        inputs["liveness"] = {
            "T4": {"fraction_zero_activity_trials": 0.0, "spike_saturation_fraction": 1e-4},
        }
        gates = evaluate_gates(**inputs)
        assert gates["per_gate"]["G3"]["status"] == "FAIL"
        assert gates["per_gate"]["G3"]["reason"] == "malformed_liveness_payload"
        assert "G3" in gates["blocking_gates"]

    def test_g3s_all_condition_pooled_logic_is_arithmetic_not_threshold_change(self) -> None:
        # Ten per-trial counts in which six are zero: zero fraction 0.6 must fail
        # even if every single spike value is nonzero and below saturation.
        record = {
            "fraction_zero_activity_trials": 0.6,
            "spike_saturation_fraction": 1e-4,
        }
        inputs = _healthy_inputs()
        inputs["liveness"] = {
            PRIMARY_ARM: {"T4": dict(record)},
            PRIMARY_CONTROL: {"T4": dict(record)},
            SECONDARY_CONTROL: {"T4": dict(record)},
        }
        assert evaluate_gates(**inputs)["per_gate"]["G3"]["status"] == "FAIL"

    def test_nan_fold_margins_are_blocked_and_not_counted_as_zero(self) -> None:
        primary_shift = [0.25, float("nan"), -0.1, 0.05]
        control_shift = [0.00, 0.00, -0.20, 0.00]
        result = _load_script()._phase5a_paired_margin(
            primary_shift, control_shift, n_permutations=5, seed=0
        )
        assert result["status"] == "nan_folds_present"
        assert result["n_nonfinite_folds"] == 1
        assert result["effect_supported"] is False
        assert "No finite effect" in result["reason"] or "no finite effect" in result["reason"].lower()

        clean = _load_script()._phase5a_paired_margin(
            [0.2, 0.3, -0.1, 0.1], [0.0, 0.1, -0.2, 0.0], n_permutations=5, seed=0
        )
        assert clean["status"] == "ok"
        assert clean["nan_audit"]["n_nonfinite_folds"] == 0

    def test_the_frozen_real_data_artifact_has_the_declared_shape_but_failed_gates(self) -> None:
        artifact = REPO / "outputs" / "mcns_phase5a_pilot" / "phase5a_pilot.json"
        if not artifact.is_file():
            pytest.skip("frozen real-data Phase 5A artifact is unavailable in this checkout")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert set(payload["liveness"]) == set(ARMS)
        for arm in ARMS:
            assert set(payload["liveness"][arm]) == {"T4", "T5"}
            for group in ("T4", "T5"):
                record = payload["liveness"][arm][group]
                assert "fraction_zero_activity_trials" in record
                assert "spike_saturation_fraction" in record
        assert set(payload["gates"]["per_gate"]) == set(GATE_ORDER)

        # The pre-fix run happened, and it reported an uninterpretable gate product.
        gates = payload["gates"]["per_gate"]
        assert gates["G1"]["status"] == "FAIL"
        assert gates["G2"]["status"] == "pass"
        assert gates["G2"]["direction_energy_equal"] is True
        assert gates["G3"]["status"] == "FAIL"
        # Frozen diagnostic: this object was built from an arm-level payload while
        # keying the result as though it were group-level. Keep the failure visible.
        assert set(gates["G3"]["per_group"]) == set(ARMS)
        assert gates["G4"]["status"] == "FAIL"
        assert gates["G5"]["status"] == "FAIL"

        primary_margin = payload["primary_margins"][PRIMARY_CONTROL]
        assert primary_margin["status"] == "ok"
        assert primary_margin["effect_supported"] is False
        assert any(np.isnan(v) for v in primary_margin["per_fold_margin"])
        assert np.isnan(primary_margin["paired_t_statistic"])


