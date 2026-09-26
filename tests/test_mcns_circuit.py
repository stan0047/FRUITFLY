"""Tests for the MCNS circuit representation, sparse LIF simulation, and figures.

Runs entirely on the tiny hand-made Feather fixtures, so no network access, no
neuPrint token, and no dependency on the real 1.05 GB MaleCNS file.

A few of these tests are specifically about not overclaiming. The circuit, the
simulator, and the figures each have a way they could quietly imply that
simulated output is recorded biology or that T4/T5 direction selectivity was
demonstrated. Those cases are asserted here, not just documented.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flybrain.brain.mcns_circuit import (
    CHAIN_EDGES,
    DENSE_REFUSE_BYTES,
    FEEDFORWARD_CHAINS,
    MODE_FEEDFORWARD,
    MODE_RECURRENT,
    MCNSCircuit,
    MCNSCircuitError,
    candidate_metadata,
    cell_type_summary,
)
from flybrain.brain.mcns_simulation import (
    MotionInputEncoder,
    Stimulus,
    active_body_ids,
    activity_by_group,
    run_simulation,
)
from flybrain.brain.neuron import NeuronParams
from flybrain.data.motion_candidate import (
    MOTION_CELL_TYPES,
    build_populations,
    extract_candidate_edges,
    load_annotations,
)
from flybrain.visualization import mcns_activity

FIXTURES = Path(__file__).parent / "fixtures"
ANN = FIXTURES / "sample_mcns_motion_annotations.feather"
EDGES = FIXTURES / "sample_mcns_motion_edges.feather"


@pytest.fixture(scope="module")
def annotations() -> dict[str, np.ndarray]:
    return load_annotations(ANN)


@pytest.fixture(scope="module")
def populations(annotations):
    return build_populations(annotations, MOTION_CELL_TYPES, require_superclass=True)


@pytest.fixture(scope="module")
def extraction(populations):
    return extract_candidate_edges(EDGES, populations, direction="both")


@pytest.fixture(scope="module")
def circuit(populations, extraction, annotations) -> MCNSCircuit:
    return MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT)


@pytest.fixture(scope="module")
def feedforward(populations, extraction, annotations) -> MCNSCircuit:
    return MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_FEEDFORWARD)


# ------------------------------------------------------------- circuit shape


def test_circuit_nodes_and_types_come_from_the_population(circuit, populations) -> None:
    assert circuit.num_neurons == populations.num_bodies
    assert np.array_equal(circuit.body_ids, np.sort(populations.body_ids))
    # Exact labels, subtypes never merged.
    for name in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        assert circuit.has(name)
    assert "T4" not in circuit.cell_type_names
    assert "T5" not in circuit.cell_type_names


def test_index_and_body_id_maps_are_exact_inverses(circuit) -> None:
    for index in range(circuit.num_neurons):
        body = circuit.to_body_id(index)
        assert circuit.to_index(body) == index
    assert circuit.body_of[circuit.to_index(circuit.body_ids[0])] == int(circuit.body_ids[0])


def test_index_lookup_rejects_bodies_outside_the_circuit(circuit) -> None:
    with pytest.raises(MCNSCircuitError, match="not in this circuit"):
        circuit.to_index(int(circuit.body_ids.max()) + 10_000)
    with pytest.raises(MCNSCircuitError, match="out of range"):
        circuit.to_body_id(circuit.num_neurons)


def test_circuit_rejects_mismatched_shapes(populations) -> None:
    with pytest.raises(MCNSCircuitError, match="does not match"):
        MCNSCircuit(
            body_ids=populations.body_ids,
            cell_type=np.asarray(populations.cell_types, dtype=object)[populations.type_of],
            csr=sp.csr_matrix((3, 3), dtype=np.int64),
        )
    with pytest.raises(MCNSCircuitError, match="one label per body"):
        MCNSCircuit(
            body_ids=populations.body_ids,
            cell_type=np.asarray(["L1"] * 2, dtype=object),
            csr=sp.csr_matrix((populations.num_bodies, populations.num_bodies), dtype=np.int64),
        )


def test_circuit_rejects_non_ascending_body_ids() -> None:
    with pytest.raises(MCNSCircuitError, match="strictly ascending"):
        MCNSCircuit(
            body_ids=np.asarray([5, 3], dtype=np.int64),
            cell_type=np.asarray(["L1", "L1"], dtype=object),
            csr=sp.csr_matrix((2, 2), dtype=np.int64),
        )


def test_unknown_mode_is_rejected(populations, extraction) -> None:
    with pytest.raises(MCNSCircuitError, match="mode must be one of"):
        MCNSCircuit.from_extraction(populations, extraction, None, mode="everything")


# -------------------------------------------------- counts are not strengths


def test_biological_synapse_counts_stay_exact_integers(circuit, extraction) -> None:
    """The data layer stores counts, not a rescaled or normalised 'strength'."""
    assert np.issubdtype(circuit.csr.dtype, np.integer)
    stored = dict(zip(zip(circuit.csr.tocoo().row, circuit.csr.tocoo().col), circuit.csr.data))
    for row, col, count in zip(extraction.pre_index, extraction.post_index, extraction.synapse_count):
        assert stored[(int(row), int(col))] == int(count)
    assert circuit.total_synapse_count == int(extraction.synapse_count.sum())


def test_coupling_is_separate_from_the_count(circuit) -> None:
    """`coupling_matrix` is float and scaled; `csr` is untouched by it."""
    before_dtype = circuit.csr.dtype
    before_total = circuit.total_synapse_count

    linear = circuit.coupling_matrix(synapse_scale=0.5)
    assert np.issubdtype(linear.dtype, np.floating)
    assert np.allclose(linear.toarray(), circuit.csr.toarray() * 0.5)

    log = circuit.coupling_matrix(synapse_scale=1.0, coupling="log1p")
    assert np.allclose(log.toarray(), np.log1p(circuit.csr.toarray()))

    # Asking for coupling must not have rewritten the measured counts.
    assert circuit.csr.dtype == before_dtype
    assert circuit.total_synapse_count == before_total


def test_coupling_rejects_bad_parameters(circuit) -> None:
    with pytest.raises(MCNSCircuitError, match="synapse_scale must be"):
        circuit.coupling_matrix(synapse_scale=-1.0)
    with pytest.raises(MCNSCircuitError, match="coupling must be"):
        circuit.coupling_matrix(coupling="quadratic")


def test_coupling_stays_sparse(circuit) -> None:
    coupling = circuit.coupling_matrix()
    assert sp.isspmatrix_csr(coupling)
    assert coupling.nnz == circuit.csr.nnz


def test_propagation_matrix_delivers_a_spike_to_the_target(circuit) -> None:
    """Orientation guard: `csr[source, target]` must not be used as-is for `M @ spikes`.

    Using the untransposed matrix is silent and wrong, so this checks the
    direction of one known edge rather than only checking shapes.
    """
    coo = circuit.csr.tocoo()
    edges = set(zip(coo.row.tolist(), coo.col.tolist()))
    # Need an edge with no reverse, so the wrong orientation is unambiguously 0.
    source, target, count = next(
        (int(r), int(c), int(v)) for r, c, v in zip(coo.row, coo.col, coo.data) if (int(c), int(r)) not in edges
    )

    spikes = np.zeros(circuit.num_neurons)
    spikes[source] = 1.0

    correct = circuit.propagation_matrix(synapse_scale=1.0) @ spikes
    assert correct[target] == count, "the target must receive the source's coupling"
    assert correct[source] == 0.0, "the source must not receive its own outgoing coupling"

    # Untransposed, the same spike is delivered to the source's own incoming
    # neighbours instead of to the target, and the target hears nothing.
    wrong = circuit.coupling_matrix(synapse_scale=1.0) @ spikes
    assert wrong[target] == 0.0
    assert not np.array_equal(wrong, correct)


def test_propagation_matrix_is_the_transpose_of_coupling(circuit) -> None:
    propagation = circuit.propagation_matrix(synapse_scale=0.5)
    coupling = circuit.coupling_matrix(synapse_scale=0.5)
    assert sp.isspmatrix_csr(propagation)
    assert propagation.nnz == coupling.nnz
    assert np.allclose(propagation.toarray(), coupling.toarray().T)


# --------------------------------------------------------------- sparsity


def test_dense_is_refused_for_a_realistic_circuit() -> None:
    """A guard that permits a multi-gigabyte allocation is not a guard."""
    big = MCNSCircuit(
        body_ids=np.arange(1, 22_452, dtype=np.int64),
        cell_type=np.asarray(["L1"] * 22_451, dtype=object),
        csr=sp.csr_matrix((22_451, 22_451), dtype=np.int64),
    )
    assert 22_451**2 * 8 > DENSE_REFUSE_BYTES
    with pytest.raises(MCNSCircuitError, match="refusing to densify"):
        big.to_dense()
    assert sp.issparse(big.coupling_matrix())


def test_sub_matrix_stays_sparse(circuit) -> None:
    idx = circuit.indices_of_type("Mi1")
    sub = circuit.sub_matrix(idx)
    assert sp.isspmatrix_csr(sub)
    assert sub.shape == (idx.size, idx.size)
    assert np.array_equal(sub.toarray(), circuit.csr[np.ix_(idx, idx)].toarray())


# ------------------------------------------------------- the two modes


def test_both_modes_keep_the_same_nodes_and_index_mapping(circuit, feedforward) -> None:
    """Switching mode changes edges, not the node set, so runs stay comparable."""
    assert np.array_equal(circuit.body_ids, feedforward.body_ids)
    assert circuit.cell_type_names == feedforward.cell_type_names
    for index in range(circuit.num_neurons):
        assert circuit.to_body_id(index) == feedforward.to_body_id(index)


def test_feedforward_keeps_only_documented_chains(circuit, feedforward) -> None:
    assert feedforward.num_edges <= circuit.num_edges
    kept = set(zip(*feedforward.csr.nonzero()))
    full = set(zip(*circuit.csr.nonzero()))
    assert kept <= full
    # Every surviving edge must be one of the documented type pairs.
    for row, col in kept:
        pair = (str(circuit.cell_type[row]), str(circuit.cell_type[col]))
        assert pair in CHAIN_EDGES, pair
    assert feedforward.metadata["edges_removed_for_mode"]["not_on_a_documented_chain"] == (
        circuit.num_edges - feedforward.num_edges
    )


def test_feedforward_removes_the_measured_feedback(circuit, feedforward) -> None:
    removed = circuit.num_edges - feedforward.num_edges
    assert removed > 0, "fixture should contain off-chain feedback to remove"
    # A T4 -> Mi1 or T4 -> T4 edge is feedback, and must not survive.
    for row, col in zip(*feedforward.csr.nonzero()):
        source = str(circuit.cell_type[row])
        assert not (source.startswith("T4") or source.startswith("T5"))


def test_recurrent_mode_keeps_every_measured_edge(circuit, extraction) -> None:
    assert circuit.num_edges == extraction.num_edges
    assert circuit.mode == MODE_RECURRENT
    assert circuit.metadata["edges_removed_for_mode"] == {}


def test_chains_cover_both_documented_pathways() -> None:
    on_pairs = set(FEEDFORWARD_CHAINS["ON"])
    off_pairs = set(FEEDFORWARD_CHAINS["OFF"])
    assert ("L1", "Mi1") in on_pairs
    assert {("Mi1", t) for t in ("T4a", "T4b", "T4c", "T4d")} <= on_pairs
    assert ("L2", "Tm1") in off_pairs
    assert ("L2", "Tm2") in off_pairs
    assert {("Tm1", t) for t in ("T5a", "T5b", "T5c", "T5d")} <= off_pairs
    assert {("Tm2", t) for t in ("T5a", "T5b", "T5c", "T5d")} <= off_pairs


# --------------------------------------------------------------- metadata


def test_metadata_is_aligned_to_the_body_order(circuit, annotations) -> None:
    by_body = {int(b): i for i, b in enumerate(annotations["bodyId"])}
    for index in range(circuit.num_neurons):
        body = int(circuit.body_ids[index])
        source = by_body[body]
        assert circuit.cell_type[index] == str(annotations["type"][source])
        side = annotations["somaSide"][source]
        expected = "" if side is None or side != side else str(side)
        assert circuit.soma_side[index] == expected


def test_candidate_metadata_loads_only_requested_bodies() -> None:
    everything = candidate_metadata(ANN, load_annotations(ANN)["bodyId"])
    subset = candidate_metadata(ANN, everything["bodyId"].to_numpy()[:3])
    assert len(subset) == 3
    assert set(subset.columns) == {
        "bodyId",
        "superclass",
        "somaSide",
        "soma_x",
        "soma_y",
        "soma_z",
    }
    assert subset["bodyId"].is_monotonic_increasing


def test_candidate_metadata_reports_a_missing_file() -> None:
    with pytest.raises(MCNSCircuitError, match="annotation file not found"):
        candidate_metadata(FIXTURES / "nope.feather", [1])


def test_soma_locations_align_and_mark_missing_as_minus_one(circuit) -> None:
    frame = candidate_metadata(ANN, circuit.body_ids)
    circuit.with_soma_locations(frame)
    assert circuit.soma_location.shape == (circuit.num_neurons, 3)
    present = (circuit.soma_location[:, 0] >= 0).sum()
    assert present == frame["soma_x"].notna().sum()
    # -1 must never be confused with a real coordinate.
    assert ((circuit.soma_location < -1).sum() == 0)


def test_summary_reports_sides_and_types(circuit) -> None:
    summary = circuit.summary()
    assert summary["neurons"] == circuit.num_neurons
    assert summary["edges"] == circuit.num_edges
    assert summary["total_biological_synapse_count"] == circuit.total_synapse_count
    assert "sparse" in summary["storage"]
    for name in ("L1", "L2", "Mi1", "Tm1", "Tm2", "T4a", "T5d"):
        assert summary["cell_types"][name]["neurons"] == circuit.indices_of_type(name).size


def test_edge_type_matrix_totals_match_the_circuit(circuit) -> None:
    frame = circuit.edge_type_matrix()
    assert frame.loc["L1", "Mi1"] > 0
    assert int(frame.to_numpy().sum()) == circuit.total_synapse_count
    assert "T4" not in frame.index and "T4a" in frame.index


# ------------------------------------------------------------------ encoder


def test_encoder_injects_only_into_the_input_populations(circuit) -> None:
    encoder = MotionInputEncoder()
    current = encoder.encode(circuit, Stimulus.LEFTWARD_MOTION)
    touched = set(circuit.types_of(np.flatnonzero(current > 0)))
    assert touched <= {"L1", "L2"}


def test_encoder_is_deterministic(circuit) -> None:
    encoder = MotionInputEncoder(seed=7)
    first = encoder.encode(circuit, Stimulus.RIGHTWARD_MOTION)
    second = encoder.encode(circuit, Stimulus.RIGHTWARD_MOTION)
    assert np.array_equal(first, second)


def test_stimuli_produce_different_drive(circuit) -> None:
    encoder = MotionInputEncoder()
    vectors = {s: encoder.encode(circuit, s) for s in Stimulus}
    assert not np.array_equal(vectors[Stimulus.LEFTWARD_MOTION], vectors[Stimulus.RIGHTWARD_MOTION])
    assert not np.array_equal(vectors[Stimulus.NO_MOTION], vectors[Stimulus.LEFTWARD_MOTION])


def test_left_and_right_stimuli_mirror_across_hemispheres(circuit) -> None:
    """The spatial drive is a convention, but it should at least be antisymmetric."""
    encoder = MotionInputEncoder()
    left = encoder.encode(circuit, Stimulus.LEFTWARD_MOTION)
    right = encoder.encode(circuit, Stimulus.RIGHTWARD_MOTION)
    sides = np.asarray([str(s) for s in circuit.soma_side])
    on = np.concatenate([circuit.indices_of_type("L1")])
    l_idx = on[sides[on] == "L"]
    r_idx = on[sides[on] == "R"]
    if l_idx.size and r_idx.size:
        assert left[l_idx].mean() > left[r_idx].mean()
        assert right[r_idx].mean() > right[l_idx].mean()


def test_encoder_describes_itself_as_synthetic() -> None:
    described = MotionInputEncoder().describe()
    assert described["is_measured"] is False
    assert "synthetic" in described["kind"].lower()
    assert "direction selectivity" in described["note"].lower()


def test_unknown_stimulus_name_is_rejected(circuit) -> None:
    with pytest.raises(ValueError):
        MotionInputEncoder().encode(circuit, "SOMETHING_ELSE")


# ---------------------------------------------------------------- simulation


def test_simulation_records_one_column_per_body(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.LEFTWARD_MOTION, n_steps=20)
    assert result.spikes.shape == (20, circuit.num_neurons)
    assert result.spikes.dtype == bool
    assert result.times.size == 20
    assert result.total_spikes() == int(np.count_nonzero(result.spikes))


def test_simulation_rejects_bad_step_counts(circuit) -> None:
    with pytest.raises(ValueError, match="n_steps must be positive"):
        run_simulation(circuit, MotionInputEncoder(), Stimulus.NO_MOTION, n_steps=0)


def test_no_spikes_outside_the_input_population(circuit) -> None:
    """A cell with no incoming drive should stay silent; that is a wiring check."""
    result = run_simulation(
        circuit, MotionInputEncoder(amplitude=0.0, baseline=0.0), Stimulus.NO_MOTION, n_steps=25
    )
    assert result.total_spikes() == 0


def test_refractory_period_limits_firing_rate(circuit) -> None:
    strong = MotionInputEncoder(amplitude=5.0, baseline=5.0)
    with_refractory = run_simulation(
        circuit, strong, Stimulus.LEFTWARD_MOTION, n_steps=200, params=NeuronParams(refractory_periods=5)
    )
    without = run_simulation(
        circuit, strong, Stimulus.LEFTWARD_MOTION, n_steps=200, params=NeuronParams(refractory_periods=0)
    )
    assert with_refractory.total_spikes() < without.total_spikes()
    # A neuron cannot fire twice inside its refractory window.
    for index in np.flatnonzero(with_refractory.spikes.any(axis=0)):
        times = np.flatnonzero(with_refractory.spikes[:, index])
        assert np.all(np.diff(times) > 5)


def test_stimulus_changes_the_response(circuit) -> None:
    encoder = MotionInputEncoder(amplitude=2.0, baseline=1.0)
    left = run_simulation(circuit, encoder, Stimulus.LEFTWARD_MOTION, n_steps=150)
    right = run_simulation(circuit, encoder, Stimulus.RIGHTWARD_MOTION, n_steps=150)
    assert left.total_spikes() != right.total_spikes()


def test_recurrent_coupling_changes_the_response(feedforward) -> None:
    """A circuit with real feedback should not behave like the feedforward cut."""
    populations, extraction, annotations = _rebuild()
    recurrent = MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT)
    encoder = MotionInputEncoder(amplitude=1.0, baseline=0.5)
    a = run_simulation(feedforward, encoder, Stimulus.LEFTWARD_MOTION, n_steps=150, synapse_scale=0.5)
    b = run_simulation(recurrent, encoder, Stimulus.LEFTWARD_MOTION, n_steps=150, synapse_scale=0.5)
    assert a.total_spikes() != b.total_spikes()


def test_result_records_that_it_is_simulated(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.LEFTWARD_MOTION, n_steps=5)
    assert result.stimulus == "LEFTWARD_MOTION"
    assert result.circuit_mode == MODE_RECURRENT
    assert "sparse mat-vec" in result.config["recurrent_update"]
    assert result.config["stimulus"]["is_measured"] is False
    assert "synapse_scale" in result.config


def test_grouped_activity_keeps_subtypes_separate(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(amplitude=2.0), Stimulus.LEFTWARD_MOTION, n_steps=100)
    grouped = activity_by_group(circuit, result)
    by_type = grouped["by_cell_type"]
    for name in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        assert name in by_type
    assert "T4" not in by_type and "T5" not in by_type
    assert sum(v["neurons"] for v in by_type.values()) == circuit.num_neurons
    assert set(grouped["by_hemisphere"]) == {"L", "R", "unlabelled"}


def test_cell_type_summary_table_matches_spike_counts(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(amplitude=2.0), Stimulus.LEFTWARD_MOTION, n_steps=100)
    frame = cell_type_summary(circuit, result.spikes, result.times)
    counts = result.spikes.astype(np.int64).sum(axis=0)
    assert int(frame["spikes"].sum()) == int(counts.sum())
    assert "T4a" in frame.index and "T4" not in frame.index
    assert frame.loc["L1", "neurons"] == circuit.indices_of_type("L1").size


def test_active_body_ids_are_real_mcns_ids(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(amplitude=3.0), Stimulus.LEFTWARD_MOTION, n_steps=60)
    step = int(np.argmax(result.spikes.sum(axis=1)))
    ids = active_body_ids(circuit, result, step)
    assert set(ids) <= set(int(b) for b in circuit.body_ids)
    assert ids == [int(circuit.to_body_id(i)) for i in result.active_indices(step)]


def test_active_indices_rejects_out_of_range_steps(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.NO_MOTION, n_steps=5)
    with pytest.raises(IndexError):
        result.active_indices(5)
    with pytest.raises(IndexError):
        result.active_indices(-1)


def _rebuild():
    annotations = load_annotations(ANN)
    populations = build_populations(annotations, MOTION_CELL_TYPES, require_superclass=True)
    extraction = extract_candidate_edges(EDGES, populations, direction="both")
    return populations, extraction, annotations


# ------------------------------------------------------------- figures


def test_figures_are_written_with_the_required_banner(circuit, tmp_path: Path) -> None:
    result = run_simulation(circuit, MotionInputEncoder(amplitude=2.0), Stimulus.LEFTWARD_MOTION, n_steps=60)
    paths = {
        "raster": mcns_activity.plot_raster(circuit, result, tmp_path / "raster.png"),
        "population": mcns_activity.plot_population_activity(circuit, result, tmp_path / "population.png"),
        "types": mcns_activity.plot_cell_type_activity(
            circuit, {s.value: result for s in Stimulus}, tmp_path / "types.png"
        ),
    }
    for name, path in paths.items():
        assert Path(path).is_file() and Path(path).stat().st_size > 0, name


def test_no_figure_claims_biological_activity() -> None:
    assert mcns_activity.FIGURE_BANNER == "MCNS connectivity + simulated LIF dynamics"
    caveat = mcns_activity.NOT_BIOLOGICAL_ACTIVITY.lower()
    assert "not neural activity recorded from a fly" in caveat
    assert "direction selectivity is not demonstrated" in caveat

    source = Path(mcns_activity.__file__).read_text(encoding="utf-8")
    # The banner and caveat are applied centrally, so the invariant to check is
    # that every plot routes through the shared finisher that stamps them.
    assert "FIGURE_BANNER" in source.split("def _footer", 1)[1].split("\ndef ", 1)[0]
    assert "NOT_BIOLOGICAL_ACTIVITY" in source.split("def _footer", 1)[1].split("\ndef ", 1)[0]
    for figure_name in ("plot_raster", "plot_population_activity", "plot_cell_type_activity"):
        body = source.split(f"def {figure_name}", 1)[1].split("\ndef ", 1)[0]
        assert "_finish(figure, path)" in body, figure_name


def test_figures_actually_carry_the_banner_and_caveat(circuit) -> None:
    """Check the rendered text, not just the source, so a refactor cannot drop it."""
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.NO_MOTION, n_steps=5)
    figure = mcns_activity.plot_population_activity(circuit, result)
    rendered = " ".join(t.get_text() for t in figure.findobj(matplotlib.text.Text)).lower()
    assert "simulated lif dynamics" in rendered
    assert "not neural activity recorded from a fly" in rendered
    assert "direction selectivity is not demonstrated" in rendered
    matplotlib.pyplot.close(figure)


def test_cell_type_figure_never_merges_subtypes(circuit, tmp_path: Path) -> None:
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.NO_MOTION, n_steps=10)
    figure = mcns_activity.plot_cell_type_activity(
        circuit, {s.value: result for s in Stimulus}
    )
    labels = {str(t.get_text()) for t in figure.axes[0].get_xticklabels()}
    assert "T4a" in labels and "T5d" in labels
    assert "T4" not in labels and "T5" not in labels
    matplotlib.pyplot.close(figure)


def test_opposed_stimulus_figure_carries_the_mirror_warning(circuit) -> None:
    """A T4/T5 left-right split is the easiest result here to misread. Guard it."""
    result = run_simulation(circuit, MotionInputEncoder(), Stimulus.NO_MOTION, n_steps=5)
    opposed = mcns_activity.plot_cell_type_activity(
        circuit, {Stimulus.LEFTWARD_MOTION.value: result, Stimulus.RIGHTWARD_MOTION.value: result}
    )
    title = opposed.axes[0].get_title().lower()
    assert "mirrors the drive" in title
    assert "not a computed direction-selective response" in title
    matplotlib.pyplot.close(opposed)

    single = mcns_activity.plot_cell_type_activity(circuit, {Stimulus.NO_MOTION.value: result})
    assert "mirrors the drive" not in single.axes[0].get_title().lower()
    matplotlib.pyplot.close(single)


def test_active_body_ids_series_is_one_list_per_step(circuit) -> None:
    result = run_simulation(circuit, MotionInputEncoder(amplitude=2.0), Stimulus.LEFTWARD_MOTION, n_steps=12)
    series = mcns_activity.active_body_ids_series(circuit, result, max_steps=5)
    assert len(series) == 5
    known = set(int(b) for b in circuit.body_ids)
    for step, ids in enumerate(series):
        assert set(ids) <= known
        assert set(ids) == set(int(circuit.to_body_id(i)) for i in result.active_indices(step))
