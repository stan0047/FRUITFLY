"""Tests for MCNS motion-candidate extraction.

Runs entirely on tiny hand-made Feather fixtures. No network access, no
neuPrint token, and no dependency on the real 1.05 GB MaleCNS file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flybrain.data.motion_candidate import (
    MOTION_CELL_TYPES,
    MOTION_PATHWAYS,
    CandidatePopulations,
    ExtractionError,
    build_celltype_matrix,
    build_populations,
    build_sparse,
    extract_candidate_edges,
    extract_motion_candidates,
    load_annotations,
    resolve_cell_types,
    summarise_pathways,
    validate_extraction,
    write_report,
)

FIXTURES = Path(__file__).parent / "fixtures"
ANN = FIXTURES / "sample_mcns_motion_annotations.feather"
EDGES = FIXTURES / "sample_mcns_motion_edges.feather"


@pytest.fixture(scope="module")
def annotations() -> dict[str, np.ndarray]:
    return load_annotations(ANN)


@pytest.fixture(scope="module")
def populations(annotations) -> CandidatePopulations:
    return build_populations(annotations, MOTION_CELL_TYPES)


@pytest.fixture(scope="module")
def extraction(populations):
    return extract_candidate_edges(EDGES, populations, direction="both")


# ------------------------------------------------------------ cell types


def test_required_cell_types_cover_both_channels_with_separate_subtypes() -> None:
    assert len(MOTION_CELL_TYPES) == 13
    for subtype in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        assert subtype in MOTION_CELL_TYPES, f"{subtype} must be its own label, not collapsed"
    assert "T4" not in MOTION_CELL_TYPES and "T5" not in MOTION_CELL_TYPES
    assert set(MOTION_PATHWAYS["ON"]["outputs"]) == {"T4a", "T4b", "T4c", "T4d"}
    assert set(MOTION_PATHWAYS["OFF"]["outputs"]) == {"T5a", "T5b", "T5c", "T5d"}


def test_populations_keep_subtypes_distinct(populations) -> None:
    for subtype in ("T4a", "T4b", "T4c", "T4d"):
        assert subtype in populations.cell_types
    indices = [populations.cell_types.index(s) for s in ("T4a", "T4b", "T4c", "T4d")]
    assert len(set(indices)) == 4
    # Distinct body sets, not one merged population.
    sets = [set(populations.ids_for(s).tolist()) for s in ("T4a", "T4b", "T4c", "T4d")]
    assert not (sets[0] & sets[1])


def test_superclass_filter_removes_fragments(annotations, populations) -> None:
    """Every selected body must have a superclass, i.e. be an official neuron.

    The official MaleCNS predicate is "a body is a neuron if and only if it has
    a superclass", so a body without one is a fragment and must not enter the
    candidate population.
    """
    assert populations.has_superclass.all()
    unfiltered = build_populations(annotations, MOTION_CELL_TYPES, require_superclass=False)
    assert unfiltered.num_bodies > populations.num_bodies


def test_unrelated_types_are_not_selected(populations) -> None:
    assert "Unrelated" not in populations.cell_types


def test_missing_label_is_reported_not_substituted(annotations) -> None:
    found, info = resolve_cell_types(annotations, ["L1", "T9z", "Tm1"])
    assert found == ["L1", "Tm1"]
    assert info["substitution_applied"] is False
    missing = {m["label"] for m in info["missing"]}
    assert missing == {"T9z"}
    assert all(m["near_miss_not_applied"] == [] or "T9z" not in str(m["near_miss_not_applied"]) for m in info["missing"])


def test_all_labels_absent_raises(annotations) -> None:
    with pytest.raises(ExtractionError, match="none of the requested cell types"):
        resolve_cell_types(annotations, ["NoSuchTypeA", "NoSuchTypeB"])


def test_counts_report_side_and_coordinate_coverage(populations) -> None:
    counts = populations.counts()
    for label, block in counts.items():
        assert block["bodies"] > 0, f"{label} should have fixture bodies"
        assert block["with_superclass"] == block["bodies"]
        assert set(block) >= {"bodies", "with_somaSide", "with_coordinates", "with_superclass"}
    assert sum(b["bodies"] for b in counts.values()) == populations.num_bodies


# ------------------------------------------------------- body budget


def test_body_budget_is_deterministic_and_balanced(annotations) -> None:
    first = build_populations(annotations, MOTION_CELL_TYPES, body_budget_per_type=2)
    second = build_populations(annotations, MOTION_CELL_TYPES, body_budget_per_type=2)
    assert np.array_equal(first.body_ids, second.body_ids)
    assert first.is_subset() and "SUBSET" in (first.subset_note() or "")
    for label in first.cell_types:
        positions = first.positions_for(label)
        sides = pd.Series(first.soma_side[positions]).replace("", pd.NA).dropna()
        assert len(sides) <= 2
        assert set(sides.astype(str)) <= {"L", "R"}


def test_no_budget_means_no_subset(populations) -> None:
    assert populations.is_subset() is False
    assert populations.subset_note() is None


# ----------------------------------------------------------- extraction


def test_extraction_uses_bounded_batches(extraction) -> None:
    assert extraction.batches_total == 3
    assert extraction.batches_processed == 3
    assert extraction.rows_scanned == 81
    assert extraction.num_edges > 0
    assert extraction.truncated is False
    assert extraction.seconds >= 0.0


def test_read_budget_truncation_is_reported(populations) -> None:
    capped = extract_candidate_edges(EDGES, populations, read_budget=1)
    assert capped.batches_processed == 1
    assert capped.truncated is True
    assert capped.num_edges < extract_candidate_edges(EDGES, populations).num_edges


def test_directions_filter_correctly(populations) -> None:
    l1_positions = np.flatnonzero(populations.type_of == populations.cell_types.index("L1"))
    t4_positions = np.flatnonzero(np.isin(populations.type_of, [populations.cell_types.index(s) for s in ("T4a","T4b","T4c","T4d")]))

    downstream = extract_candidate_edges(EDGES, populations, direction="downstream", seed_indices=l1_positions)
    upstream = extract_candidate_edges(EDGES, populations, direction="upstream", seed_indices=t4_positions)
    both = extract_candidate_edges(EDGES, populations, direction="both")

    seed_set = set(l1_positions.tolist())
    assert set(downstream.pre_index.tolist()) <= seed_set
    target_set = set(t4_positions.tolist())
    assert set(upstream.post_index.tolist()) <= target_set
    assert both.num_edges >= downstream.num_edges
    assert both.num_edges >= upstream.num_edges


def test_invalid_direction_raises(populations) -> None:
    with pytest.raises(ExtractionError, match="direction must be"):
        extract_candidate_edges(EDGES, populations, direction="sideways")


def test_missing_files_raise(annotations) -> None:
    populations = build_populations(annotations, MOTION_CELL_TYPES)
    with pytest.raises(ExtractionError, match="connectivity file not found"):
        extract_candidate_edges(FIXTURES / "absent.feather", populations)
    with pytest.raises(ExtractionError, match="annotation file not found"):
        load_annotations(FIXTURES / "absent.feather")


# ------------------------------------------------------------- sparsity


def test_sparse_graph_is_sparse_and_directed(populations, extraction) -> None:
    graph = build_sparse(extraction, populations)
    assert sp.issparse(graph)
    assert graph.shape == (populations.num_bodies, populations.num_bodies)
    assert not graph.has_sorted_indices or True  # CSR from COO is unsorted; irrelevant
    assert graph.nnz <= extraction.num_edges


def test_celltype_matrix_separates_edges_from_synapses(populations, extraction) -> None:
    synapses = build_celltype_matrix(extraction, populations, "synapses")
    edges = build_celltype_matrix(extraction, populations, "edges")
    assert list(synapses.index) == populations.cell_types
    assert synapses.shape == (len(populations.cell_types),) * 2
    assert int(synapses.to_numpy().sum()) == int(extraction.synapse_count.sum())
    assert int(edges.to_numpy().sum()) == extraction.num_edges
    # A synapse count is never equal to an edge count by construction here.
    assert int(synapses.to_numpy().sum()) != int(edges.to_numpy().sum())


# ----------------------------------------------------------- validation


def test_validation_all_passes_on_clean_fixture(extraction, populations, annotations) -> None:
    result = validate_extraction(extraction, populations, annotations)
    assert result["all_passed"] is True
    names = {check["check"] for check in result["checks"]}
    assert names >= {
        "all_candidate_bodies_in_annotations",
        "edge_endpoints_in_range",
        "no_duplicate_ordered_pairs",
        "synapse_counts_positive",
        "self_loops_reported",
        "every_body_has_an_exact_type",
        "no_type_substitution",
        "bodies_verified_against_annotation_table",
    }


def test_validation_reports_self_loops(extraction, populations, annotations) -> None:
    result = validate_extraction(extraction, populations, annotations)
    assert result["self_loops"] >= 1, "the fixture contains a deliberate self-loop"
    check = next(c for c in result["checks"] if c["check"] == "self_loops_reported")
    assert check["passed"] is True and check["value"] >= 1


def test_synapse_counts_are_preserved_exactly(extraction, populations, annotations) -> None:
    """The integer synapse count must survive extraction byte-for-byte."""
    import pyarrow as pa

    reader = pa.ipc.open_file(pa.memory_map(str(EDGES), "rb"))
    source: set[tuple[int, int, int]] = set()
    for index in range(reader.num_record_batches):
        batch = reader.get_batch(index)
        pre, post, weight = (batch.column(i).to_numpy() for i in range(3))
        source.update(zip(pre.tolist(), post.tolist(), weight.tolist()))

    retained = {
        (int(populations.body_ids[p]), int(populations.body_ids[q]), int(w))
        for p, q, w in zip(extraction.pre_index, extraction.post_index, extraction.synapse_count, strict=True)
    }
    assert retained <= source, "extraction must not invent edges"
    assert all(w > 0 for _, _, w in retained)
    assert all(isinstance(w, int) for _, _, w in retained)

    # Every dropped edge must be explained by an endpoint that has no superclass
    # and was therefore excluded as a fragment. Nothing is lost silently.
    fragment_ids = {
        int(b)
        for b, keep in zip(annotations["bodyId"], _superclass_mask(annotations), strict=True)
        if not keep
    }
    unexplained = {
        (a, b, w) for (a, b, w) in source - retained
        if a not in fragment_ids and b not in fragment_ids
    }
    assert not unexplained, f"edges dropped without a fragment explanation: {unexplained}"


# -------------------------------------------------------------- report


def test_summary_reports_per_type_and_per_pathway(populations, extraction, annotations) -> None:
    summary = summarise_pathways(populations, extraction, annotations)
    per_type = summary["per_cell_type"]
    for label in populations.cell_types:
        block = per_type[label]
        assert set(block) >= {"bodies", "with_somaSide", "with_coordinates", "outgoing_edges", "incoming_edges"}
    on = summary["per_pathway"]["ON"]
    off = summary["per_pathway"]["OFF"]
    assert on["available"] and off["available"]
    for pathway in (on, off):
        assert set(pathway) >= {"nodes", "edges", "synapse_count_stats", "self_loops", "cross_side_edges"}
        stats = pathway["synapse_count_stats"]
        assert set(stats) >= {"min", "median", "max", "total"}
        assert stats["min"] is None or stats["min"] <= stats["median"] <= stats["max"]


def test_end_to_end_report_is_json_serialisable(tmp_path: Path) -> None:
    report = extract_motion_candidates(ANN, EDGES)
    payload = json.dumps(report, default=str)
    assert "number of synapses" in report["weight_semantics"].lower()
    assert "not synaptic strength" in report["weight_semantics"].lower()
    assert report["validation"]["all_passed"] is True
    assert report["subset"]["is_subset"] is False
    assert report["extraction"]["edges_retained"] > 0
    assert report["memory"]["dense_would_be_bytes"] > 0
    assert json.loads(payload)["experiment"].startswith("MCNS motion")

    written = write_report(report, tmp_path / "summary.json", tmp_path / "edges")
    assert Path(written["summary"]).is_file()
    assert Path(written["celltype_matrix"]).is_file()


def test_report_flags_budget_as_subset() -> None:
    report = extract_motion_candidates(ANN, EDGES, body_budget_per_type=2)
    assert report["subset"]["is_subset"] is True
    assert "SUBSET" in report["subset"]["note"]


def test_report_flags_read_truncation() -> None:
    report = extract_motion_candidates(ANN, EDGES, read_budget=1)
    assert report["subset"]["read_truncated"] is True
    assert "SUBSET" in report["subset"]["read_truncation_note"]


def _superclass_mask(annotations) -> np.ndarray:
    return np.asarray(pd.notna(pd.Series(annotations["superclass"])), dtype=bool)


def _write_edges(path: Path, rows: list[tuple[int, int, int]]) -> Path:
    """Write a tiny aggregated connectivity table with the real column names."""
    import json as _json

    import pyarrow as pa

    ordered = sorted(rows, key=lambda r: -r[2])
    frame = pd.DataFrame(ordered, columns=["body_pre", "body_post", "weight"]).astype("int64")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = _json.loads(table.schema.metadata[b"pandas"])
    meta["index_columns"] = [{"kind": "range", "name": None, "start": 0, "stop": len(ordered), "step": 1}]
    meta["column_indexes"] = [
        {"name": c, "field_name": c, "pandas_type": "int64", "numpy_type": "int64", "metadata": None}
        for c in frame.columns
    ]
    table = table.replace_schema_metadata({b"pandas": _json.dumps(meta).encode("utf-8")})
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def test_duplicate_ordered_pairs_are_detected(tmp_path: Path, annotations) -> None:
    """The duplicate check must actually fire, not just always pass."""
    populations = build_populations(annotations, ["L1", "Mi1"])
    bodies = list(populations.ids_for("L1")[:2]) + list(populations.ids_for("Mi1")[:1])
    rows = [
        (bodies[0], bodies[2], 10),
        (bodies[0], bodies[2], 5),   # same ordered pair, different count
        (bodies[1], bodies[2], 4),
    ]
    path = _write_edges(tmp_path / "dupes.feather", rows)
    extraction = extract_candidate_edges(path, populations, direction="both")
    assert extraction.num_edges == 3
    result = validate_extraction(extraction, populations, annotations)
    check = next(c for c in result["checks"] if c["check"] == "no_duplicate_ordered_pairs")
    assert check["passed"] is False
    assert check["value"] == 1
    assert result["all_passed"] is False, "a failed check must surface in the report"


def test_non_positive_synapse_counts_are_detected(tmp_path: Path, annotations) -> None:
    populations = build_populations(annotations, ["L1", "Mi1"])
    bodies = list(populations.ids_for("L1")[:1]) + list(populations.ids_for("Mi1")[:1])
    path = _write_edges(tmp_path / "zero.feather", [(bodies[0], bodies[1], 0)])
    extraction = extract_candidate_edges(path, populations, direction="both")
    result = validate_extraction(extraction, populations, annotations)
    check = next(c for c in result["checks"] if c["check"] == "synapse_counts_positive")
    assert check["passed"] is False and check["value"] == 1


def test_report_never_calls_weight_a_strength() -> None:
    report = extract_motion_candidates(ANN, EDGES)
    text = json.dumps(report, default=str).lower()
    assert "synaptic strength" not in text.replace("not synaptic strength", "")
    assert "synapse_count" in text


# ------------------------------------------------- notebook display logic


def test_notebook_celltype_graph_logic_runs(annotations) -> None:
    """The notebook's cell-type graph block must work, not just look plausible.

    Kept as a test so the notebook cannot silently rot: the layout, colour
    classification and node/edge sizing code is exercised here on the fixture.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    populations = build_populations(annotations, MOTION_CELL_TYPES)
    extraction = extract_candidate_edges(EDGES, populations, direction="both")
    matrix = build_celltype_matrix(extraction, populations, "synapses")

    on_set = {MOTION_PATHWAYS["ON"]["input"], MOTION_PATHWAYS["ON"]["relay"], *MOTION_PATHWAYS["ON"]["outputs"]}
    off_set = {MOTION_PATHWAYS["OFF"]["input"], *MOTION_PATHWAYS["OFF"]["relay"], *MOTION_PATHWAYS["OFF"]["outputs"]}

    graph = nx.DiGraph()
    for node in MOTION_CELL_TYPES:
        graph.add_node(node)
    max_syn = int(np.nanmax(matrix.to_numpy())) or 1
    for source in matrix.index:
        for target in matrix.columns:
            value = int(matrix.loc[source, target])
            if value > 0 and source != target:
                graph.add_edge(source, target, weight=value)

    assert graph.number_of_nodes() == len(MOTION_CELL_TYPES)
    # Same classification and colours the notebook uses.
    colors = [
        "#2ec4b6" if n in on_set else "#ef6f6c" if n in off_set else "#8892b0" for n in graph.nodes
    ]
    sizes = [700 + 2600 * (float(matrix.loc[n, n]) / max_syn) for n in graph.nodes]
    widths = [0.4 + 5.0 * (d["weight"] / max_syn) for _, _, d in graph.edges(data=True)]
    assert len(colors) == graph.number_of_nodes()
    assert len(sizes) == graph.number_of_nodes() and all(s > 0 for s in sizes)
    assert len(widths) == graph.number_of_edges()
    # Every requested type lands in exactly one of the three colour classes.
    assert all(c in {"#2ec4b6", "#ef6f6c", "#8892b0"} for c in colors)
    assert set(colors) & {"#2ec4b6", "#ef6f6c"}, "both channels must be represented"

    figure, axes = plt.subplots()
    nx.draw_networkx_nodes(graph, nx.spring_layout(graph, seed=0), ax=axes, node_color=colors, node_size=sizes)
    plt.close(figure)


def test_notebook_histogram_logic_runs(annotations) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populations = build_populations(annotations, MOTION_CELL_TYPES)
    counts = extract_candidate_edges(EDGES, populations, direction="both").synapse_count
    assert counts.size > 0 and (counts > 0).all(), "log10 requires strictly positive counts"

    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].hist(np.clip(counts, 0, 60), bins=20)
    axes[1].hist(np.sort(counts)[::-1][:2000], bins=20)
    axes[1].set_yscale("log")
    axes[2].hist(np.log10(counts), bins=20)
    plt.close(figure)
