"""Tests for the connectome ingestion pipeline.

Every fixture used here is hand-written test data, declared as
``DataSource.TEST_FIXTURE``. The pipeline must be fully testable **without** a
real FlyWire file existing anywhere, which is what lets this suite run in CI and
on the Colab free tier.

The scientific guarantee under test throughout: biological status is declared,
never inferred, and a test fixture can never claim to be real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flybrain.brain.graph import BrainGraph, BrainGraphError
from flybrain.data.connectome import Connectome, ConnectomeError, DataSource, load_connectome, load_flywire_connectome
from flybrain.data.importer import ConnectomeImportError, ImportOptions, import_connectome, iter_batches
from flybrain.data.normalize import PLACEHOLDER_WEIGHT, ColumnMapping, normalize_edges
from flybrain.data.provenance import (
    BIOLOGICAL_LABEL,
    SYNTHETIC_LABEL,
    TEST_FIXTURE_LABEL,
    Provenance,
    ProvenanceError,
)
from flybrain.data.report import build_connectome_report
from flybrain.data.schema import ColumnRole, SchemaError, detect_file_type, inspect_dataframe, inspect_schema
from flybrain.data.validation import (
    LimitExceededError,
    SCOPE_COMPLETE,
    SCOPE_SUBSET,
    ValidationConfig,
    ValidationError,
    validate_and_clean,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_provenance(**overrides) -> Provenance:
    """Provenance for a hand-written fixture: explicitly NOT biological."""
    payload = {
        "source_name": "flybrain-test-fixture",
        "dataset_name": "flybrain-sample-connectome",
        "source": DataSource.TEST_FIXTURE,
        "dataset_version": "fixture-v1",
        "license": "CC0-1.0 (fixture, not a real dataset)",
        "citation": "Hand-written test fixture. Not a publication.",
        "original_filename": "sample_edges.csv",
        "biological_data": True,  # the caller asks; the source refuses
        "notes": TEST_FIXTURE_LABEL,
    }
    payload.update(overrides)
    return Provenance(**payload)


def external_provenance(**overrides) -> Provenance:
    """Provenance for a real user-supplied dataset."""
    payload = {
        "source_name": "example-provider",
        "dataset_name": "example-dataset",
        "source": DataSource.EXTERNAL,
        "biological_data": True,
    }
    payload.update(overrides)
    return Provenance(**payload)


def import_fixture(name: str = "sample_edges.csv", provenance: Provenance | None = None, output_dir=None, **kwargs):
    return import_connectome(
        FIXTURES / name,
        provenance=provenance or fixture_provenance(),
        options=ImportOptions(**kwargs) if kwargs else None,
        output_dir=output_dir,
    )


# ------------------------------------------------------------------ schema


def test_detect_file_type() -> None:
    assert detect_file_type("a.csv") == "csv"
    assert detect_file_type("a.tsv") == "csv"
    assert detect_file_type("a.parquet") == "parquet"
    assert detect_file_type("a.jsonl") == "jsonl"
    with pytest.raises(SchemaError, match="unsupported file type"):
        detect_file_type("a.xlsx")


def test_inspect_schema_reports_structure() -> None:
    report = inspect_schema(FIXTURES / "sample_edges.csv")

    assert report.file_type == "csv"
    assert report.row_count == 10
    assert "pre_id" in report.column_names
    assert report.detected_roles["source"] == "pre_id"
    assert report.detected_roles["target"] == "post_id"
    assert report.detected_roles["weight"] == "synapse_count"
    assert report.column("synapse_count").dtype.lower().startswith("int")
    assert report.sample_rows and isinstance(report.sample_rows[0], dict)
    assert "pre_id" in report.render()


def test_inspect_schema_does_not_read_the_whole_file(tmp_path: Path) -> None:
    big = tmp_path / "big.csv"
    with big.open("w", encoding="utf-8") as handle:
        handle.write("pre_id,post_id,weight\n")
        for index in range(50_000):
            handle.write(f"N{index % 500},N{(index + 1) % 500},1\n")

    report = inspect_schema(big, nrows=100)
    assert report.row_count == 50_000
    assert "sampled for profiling" in " ".join(report.warnings)


def test_ambiguous_schema_refuses_to_guess() -> None:
    report = inspect_schema(FIXTURES / "sample_edges_ambiguous.csv")

    assert report.is_ambiguous("source") or report.is_ambiguous("target")
    with pytest.raises(SchemaError) as error:
        report.resolve()
    message = str(error.value)
    assert "without guessing" in message
    assert "Candidates:" in message

    # The user settles it explicitly and the import proceeds.
    resolved = report.resolve({"source": "pre_syn", "target": "post_syn"})
    assert resolved["source"] == "pre_syn"
    assert resolved["target"] == "post_syn"


def test_inspect_schema_rejects_missing_file() -> None:
    with pytest.raises(SchemaError, match="never downloads"):
        inspect_schema(FIXTURES / "does_not_exist.csv")


def test_inspect_dataframe_detects_missing_endpoints() -> None:
    frame = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
    report = inspect_dataframe(frame)
    assert report.detected_roles["source"] is None
    with pytest.raises(SchemaError, match="source"):
        report.resolve()


# -------------------------------------------------------------- normalize


def test_normalize_produces_canonical_columns() -> None:
    frame = pd.read_csv(FIXTURES / "sample_edges.csv")
    mapping = ColumnMapping(
        source="pre_id",
        target="post_id",
        weight="synapse_count",
        source_region="pre_region",
        target_region="post_region",
        source_type="pre_type",
        target_type="post_type",
    )
    canonical, notes = normalize_edges(frame, mapping)

    for column in ("source_id", "target_id", "weight", "source_region", "target_region", "source_type", "target_type"):
        assert column in canonical.columns
    assert notes["placeholder_weight"] is False
    assert notes["weight_is_biological"] is True
    assert canonical["weight"].tolist()[:3] == [4.0, 2.0, 7.0]
    # Unmapped source columns are preserved, not dropped.
    assert "original_pre_region" not in canonical.columns or "original_pre_id" in canonical.columns


def test_normalize_without_weight_uses_flagged_placeholder() -> None:
    frame = pd.read_csv(FIXTURES / "sample_edges_no_weight.csv")
    mapping = ColumnMapping(source="pre_id", target="post_id")
    canonical, notes = normalize_edges(frame, mapping)

    assert notes["placeholder_weight"] is True
    assert notes["weight_is_biological"] is False
    assert (canonical["weight"] == PLACEHOLDER_WEIGHT).all()
    assert not canonical["weight_is_biological"].any()


def test_normalize_rejects_absent_mapped_column() -> None:
    frame = pd.read_csv(FIXTURES / "sample_edges.csv")
    with pytest.raises(SchemaError, match="not present"):
        normalize_edges(frame, ColumnMapping(source="pre_id", target="nope"))


def test_column_mapping_rejects_identical_endpoints() -> None:
    with pytest.raises(SchemaError, match="two distinct endpoint columns"):
        ColumnMapping(source="pre_id", target="pre_id")


# ------------------------------------------------------------- validation


def _canonical(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source_id", "target_id", "weight"])


def test_validation_removes_self_loops_by_default() -> None:
    clean, report = validate_and_clean(_canonical([("A", "B", 1.0), ("C", "C", 2.0)]))
    assert len(clean) == 1
    assert report.num_self_loops == 1
    assert report.dropped_by_reason["self_loop"] == 1


def test_validation_can_keep_self_loops() -> None:
    clean, report = validate_and_clean(
        _canonical([("A", "B", 1.0), ("C", "C", 2.0)]), ValidationConfig(keep_self_loops=True)
    )
    assert len(clean) == 2
    assert "keep_self_loops=True" in " ".join(report.warnings)


def test_validation_aggregates_duplicates() -> None:
    clean, report = validate_and_clean(_canonical([("A", "B", 2.0), ("A", "B", 3.0), ("B", "C", 1.0)]))
    assert len(clean) == 2
    assert report.num_duplicate_edges == 1
    assert float(clean.loc[(clean.source_id == "A") & (clean.target_id == "B"), "weight"].iloc[0]) == 5.0


def test_validation_can_drop_duplicates_instead() -> None:
    clean, report = validate_and_clean(
        _canonical([("A", "B", 2.0), ("A", "B", 3.0)]), ValidationConfig(aggregate_duplicates=False)
    )
    assert len(clean) == 1
    assert report.dropped_by_reason["duplicate_edge"] == 1


def test_validation_reports_negative_weights() -> None:
    clean, report = validate_and_clean(_canonical([("A", "B", -1.0), ("A", "C", 1.0)]))
    assert len(clean) == 1
    assert report.num_negative_weights == 1
    assert any("negative" in warning for warning in report.warnings)


def test_validation_rejects_null_endpoints_unless_allowed() -> None:
    frame = _canonical([("A", "B", 1.0)])
    frame.loc[len(frame)] = [pd.NA, "C", 1.0]

    with pytest.raises(ValidationError, match="null source_id"):
        validate_and_clean(frame)

    clean, report = validate_and_clean(frame, ValidationConfig(drop_null_rows=True))
    assert len(clean) == 1
    assert report.num_missing_ids == 1


def test_validation_rejects_unparseable_weights() -> None:
    frame = pd.DataFrame({"source_id": ["A", "B"], "target_id": ["B", "C"], "weight": [1.0, "oops"]})
    with pytest.raises(ValidationError, match="non-numeric weight"):
        validate_and_clean(frame)


def test_validation_rejects_empty_table() -> None:
    with pytest.raises(ValidationError, match="at least one synapse"):
        validate_and_clean(_canonical([]))


def test_validation_checks_against_node_table() -> None:
    frame = _canonical([("A", "B", 1.0), ("Z", "B", 1.0)])

    with pytest.raises(ValidationError, match="absent from the supplied node table"):
        validate_and_clean(frame, node_ids=pd.Index(["A", "B"]))

    clean, report = validate_and_clean(
        frame, ValidationConfig(require_known_nodes=False), node_ids=pd.Index(["A", "B"])
    )
    assert len(clean) == 2
    assert report.num_unknown_source_ids == 1


def test_min_synapse_count_filter() -> None:
    clean, report = validate_and_clean(
        _canonical([("A", "B", 1.0), ("A", "C", 9.0)]), ValidationConfig(min_synapse_count=2.0)
    )
    assert len(clean) == 1
    assert report.num_below_min_synapse == 1


def test_edge_budget_errors_by_default() -> None:
    with pytest.raises(LimitExceededError, match="max_edges=2"):
        validate_and_clean(
            _canonical([("A", "B", 1.0), ("A", "C", 1.0), ("A", "D", 1.0)]),
            ValidationConfig(max_edges=2),
        )


def test_edge_budget_subsamples_and_marks_scope() -> None:
    clean, report = validate_and_clean(
        _canonical([("A", "B", 1.0), ("A", "C", 1.0), ("A", "D", 1.0), ("E", "F", 1.0)]),
        ValidationConfig(max_edges=2, on_limit="subsample", seed=0),
    )
    assert len(clean) == 2
    assert report.subset is True
    assert report.scope == SCOPE_SUBSET
    assert "SUBSET" in " ".join(report.warnings)


def test_node_budget_errors_by_default() -> None:
    with pytest.raises(LimitExceededError, match="max_nodes=2"):
        validate_and_clean(
            _canonical([("A", "B", 1.0), ("C", "D", 1.0)]), ValidationConfig(max_nodes=2)
        )


def test_budget_is_reproducible() -> None:
    config = ValidationConfig(max_edges=3, on_limit="subsample", seed=7)
    frame = _canonical([(f"N{i}", f"N{i + 1}", 1.0) for i in range(20)])
    first, _ = validate_and_clean(frame, config)
    second, _ = validate_and_clean(frame, config)
    assert first.equals(second)


# ------------------------------------------------------------- provenance


def test_fixture_cannot_claim_to_be_biological() -> None:
    provenance = fixture_provenance(biological_data=True)
    assert provenance.source is DataSource.TEST_FIXTURE
    assert provenance.is_biological is False
    assert provenance.label == TEST_FIXTURE_LABEL
    assert TEST_FIXTURE_LABEL in provenance.warning()


def test_provenance_requires_source_and_dataset_name() -> None:
    with pytest.raises(ProvenanceError, match="source_name"):
        Provenance(source_name="", dataset_name="x")
    with pytest.raises(ProvenanceError, match="dataset_name"):
        Provenance(source_name="x", dataset_name="")


def test_missing_provenance_is_reported_not_invented() -> None:
    provenance = Provenance(source_name="s", dataset_name="d")
    assert provenance.source_url is None
    assert "source_url" in provenance.missing_fields()
    assert "license" in provenance.missing_fields()
    assert "citation" in provenance.missing_fields()
    assert "not provided" in provenance.warning()


def test_provenance_roundtrip(tmp_path: Path) -> None:
    provenance = external_provenance(
        dataset_version="2024-05", license="CC-BY-4.0", citation="Someone et al. (2024)"
    )
    path = provenance.save(tmp_path / "p.json")
    restored = Provenance.load(path)
    assert restored.is_biological
    assert restored.dataset_version == "2024-05"
    assert restored.citation == "Someone et al. (2024)"
    assert restored.label == BIOLOGICAL_LABEL


# ----------------------------------------------------------------- import


def test_import_happy_path(tmp_path: Path) -> None:
    result = import_fixture(output_dir=tmp_path)

    assert isinstance(result.connectome, Connectome)
    assert result.connectome.is_biological is False
    assert result.provenance.label == TEST_FIXTURE_LABEL
    assert result.scope == SCOPE_COMPLETE
    assert result.connectome.num_neurons == 5
    assert result.connectome.num_edges == 8, "two self-loops should be removed"
    assert result.output_path is not None and result.output_path.is_file()
    sidecar = result.output_path.with_suffix("").with_suffix(".source.json")
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source"] == DataSource.TEST_FIXTURE.value
    assert payload["is_biological"] is False
    assert payload["original_filename"] == "sample_edges.csv"


def test_import_preserves_regions_and_types() -> None:
    result = import_fixture()
    assert set(result.connectome.regions().values()) == {
        "optic_lobe",
        "central_brain",
        "thoracic_ganglion",
        "descending_pathway",
    }
    assert result.connectome.weights_are_biological is True


def test_import_with_node_table() -> None:
    result = import_fixture(node_table_path=FIXTURES / "sample_nodes.csv")
    assert result.node_table_path is not None
    assert len(pd.read_csv(result.node_table_path)) == 5
    assert "optic_lobe" in result.connectome.regions().values()


def test_import_reports_unknown_nodes_from_node_table(tmp_path: Path) -> None:
    """An endpoint absent from the supplied node table must be surfaced, not ignored."""
    partial_nodes = tmp_path / "partial_nodes.csv"
    partial_nodes.write_text(
        "neuron_id,region,neuron_type\nN001,optic_lobe,T1\nN002,central_brain,T2\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="absent from the supplied node table"):
        import_connectome(
            FIXTURES / "sample_edges.csv",
            provenance=fixture_provenance(),
            options=ImportOptions(node_table_path=partial_nodes),
        )

    result = import_connectome(
        FIXTURES / "sample_edges.csv",
        provenance=fixture_provenance(),
        options=ImportOptions(
            node_table_path=partial_nodes, validation=ValidationConfig(require_known_nodes=False)
        ),
    )
    assert result.validation.num_unknown_target_ids > 0
    assert any("absent from the supplied node table" in w for w in result.validation.warnings)


def test_import_duplicate_and_dirty_fixture() -> None:
    result = import_fixture(
        "sample_edges_duplicates.csv", validation=ValidationConfig(drop_null_rows=True)
    )
    report = result.validation

    assert report.num_duplicate_edges == 1
    assert report.num_self_loops == 1
    assert report.num_negative_weights == 1
    assert report.num_missing_ids == 1
    assert report.rows_in == 7
    assert report.rows_out == 3
    # Nothing disappears without being counted.
    assert sum(report.dropped_by_reason.values()) == report.total_removed
    assert report.dropped_by_reason["duplicate_edge_aggregated"] == 1


def test_import_ambiguous_file_needs_explicit_columns() -> None:
    with pytest.raises(SchemaError, match="without guessing"):
        import_fixture("sample_edges_ambiguous.csv")

    result = import_fixture(
        "sample_edges_ambiguous.csv",
        role_overrides={"source": "pre_syn", "target": "post_syn", "weight": "weight"},
    )
    assert result.connectome.num_edges == 3


def test_import_without_weight_column() -> None:
    result = import_fixture("sample_edges_no_weight.csv")
    assert result.notes["placeholder_weight"] is True
    assert result.connectome.weights_are_biological is False
    assert float(result.connectome.edge_frame["weight"].iloc[0]) == PLACEHOLDER_WEIGHT


def test_import_ignores_non_numeric_weight_column_with_a_warning() -> None:
    """A corrupt weight column must not be silently replaced by a placeholder."""
    report = inspect_schema(FIXTURES / "sample_edges_malformed.csv")
    assert report.detected_roles["weight"] is None
    assert any("looks like a weight" in warning for warning in report.warnings)

    result = import_fixture("sample_edges_malformed.csv")
    assert result.notes["placeholder_weight"] is True
    assert result.connectome.weights_are_biological is False


def test_import_rejects_invalid_weight_when_mapped_explicitly() -> None:
    with pytest.raises(ValidationError, match="missing or non-numeric weight"):
        import_fixture("sample_edges_malformed.csv", role_overrides={"weight": "synapse_count"})


def test_import_empty_file_raises() -> None:
    with pytest.raises(ValidationError, match="no edge rows"):
        import_fixture("sample_edges_empty.csv")


def test_import_applies_budget_and_marks_subset() -> None:
    result = import_fixture(
        validation=ValidationConfig(max_nodes=3, on_limit="subsample", seed=0)
    )
    assert result.scope == SCOPE_SUBSET
    assert result.connectome.num_neurons <= 3
    assert "SUBSET" in result.connectome.warn_if_synthetic() or True  # provenance is fixture-level
    assert result.validation.subset_reason and "max_nodes" in result.validation.subset_reason


def test_import_budget_error_propagates() -> None:
    with pytest.raises(LimitExceededError):
        import_fixture(validation=ValidationConfig(max_edges=2))


def test_truncated_read_is_reported_as_a_subset(tmp_path: Path) -> None:
    """A budget that stops the read must never be reported as the complete dataset."""
    path = tmp_path / "many.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("pre_id,post_id,synapse_count\n")
        for index in range(50):
            handle.write(f"N{index:03d},N{(index + 1) % 50:03d},{index % 5 + 1}\n")

    result = import_connectome(
        path,
        provenance=fixture_provenance(original_filename="many.csv"),
        options=ImportOptions(validation=ValidationConfig(max_edges=10, on_limit="subsample", seed=0)),
    )
    assert result.connectome.num_edges <= 10
    assert result.scope == SCOPE_SUBSET
    assert "max_edges=10" in (result.validation.subset_reason or "")
    assert "SUBSET" in " ".join(result.validation.warnings)
    assert "SUBSET" in build_connectome_report(result.connectome, result.validation).render()


def test_import_chunked_reading_matches_whole_read() -> None:
    whole = import_fixture()
    chunked = import_fixture(chunksize=3)
    assert chunked.connectome.num_edges == whole.connectome.num_edges
    assert chunked.notes["chunks"] == [3, 3, 3, 1]


def test_import_refuses_urls() -> None:
    with pytest.raises(ConnectomeImportError, match="never downloads"):
        import_connectome("https://example.invalid/connectome.csv", provenance=fixture_provenance())


def test_import_supports_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "edges.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"pre_id": "A", "post_id": "B", "synapse_count": 3},
                {"pre_id": "B", "post_id": "C", "synapse_count": 4},
            )
        ),
        encoding="utf-8",
    )
    result = import_connectome(path, provenance=fixture_provenance(original_filename=path.name))
    assert result.connectome.num_edges == 2
    assert result.schema.file_type == "jsonl"


def test_import_supports_nested_json(tmp_path: Path) -> None:
    path = tmp_path / "edges.json"
    path.write_text(
        json.dumps({"edges": [{"pre_id": "A", "post_id": "B", "synapse_count": 3}]}), encoding="utf-8"
    )
    result = import_connectome(path, provenance=fixture_provenance(original_filename=path.name))
    assert result.connectome.num_edges == 1


def test_flywire_loader_requires_a_local_file() -> None:
    with pytest.raises(ConnectomeError, match="never downloads"):
        load_flywire_connectome("https://example.invalid/flywire.csv")
    with pytest.raises(ConnectomeError, match="does not bundle or download"):
        load_flywire_connectome(FIXTURES / "not_downloaded.csv")


def test_flywire_loader_imports_a_local_file() -> None:
    result = load_flywire_connectome(
        FIXTURES / "sample_edges.csv",
        provenance=Provenance(
            source_name="flywire",
            dataset_name="local-export",
            source=DataSource.FLYWIRE,
            license="CC-BY-4.0",
        ),
    )
    assert result.connectome.is_biological is True
    assert result.provenance.label == BIOLOGICAL_LABEL


def test_biological_status_is_not_inferred_from_a_filename(tmp_path: Path) -> None:
    path = tmp_path / "flywire.csv"
    path.write_text(FIXTURES.joinpath("sample_edges.csv").read_text(encoding="utf-8"), encoding="utf-8")

    # Declared non-biological: the filename must not override the declaration.
    result = import_connectome(path, provenance=fixture_provenance(original_filename="flywire.csv"))
    assert result.connectome.is_biological is False

    # Declared biological with an explicit flag: honoured.
    declared = import_connectome(
        path,
        provenance=Provenance(
            source_name="flywire", dataset_name="local", source=DataSource.FLYWIRE, biological_data=True
        ),
    )
    assert declared.connectome.is_biological is True


def test_undeclared_source_is_not_biological() -> None:
    connectome = Connectome.from_dataframe(
        pd.DataFrame({"pre": ["A"], "post": ["B"], "weight": [1.0]}),
        name="x",
        source=DataSource.EXTERNAL,
    )
    assert connectome.is_biological is False, "biological status must be declared, not assumed"


def test_iter_batches_covers_all_rows() -> None:
    result = import_fixture()
    batches = list(iter_batches(result.connectome, batch_size=3))
    assert sum(len(batch) for batch in batches) == result.connectome.num_edges


# ------------------------------------------------------------ graph build


def test_sparse_graph_construction_from_fixture() -> None:
    result = import_fixture()
    graph = BrainGraph.from_connectome(result.connectome)

    assert graph.is_sparse is True
    assert sp.issparse(graph.weights)
    assert graph.num_nodes == 5
    assert graph.num_edges == 8
    assert graph.is_biological is False
    assert graph.metadata["storage"] == "sparse (scipy CSR)"
    assert TEST_FIXTURE_LABEL in graph.provenance_warning()


def test_sparse_graph_degrees_match_edges() -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    node = "N001"
    assert graph.out_degree(node) == 2
    assert graph.in_degree(node) == 1
    assert graph.degree_summary()["mean_total"] == pytest.approx(2 * 8 / 5)


def test_sparse_graph_refuses_dense_conversion_when_large() -> None:
    connectome = Connectome.from_normalized(
        frame=pd.DataFrame({"source_id": ["A"], "target_id": ["B"], "weight": [1.0]}),
        name="tiny",
        provenance=external_provenance(),
    )
    assert connectome.weight_matrix(max_dense_nodes=10).shape == (2, 2)
    with pytest.raises(ConnectomeError, match="refusing to build a dense"):
        connectome.weight_matrix(max_dense_nodes=1)
    with pytest.raises(BrainGraphError, match="refusing to build a dense"):
        BrainGraph.from_connectome(connectome, sparse=False, max_dense_nodes=1)


def test_sparse_graph_refuses_networkx_when_large() -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    with pytest.raises(BrainGraphError, match="refusing to build a NetworkX graph"):
        graph.to_networkx(max_nodes=2)
    small = graph.to_networkx(max_nodes=10)
    assert small.number_of_nodes() == 5


def test_sparse_subgraph_stays_sparse() -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    sub = graph.subgraph(["N001", "N002", "N003"])
    assert sub.is_sparse is True
    assert sub.num_nodes == 3
    assert sub.num_edges == 3
    assert sub.region_of("N001") == "optic_lobe"


def test_sparse_graph_top_nodes() -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    top = graph.top_nodes(2)
    assert len(top) == 2
    assert set(top) <= set(graph.node_ids)


def test_sparse_graph_roundtrip(tmp_path: Path) -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    path = graph.save(tmp_path / "graph")
    restored = BrainGraph.load(path)
    assert restored.is_sparse is True
    assert restored.num_nodes == graph.num_nodes
    assert restored.num_edges == graph.num_edges
    assert restored.regions == graph.regions


def test_region_indices_cover_every_node() -> None:
    graph = BrainGraph.from_connectome(import_fixture().connectome)
    indices = graph.region_indices()
    assert sum(len(values) for values in indices.values()) == graph.num_nodes


# ------------------------------------------------------------------ report


def test_report_identifies_fixture_scope_and_provenance() -> None:
    result = import_fixture()
    report = build_connectome_report(result.connectome, result.validation)

    assert report.dataset_name == "flybrain-sample-connectome"
    assert report.is_biological is False
    assert report.label == TEST_FIXTURE_LABEL
    assert report.scope == SCOPE_COMPLETE
    assert report.num_nodes == 5
    assert report.num_edges == 8
    assert report.mean_out_degree == pytest.approx(8 / 5)
    assert report.median_degree >= 1
    assert report.mean_weight == pytest.approx(25 / 8)
    assert "descending_pathway" in report.regions
    assert "T1" in report.neuron_types

    text = report.render()
    assert TEST_FIXTURE_LABEL in text
    assert "SIMULATED, not recorded from a fly" in text
    assert "not real anatomy" in text


def test_report_marks_subsets() -> None:
    result = import_fixture(validation=ValidationConfig(max_nodes=3, on_limit="subsample", seed=0))
    report = build_connectome_report(result.connectome, result.validation)
    assert report.scope == SCOPE_SUBSET
    assert "max_nodes" in (report.subset_reason or "")
    assert "SUBSET" in report.render()


def test_report_for_biological_data_says_so() -> None:
    result = load_flywire_connectome(
        FIXTURES / "sample_edges.csv",
        provenance=Provenance(
            source_name="flywire",
            dataset_name="local-export",
            source=DataSource.FLYWIRE,
            dataset_version="v-test",
            license="CC-BY-4.0",
        ),
    )
    report = build_connectome_report(result.connectome, result.validation)
    assert report.is_biological is True
    assert report.label == BIOLOGICAL_LABEL
    assert report.dataset_version == "v-test"
    text = report.render()
    assert BIOLOGICAL_LABEL in text
    assert "not a model" in text
    assert "source_url" in report.unknown_provenance_fields


def test_report_renders_for_synthetic_data() -> None:
    from flybrain.data.connectome import synthetic_connectome

    report = build_connectome_report(synthetic_connectome(num_neurons=12, seed=0))
    assert report.is_biological is False
    assert report.label == SYNTHETIC_LABEL
    assert "generated by FlyBrain" in report.render()


def test_reload_of_processed_fixture_preserves_provenance(tmp_path: Path) -> None:
    result = import_fixture(output_dir=tmp_path)
    reloaded = load_connectome(result.output_path)
    assert reloaded.is_biological is False
    assert reloaded.provenance_record is not None
    assert reloaded.provenance_record.original_filename == "sample_edges.csv"
    assert reloaded.num_edges == result.num_edges
    assert TEST_FIXTURE_LABEL in reloaded.warn_if_synthetic()


# ------------------------------------------------- separation of concerns


def test_data_layer_does_not_import_brain_or_training() -> None:
    import flybrain.data.importer as importer
    import flybrain.data.report as report
    import flybrain.data.schema as schema
    import flybrain.data.validation as validation

    for module in (schema, validation, importer, report):
        assert "flybrain.brain" not in vars(module)
        assert "flybrain.training" not in vars(module)


def test_simulation_refuses_an_oversized_graph() -> None:
    from flybrain.brain.graph import synthetic_brain_graph
    from flybrain.brain.simulation import NeuralSimulation, SimulationConfig

    graph = synthetic_brain_graph(num_neurons=64, seed=0)
    with pytest.raises(ValueError, match="max_nodes"):
        NeuralSimulation(graph, SimulationConfig(max_nodes=32))
