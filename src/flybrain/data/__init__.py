"""Biological data access layer.

This package is the *only* place where third-party data enters FlyBrain, and it
is deliberately ignorant of neural simulation: it can import neither
``flybrain.brain`` nor ``flybrain.training``.

Provenance is a first-class field. Anything produced by
:func:`synthetic_connectome` is flagged ``is_biological=False`` with
``source="synthetic"`` so it can never be mistaken for published data, and a
``TEST FIXTURE`` source is structurally incapable of claiming biological status.

The ingestion path is::

    user-provided file
        -> inspect_schema      (flybrain.data.schema)
        -> ColumnMapping       (flybrain.data.normalize)
        -> normalize_edges     (flybrain.data.normalize)
        -> validate_and_clean  (flybrain.data.validation)
        -> Connectome          (flybrain.data.connectome)
        -> BrainGraph          (flybrain.brain.graph, sparse by default)
        -> report              (flybrain.data.report)

Nothing here downloads data. The user supplies a local file.
"""

from flybrain.data.connectome import (
    Connectome,
    ConnectomeEdge,
    ConnectomeError,
    load_connectome,
    load_flywire_connectome,
    save_connectome,
    synthetic_connectome,
)
from flybrain.data.importer import (
    ConnectomeImportError,
    ImportOptions,
    ImportResult,
    import_connectome,
    iter_batches,
)
from flybrain.data.mcns_provenance import (
    MCNS_CITATION,
    MCNS_DATASET,
    MCNS_LICENSE,
    MCNS_SOURCE,
    MCNS_SOURCE_URL,
    MCNS_UNRESOLVED,
    mcns_provenance,
)
from flybrain.data.motion_candidate import (
    MOTION_CELL_TYPES,
    MOTION_PATHWAYS,
    CandidatePopulations,
    EdgeExtraction,
    ExtractionError,
    build_celltype_matrix,
    build_populations,
    build_sparse,
    extract_candidate_edges,
    extract_motion_candidates,
    validate_extraction,
    write_report,
)
from flybrain.data.normalize import (
    CANONICAL_EDGE_COLUMNS,
    CANONICAL_NODE_COLUMNS,
    OPTIONAL_EDGE_COLUMNS,
    ColumnMapping,
    normalize_edges,
    normalize_node_table,
)
from flybrain.data.provenance import (
    BIOLOGICAL_LABEL,
    SYNTHETIC_LABEL,
    TEST_FIXTURE_LABEL,
    DataSource,
    Provenance,
    ProvenanceError,
)
from flybrain.data.report import ConnectomeReport, build_connectome_report, render_connectome_report
from flybrain.data.schema import (
    ROLES,
    ColumnProfile,
    SchemaError,
    SchemaReport,
    detect_file_type,
    inspect_dataframe,
    inspect_schema,
)
from flybrain.data.validation import (
    LimitExceededError,
    ValidationConfig,
    ValidationError,
    ValidationReport,
    validate_and_clean,
)

__all__ = [
    "BIOLOGICAL_LABEL",
    "CANONICAL_EDGE_COLUMNS",
    "CANONICAL_NODE_COLUMNS",
    "MCNS_CITATION",
    "MCNS_DATASET",
    "MCNS_LICENSE",
    "MCNS_SOURCE",
    "MCNS_SOURCE_URL",
    "MCNS_UNRESOLVED",
    "MOTION_CELL_TYPES",
    "MOTION_PATHWAYS",
    "CandidatePopulations",
    "Connectome",
    "ConnectomeEdge",
    "ConnectomeError",
    "ConnectomeImportError",
    "ConnectomeReport",
    "ColumnMapping",
    "ColumnProfile",
    "DataSource",
    "EdgeExtraction",
    "ExtractionError",
    "ImportOptions",
    "ImportResult",
    "LimitExceededError",
    "OPTIONAL_EDGE_COLUMNS",
    "Provenance",
    "ProvenanceError",
    "ROLES",
    "SYNTHETIC_LABEL",
    "SchemaError",
    "SchemaReport",
    "TEST_FIXTURE_LABEL",
    "ValidationConfig",
    "ValidationError",
    "ValidationReport",
    "build_connectome_report",
    "build_celltype_matrix",
    "build_populations",
    "build_sparse",
    "detect_file_type",
    "extract_candidate_edges",
    "extract_motion_candidates",
    "import_connectome",
    "inspect_dataframe",
    "inspect_schema",
    "iter_batches",
    "load_connectome",
    "load_flywire_connectome",
    "mcns_provenance",
    "normalize_edges",
    "normalize_node_table",
    "render_connectome_report",
    "save_connectome",
    "synthetic_connectome",
    "validate_and_clean",
    "validate_extraction",
    "write_report",
]
