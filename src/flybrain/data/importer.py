"""The ingestion pipeline.

::

    USER-PROVIDED FILE
            ↓
    schema inspection
            ↓
    validation
            ↓
    normalization
            ↓
    processed connectome
            ↓
    BrainGraph
            ↓
    provenance metadata

Design commitments
------------------
* **The user supplies the file.** Nothing here downloads, scrapes, or
  authenticates against anything. A path that looks like a URL is rejected with
  an explanation rather than fetched.
* **Nothing is guessed.** If the source and target columns cannot be identified
  with confidence, the import stops and lists the candidates.
* **Nothing is invented.** Missing provenance stays ``None``; a missing weight
  column produces a flagged placeholder rather than a fabricated measurement.
* **Statistics state their scope.** A subset produced by a node/edge budget is
  labelled as a subset everywhere it appears.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from flybrain.data.connectome import Connectome, DataSource, save_connectome
from flybrain.data.normalize import (
    ColumnMapping,
    canonical_edge_columns,
    normalize_edges,
    normalize_node_table,
    summarise_canonical_frame,
)
from flybrain.data.provenance import Provenance
from flybrain.data.schema import SchemaReport, detect_file_type, inspect_schema, read_table
from flybrain.data.validation import (
    SCOPE_COMPLETE,
    SCOPE_SUBSET,
    ValidationConfig,
    ValidationError,
    ValidationReport,
    known_node_ids,
    validate_and_clean,
)

__all__ = ["ConnectomeImportError", "ImportOptions", "ImportResult", "import_connectome", "iter_batches", "top_regions"]


class ConnectomeImportError(RuntimeError):
    """Raised when a connectome cannot be imported."""


@dataclass
class ImportOptions:
    """Everything the importer needs beyond the file and the provenance."""

    column_mapping: ColumnMapping | None = None
    role_overrides: dict[str, str] = field(default_factory=dict)
    node_table_roles: dict[str, str] = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    chunksize: int | None = None
    keep_extra_columns: bool = True
    weight_is_biological: bool = True
    n_profile_rows: int = 10_000
    node_table_path: str | Path | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ImportOptions":
        if not mapping:
            return cls()
        known = set(cls.__dataclass_fields__) - {"column_mapping", "role_overrides", "validation", "node_table_roles"}
        kwargs: dict[str, Any] = {str(k): v for k, v in mapping.items() if str(k) in known}
        role_overrides = {
            str(k): str(v) for k, v in dict(mapping.get("role_overrides") or {}).items() if v
        }
        node_table_roles = {
            str(k): str(v) for k, v in dict(mapping.get("node_table_roles") or {}).items() if v
        }
        return cls(
            role_overrides=role_overrides,
            node_table_roles=node_table_roles,
            validation=ValidationConfig.from_mapping(mapping.get("validation")),
            **kwargs,
        )


@dataclass
class ImportResult:
    """Everything the pipeline learned, for reporting and tests."""

    connectome: Connectome
    schema: SchemaReport
    validation: ValidationReport
    provenance: Provenance
    column_mapping: ColumnMapping
    notes: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    node_table_path: Path | None = None

    @property
    def scope(self) -> str:
        return SCOPE_SUBSET if self.validation.subset else SCOPE_COMPLETE

    @property
    def num_nodes(self) -> int:
        return self.connectome.num_neurons

    @property
    def num_edges(self) -> int:
        return self.connectome.num_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "scope": self.scope,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "column_mapping": self.column_mapping.to_dict(),
            "schema": self.schema.to_dict(),
            "validation": self.validation.to_dict(),
            "notes": self.notes,
            "output_path": str(self.output_path) if self.output_path else None,
            "node_table_path": str(self.node_table_path) if self.node_table_path else None,
        }


def _reject_urls(path: Path) -> None:
    """Refuse anything that looks like a remote location.

    The project downloads nothing. A user who typed a URL gets told to download
    the dataset themselves, from the official source, and pass a local path.
    """
    # Check the raw string before Path() normalises it: on Windows
    # Path("https://host/x") becomes "https:\\host\\x" and the "://" test
    # would silently miss it.
    text = str(path)
    if "://" in text or text.lower().startswith(("http:", "https:", "ftp:", "s3:", "gs:")):
        raise ConnectomeImportError(
            f"refusing to fetch '{text}'.\n"
            "FlyBrain never downloads connectome data and does not implement an API client. "
            "Download the dataset manually from the official provider, save it locally, then "
            "pass the local file path."
        )
    return None


def _resolve_mapping(schema: SchemaReport, options: ImportOptions) -> ColumnMapping:
    """Resolve a column mapping, preferring an explicit user mapping."""
    if options.column_mapping is not None:
        return options.column_mapping
    return ColumnMapping.from_roles(schema.resolve(options.role_overrides or None))


def _read_edges(
    path: Path,
    file_type: str,
    mapping: ColumnMapping,
    available_columns: Sequence[str],
    options: ImportOptions,
) -> pd.DataFrame:
    """Read the mapped edge columns, bounded by the edge budget when one is set."""
    usecols = sorted(set(mapping.edge_columns().values()))
    missing = [column for column in usecols if column not in available_columns]
    if missing:
        raise ConnectomeImportError(
            f"mapped column(s) not in the file: {', '.join(missing)}. "
            f"Available columns: {', '.join(available_columns)}"
        )

    if not options.chunksize:
        budget = options.validation.max_edges
        nrows = budget if (budget and options.validation.on_limit == "subsample") else None
        frame = read_table(path, file_type=file_type, nrows=nrows, usecols=usecols)
        if not isinstance(frame, pd.DataFrame):  # pragma: no cover - chunksize is None here
            raise ConnectomeImportError("unexpected chunked read")
        return frame

    frames: list[pd.DataFrame] = []
    total = 0
    for chunk in read_table(path, file_type=file_type, usecols=usecols, chunksize=options.chunksize):
        frames.append(chunk)
        total += len(chunk)
        if options.validation.max_edges and total >= options.validation.max_edges:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)


def _infer_node_roles(frame: pd.DataFrame, name: str) -> dict[str, str | None]:
    """Infer the neuron id / region / type columns of a node table.

    A node table is a different file from the edge table and has its own column
    names, so it gets inspected independently rather than inheriting the edge
    file's mapping.
    """
    from flybrain.data.schema import inspect_dataframe

    report = inspect_dataframe(frame, file=name, file_type="node-table")
    detected = report.detected_roles
    return {
        "neuron_id": detected.get("neuron_id"),
        "region": detected.get("region"),
        "neuron_type": detected.get("neuron_type"),
    }


def _load_node_table(
    path: Path | None,
    overrides: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, Path | None, dict[str, str | None]]:
    """Load an optional neuron-metadata table with its own column mapping."""
    empty = pd.DataFrame(columns=["neuron_id", "region", "neuron_type"])
    if path is None:
        return empty, None, {}
    resolved = Path(path)
    _reject_urls(resolved)
    if not resolved.is_file():
        raise ConnectomeImportError(f"node table not found: {resolved}")
    kind = detect_file_type(resolved)
    raw = read_table(resolved, file_type=kind)
    if not isinstance(raw, pd.DataFrame):
        raise ConnectomeImportError(f"could not read node table {resolved.name} as a table")

    roles = _infer_node_roles(raw, resolved.name)
    for role, column in (overrides or {}).items():
        if column not in raw.columns:
            raise ConnectomeImportError(
                f"node table column {column!r} (for role {role!r}) is not in {resolved.name}. "
                f"Available columns: {', '.join(str(c) for c in raw.columns)}"
            )
        roles[role] = column

    if not roles.get("neuron_id"):
        raise ConnectomeImportError(
            f"could not identify the neuron id column in {resolved.name}. "
            f"Available columns: {', '.join(str(c) for c in raw.columns)}. "
            "Pass it explicitly, e.g. --node-id-column <name>."
        )
    return normalize_node_table(raw, roles), resolved, roles


def _iter_frames(frame: pd.DataFrame, chunksize: int | None) -> Iterator[pd.DataFrame]:
    if not chunksize or chunksize >= len(frame):
        yield frame
        return
    for start in range(0, len(frame), chunksize):
        yield frame.iloc[start : start + chunksize]


def import_connectome(
    path: str | Path,
    provenance: Provenance,
    options: ImportOptions | None = None,
    output_dir: str | Path | None = None,
) -> ImportResult:
    """Import a user-provided connectome file.

    Parameters
    ----------
    path:
        A local file: CSV, TSV, Parquet, JSON or JSON Lines.
    provenance:
        Attribution supplied by the user. Missing fields stay ``None``.
    options:
        Column mapping, validation policy and memory limits.
    output_dir:
        When given, writes ``<dataset_name>.csv`` plus the provenance sidecar
        and a ``<dataset_name>.nodes.csv`` neuron table.

    Returns
    -------
    ImportResult
    """
    settings = options or ImportOptions()
    source_path = Path(path)
    _reject_urls(source_path)

    if provenance.original_filename is None:
        provenance.original_filename = source_path.name

    file_type = detect_file_type(source_path)
    schema = inspect_schema(source_path, nrows=settings.n_profile_rows)
    mapping = _resolve_mapping(schema, settings)

    raw = _read_edges(source_path, file_type, mapping, schema.column_names, settings)
    if raw.empty:
        raise ValidationError(f"{source_path.name} contains a header but no edge rows")

    # The edge budget can be enforced by not reading the whole file. That is the
    # memory-efficient path, but it must never be silent: a truncated read would
    # otherwise be reported as the complete dataset.
    truncated_by_read = bool(
        settings.validation.max_edges
        and schema.row_count is not None
        and schema.row_count > len(raw)
    )

    node_table, node_path, node_roles = _load_node_table(settings.node_table_path, settings.node_table_roles)

    # Normalise in chunks, then validate the whole table. Cross-row checks
    # (duplicates, budgets) cannot be done per chunk, so the canonical frame is
    # accumulated; chunking therefore bounds parse memory and enables early exit
    # on the edge budget rather than eliminating the final copy.
    canonical_parts: list[pd.DataFrame] = []
    notes: dict[str, Any] = {"chunks": []}
    for part in _iter_frames(raw, settings.chunksize):
        canonical, part_notes = normalize_edges(
            part,
            mapping,
            weight_is_biological=settings.weight_is_biological,
            keep_extra_columns=settings.keep_extra_columns,
        )
        canonical_parts.append(canonical)
        if "invalid_weight_values" in part_notes:
            notes["invalid_weight_values"] = notes.get("invalid_weight_values", 0) + part_notes["invalid_weight_values"]
        notes["placeholder_weight"] = part_notes["placeholder_weight"]
        notes["weight_is_biological"] = part_notes["weight_is_biological"]
        notes.setdefault("weight_column", part_notes.get("weight_column"))
        notes["chunks"].append(len(part))
        if settings.validation.max_edges and sum(len(f) for f in canonical_parts) >= settings.validation.max_edges:
            break

    canonical = (
        pd.concat(canonical_parts, ignore_index=True) if len(canonical_parts) > 1 else canonical_parts[0]
    )
    del canonical_parts, raw

    clean, report = validate_and_clean(canonical, settings.validation, known_node_ids(node_table))
    del canonical

    if truncated_by_read and not report.subset:
        report.subset = True
        report.scope = SCOPE_SUBSET
        report.subset_reason = f"max_edges={settings.validation.max_edges} (file has ~{schema.row_count} rows)"
        report.warnings.append(
            "reading stopped at the edge budget, so these statistics describe a SUBSET, "
            "not the complete source dataset"
        )

    regions = _regions_from_nodes(clean, node_table)
    connectome = Connectome.from_normalized(
        frame=clean,
        name=provenance.dataset_name,
        provenance=provenance,
        node_regions=regions,
        is_subset=report.subset,
        subset_reason=report.subset_reason,
    )

    result = ImportResult(
        connectome=connectome,
        schema=schema,
        validation=report,
        provenance=provenance,
        column_mapping=mapping,
        notes={**notes, **summarise_canonical_frame(clean)},
        node_table_path=node_path,
    )
    if node_roles:
        result.notes["node_table_roles"] = node_roles

    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(provenance.dataset_name)
        result.output_path = save_connectome(connectome, target_dir / f"{stem}.csv")
        if not node_table.empty:
            nodes_path = target_dir / f"{stem}.nodes.csv"
            node_table.to_csv(nodes_path, index=False)
            result.node_table_path = nodes_path
        result.notes["canonical_columns"] = canonical_edge_columns()

    return result


def _regions_from_nodes(edges: pd.DataFrame, node_table: pd.DataFrame) -> dict[str, str] | None:
    """Neuron -> region map from the node table, when the source supplied one."""
    if node_table is None or node_table.empty or "region" not in node_table.columns:
        return None
    regions = node_table.dropna(subset=["region"])
    if regions.empty:
        return None
    return dict(zip(regions["neuron_id"].astype(str), regions["region"].astype(str), strict=True))


def _safe_stem(name: str) -> str:
    """Filesystem-safe stem derived from the dataset name."""
    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in str(name))
    cleaned = cleaned.strip("-._") or "connectome"
    return cleaned.lower()


def write_node_table(node_table: pd.DataFrame, path: str | Path) -> Path:
    """Write a canonical neuron-metadata table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    node_table.to_csv(target, index=False)
    return target


def iter_batches(connectome: Connectome, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield the canonical edges of a connectome in row batches.

    The memory-safe way to walk a large imported connectome: the frame is never
    duplicated, and callers can aggregate incrementally.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frame = connectome.edge_frame
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def summarize_source(result: ImportResult, top_n: int = 10) -> dict[str, Any]:
    """Compact, log-friendly summary of an import."""
    return {
        "dataset": result.provenance.dataset_name,
        "source": result.provenance.source_name,
        "is_biological": result.provenance.is_biological,
        "scope": result.scope,
        "nodes": result.num_nodes,
        "edges": result.num_edges,
        "removed": result.validation.total_removed,
        "placeholder_weight": result.notes.get("placeholder_weight", False),
        "top_regions": top_regions(result.connectome, top_n),
    }


def top_regions(connectome: Connectome, top_n: int = 10) -> list[tuple[str, int]]:
    """Most common region labels, when the source provided them."""
    regions = connectome.regions()
    if not regions:
        return []
    counts: dict[str, int] = {}
    for region in regions.values():
        counts[region] = counts.get(region, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
