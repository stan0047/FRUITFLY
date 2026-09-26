"""Schema inspection for user-provided connectome files.

The importer must not guess. Real exports differ in column naming, in whether
they store one row per synapse or per connection, and in which of the
biologically interesting attributes they bother to include. So this module
*reports* what a file looks like and what each column might mean, and refuses
to pick a mapping when the evidence is ambiguous.

Nothing here downloads anything and nothing here interprets biology: it is a
table profiler with domain-flavoured heuristics.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ROLES",
    "ColumnProfile",
    "ColumnRole",
    "SchemaError",
    "SchemaReport",
    "detect_file_type",
    "inspect_dataframe",
    "inspect_schema",
    "read_table",
]

#: Canonical roles the importer can map onto.
ColumnRole = str
ROLES: tuple[ColumnRole, ...] = (
    "source",
    "target",
    "weight",
    "neuron_id",
    "region",
    "source_region",
    "target_region",
    "neuron_type",
    "source_type",
    "target_type",
)

#: Trailing tokens that describe an *attribute* of an entity rather than the
#: entity itself. ``pre_region`` is a property of the pre-synaptic neuron; only
#: ``pre_id`` names the neuron. This distinction is what keeps a perfectly clear
#: file from looking ambiguous.
_ATTRIBUTE_SUFFIXES: frozenset[str] = frozenset(
    {
        "id",
        "ids",
        "neuron",
        "neurons",
        "cell",
        "cells",
        "name",
        "names",
        "root",
        "line",
        "type",
        "types",
        "class",
        "classes",
        "label",
        "labels",
        "region",
        "regions",
        "neuropil",
        "roi",
        "division",
        "cluster",
        "compartment",
        "structure",
        "count",
        "counts",
        "synapses",
        "synapse",
        "weight",
        "weights",
        "strength",
        "size",
        "num",
        "n",
    }
)

#: Which canonical role a trailing attribute implies.
_ATTRIBUTE_ROLES: dict[str, ColumnRole] = {
    "region": "region",
    "regions": "region",
    "neuropil": "region",
    "roi": "region",
    "division": "region",
    "cluster": "region",
    "compartment": "region",
    "structure": "region",
    "type": "neuron_type",
    "types": "neuron_type",
    "class": "neuron_type",
    "classes": "neuron_type",
    "label": "neuron_type",
    "labels": "neuron_type",
    "count": "weight",
    "counts": "weight",
    "synapses": "weight",
    "synapse": "weight",
    "weight": "weight",
    "weights": "weight",
    "strength": "weight",
    "size": "weight",
    "num": "weight",
    "n": "weight",
}

#: Words naming the pre-synaptic side.
_SOURCE_WORDS: frozenset[str] = frozenset(
    {"pre", "presyn", "presynaptic", "from", "src", "source", "upstream", "origin", "sender", "input", "antecedent"}
)
#: Words naming the post-synaptic side.
_TARGET_WORDS: frozenset[str] = frozenset(
    {"post", "postsyn", "postsynaptic", "to", "dst", "target", "downstream", "receiver", "dendrite"}
)
#: Words naming a neuron in its own right, with no side implied.
_IDENTITY_WORDS: frozenset[str] = frozenset({"root", "neuron", "cell", "global", "line", "id", "name"})

_SUPPORTED_SUFFIXES = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".csv.gz": "csv", ".parquet": "parquet", ".pq": "parquet", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl"}


class SchemaError(ValueError):
    """Raised when a file cannot be interpreted, with a message meant for a human."""


def detect_file_type(path: str | Path) -> str:
    """Classify a path as ``csv``, ``parquet``, ``json`` or ``jsonl``."""
    resolved = Path(path)
    name = resolved.name.lower()
    if name.endswith(".csv.gz") or name.endswith(".tsv.gz"):
        return "csv"
    suffix = resolved.suffix.lower()
    if suffix in _SUPPORTED_SUFFIXES:
        return _SUPPORTED_SUFFIXES[suffix]
    raise SchemaError(
        f"unsupported file type '{resolved.suffix or resolved.name}'. "
        f"Supported: {', '.join(sorted(set(_SUPPORTED_SUFFIXES.values())))}."
    )


def read_table(
    path: str | Path,
    file_type: str | None = None,
    nrows: int | None = None,
    usecols: Sequence[str] | Mapping[str, str] | None = None,
    chunksize: int | None = None,
) -> Any:
    """Read a tabular connectome file with pandas.

    Returns a :class:`pandas.DataFrame`, or a chunk iterator when ``chunksize``
    is given. Parquet needs an engine (``pyarrow`` or ``fastparquet``); the error
    explains how to install one rather than surfacing an opaque import failure.
    """
    resolved = Path(path)
    kind = file_type or detect_file_type(resolved)

    if kind == "csv":
        kwargs: dict[str, Any] = {"nrows": nrows, "usecols": usecols, "chunksize": chunksize}
        if resolved.suffix.lower() in {".tsv", ".tsv.gz"}:
            kwargs["sep"] = "\t"
        return pd.read_csv(resolved, **kwargs)

    if kind == "parquet":
        try:
            return pd.read_parquet(resolved, columns=list(usecols) if usecols else None)
        except ImportError as exc:
            raise SchemaError(
                "reading Parquet requires an engine. Install one with "
                "`pip install pyarrow` (free, open source) or `pip install -e .[parquet]`, "
                "or convert the file to CSV first."
            ) from exc
        except ValueError as exc:  # engine present but broken
            raise SchemaError(f"could not read Parquet file {resolved.name}: {exc}") from exc

    if kind in {"json", "jsonl"}:
        return _read_json(resolved, nrows=nrows, lines=kind == "jsonl")

    raise SchemaError(f"unsupported file type '{kind}'")


def _read_json(path: Path, nrows: int | None, lines: bool) -> pd.DataFrame:
    """Read JSON Lines or a single JSON document holding a list of records."""
    try:
        if lines:
            return pd.read_json(path, lines=True, nrows=nrows)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{path.name} is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        for key in ("edges", "data", "rows", "records", "connections"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            raise SchemaError(
                f"{path.name} is a JSON object without a record list. Expected one of "
                "'edges', 'data', 'rows', 'records' or 'connections'."
            )
    if not isinstance(payload, list):
        raise SchemaError(f"{path.name} must contain a list of records")
    frame = pd.DataFrame(payload)
    return frame.head(nrows) if nrows else frame


def _count_csv_rows(path: Path) -> int | None:
    """Stream-count data rows in a CSV without parsing or loading it.

    Approximate for files with quoted embedded newlines; reported as such.
    """
    try:
        newlines = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                newlines += chunk.count(b"\n")
    except OSError:
        return None
    if newlines == 0:
        return 0
    return max(newlines - 1, 0)


def _row_count(path: Path, file_type: str) -> tuple[int | None, str]:
    """Best-effort row count plus a note on how it was obtained."""
    if file_type == "csv":
        return _count_csv_rows(path), "streamed line count (approximate; quoted newlines are not accounted for)"
    if file_type == "parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows), "parquet metadata (exact)"
        except ImportError:
            return None, "unavailable (install pyarrow to read Parquet row counts)"
        except Exception:  # pragma: no cover - corrupt or unreadable footer
            return None, "unavailable (could not read Parquet metadata)"
    return None, "not counted (JSON is loaded in full)"


@dataclass(frozen=True)
class ColumnProfile:
    """What one column looks like, and what it might mean."""

    name: str
    dtype: str
    null_count: int
    unique_estimate: int
    sample_values: list[str]
    role: ColumnRole | None
    role_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "unique_estimate": self.unique_estimate,
            "sample_values": self.sample_values,
            "role": self.role,
            "role_candidates": self.role_candidates,
        }


@dataclass
class SchemaReport:
    """The result of inspecting a file. Purely descriptive."""

    file: str
    file_type: str
    file_size_bytes: int
    row_count: int | None
    row_count_method: str
    columns: list[ColumnProfile]
    detected_roles: dict[ColumnRole, str | None]
    ambiguities: dict[ColumnRole, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------- accessors

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise SchemaError(f"no column named {name!r} in {self.file}")

    def role_candidates(self, role: ColumnRole) -> list[str]:
        """Column names that could serve ``role``, best first."""
        return [column.name for column in self.columns if role in column.role_candidates]

    def is_ambiguous(self, role: ColumnRole) -> bool:
        return role in self.ambiguities

    # -------------------------------------------------------------- contract

    def required_roles(self, mapping: Mapping[ColumnRole, str | None] | None = None) -> list[ColumnRole]:
        source = self.detected_roles if mapping is None else mapping
        return [role for role in ("source", "target") if not source.get(role)]

    def resolve(self, overrides: Mapping[ColumnRole, str] | None = None) -> dict[ColumnRole, str | None]:
        """Return the column mapping, raising if any required role is unresolved.

        Overrides are applied first, so a user can always settle an ambiguity
        explicitly instead of letting a heuristic decide.
        """
        mapping: dict[ColumnRole, str | None] = dict(self.detected_roles)
        for role, column in (overrides or {}).items():
            if column not in self.column_names:
                raise SchemaError(
                    f"column {column!r} was given for role {role!r} but is not in the file. "
                    f"Available columns: {', '.join(self.column_names)}"
                )
            mapping[role] = column

        missing = self.required_roles(mapping)
        if missing:
            details = []
            for role in missing:
                candidates = self.role_candidates(role)
                hint = f" Candidates: {', '.join(candidates)}." if candidates else " No column name looks plausible."
                details.append(f"  - {role}: unresolved.{hint}")
            raise SchemaError(
                "cannot map this file onto a connectome without guessing.\n"
                + "\n".join(details)
                + "\nRe-run with explicit column names, e.g. --source-column <name> --target-column <name>."
            )
        return mapping

    # ---------------------------------------------------------------- output

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "row_count_method": self.row_count_method,
            "columns": [column.to_dict() for column in self.columns],
            "detected_roles": self.detected_roles,
            "ambiguities": self.ambiguities,
            "warnings": list(self.warnings),
            "sample_rows": list(self.sample_rows),
        }

    def render(self, max_sample_values: int = 3) -> str:
        """Human-readable report, used by ``scripts/inspect_connectome.py``."""
        lines = [
            "FlyBrain Schema Inspection",
            "-" * 78,
            f"File      : {self.file}",
            f"Type      : {self.file_type}",
            f"Size      : {_human_bytes(self.file_size_bytes)}",
            f"Rows      : {self.row_count if self.row_count is not None else 'unknown'}  ({self.row_count_method})",
            f"Columns   : {len(self.columns)}",
            "",
            f"{'column':<28}{'dtype':<14}{'nulls':>7}{'unique':>9}  role",
            "-" * 78,
        ]
        for column in self.columns:
            role = column.role or ("ambiguous: " + "/".join(column.role_candidates) if column.role_candidates else "-")
            lines.append(
                f"{column.name[:27]:<28}{column.dtype[:13]:<14}"
                f"{column.null_count:>7}{column.unique_estimate:>9}  {role}"
            )

        lines += ["", "Detected roles:"]
        for role in ROLES:
            detected = self.detected_roles.get(role)
            if detected:
                lines.append(f"  {role:<12} -> {detected}")
            else:
                candidates = self.role_candidates(role)
                note = f" (candidates: {', '.join(candidates)})" if candidates else " (none found)"
                lines.append(f"  {role:<12} -> unresolved{note}")

        if self.ambiguities:
            lines += ["", "Ambiguous roles (not guessed):"]
            for role, candidates in self.ambiguities.items():
                lines.append(f"  {role}: {', '.join(candidates)}")
            lines.append("  Supply these explicitly to import.")

        if self.warnings:
            lines += ["", "Warnings:"]
            lines += [f"  - {warning}" for warning in self.warnings]

        lines += ["", "Sample rows:"]
        if self.sample_rows:
            keys = list(self.sample_rows[0].keys())
            lines.append("  " + " | ".join(str(key)[:18].ljust(18) for key in keys))
            for row in self.sample_rows:
                lines.append("  " + " | ".join(str(row.get(key, ""))[:18].ljust(18) for key in keys))
        else:
            lines.append("  (none)")
        return "\n".join(lines)


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover


def _tokens(name: str) -> list[str]:
    """Split a column name into lower-case alphanumeric tokens."""
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _split_entity(name: str) -> tuple[str | None, str | None]:
    """Split ``pre_region`` into ``("pre", "region")``; ``pre_id`` into ``("pre", "id")``.

    Returns ``(None, None)`` when the whole name denotes an entity, e.g. ``pre``.
    """
    tokens = _tokens(name)
    if len(tokens) >= 2 and tokens[-1] in _ATTRIBUTE_SUFFIXES:
        return "_".join(tokens[:-1]), tokens[-1]
    return None, None


def _entity_role(entity: str) -> ColumnRole | None:
    """Role implied by the entity part of a column name."""
    tokens = _tokens(entity)
    if not tokens:
        return None
    for token in reversed(tokens):
        if token in _TARGET_WORDS:
            return "target"
        if token in _SOURCE_WORDS:
            return "source"
        if token in _IDENTITY_WORDS:
            return "neuron_id"
    return None


def _side_prefixed_role(entity: str, base: ColumnRole) -> ColumnRole | None:
    """``pre_region`` -> ``source_region``; unprefixed -> ``None``."""
    tokens = _tokens(entity)
    if not tokens:
        return None
    if tokens[0] in _SOURCE_WORDS:
        return f"source_{base}"
    if tokens[0] in _TARGET_WORDS:
        return f"target_{base}"
    return None


def _looks_like_count(name: str, series: pd.Series) -> bool:
    """True when a column name and dtype suggest a synaptic count."""
    lowered = name.lower()
    if not any(hint in lowered for hint in ("count", "num", "syn", "size", "weight", "strength")):
        return False
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _candidate_roles(name: str, series: pd.Series) -> list[ColumnRole]:
    """Roles this column could plausibly serve.

    The rule is structural rather than a substring search: a column is split into
    an entity and an optional attribute, so ``pre_id`` is a source endpoint while
    ``pre_region`` is a region attribute of the source. Anything unrecognised
    yields no candidate, which surfaces as "unresolved" rather than a bad guess.
    """
    lowered = name.lower().strip()
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    entity, attribute = _split_entity(lowered)

    if attribute is not None:
        base = _ATTRIBUTE_ROLES.get(attribute)
        if base == "weight":
            return ["weight"] if numeric else []
        if base is not None:
            if base == "region":
                side = _side_prefixed_role(entity, "region")
                return [side] if side else ["region"]
            if base == "neuron_type":
                side = _side_prefixed_role(entity, "type")
                return [side] if side else ["neuron_type"]
            return [base]
        # An identity attribute: the entity itself is the neuron.
        role = _entity_role(entity)
        return [role] if role else []

    # No attribute suffix: the whole name is either a quantity or an entity.
    if lowered in _ATTRIBUTE_ROLES and _ATTRIBUTE_ROLES[lowered] == "weight":
        return ["weight"] if numeric else []
    role = _entity_role(lowered)
    return [role] if role else []


def _profile_columns(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, ColumnRole]]:
    """Collect raw per-column statistics and role candidates.

    Returns untyped dicts rather than :class:`ColumnProfile` objects, because
    the detected roles are only known once every column has been examined, and
    ``ColumnProfile`` is frozen.
    """
    stats: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        try:
            unique_estimate = int(series.nunique(dropna=True))
        except TypeError:  # unhashable values, e.g. nested JSON
            unique_estimate = -1
        stats.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "unique_estimate": unique_estimate,
                "sample_values": [str(value) for value in series.dropna().head(3).tolist()],
                "role_candidates": _candidate_roles(str(column), series),
            }
        )
    return stats, {str(entry["name"]): entry["role_candidates"] for entry in stats}


def _by_role(candidates: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """Invert a ``column -> roles`` mapping into ``role -> columns``."""
    grouped: dict[str, list[str]] = {}
    for column, roles in candidates.items():
        for role in roles:
            grouped.setdefault(role, []).append(column)
    return grouped


def _assign_roles(by_role: Mapping[str, list[str]]) -> tuple[dict[ColumnRole, str | None], dict[ColumnRole, list[str]]]:
    """Pick a column per role, or record the ambiguity instead of guessing."""
    assigned: dict[ColumnRole, str | None] = {role: None for role in ROLES}
    ambiguities: dict[ColumnRole, list[str]] = {}
    used: set[str] = set()

    def claim(role: ColumnRole) -> None:
        candidates = [name for name in by_role.get(role, []) if name not in used]
        if not candidates:
            return
        if len(candidates) == 1:
            assigned[role] = candidates[0]
            used.add(candidates[0])
        else:
            # Several plausible names: never pick silently, even for the
            # lower-stakes roles, because a wrong region column quietly
            # mislabels every figure derived from it.
            ambiguities[role] = candidates

    # Endpoints first: they are the only required roles, so they get first pick
    # of any column that several roles could otherwise claim.
    for role in ("source", "target"):
        claim(role)
    for role in ROLES:
        if role not in ("source", "target"):
            claim(role)

    return assigned, ambiguities


def _sanitise_roles(report: SchemaReport) -> None:
    """Flag roles where a column was chosen but the guess looks weak."""
    for role, candidates in report.ambiguities.items():
        if report.detected_roles.get(role) and role in {"weight", "region", "neuron_type"}:
            report.warnings.append(
                f"role {role!r} was inferred from name matching among {candidates}; confirm it is correct"
            )


def inspect_dataframe(
    frame: pd.DataFrame,
    file: str = "<in-memory>",
    file_type: str = "dataframe",
    file_size_bytes: int = 0,
    row_count: int | None = None,
    row_count_method: str = "in-memory length",
    n_samples: int = 5,
) -> SchemaReport:
    """Build a :class:`SchemaReport` for a loaded table."""
    if frame is None or len(frame.columns) == 0:
        raise SchemaError(f"{file} has no columns; this does not look like a tabular connectome file")

    stats, candidates = _profile_columns(frame)
    assigned, ambiguities = _assign_roles(_by_role(candidates))
    role_of_column: dict[str, ColumnRole] = {
        column: role for role, column in assigned.items() if column
    }
    profiles = [
        ColumnProfile(
            name=entry["name"],
            dtype=entry["dtype"],
            null_count=entry["null_count"],
            unique_estimate=entry["unique_estimate"],
            sample_values=entry["sample_values"],
            role=role_of_column.get(entry["name"]),
            role_candidates=entry["role_candidates"],
        )
        for entry in stats
    ]

    warnings: list[str] = []
    if row_count is not None and row_count != len(frame):
        warnings.append(
            f"file has ~{row_count} rows but only {len(frame)} were sampled for profiling; "
            "dtypes and null counts describe the sample only"
        )
    if len(frame) == 0:
        warnings.append("file contains a header but no data rows")
    if all(profile.null_count == len(frame) for profile in profiles):
        warnings.append("every column is null in the sampled rows")

    # A weight-shaped column that is not numeric is a trap: it would otherwise
    # be ignored without comment and the import would quietly fall back to a
    # placeholder weight.
    for profile in profiles:
        lowered = profile.name.lower()
        looks_like_weight = any(
            hint in lowered for hint in ("weight", "count", "synapse", "syn_count", "strength", "size", "num")
        )
        numeric = not profile.dtype.lower().startswith(("object", "str", "string", "category"))
        if looks_like_weight and not numeric:
            warnings.append(
                f"column '{profile.name}' looks like a weight but its dtype is {profile.dtype}; "
                "it will be ignored, and the import will use a flagged placeholder weight. "
                "Map it explicitly with --weight-column to find out which values fail to parse."
            )

    report = SchemaReport(
        file=file,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        row_count=row_count if row_count is not None else len(frame),
        row_count_method=row_count_method,
        columns=profiles,
        detected_roles=assigned,
        ambiguities=ambiguities,
        warnings=warnings,
        sample_rows=frame.head(max(n_samples, 0)).to_dict(orient="records"),
    )
    _sanitise_roles(report)
    return report


def inspect_schema(path: str | Path, nrows: int = 10_000, n_samples: int = 5) -> SchemaReport:
    """Inspect a local file without loading it in full.

    Only ``nrows`` rows are parsed, so a multi-gigabyte export can be inspected
    on a laptop. The full row count is obtained separately, by streaming.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise SchemaError(
            f"file not found: {resolved}\n"
            "FlyBrain never downloads connectome data. Download the dataset yourself from the "
            "official source, then pass the local path to this script."
        )
    if not resolved.stat().st_size:
        raise SchemaError(f"{resolved.name} is empty (0 bytes)")

    file_type = detect_file_type(resolved)
    try:
        frame = read_table(resolved, file_type=file_type, nrows=nrows)
    except SchemaError:
        raise
    except Exception as exc:  # pandas parse errors, bad gzip, etc.
        raise SchemaError(f"could not parse {resolved.name}: {type(exc).__name__}: {exc}") from exc

    if isinstance(frame, pd.DataFrame) and frame.shape[1] == 1 and file_type == "csv":
        # A single column usually means a wrong separator, which is a common and
        # confusing failure for a tab-separated export saved as .csv.
        with resolved.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().rstrip("\n")
        if header.count("\t") > 0:
            raise SchemaError(
                f"{resolved.name} appears to be tab-separated but has a .csv/.tsv name. "
                f"Header line: {header[:200]!r}"
            )

    row_count, method = _row_count(resolved, file_type)
    return inspect_dataframe(
        frame,
        file=str(resolved),
        file_type=file_type,
        file_size_bytes=resolved.stat().st_size,
        row_count=row_count,
        row_count_method=method,
        n_samples=n_samples,
    )
