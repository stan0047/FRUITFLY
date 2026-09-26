"""Canonical internal edge format.

Every imported dataset is reduced to the same three required columns plus a set
of optional ones, so that downstream code never has to know which export it came
from:

===============  ========  =====================================================
Column           Required  Meaning
===============  ========  =====================================================
``source_id``    yes       pre-synaptic neuron identifier, as published
``target_id``    yes       post-synaptic neuron identifier, as published
``weight``       yes       edge weight; see the note on placeholder weights
``source_region``  no      brain region of the source neuron, if reported
``target_region``  no      brain region of the target neuron, if reported
``source_type``    no      neuron type / class of the source, if reported
``target_type``    no      neuron type / class of the target, if reported
``weight_is_biological`` no ``False`` when ``weight`` is a placeholder
===============  ========  =====================================================

Two rules govern this module:

* **Optional biological metadata is never required.** A file with only three
  columns imports fine.
* **Original information is preserved.** Columns that are not part of the
  canonical set are carried through untouched rather than dropped, and the
  mapping used is recorded in the provenance record.

On weights
----------
``weight`` is whatever the source reports as edge strength or synapse count. If
the file has no such column, every edge gets a uniform placeholder weight of
``1.0`` and ``weight_is_biological`` is set to ``False``. That keeps the graph
usable without fabricating a claim about synaptic strength.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from flybrain.data.schema import ROLES, SchemaError

__all__ = [
    "CANONICAL_EDGE_COLUMNS",
    "CANONICAL_NODE_COLUMNS",
    "OPTIONAL_EDGE_COLUMNS",
    "PLACEHOLDER_WEIGHT",
    "ColumnMapping",
    "canonical_edge_columns",
    "normalize_edges",
    "normalize_node_table",
]

#: Required canonical edge columns.
CANONICAL_EDGE_COLUMNS: tuple[str, ...] = ("source_id", "target_id", "weight")
#: Optional canonical edge columns, all biological metadata that may be absent.
OPTIONAL_EDGE_COLUMNS: tuple[str, ...] = (
    "source_region",
    "target_region",
    "source_type",
    "target_type",
    "weight_is_biological",
)
#: Canonical neuron-metadata columns.
CANONICAL_NODE_COLUMNS: tuple[str, ...] = ("neuron_id", "region", "neuron_type")

#: Weight assigned to every edge when the source reports no weight column.
PLACEHOLDER_WEIGHT = 1.0


def canonical_edge_columns() -> list[str]:
    """Required followed by optional canonical column names."""
    return [*CANONICAL_EDGE_COLUMNS, *OPTIONAL_EDGE_COLUMNS]


@dataclass(frozen=True)
class ColumnMapping:
    """Which source column fills which canonical role.

    Built from :meth:`~flybrain.data.schema.SchemaReport.resolve`, or supplied
    directly by a user who knows their file. ``weight`` and all metadata columns
    are optional; ``source`` and ``target`` are not.
    """

    source: str
    target: str
    weight: str | None = None
    source_region: str | None = None
    target_region: str | None = None
    source_type: str | None = None
    target_type: str | None = None
    neuron_id: str | None = None
    region: str | None = None
    neuron_type: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise SchemaError("ColumnMapping requires both a source and a target column")
        if self.source == self.target:
            raise SchemaError(
                f"source and target are both {self.source!r}; a connectome needs two distinct endpoint columns"
            )

    @classmethod
    def from_roles(cls, roles: Mapping[str, str | None]) -> "ColumnMapping":
        """Build from a ``{role: column}`` mapping as returned by the inspector.

        Side-prefixed metadata roles win over the generic ones, so
        ``pre_region`` fills ``source_region`` rather than leaving both ends
        sharing one column.
        """
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise SchemaError(f"unknown role(s): {', '.join(sorted(unknown))}")
        region = roles.get("source_region") or roles.get("region")
        neuron_type = roles.get("source_type") or roles.get("neuron_type")
        return cls(
            source=str(roles["source"]),
            target=str(roles["target"]),
            weight=roles.get("weight"),
            source_region=region,
            target_region=roles.get("target_region"),
            source_type=neuron_type,
            target_type=roles.get("target_type"),
            neuron_id=roles.get("neuron_id"),
            region=roles.get("region"),
            neuron_type=roles.get("neuron_type"),
        )

    def edge_columns(self) -> dict[str, str]:
        """``canonical column -> source column`` for the edge table."""
        pairs = {
            "source_id": self.source,
            "target_id": self.target,
            "weight": self.weight,
            "source_region": self.source_region,
            "target_region": self.target_region,
            "source_type": self.source_type,
            "target_type": self.target_type,
        }
        return {canonical: column for canonical, column in pairs.items() if column}

    def node_columns(self) -> dict[str, str]:
        """``canonical column -> source column`` for the neuron metadata table."""
        pairs = {
            "neuron_id": self.neuron_id,
            "region": self.region,
            "neuron_type": self.neuron_type,
        }
        return {canonical: column for canonical, column in pairs.items() if column}

    def node_roles(self) -> dict[str, str | None]:
        """``role -> source column`` for a neuron metadata table.

        A node table is a separate file with its own columns, so it needs its own
        mapping rather than borrowing the edge file's.
        """
        return {"neuron_id": self.neuron_id, "region": self.region, "neuron_type": self.neuron_type}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "source_region": self.source_region,
            "target_region": self.target_region,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "neuron_id": self.neuron_id,
            "region": self.region,
            "neuron_type": self.neuron_type,
        }


def _clean_ids(series: pd.Series) -> pd.Series:
    """Trim whitespace and render identifiers as strings, leaving nulls as nulls."""
    as_text = series.astype("string").str.strip()
    return as_text.mask(as_text.eq(""), other=pd.NA)


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim whitespace in optional metadata, leaving nulls as nulls."""
    as_text = series.astype("string").str.strip()
    return as_text.mask(as_text.eq(""), other=pd.NA)


def _coerce_weight(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert a weight column to float, counting values that failed to parse."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = int(numeric.isna().sum() - series.isna().sum())
    return numeric.astype("float64"), max(invalid, 0)


def normalize_edges(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    weight_is_biological: bool = True,
    keep_extra_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project a raw edge table onto the canonical format.

    Parameters
    ----------
    frame:
        Raw rows, one per synapse or per connection.
    mapping:
        Which column plays which role.
    weight_is_biological:
        Whether the mapped weight column is a *measured* quantity. Defaults to
        ``True`` because the user chose the column; pass ``False`` when mapping a
        column that is merely a count or an author-chosen proxy.
    keep_extra_columns:
        Carry unmapped source columns through to the output so nothing from the
        original file is lost. Set ``False`` to reduce memory on very large files.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The canonical frame and a notes mapping describing what happened.
    """
    missing = [column for column in (mapping.source, mapping.target) if column not in frame.columns]
    if missing:
        raise SchemaError(
            f"mapped column(s) not present in the data: {', '.join(missing)}. "
            f"Available columns: {', '.join(str(c) for c in frame.columns)}"
        )

    notes: dict[str, Any] = {"placeholder_weight": False, "invalid_weight_values": 0, "dropped_extra_columns": []}
    canonical = pd.DataFrame(index=frame.index)
    canonical["source_id"] = _clean_ids(frame[mapping.source])
    canonical["target_id"] = _clean_ids(frame[mapping.target])

    mapped = set(mapping.edge_columns().values())
    if mapping.weight and mapping.weight in frame.columns:
        weight, invalid = _coerce_weight(frame[mapping.weight])
        notes["invalid_weight_values"] = invalid
        notes["weight_column"] = mapping.weight
        notes["weight_is_biological"] = bool(weight_is_biological)
    else:
        if mapping.weight:
            raise SchemaError(
                f"weight column {mapping.weight!r} was mapped but is not in the data. "
                f"Available columns: {', '.join(str(c) for c in frame.columns)}"
            )
        weight = pd.Series(PLACEHOLDER_WEIGHT, index=frame.index, dtype="float64")
        notes["placeholder_weight"] = True
        notes["weight_is_biological"] = False

    canonical["weight"] = weight
    canonical["weight_is_biological"] = bool(weight_is_biological and not notes["placeholder_weight"])

    for canonical_name, source_column in (
        ("source_region", mapping.source_region),
        ("target_region", mapping.target_region),
        ("source_type", mapping.source_type),
        ("target_type", mapping.target_type),
    ):
        if source_column and source_column in frame.columns:
            canonical[canonical_name] = _clean_text(frame[source_column])
            mapped.add(source_column)
        else:
            canonical[canonical_name] = pd.Series(pd.NA, index=frame.index, dtype="string")

    if keep_extra_columns:
        extras = [column for column in frame.columns if column not in mapped]
        for column in extras:
            canonical[f"original_{column}"] = frame[column].to_numpy()
    else:
        notes["dropped_extra_columns"] = [str(column) for column in frame.columns if column not in mapped]

    return canonical, notes


def normalize_node_table(
    frame: pd.DataFrame,
    roles: Mapping[str, str | None] | ColumnMapping,
) -> pd.DataFrame:
    """Project a raw neuron-metadata table onto the canonical node format.

    ``roles`` maps ``neuron_id``/``region``/``neuron_type`` to source column
    names, either as a plain mapping or via
    :meth:`ColumnMapping.node_roles`.

    Returns an empty frame when no neuron id column is available, which means the
    source supplied no node metadata. That is acceptable: the edge table alone
    defines the node set.
    """
    if isinstance(roles, ColumnMapping):
        roles = roles.node_roles()

    id_column = roles.get("neuron_id")
    if not id_column or id_column not in frame.columns:
        return pd.DataFrame(columns=list(CANONICAL_NODE_COLUMNS))

    region_column = roles.get("region")
    type_column = roles.get("neuron_type")

    nodes = pd.DataFrame(index=frame.index)
    nodes["neuron_id"] = _clean_ids(frame[id_column])
    nodes["region"] = _clean_text(frame[region_column]) if region_column in frame.columns else pd.NA
    nodes["neuron_type"] = _clean_text(frame[type_column]) if type_column in frame.columns else pd.NA
    return nodes.dropna(subset=["neuron_id"]).drop_duplicates(subset=["neuron_id"]).reset_index(drop=True)


def edge_frame_memory_bytes(frame: pd.DataFrame) -> int:
    """Approximate resident size of a canonical edge frame, for reporting."""
    return int(frame.memory_usage(deep=True).sum())


def summarise_canonical_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Small structural summary used in reports and log lines."""
    if frame.empty:
        return {"rows": 0, "columns": list(frame.columns), "memory_bytes": 0}
    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "unique_source_ids": int(frame["source_id"].nunique(dropna=True)) if "source_id" in frame else 0,
        "unique_target_ids": int(frame["target_id"].nunique(dropna=True)) if "target_id" in frame else 0,
        "weight_sum": float(np.nansum(frame["weight"].to_numpy(dtype="float64"))) if "weight" in frame else 0.0,
        "memory_bytes": edge_frame_memory_bytes(frame),
    }
