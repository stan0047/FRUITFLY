"""Validation, cleaning and budgeting for imported edges.

The governing rule is **report, do not silently delete**. Every row removed is
counted and named in a :class:`ValidationReport`, so a report can always state
how many edges were dropped and why. Nothing is discarded quietly, and a
problem severe enough to make the data meaningless (an empty table, a
non-numeric weight column) raises instead of degrading quietly.

Defaults are conservative:

* self-loops are removed (a chemical synapse from a neuron to itself is not
  meaningful for the circuits we intend to simulate);
* duplicate ``(source, target)`` rows are aggregated by summing their weights,
  which is the right operation when the weight is a synapse count;
* no node/edge budget is applied unless the user configures one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from flybrain.data.normalize import CANONICAL_EDGE_COLUMNS

__all__ = [
    "LimitExceededError",
    "SCOPE_COMPLETE",
    "SCOPE_SUBSET",
    "ValidationConfig",
    "ValidationError",
    "ValidationReport",
    "validate_and_clean",
]


class ValidationError(ValueError):
    """Raised when input data cannot be interpreted as a connectome at all."""


class LimitExceededError(ValidationError):
    """Raised when a dataset exceeds its configured budget and cannot be cut."""


SCOPE_COMPLETE = "complete dataset"
SCOPE_SUBSET = "subset of the source dataset"

_ON_LIMIT = ("error", "subsample")


@dataclass(frozen=True)
class ValidationConfig:
    """Explicit import policy. Every field is opt-in except the two defaults."""

    keep_self_loops: bool = False
    aggregate_duplicates: bool = True
    min_synapse_count: float | None = None
    max_nodes: int = 0
    max_edges: int = 0
    on_limit: str = "error"
    require_known_nodes: bool = True
    drop_null_rows: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.on_limit not in _ON_LIMIT:
            raise ValidationError(f"on_limit must be one of {_ON_LIMIT}, got {self.on_limit!r}")
        if self.max_nodes < 0 or self.max_edges < 0:
            raise ValidationError("max_nodes and max_edges must be >= 0 (0 means unlimited)")
        if self.min_synapse_count is not None and self.min_synapse_count < 0:
            raise ValidationError("min_synapse_count must be >= 0")
        if not self.drop_null_rows and self.min_synapse_count is None and self.max_edges == 0 and self.max_nodes == 0:
            pass  # defaults; nothing to do

    @property
    def limits_enabled(self) -> bool:
        return bool(self.max_nodes or self.max_edges)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ValidationConfig":
        """Build from the ``connectome.import`` config section."""
        if not mapping:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{str(k): v for k, v in mapping.items() if str(k) in known})


@dataclass
class ValidationReport:
    """What was checked, what was found, and what was removed."""

    rows_in: int = 0
    rows_out: int = 0
    num_null_values: int = 0
    num_missing_ids: int = 0
    num_invalid_weights: int = 0
    num_negative_weights: int = 0
    num_self_loops: int = 0
    num_duplicate_edges: int = 0
    num_below_min_synapse: int = 0
    num_unknown_source_ids: int = 0
    num_unknown_target_ids: int = 0
    num_unknown_node_rows: int = 0
    subset: bool = False
    scope: str = SCOPE_COMPLETE
    subset_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dropped_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_removed(self) -> int:
        return max(self.rows_in - self.rows_out, 0)

    def record_drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + int(count)

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValidationError(
                "connectome validation failed:\n" + "\n".join(f"  - {error}" for error in self.errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "total_removed": self.total_removed,
            "dropped_by_reason": dict(self.dropped_by_reason),
            "counts": {
                "null_values": self.num_null_values,
                "missing_ids": self.num_missing_ids,
                "invalid_weights": self.num_invalid_weights,
                "negative_weights": self.num_negative_weights,
                "self_loops": self.num_self_loops,
                "duplicate_edges": self.num_duplicate_edges,
                "below_min_synapse_count": self.num_below_min_synapse,
                "unknown_source_ids": self.num_unknown_source_ids,
                "unknown_target_ids": self.num_unknown_target_ids,
                "unknown_node_rows": self.num_unknown_node_rows,
            },
            "scope": self.scope,
            "subset": self.subset,
            "subset_reason": self.subset_reason,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "ok": self.ok,
        }

    def render(self, width: int = 78) -> str:
        lines = [
            "Validation report",
            "-" * width,
            f"rows in              : {self.rows_in}",
            f"rows out             : {self.rows_out}",
            f"total removed        : {self.total_removed}",
            f"scope                : {self.scope}" + (f" ({self.subset_reason})" if self.subset_reason else ""),
        ]
        if self.dropped_by_reason:
            lines.append("removed by reason:")
            for reason, count in sorted(self.dropped_by_reason.items()):
                lines.append(f"  {reason:<20}{count}")
        else:
            lines.append("removed by reason    : nothing removed")
        if self.warnings:
            lines.append("warnings:")
            lines += [f"  - {warning}" for warning in self.warnings]
        if self.errors:
            lines.append("errors:")
            lines += [f"  - {error}" for error in self.errors]
        return "\n".join(lines)


def _clean_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column].astype("string").str.strip()
    return series.mask(series.eq(""), other=pd.NA)


def validate_and_clean(
    frame: pd.DataFrame,
    config: ValidationConfig | None = None,
    node_ids: Iterable[str] | pd.Index | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a canonical edge frame.

    Parameters
    ----------
    frame:
        Canonical frame with at least ``source_id``, ``target_id``, ``weight``.
    config:
        Import policy. Defaults are conservative.
    node_ids:
        Known neuron identifiers, when the source supplied a node table. When
        given, edges referring to unknown neurons are reported; with
        ``require_known_nodes=True`` they are also an error.

    Returns
    -------
    tuple[pandas.DataFrame, ValidationReport]
    """
    settings = config or ValidationConfig()
    report = ValidationReport()

    missing = [column for column in CANONICAL_EDGE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationError(
            f"canonical edge frame is missing required column(s): {', '.join(missing)}"
        )
    if len(frame) == 0:
        raise ValidationError("edge table is empty: a connectome needs at least one synapse")

    report.rows_in = int(len(frame))
    work = frame.copy(deep=False)
    work["source_id"] = _clean_text_column(work, "source_id")
    work["target_id"] = _clean_text_column(work, "target_id")
    weights = pd.to_numeric(work["weight"], errors="coerce")

    # 1. A missing or non-numeric weight is a hard error. Defaulting it would
    #    invent a synaptic strength, which is exactly what this project must not
    #    do, so the import stops and the user maps a real column instead.
    unparsed = int(weights.isna().sum())
    if unparsed:
        report.num_invalid_weights = unparsed
        report.errors.append(
            f"{unparsed} row(s) have a missing or non-numeric weight; map a numeric column as the "
            "weight, or drop the weight column to receive a flagged placeholder weight"
        )
    report.num_null_values = int(work[["source_id", "target_id"]].isna().sum().sum())

    # 2. Null endpoints.
    null_rows = work["source_id"].isna() | work["target_id"].isna()
    if int(null_rows.sum()):
        report.num_missing_ids = int(null_rows.sum())
        report.record_drop("missing_id", report.num_missing_ids)
        if settings.drop_null_rows:
            work = work.loc[~null_rows]
        else:
            report.errors.append(
                f"{report.num_missing_ids} row(s) have a null source_id or target_id; "
                "set drop_null_rows=True to remove them"
            )

    if len(work) == 0:
        report.rows_out = 0
        report.errors.append("no usable rows remain after removing null endpoints")
        report.raise_if_invalid()

    work["weight"] = weights

    # 3. Negative weights.
    negative = work["weight"] < 0
    if int(negative.sum()):
        report.num_negative_weights = int(negative.sum())
        report.record_drop("negative_weight", report.num_negative_weights)
        work = work.loc[~negative]
        report.warnings.append(
            f"{report.num_negative_weights} row(s) had a negative weight and were removed; "
            "chemical synaptic weights are non-negative"
        )

    # 4. Self-loops.
    self_loops = work["source_id"] == work["target_id"]
    if int(self_loops.sum()):
        report.num_self_loops = int(self_loops.sum())
        if settings.keep_self_loops:
            report.warnings.append(
                f"{report.num_self_loops} self-loop(s) kept because keep_self_loops=True"
            )
        else:
            report.record_drop("self_loop", report.num_self_loops)
            work = work.loc[~self_loops]

    # 5. Minimum synapse count / weight.
    if settings.min_synapse_count is not None and len(work):
        below = work["weight"] < float(settings.min_synapse_count)
        if int(below.sum()):
            report.num_below_min_synapse = int(below.sum())
            report.record_drop("below_min_synapse_count", report.num_below_min_synapse)
            work = work.loc[~below]
            report.warnings.append(
                f"{report.num_below_min_synapse} row(s) below min_synapse_count="
                f"{settings.min_synapse_count} were removed"
            )

    # 6. Duplicate edges.
    if len(work):
        key = work[["source_id", "target_id"]]
        duplicate_mask = key.duplicated(keep="first")
        duplicates = int(duplicate_mask.sum())
        if duplicates:
            report.num_duplicate_edges = duplicates
            if settings.aggregate_duplicates:
                work = _aggregate(work, report)
            else:
                report.record_drop("duplicate_edge", duplicates)
                work = work.loc[~duplicate_mask]
                report.warnings.append(f"{duplicates} duplicate edge(s) removed without aggregation")

    # 7. Node-table cross-check.
    if node_ids is not None and len(work):
        known = pd.Index(pd.unique(pd.Series(list(node_ids), dtype="string")))
        unknown_source = ~work["source_id"].isin(known)
        unknown_target = ~work["target_id"].isin(known)
        report.num_unknown_source_ids = int(unknown_source.sum())
        report.num_unknown_target_ids = int(unknown_target.sum())
        if report.num_unknown_source_ids or report.num_unknown_target_ids:
            message = (
                f"{report.num_unknown_source_ids} source and {report.num_unknown_target_ids} target "
                f"identifiers are absent from the supplied node table (of {len(known)} known neurons)"
            )
            if settings.require_known_nodes:
                report.errors.append(message + "; set require_known_nodes=False to import anyway")
            else:
                report.warnings.append(message + "; edges kept because require_known_nodes=False")

    if len(work) == 0:
        report.rows_out = 0
        report.errors.append("no edges remain after validation")
        report.raise_if_invalid()

    # 8. Node/edge budget.
    work = _apply_limits(work, settings, report)

    report.rows_out = int(len(work))
    report.raise_if_invalid()
    return work.reset_index(drop=True), report


def _aggregate(frame: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Collapse duplicate edges by summing their weights."""
    extra = report.num_duplicate_edges
    report.record_drop("duplicate_edge_aggregated", extra)

    value_columns = [column for column in frame.columns if column not in ("source_id", "target_id")]
    aggregations: dict[str, Any] = {}
    for column in value_columns:
        if column == "weight":
            aggregations[column] = "sum"
        elif pd.api.types.is_numeric_dtype(frame[column]):
            aggregations[column] = "first"
        else:
            aggregations[column] = "first"

    grouped = frame.groupby(["source_id", "target_id"], as_index=False, sort=False, dropna=False).agg(aggregations)
    report.warnings.append(
        f"{extra} duplicate edge row(s) aggregated into existing edges by summing weights "
        f"({report.rows_in} rows -> {len(grouped)})"
    )
    return grouped


def _apply_limits(frame: pd.DataFrame, settings: ValidationConfig, report: ValidationReport) -> pd.DataFrame:
    """Enforce ``max_edges``/``max_nodes``, erroring or subsetting as configured."""
    if not settings.limits_enabled:
        return frame

    reason: list[str] = []
    work = frame

    if settings.max_edges and len(work) > settings.max_edges:
        reason.append(f"max_edges={settings.max_edges} (file has {len(work)})")
        if settings.on_limit == "error":
            raise LimitExceededError(
                "dataset exceeds the configured edge budget: " + "; ".join(reason) + ". "
                "Raise --max-edges, or pass --on-limit subsample to take a reproducible subset."
            )
        rng = np.random.default_rng(settings.seed)
        picks = np.sort(rng.choice(len(work), size=settings.max_edges, replace=False))
        work = work.iloc[picks]
        report.record_drop("over_max_edges", len(frame) - len(work))

    if settings.max_nodes:
        node_count = int(pd.unique(pd.concat([work["source_id"], work["target_id"]], ignore_index=True)).size)
        if node_count > settings.max_nodes:
            reason.append(f"max_nodes={settings.max_nodes} (subset has {node_count})")
            if settings.on_limit == "error":
                raise LimitExceededError(
                    "dataset exceeds the configured node budget: " + "; ".join(reason) + ". "
                    "Raise --max-nodes, or pass --on-limit subsample."
                )
            # Keep only edges whose endpoints are among the most connected nodes,
            # so the subset stays a connected-ish core rather than random noise.
            degree = pd.concat(
                [
                    work["source_id"].value_counts(),
                    work["target_id"].value_counts(),
                ]
            )
            degree = degree.groupby(level=0).sum().sort_values(ascending=False)
            keep_nodes = set(degree.index[: settings.max_nodes].astype(str))
            mask = work["source_id"].isin(keep_nodes) & work["target_id"].isin(keep_nodes)
            work = work.loc[mask]
            report.record_drop("over_max_nodes", int((~mask).sum()))

    if reason:
        report.subset = True
        report.scope = SCOPE_SUBSET
        report.subset_reason = "; ".join(reason)
        report.warnings.append(
            "statistics in this report describe a SUBSET, not the complete source dataset"
        )
    return work


def known_node_ids(node_table: pd.DataFrame) -> pd.Index | None:
    """Extract the identifier column from a canonical node table."""
    if node_table is None or node_table.empty or "neuron_id" not in node_table.columns:
        return None
    return pd.Index(node_table["neuron_id"].dropna().unique())
