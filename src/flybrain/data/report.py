"""Dataset reports: what is in this file, and what may be claimed about it.

A report has one job beyond printing numbers: it must be impossible to read one
without knowing **where the data came from** and **whether the figures describe
the whole dataset or a subset**. Both appear in the header, not in a footnote.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from flybrain.data.connectome import Connectome
from flybrain.data.provenance import BIOLOGICAL_LABEL, SYNTHETIC_LABEL, TEST_FIXTURE_LABEL
from flybrain.data.validation import SCOPE_COMPLETE, SCOPE_SUBSET, ValidationReport

__all__ = ["ConnectomeReport", "build_connectome_report", "render_connectome_report"]

_WIDTH = 78


@dataclass
class ConnectomeReport:
    """Statistics for one connectome, with its scope and provenance attached."""

    dataset_name: str
    source_name: str
    scope: str
    label: str
    is_biological: bool
    dataset_version: str | None = None
    source_url: str | None = None
    license: str | None = None
    citation: str | None = None
    imported_at: str | None = None
    original_filename: str | None = None
    notes: str = ""
    num_nodes: int = 0
    num_edges: int = 0
    unique_source_neurons: int = 0
    unique_target_neurons: int = 0
    num_self_loops: int = 0
    mean_weight: float = 0.0
    median_weight: float = 0.0
    total_weight: float = 0.0
    mean_out_degree: float = 0.0
    mean_in_degree: float = 0.0
    mean_degree: float = 0.0
    median_degree: float = 0.0
    max_degree: int = 0
    density: float = 0.0
    weight_is_biological: bool = False
    regions: dict[str, int] = field(default_factory=dict)
    neuron_types: dict[str, int] = field(default_factory=dict)
    unknown_provenance_fields: list[str] = field(default_factory=list)
    subset_reason: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {key: value for key, value in self.__dict__.items()}
        payload["provenance"] = {
            "source_name": self.source_name,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "license": self.license,
            "citation": self.citation,
            "imported_at": self.imported_at,
            "original_filename": self.original_filename,
            "is_biological": self.is_biological,
            "unknown_fields": list(self.unknown_provenance_fields),
        }
        return payload

    def render(self, top_n: int = 10) -> str:
        """Human-readable report, as printed by ``scripts/import_connectome.py``."""
        lines = [
            "",
            "FlyBrain Connectome Report",
            "-" * _WIDTH,
            f"Dataset      : {self.dataset_name}",
            f"Version      : {self.dataset_version or 'not provided'}",
            f"Source       : {self.source_name}",
            f"Biological   : {self.is_biological}   [{self.label}]",
            f"Statistics   : {'SUBSET' if self.scope == SCOPE_SUBSET else 'complete dataset'}"
            + (f" — {self.subset_reason}" if self.subset_reason else ""),
        ]
        if self.source_url:
            lines.append(f"Source URL   : {self.source_url}")
        if self.original_filename:
            lines.append(f"Imported from: {self.original_filename} at {self.imported_at or 'unknown time'}")
        if self.license:
            lines.append(f"License      : {self.license}")
        if self.citation:
            lines.append(f"Citation     : {self.citation}")
        if self.unknown_provenance_fields:
            lines.append(f"Unverified   : {', '.join(self.unknown_provenance_fields)} (not provided; not guessed)")

        lines += [
            "",
            f"Nodes                    : {self.num_nodes}",
            f"Edges                    : {self.num_edges}",
            f"Unique source neurons    : {self.unique_source_neurons}",
            f"Unique target neurons    : {self.unique_target_neurons}",
            f"Mean out-degree          : {self.mean_out_degree:.3f}",
            f"Mean in-degree           : {self.mean_in_degree:.3f}",
            f"Mean degree (in + out)   : {self.mean_degree:.3f}",
            f"Median degree            : {self.median_degree:.1f}",
            f"Max degree               : {self.max_degree}",
            f"Density                  : {self.density:.3e}",
            f"Mean edge weight         : {self.mean_weight:.4f}",
            f"Median edge weight       : {self.median_weight:.4f}",
            f"Total weight             : {self.total_weight:.1f}",
            f"Edge weight is biological: {self.weight_is_biological}",
        ]
        if self.num_self_loops:
            lines.append(f"Self-loops               : {self.num_self_loops}")

        lines += ["", f"Regions ({len(self.regions)} distinct, top {min(top_n, len(self.regions))}):"]
        lines += _format_counts(self.regions, top_n, "    (none reported by the source)")
        lines += ["", f"Neuron types ({len(self.neuron_types)} distinct, top {min(top_n, len(self.neuron_types))}):"]
        lines += _format_counts(self.neuron_types, top_n, "    (none reported by the source)")

        if self.notes:
            lines += ["", f"Notes: {self.notes}"]

        lines += ["", "Provenance statement:"]
        lines += [f"  - {line}" for line in _provenance_statement(self).splitlines()]
        lines += [
            "  - Neural activity derived from this graph is SIMULATED, not recorded from a fly.",
            "",
        ]
        return "\n".join(lines)


def _format_counts(counts: Mapping[str, int], top_n: int, empty: str) -> list[str]:
    if not counts:
        return [empty]
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    width = max(len(str(name)) for name, _ in ordered)
    return [f"    {str(name)[:width]:<{width}}  {count}" for name, count in ordered]


def _provenance_statement(report: ConnectomeReport) -> str:
    if report.label == TEST_FIXTURE_LABEL:
        return "This report describes a test fixture, not real anatomy. Its numbers are not biology."
    if report.label == SYNTHETIC_LABEL:
        return "This report describes synthetic data generated by FlyBrain, not a measurement."
    return (
        f"Topology described here originates from '{report.source_name}' "
        f"({report.label}). It is a measurement of connectivity, not a model."
    )


def _degree_stats(connectome: Connectome) -> dict[str, Any]:
    """Degree statistics computed on a sparse matrix, never a dense one.

    Degree counts distinct connections, not summed weight, so it matches the
    graph-theoretic meaning of "degree". Returned keys are the
    :class:`ConnectomeReport` field names, so the caller can splat them in.
    """
    graph = connectome.sparse_matrix()
    num_nodes = int(graph.shape[0])
    if num_nodes == 0:
        return {
            "mean_in_degree": 0.0,
            "mean_out_degree": 0.0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "max_degree": 0,
            "density": 0.0,
        }

    binary = graph.copy()
    binary.data = np.ones_like(binary.data, dtype=np.int64)
    in_degree = np.asarray(binary.sum(axis=0)).ravel()
    out_degree = np.asarray(binary.sum(axis=1)).ravel()
    total = in_degree + out_degree
    possible = float(num_nodes) * float(num_nodes - 1)
    return {
        "mean_in_degree": float(in_degree.mean()),
        "mean_out_degree": float(out_degree.mean()),
        "mean_degree": float(total.mean()),
        "median_degree": float(np.median(total)),
        "max_degree": int(total.max()),
        "density": float(graph.nnz / possible) if possible > 0 else 0.0,
    }


def _count_column(frame: pd.DataFrame, column: str, top_n: int) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    series = frame[column].dropna().astype(str)
    if series.empty:
        return {}
    counts = series.value_counts().head(top_n)
    return {str(name): int(count) for name, count in counts.items()}


def build_connectome_report(
    connectome: Connectome,
    validation: ValidationReport | None = None,
    top_n: int = 10,
) -> ConnectomeReport:
    """Compute report statistics for a connectome.

    Uses a sparse adjacency matrix throughout, so the cost is O(edges) rather
    than O(nodes^2). Nothing dense is ever allocated for a real-size dataset.
    """
    frame = connectome.edge_frame
    provenance = connectome.provenance_record
    weights = frame["weight"].to_numpy(dtype="float64") if "weight" in frame else np.zeros(0)

    subset = bool(validation.subset) if validation is not None else connectome.is_subset
    report = ConnectomeReport(
        dataset_name=connectome.name,
        source_name=provenance.source_name if provenance else connectome.source.value,
        scope=SCOPE_SUBSET if subset else SCOPE_COMPLETE,
        label=provenance.label if provenance else connectome.source.label,
        is_biological=connectome.is_biological,
        dataset_version=provenance.dataset_version if provenance else connectome.version,
        source_url=provenance.source_url if provenance else connectome.url,
        license=provenance.license if provenance else connectome.license,
        citation=provenance.citation if provenance else connectome.citation,
        imported_at=provenance.imported_at if provenance else None,
        original_filename=provenance.original_filename if provenance else None,
        notes=provenance.notes if provenance else connectome.notes,
        num_nodes=connectome.num_neurons,
        num_edges=connectome.num_edges,
        unique_source_neurons=int(frame["source_id"].nunique(dropna=True)) if len(frame) else 0,
        unique_target_neurons=int(frame["target_id"].nunique(dropna=True)) if len(frame) else 0,
        num_self_loops=int((frame["source_id"] == frame["target_id"]).sum()) if len(frame) else 0,
        mean_weight=float(weights.mean()) if weights.size else 0.0,
        median_weight=float(np.median(weights)) if weights.size else 0.0,
        total_weight=float(weights.sum()) if weights.size else 0.0,
        weight_is_biological=connectome.weights_are_biological,
        regions=connectome.region_counts(top_n),
        neuron_types=_count_column(frame, "source_type", top_n),
        unknown_provenance_fields=provenance.missing_fields() if provenance else [],
        subset_reason=(validation.subset_reason if validation else None) or connectome.subset_reason,
        validation=validation.to_dict() if validation else {},
    )
    report.__dict__.update(_degree_stats(connectome))
    return report


def render_connectome_report(connectome: Connectome, validation: ValidationReport | None = None) -> str:
    """Convenience wrapper returning :meth:`ConnectomeReport.render` output."""
    return build_connectome_report(connectome, validation).render()


def label_for(is_biological: bool) -> str:
    """Banner for a boolean biological flag."""
    return BIOLOGICAL_LABEL if is_biological else SYNTHETIC_LABEL
