"""Bounded seeded connectivity exploration over a MaleCNS connectome table.

Given a set of seed body ids, retrieve their local connectivity and expand it to a
bounded neighborhood. The result is a **seeded connectivity neighborhood**: a
data-structure feasibility artifact, not a circuit and not a visual pathway.

Why a bounded read
------------------
The MaleCNS ``connectome-weights`` table is sorted by ``weight`` descending, so
the strongest N edges are the first N rows. This module reads a prefix of the
file and is explicit about the resulting truncation. That matters scientifically:
a prefix-limited neighborhood is a **synapse-count-biased** sample of the true
neighborhood, and weak connections are systematically absent from it. The
truncation is reported, never hidden.

Getting an *unbiased* neighborhood for arbitrary seeds would require indexing the
whole file once. That is a deliberate, separate step and is not performed here.

Weight semantics
----------------
``weight`` is the **number of synapses** from the pre-synaptic to the
post-synaptic body, as provided by the dataset. It is a structural count. It is
**not** synaptic efficacy and is never normalised and relabelled as biological
strength. The ``min_weight`` parameter is a *minimum synapse-count threshold*.

Provenance
----------
Provided by the dataset: body ids, direction, synapse counts, and the neuron
annotations joined onto the result.
Derived by FlyBrain: the prefix, the neighborhood, the budget truncation, and
every coverage percentage.
Assumed by FlyBrain: nothing. No weight is synthesised and no neuron is given a
category it does not have.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp

__all__ = [
    "DIRECTIONS",
    "NeighborhoodResult",
    "PrefixIndex",
    "SeedSet",
    "build_prefix_index",
    "explore",
    "load_annotation_table",
    "load_seed_file",
    "seeds_from_types",
]

DIRECTIONS = ("downstream", "upstream", "both")

#: Refuse to densify a neighborhood above this node count.
DENSE_REFUSE_NODES = 25_000

#: Arrow record batch rows in the MaleCNS connectivity table.
_BATCH_ROWS_HINT = 65_536


# --------------------------------------------------------------------- index


@dataclass
class PrefixIndex:
    """A weight-sorted prefix of the connectome table, held sparsely.

    ``csr`` is square but never dense: it stores only the edges in the prefix.
    """

    node_ids: np.ndarray
    csr: sp.csr_matrix
    weights: sp.csr_matrix
    edges_read: int
    edges_after_threshold: int
    file_total_rows: int
    record_batches_read: int
    record_batches_total: int
    min_weight: int
    truncated: bool

    @property
    def num_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def record_batches_untouched(self) -> int:
        """Batches that were never read, and therefore never paged into memory."""
        return self.record_batches_total - self.record_batches_read

    def dense_bytes_if_materialised(self) -> int:
        """What a dense float32 matrix would cost. Reported, never allocated."""
        return self.num_nodes * self.num_nodes * 4

    def index_of(self, body_ids: Iterable[int]) -> np.ndarray:
        """Positions in ``node_ids`` for the ids that are present."""
        wanted = np.asarray(sorted(set(int(b) for b in body_ids)), dtype=np.int64)
        positions = np.searchsorted(self.node_ids, wanted)
        positions = np.clip(positions, 0, max(self.num_nodes - 1, 0))
        found = self.node_ids[positions] == wanted
        return positions[found]

    def neighbours(self, node: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour positions and synapse counts for one node."""
        if direction in ("downstream", "both"):
            out_start, out_end = self.csr.indptr[node], self.csr.indptr[node + 1]
            targets = self.csr.indices[out_start:out_end]
            target_weights = self.csr.data[out_start:out_end]
        else:
            targets = targets_empty = np.zeros(0, dtype=np.int64)
            target_weights = targets_empty
        if direction in ("upstream", "both"):
            column = self.csr.tocsc()
            in_start, in_end = column.indptr[node], column.indptr[node + 1]
            sources = column.indices[in_start:in_end]
            source_weights = column.data[in_start:in_end]
        else:
            sources = np.zeros(0, dtype=np.int64)
            source_weights = np.zeros(0, dtype=np.float32)
        if direction == "both":
            return (
                np.concatenate([targets, sources]),
                np.concatenate([target_weights, source_weights]),
            )
        if direction == "upstream":
            return sources, source_weights
        return targets, target_weights


def _file_total_rows(reader: pa.ipc.RecordBatchFileReader) -> int:
    metadata = reader.schema.metadata or {}
    if b"pandas" not in metadata:
        return -1
    try:
        payload = json.loads(metadata[b"pandas"])
    except (json.JSONDecodeError, TypeError):
        return -1
    for column in payload.get("index_columns", []) or []:
        if column.get("kind") == "range":
            return int(column["stop"])
    return -1


def build_prefix_index(
    path: str | Path,
    max_edges: int = 2_000_000,
    min_weight: int = 1,
) -> PrefixIndex:
    """Read the strongest ``max_edges`` rows and index them sparsely.

    Parameters
    ----------
    max_edges:
        Row budget. Only this many rows are read; the file is memory-mapped so
        the remaining batches are never paged in.
    min_weight:
        Minimum synapse-count threshold applied after reading. Rows below it are
        counted and reported, not silently dropped.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"connectome table not found: {resolved}")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")

    reader = pa.ipc.open_file(pa.memory_map(str(resolved), "rb"))
    total_batches = reader.num_record_batches
    total_rows = _file_total_rows(reader)

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    read = 0
    batches_used = 0

    for index in range(total_batches):
        if read >= max_edges:
            break
        batch = reader.get_batch(index)
        take = min(batch.num_rows, max_edges - read)
        pre_parts.append(batch.column(0).to_numpy()[:take])
        post_parts.append(batch.column(1).to_numpy()[:take])
        weight_parts.append(batch.column(2).to_numpy()[:take])
        read += take
        batches_used += 1

    pre = np.concatenate(pre_parts) if pre_parts else np.zeros(0, dtype=np.int64)
    post = np.concatenate(post_parts) if post_parts else np.zeros(0, dtype=np.int64)
    weight = np.concatenate(weight_parts) if weight_parts else np.zeros(0, dtype=np.int64)

    keep = weight >= int(min_weight)
    kept = int(keep.sum())
    pre, post, weight = pre[keep], post[keep], weight[keep]

    codes, uniques = pd.factorize(np.concatenate([pre, post]))
    half = pre.size
    # factorize returns uniques in order of appearance. Sort them and remap the
    # codes so that node_ids is ascending, which is what searchsorted (used by
    # index_of and the annotation join) requires.
    order = np.argsort(uniques, kind="stable")
    node_ids = np.asarray(uniques, dtype=np.int64)[order]
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    rows = remap[codes[:half]]
    cols = remap[codes[half:]]
    size = int(node_ids.size)

    csr = sp.coo_matrix(
        (weight.astype(np.float32), (rows, cols)), shape=(size, size)
    ).tocsr()
    csr.sum_duplicates()
    ones = sp.csr_matrix(
        (np.ones_like(csr.data), csr.indices.copy(), csr.indptr.copy()), shape=csr.shape
    )
    ones.data[:] = 1.0

    return PrefixIndex(
        node_ids=node_ids,
        csr=ones,
        weights=csr,
        edges_read=int(read),
        edges_after_threshold=kept,
        file_total_rows=int(total_rows),
        record_batches_read=batches_used,
        record_batches_total=total_batches,
        min_weight=int(min_weight),
        truncated=bool(total_rows > 0 and read < total_rows),
    )


# -------------------------------------------------------------------- seeds


@dataclass
class SeedSet:
    """Seed body ids plus where they came from."""

    body_ids: np.ndarray
    origin: str
    labels: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.body_ids.size)


def seeds_from_types(
    annotations_path: str | Path,
    types: Sequence[str],
    column: str = "type",
) -> SeedSet:
    """Collect seed body ids whose annotation ``column`` matches any of ``types``.

    Matching is exact. A neuron whose label is null is never selected, and a
    neuron with no label is never assumed to belong to a type.
    """
    resolved = Path(annotations_path)
    table = feather.read_table(resolved, columns=[column, "bodyId"]).to_pandas()
    wanted = {str(t) for t in types}
    selected = table[table[column].isin(wanted)]
    body_ids = np.sort(selected["bodyId"].to_numpy().astype(np.int64))
    labels = {
        str(b): str(t)
        for b, t in zip(selected["bodyId"], selected[column], strict=True)
    }
    return SeedSet(body_ids=body_ids, origin=f"{column} in {sorted(wanted)}", labels=labels)


def load_seed_file(path: str | Path, column: str = "bodyId") -> SeedSet:
    """Load seed body ids from a text or CSV file.

    Accepts a plain one-id-per-line file, or any CSV with an id column.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"seed file not found: {resolved}")

    if resolved.suffix.lower() in {".csv", ".tsv", ".feather", ".parquet"}:
        if resolved.suffix.lower() == ".feather":
            frame = feather.read_table(resolved, columns=None).to_pandas()
        else:
            frame = pd.read_csv(resolved)
        if column in frame.columns:
            body_ids = frame[column].dropna().to_numpy().astype(np.int64)
        else:
            candidates = [c for c in frame.columns if "id" in c.lower() or "body" in c.lower()]
            if not candidates:
                raise ValueError(
                    f"{resolved.name} has no {column!r} column and no id-like column; "
                    f"columns are {list(frame.columns)}"
                )
            body_ids = frame[candidates[0]].dropna().to_numpy().astype(np.int64)
    else:
        body_ids = np.array(
            [int(line.strip()) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()],
            dtype=np.int64,
        )

    return SeedSet(body_ids=np.unique(body_ids), origin=str(resolved.name))


def load_annotation_table(path: str | Path, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Project the annotation columns needed for coverage reporting."""
    resolved = Path(path)
    projected: dict[str, Any] = {}
    for name in ("bodyId", *columns):
        try:
            projected[name] = feather.read_table(resolved, columns=[name])
        except (KeyError, pa.ArrowInvalid, ValueError):
            continue
    if "bodyId" not in projected:
        raise ValueError(f"{resolved.name} has no bodyId column")
    frame = pd.DataFrame({name: table.column(0).to_numpy() for name, table in projected.items()})
    frame = frame.drop_duplicates(subset=["bodyId"]).sort_values("bodyId")
    out: dict[str, np.ndarray] = {"bodyId": frame["bodyId"].to_numpy().astype(np.int64)}
    for name in frame.columns:
        if name != "bodyId":
            out[name] = frame[name].to_numpy()
    return out


# ----------------------------------------------------------------- explore


@dataclass
class NeighborhoodResult:
    """Growth statistics for one bounded seeded exploration."""

    seeds: int
    seeds_present_in_prefix: int
    hops: int
    direction: str
    min_weight: int
    max_nodes: int
    max_edges: int
    stages: list[dict[str, Any]] = field(default_factory=list)
    final_nodes: int = 0
    final_edges: int = 0
    budget_binding: str = "none"
    prefix: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    node_type_top: dict[str, int] = field(default_factory=dict)
    node_superclass_top: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "what_this_is": (
                "seeded connectivity neighborhood: a bounded, weight-truncated expansion from a seed "
                "set. It is NOT the complete visual circuit and no claim is made that it is one."
            ),
            "edge_count_semantics": (
                "edges here are DISCOVERY edges: the edges that first brought a new node into the "
                "neighborhood. They are not the number of induced edges among the visited nodes, "
                "which is larger."
            ),
            "seeds": self.seeds,
            "seeds_present_in_prefix": self.seeds_present_in_prefix,
            "hops": self.hops,
            "direction": self.direction,
            "minimum_synapse_count_threshold": self.min_weight,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "budget_binding": self.budget_binding,
            "stages": self.stages,
            "final_nodes": self.final_nodes,
            "final_edges": self.final_edges,
            "prefix": self.prefix,
            "coverage": self.coverage,
            "node_type_top": self.node_type_top,
            "node_superclass_top": self.node_superclass_top,
        }


def explore(
    index: PrefixIndex,
    seeds: SeedSet,
    hops: int = 1,
    direction: str = "both",
    max_nodes: int = 5_000,
    max_edges: int = 50_000,
    annotations: dict[str, np.ndarray] | None = None,
    top: int = 10,
) -> NeighborhoodResult:
    """Expand from ``seeds`` up to ``hops``, stopping deterministically at budget.

    Determinism: the frontier is always processed in ascending body-id order, so
    the same inputs always yield the same node set, independently of dict or
    set iteration order.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if hops < 0:
        raise ValueError("hops must be >= 0")

    seed_positions = index.index_of(seeds.body_ids)
    if seed_positions.size >= max_nodes:
        raise ValueError(
            f"the seed set alone ({seed_positions.size} bodies) meets or exceeds max_nodes={max_nodes}. "
            "max_nodes bounds the whole visited set, and seeds are never dropped to fit a budget. "
            "Raise --max-nodes, or use fewer seed types."
        )
    result = NeighborhoodResult(
        seeds=len(seeds),
        seeds_present_in_prefix=int(seed_positions.size),
        hops=hops,
        direction=direction,
        min_weight=index.min_weight,
        max_nodes=max_nodes,
        max_edges=max_edges,
        prefix={
            "edges_read": index.edges_read,
            "edges_after_minimum_synapse_count_threshold": index.edges_after_threshold,
            "file_total_rows": index.file_total_rows,
            "record_batches_read": index.record_batches_read,
            "record_batches_total": index.record_batches_total,
            "truncated": index.truncated,
            "prefix_nodes": index.num_nodes,
            "dense_would_be_bytes": index.dense_bytes_if_materialised(),
        },
    )

    csc = index.csr.tocsc() if direction in ("upstream", "both") else None

    visited: set[int] = set(int(p) for p in seed_positions)
    frontier = sorted(visited)
    edges_kept = 0
    result.stages.append(
        {
            "stage": "seed",
            "new_nodes": int(frontier.__len__()),
            "cumulative_nodes": len(visited),
            "cumulative_edges": 0,
        }
    )

    budget_binding = "none"
    for hop in range(1, hops + 1):
        new_frontier: list[int] = []
        for node in frontier:
            if len(visited) >= max_nodes or edges_kept >= max_edges:
                budget_binding = "max_nodes" if len(visited) >= max_nodes else "max_edges"
                break
            if direction in ("downstream", "both"):
                start, end = index.csr.indptr[node], index.csr.indptr[node + 1]
                candidates = index.csr.indices[start:end]
            else:
                candidates = np.zeros(0, dtype=np.int64)
            if direction in ("upstream", "both"):
                assert csc is not None
                cstart, cend = csc.indptr[node], csc.indptr[node + 1]
                candidates = np.concatenate([candidates, csc.indices[cstart:cend]])
            for candidate in np.sort(candidates):
                candidate = int(candidate)
                if candidate in visited:
                    continue
                if len(visited) >= max_nodes:
                    budget_binding = "max_nodes"
                    break
                if edges_kept >= max_edges:
                    budget_binding = "max_edges"
                    break
                visited.add(candidate)
                new_frontier.append(candidate)
                edges_kept += 1
            if budget_binding != "none":
                break

        result.stages.append(
            {
                "stage": f"{hop}-hop {direction}",
                "new_nodes": len(new_frontier),
                "cumulative_nodes": len(visited),
                "cumulative_edges": edges_kept,
            }
        )
        frontier = new_frontier
        if budget_binding != "none" or not frontier:
            break

    if budget_binding == "none" and hops > 0 and not frontier:
        budget_binding = "frontier_exhausted"

    result.final_nodes = len(visited)
    result.final_edges = edges_kept
    result.budget_binding = budget_binding

    if annotations is not None:
        result.coverage, result.node_type_top, result.node_superclass_top = _coverage(
            index, np.array(sorted(visited), dtype=np.int64), annotations, top
        )
    return result


def _coverage(
    index: PrefixIndex,
    positions: np.ndarray,
    annotations: dict[str, np.ndarray],
    top: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    """Annotation coverage over the neighborhood, with nulls kept separate."""
    body_ids = index.node_ids[positions] if positions.size else np.zeros(0, dtype=np.int64)
    ann_ids = annotations["bodyId"]
    lookup = np.clip(np.searchsorted(ann_ids, body_ids), 0, max(ann_ids.size - 1, 0))
    present = ann_ids[lookup] == body_ids if ann_ids.size else np.zeros(body_ids.size, dtype=bool)

    summary: dict[str, Any] = {
        "nodes": int(body_ids.size),
        "in_annotation_table": int(present.sum()),
        "percent_in_annotation_table": round(100.0 * float(present.mean()), 2) if body_ids.size else 0.0,
    }

    def _counts(field: str) -> tuple[int, dict[str, int]]:
        if field not in annotations or not present.any():
            return 0, {}
        values = annotations[field][lookup[present]]
        series = pd.Series(values)
        has_label = int(pd.Series(series).notna().sum())
        counts = series.dropna().astype(str).value_counts().head(top)
        return has_label, {str(k): int(v) for k, v in counts.items()}

    type_labelled, type_counts = _counts("type")
    superclass_labelled, superclass_counts = _counts("superclass")
    summary["with_type_label"] = type_labelled
    summary["percent_with_type_label"] = (
        round(100.0 * type_labelled / body_ids.size, 2) if body_ids.size else 0.0
    )
    summary["with_superclass_label"] = superclass_labelled
    summary["note"] = (
        "an unlabelled neuron has an unknown label; it is not evidence of non-membership in any category"
    )
    return summary, type_counts, superclass_counts
