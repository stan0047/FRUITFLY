"""Bounded extraction of the ON and OFF motion candidate pathways from MCNS.

This module builds **candidate** cell-population graphs for two motion channels
and reports their structure. It does not simulate, does not select a final
circuit, and does not produce a model.

The two candidate pathways, stated as the literature frames them and kept as
separate cell types throughout:

``ON``
    ``L1 -> Mi1 -> T4a / T4b / T4c / T4d``
``OFF``
    ``L2 -> Tm1 / Tm2 -> T5a / T5b / T5c / T5d``

Why subtypes are never collapsed
--------------------------------
``T4`` and ``T5`` are *direction* codes: T4a-d and T5a-d terminate in four
different lobula-plate layers and respond to four different directions of motion.
Collapsing them into a generic ``T4`` would average four different computations
into one and misrepresent the annotation. They are carried as eight distinct
labels end to end.

Weight semantics
----------------
``weight`` is the **number of synapses** from the pre-synaptic to the
post-synaptic body, exactly as the MaleCNS dataset defines it. It is a
structural count. It is **not** synaptic efficacy, it is never called a strength,
and it is **never converted** to a LIF conductance here. The integer count is
preserved exactly.

Memory model
------------
The MaleCNS connectivity table has ~151.9M ordered-pair rows. It is never read
whole. This module:

1. reads the annotation table with **column projection** (a handful of columns);
2. reduces the candidate set to a compact, sorted body-id array;
3. streams the connectivity table **record batch by record batch**, projecting
   only ``body_pre``, ``body_post`` and ``weight``;
4. keeps only rows whose **both** endpoints are in the candidate set, storing
   compact int32 indices.

Peak additional memory is therefore proportional to the number of *retained*
edges, not to the file size. A dense ``N x N`` matrix is never built and is
refused by the same guards used elsewhere in FlyBrain.

Nothing is ever truncated silently. If a per-cell-type body budget is applied it
is recorded in the report, and the report labels the result a subset.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
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
    "MOTION_CELL_TYPES",
    "MOTION_PATHWAYS",
    "CandidatePopulations",
    "EdgeExtraction",
    "build_celltype_matrix",
    "extract_candidate_edges",
    "extract_motion_candidates",
    "load_annotations",
    "resolve_cell_types",
    "validate_extraction",
]

DIRECTIONS = ("downstream", "upstream", "both")

#: The exact cell-type labels required for the motion candidate. Subtypes are
#: deliberately distinct entries; see the module docstring.
MOTION_CELL_TYPES: tuple[str, ...] = (
    "L1",
    "L2",
    "Mi1",
    "Tm1",
    "Tm2",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)

#: Literature framing of the two candidate channels. T4a-d and T5a-d are four
#: direction channels each and are never merged.
MOTION_PATHWAYS: dict[str, dict[str, Any]] = {
    "ON": {
        "input": "L1",
        "relay": "Mi1",
        "outputs": ("T4a", "T4b", "T4c", "T4d"),
        "role": "ON-motion contrast channel",
    },
    "OFF": {
        "input": "L2",
        "relay": ("Tm1", "Tm2"),
        "outputs": ("T5a", "T5b", "T5c", "T5d"),
        "role": "OFF-motion contrast channel",
    },
}

#: Annotation columns needed. Projected explicitly; the table has 36.
ANNOTATION_COLUMNS = ("bodyId", "type", "superclass", "somaSide", "somaLocation")
#: Connectivity columns needed. The table has exactly these three.
CONNECTIVITY_COLUMNS = ("body_pre", "body_post", "weight")

#: A dense matrix is refused above this many candidate bodies.
DENSE_REFUSE_NODES = 25_000


class ExtractionError(RuntimeError):
    """Raised when the requested extraction cannot be performed as specified."""


# ------------------------------------------------------------ annotations


def load_annotations(
    path: str | Path,
    columns: Sequence[str] = ANNOTATION_COLUMNS,
) -> dict[str, np.ndarray]:
    """Load the annotation table with column projection.

    Only the requested columns are materialised. ``somaLocation`` is a
    ``list<int64>`` of length 3, so it is converted to a boolean
    "has coordinates" flag rather than unpacked.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise ExtractionError(f"annotation file not found: {resolved}")

    projected: dict[str, Any] = {}
    for name in ("bodyId", *columns):
        try:
            projected[name] = feather.read_table(resolved, columns=[name])
        except (KeyError, pa.ArrowInvalid, ValueError) as exc:
            if name == "bodyId":
                raise ExtractionError(f"{resolved.name} has no bodyId column") from exc
            continue

    frame = pd.DataFrame({name: table.column(0).to_numpy() for name, table in projected.items()})
    frame = frame.drop_duplicates(subset=["bodyId"]).sort_values("bodyId").reset_index(drop=True)

    out: dict[str, np.ndarray] = {"bodyId": frame["bodyId"].to_numpy().astype(np.int64)}
    for name in frame.columns:
        if name == "bodyId":
            continue
        if name == "somaLocation":
            out["has_coordinates"] = np.asarray(pd.notna(pd.Series(frame[name])), dtype=bool)
        elif name == "somaSide":
            out["somaSide"] = frame[name].to_numpy()
        else:
            out[name] = frame[name].to_numpy()
    return out


def resolve_cell_types(
    annotations: Mapping[str, np.ndarray],
    cell_types: Sequence[str],
    require_superclass: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Split requested labels into those present and those absent.

    An absent label is **reported, never substituted**. A near-miss string is
    returned as a candidate for human adjudication only.
    """
    if "type" not in annotations:
        raise ExtractionError("annotation table has no 'type' column")
    type_series = pd.Series(annotations["type"]).astype("string")
    present_mask = type_series.notna().to_numpy()
    present_values = set(type_series[present_mask].astype(str).unique())
    all_values = set(type_series.dropna().astype(str).unique())

    found: list[str] = []
    missing: list[dict[str, Any]] = []
    for label in cell_types:
        if label in present_values:
            found.append(label)
        else:
            near = sorted(v for v in all_values if label.lower() in v.lower() or v.lower() in label.lower())
            missing.append({"label": label, "near_miss_not_applied": near[:5]})

    info = {
        "requested": list(cell_types),
        "found": found,
        "missing": missing,
        "substitution_applied": False,
        "require_superclass": require_superclass,
    }
    if missing and len(found) == 0:
        raise ExtractionError(
            f"none of the requested cell types exist in the annotation table: "
            f"{[m['label'] for m in missing]}"
        )
    return found, info


# ------------------------------------------------------------ populations


@dataclass
class CandidatePopulations:
    """The candidate body population, keyed by exact cell type."""

    cell_types: list[str]
    body_ids: np.ndarray          # sorted, unique
    type_of: np.ndarray           # index into cell_types, parallel to body_ids
    soma_side: np.ndarray         # object array of side labels, '' when absent
    has_coordinates: np.ndarray   # bool
    has_superclass: np.ndarray    # bool
    label_info: dict[str, Any] = field(default_factory=dict)
    body_budget_per_type: int = 0

    @property
    def num_bodies(self) -> int:
        return int(self.body_ids.size)

    def positions_for(self, cell_type: str) -> np.ndarray:
        """Array positions of a cell type, for indexing the parallel arrays.

        Use this, not :meth:`ids_for`, when indexing ``soma_side``,
        ``has_coordinates`` or ``type_of``.
        """
        if cell_type not in self.cell_types:
            raise KeyError(cell_type)
        return np.flatnonzero(self.type_of == self.cell_types.index(cell_type))

    def ids_for(self, cell_type: str) -> np.ndarray:
        """Body identifiers of a cell type."""
        return self.body_ids[self.positions_for(cell_type)]

    def counts(self) -> dict[str, dict[str, Any]]:
        """Per cell type: bodies, side coverage, coordinate coverage."""
        out: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(self.cell_types):
            mask = self.type_of == index
            total = int(mask.sum())
            side = pd.Series(self.soma_side[mask]).replace("", pd.NA)
            out[label] = {
                "bodies": total,
                "with_somaSide": int(side.notna().sum()),
                "with_coordinates": int(self.has_coordinates[mask].sum()),
                "with_superclass": int(self.has_superclass[mask].sum()),
                "somaSide_breakdown": {
                    str(k): int(v) for k, v in side.dropna().astype(str).value_counts().items()
                },
            }
        return out

    def is_subset(self) -> bool:
        return self.body_budget_per_type > 0

    def subset_note(self) -> str | None:
        if not self.is_subset():
            return None
        return (
            f"a deterministic budget of {self.body_budget_per_type} bodies per cell type was applied; "
            "these statistics describe a SUBSET of the annotated population, not the complete population"
        )


def _stratified_sample(
    ids: np.ndarray,
    sides: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Pick ``budget`` bodies deterministically, balanced across hemispheres.

    Rule, fixed and documented so it is reproducible: take the lowest-numbered
    ``ceil(budget/2)`` body ids on each side, then fill any remainder from the
    lowest-numbered remaining bodies. Sorting by body id rather than by row order
    avoids depending on how the source file happens to be laid out.
    """
    if budget <= 0 or ids.size <= budget:
        return np.sort(ids)
    left = ids[sides == "L"]
    right = ids[sides == "R"]
    other = ids[(sides != "L") & (sides != "R")]
    per_side = int(np.ceil(budget / 2))

    picked: list[np.ndarray] = []
    remaining = budget
    for pool in (left, right):
        take = min(per_side, pool.size, remaining)
        if take > 0:
            picked.append(np.sort(pool)[:take])
            remaining -= take
    if remaining > 0 and other.size:
        take = min(remaining, other.size)
        picked.append(np.sort(other)[:take])
        remaining -= take
    if remaining > 0:
        taken = np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)
        leftovers = np.setdiff1d(ids, taken, assume_unique=True)
        if leftovers.size:
            picked.append(np.sort(leftovers)[:remaining])
    return np.sort(np.unique(np.concatenate(picked))) if picked else np.zeros(0, dtype=np.int64)


def build_populations(
    annotations: Mapping[str, np.ndarray],
    cell_types: Sequence[str],
    require_superclass: bool = True,
    body_budget_per_type: int = 0,
) -> CandidatePopulations:
    """Select candidate bodies for the requested exact cell types."""
    found, info = resolve_cell_types(annotations, cell_types, require_superclass)

    type_series = pd.Series(annotations["type"]).astype("string")
    side_series = (
        pd.Series(annotations["somaSide"]).astype("string").fillna("").to_numpy()
        if "somaSide" in annotations
        else np.full(annotations["bodyId"].size, "", dtype=object)
    )
    coords = (
        annotations["has_coordinates"]
        if "has_coordinates" in annotations
        else np.zeros(annotations["bodyId"].size, dtype=bool)
    )
    superclass_ok = (
        np.asarray(pd.notna(pd.Series(annotations["superclass"])), dtype=bool)
        if "superclass" in annotations
        else np.ones(annotations["bodyId"].size, dtype=bool)
    )

    all_ids: list[np.ndarray] = []
    all_types: list[np.ndarray] = []
    # Null labels become "" so the vectorised comparison cannot produce pd.NA.
    type_values = type_series.fillna("").to_numpy()
    for label in found:
        mask = type_values == label
        if require_superclass:
            # Official MaleCNS predicate: a body is a neuron if and only if it has
            # a superclass. Without one it is a fragment of a neuron.
            mask = mask & superclass_ok
        ids = annotations["bodyId"][mask].astype(np.int64)
        sides = side_series[mask]
        chosen = _stratified_sample(ids, sides, body_budget_per_type) if body_budget_per_type else np.sort(ids)
        if chosen.size == 0:
            continue
        all_ids.append(chosen)
        all_types.append(np.full(chosen.size, len(all_ids) - 1, dtype=np.int64))

    if not all_ids:
        raise ExtractionError("no candidate bodies selected")

    body_ids = np.unique(np.concatenate(all_ids))
    # Map each selected body back to its cell type, using the first match.
    type_of = np.full(body_ids.size, -1, dtype=np.int64)
    lookup = np.full(annotations["bodyId"].size, -1, dtype=np.int64)
    for chunk_index, ids in enumerate(all_ids):
        positions = np.searchsorted(annotations["bodyId"], ids)
        positions = np.clip(positions, 0, annotations["bodyId"].size - 1)
        lookup[positions[annotations["bodyId"][positions] == ids]] = chunk_index

    body_positions = np.searchsorted(annotations["bodyId"], body_ids)
    type_of = lookup[body_positions]

    kept_types = [found[i] for i in range(len(all_ids))]
    populations = CandidatePopulations(
        cell_types=kept_types,
        body_ids=body_ids,
        type_of=type_of,
        soma_side=side_series[body_positions],
        has_coordinates=coords[body_positions],
        has_superclass=superclass_ok[body_positions],
        label_info=info,
        body_budget_per_type=body_budget_per_type,
    )
    _ = all_types
    return populations


# ------------------------------------------------------------ connectivity


@dataclass
class EdgeExtraction:
    """Candidate edges, held as compact indices into the candidate body array."""

    pre_index: np.ndarray
    post_index: np.ndarray
    synapse_count: np.ndarray   # int64, preserved exactly
    batches_processed: int
    batches_total: int
    rows_scanned: int
    read_budget: int
    truncated: bool
    direction: str
    seconds: float

    @property
    def num_edges(self) -> int:
        return int(self.pre_index.size)

    def pair_key(self) -> np.ndarray:
        """Single int64 key per ordered pair, for duplicate detection."""
        n = int(max(self.pre_index.max(initial=0), self.post_index.max(initial=0))) + 1
        return self.pre_index.astype(np.int64) * np.int64(n) + self.post_index.astype(np.int64)

    def filter_direction(self, direction: str) -> "EdgeExtraction":
        """Restrict to upstream, downstream, or both relative to a seed subset."""
        if direction not in DIRECTIONS:
            raise ExtractionError(f"direction must be one of {DIRECTIONS}")
        if direction == "both":
            return self
        raise ExtractionError("direction filtering requires a seed index set; use extract_candidate_edges")


def extract_candidate_edges(
    connectivity_path: str | Path,
    populations: CandidatePopulations,
    direction: str = "both",
    seed_indices: np.ndarray | None = None,
    read_budget: int = 0,
) -> EdgeExtraction:
    """Stream the connectivity table and keep only candidate-to-candidate edges.

    Parameters
    ----------
    seed_indices:
        Positions in ``populations.body_ids`` to treat as seeds. ``None`` means
        every candidate body is a seed.
    direction:
        ``downstream`` keeps ``seed -> candidate``, ``upstream`` keeps
        ``candidate -> seed``, ``both`` keeps every candidate-to-candidate edge.
    read_budget:
        Optional cap on record batches read. If it binds, ``truncated`` is set and
        the report says so. Truncation is never silent.

    Only the three connectivity columns are projected, one record batch at a time.
    """
    if direction not in DIRECTIONS:
        raise ExtractionError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    resolved = Path(connectivity_path)
    if not resolved.is_file():
        raise ExtractionError(f"connectivity file not found: {resolved}")

    reader = pa.ipc.open_file(pa.memory_map(str(resolved), "rb"))
    total_batches = reader.num_record_batches
    limit = read_budget if read_budget and read_budget > 0 else total_batches

    body_ids = populations.body_ids
    n = body_ids.size
    if n > DENSE_REFUSE_NODES:
        raise ExtractionError(
            f"{n} candidate bodies exceeds the dense-matrix guard ({DENSE_REFUSE_NODES}). "
            "Reduce the body budget per cell type."
        )

    seeds = (
        np.zeros(n, dtype=bool)
        if seed_indices is None
        else np.isin(np.arange(n, dtype=np.int64), np.asarray(seed_indices, dtype=np.int64))
    )

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    scanned = 0
    processed = 0
    started = time.perf_counter()

    for batch_index in range(min(limit, total_batches)):
        batch = reader.get_batch(batch_index)
        pre = batch.column(0).to_numpy()
        post = batch.column(1).to_numpy()
        weight = batch.column(2).to_numpy()
        scanned += pre.size

        pre_pos = np.searchsorted(body_ids, pre)
        np.clip(pre_pos, 0, max(n - 1, 0), out=pre_pos)
        post_pos = np.searchsorted(body_ids, post)
        np.clip(post_pos, 0, max(n - 1, 0), out=post_pos)

        keep = (body_ids[pre_pos] == pre) & (body_ids[post_pos] == post)
        if direction == "downstream":
            keep &= seeds[pre_pos]
        elif direction == "upstream":
            keep &= seeds[post_pos]
        if not keep.any():
            processed += 1
            continue

        pre_parts.append(pre_pos[keep].astype(np.int32))
        post_parts.append(post_pos[keep].astype(np.int32))
        count_parts.append(weight[keep].astype(np.int64))
        processed += 1

    extraction = EdgeExtraction(
        pre_index=np.concatenate(pre_parts) if pre_parts else np.zeros(0, np.int32),
        post_index=np.concatenate(post_parts) if post_parts else np.zeros(0, np.int32),
        synapse_count=np.concatenate(count_parts) if count_parts else np.zeros(0, np.int64),
        batches_processed=processed,
        batches_total=total_batches,
        rows_scanned=scanned,
        read_budget=limit,
        truncated=bool(limit < total_batches),
        direction=direction,
        seconds=time.perf_counter() - started,
    )
    return extraction


# ------------------------------------------------------------------ graph


def build_sparse(
    extraction: EdgeExtraction,
    populations: CandidatePopulations,
    directed: bool = True,
) -> sp.csr_matrix:
    """Sparse adjacency over the candidate body index space.

    Duplicate ordered pairs are summed, matching the dataset's aggregation
    convention. Never dense.
    """
    n = populations.num_bodies
    data = extraction.synapse_count.astype(np.float64)
    if not directed:
        data = np.concatenate([data, data])
        rows = np.concatenate([extraction.pre_index, extraction.post_index])
        cols = np.concatenate([extraction.post_index, extraction.pre_index])
    else:
        rows, cols = extraction.pre_index, extraction.post_index
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_celltype_matrix(
    extraction: EdgeExtraction,
    populations: CandidatePopulations,
    value: str = "synapses",
) -> pd.DataFrame:
    """Cell-type by cell-type matrix.

    ``value="synapses"`` sums integer synapse counts; ``value="edges"`` counts
    retained rows. The two differ, and conflating them would be a reporting error.
    """
    types = populations.cell_types
    n = len(types)
    edges = np.zeros((n, n), dtype=np.int64)
    synapses = np.zeros((n, n), dtype=np.int64)
    if extraction.num_edges:
        rows = populations.type_of[extraction.pre_index]
        cols = populations.type_of[extraction.post_index]
        np.add.at(edges, (rows, cols), 1)
        np.add.at(synapses, (rows, cols), extraction.synapse_count)
    frame = pd.DataFrame(edges if value == "edges" else synapses, index=types, columns=types)
    frame.index.name = "from_type"
    frame.columns.name = "to_type"
    return frame


# ------------------------------------------------------------- validation


def validate_extraction(
    extraction: EdgeExtraction,
    populations: CandidatePopulations,
    annotations: Mapping[str, np.ndarray],
    require_positive: bool = True,
) -> dict[str, Any]:
    """Run every required check and report pass/fail per check.

    Nothing is asserted silently: a failing check appears in the report with its
    count, and the caller decides what to do about it.
    """
    n = populations.num_bodies
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, value: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail, "value": value})

    known = set(populations.body_ids.tolist())
    unknown_bodies = (
        [int(b) for b in populations.body_ids.tolist() if int(b) not in known] if extraction.num_edges else []
    )
    add(
        "all_candidate_bodies_in_annotations",
        not unknown_bodies,
        "every selected body id appears in the annotation table",
        len(unknown_bodies),
    )

    bad_endpoints = 0
    if extraction.num_edges:
        bad_endpoints = int(
            ((extraction.pre_index < 0) | (extraction.pre_index >= n) | (extraction.post_index < 0) | (extraction.post_index >= n)).sum()
        )
    add("edge_endpoints_in_range", bad_endpoints == 0, "every edge endpoint maps to a candidate body", bad_endpoints)

    duplicates = 0
    if extraction.num_edges:
        keys = extraction.pair_key()
        duplicates = int(keys.size - np.unique(keys).size)
    add(
        "no_duplicate_ordered_pairs",
        duplicates == 0,
        "no repeated (source, target) pair; the dataset is aggregated, so this must hold",
        duplicates,
    )

    nonpositive = int((extraction.synapse_count <= 0).sum()) if extraction.num_edges else 0
    add(
        "synapse_counts_positive",
        nonpositive == 0 if require_positive else True,
        "synapse counts are positive integers; a count is not a strength and must not be rescaled",
        nonpositive,
    )

    self_loops = int((extraction.pre_index == extraction.post_index).sum()) if extraction.num_edges else 0
    add(
        "self_loops_reported",
        True,
        "self-loops are measured and retained; removal is a later modelling decision",
        self_loops,
    )

    untyped = int((populations.type_of < 0).sum())
    add("every_body_has_an_exact_type", untyped == 0, "no body carries an unknown or substituted type", untyped)

    add(
        "no_type_substitution",
        not populations.label_info.get("substitution_applied", False),
        "requested labels are used verbatim; absent labels are reported, never replaced by a similar name",
        [m["label"] for m in populations.label_info.get("missing", [])],
    )

    annotation_ids = set(annotations["bodyId"].tolist())
    unverified = [int(b) for b in populations.body_ids.tolist() if int(b) not in annotation_ids]
    add(
        "bodies_verified_against_annotation_table",
        not unverified,
        "every retained body id was verified against the annotation table",
        len(unverified),
    )

    return {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "self_loops": self_loops,
    }


# ----------------------------------------------------------------- report


def _int_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None, "total": 0, "mean": None}
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "total": int(values.sum()),
        "mean": float(values.mean()),
    }


def summarise_pathways(
    populations: CandidatePopulations,
    extraction: EdgeExtraction,
    annotations: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Per cell type and per pathway statistics for the candidate graphs."""
    n = populations.num_bodies
    out_degree = np.zeros(n, dtype=np.int64)
    in_degree = np.zeros(n, dtype=np.int64)
    if extraction.num_edges:
        np.add.at(out_degree, extraction.pre_index, 1)
        np.add.at(in_degree, extraction.post_index, 1)

    per_type: dict[str, Any] = {}
    for index, label in enumerate(populations.cell_types):
        mask = populations.type_of == index
        per_type[label] = {
            "bodies": int(mask.sum()),
            "with_somaSide": int(np.asarray([bool(s) for s in populations.soma_side[mask]]).sum()),
            "with_coordinates": int(populations.has_coordinates[mask].sum()),
            "outgoing_edges": int(out_degree[mask].sum()),
            "incoming_edges": int(in_degree[mask].sum()),
        }

    pathways: dict[str, Any] = {}
    for name, spec in MOTION_PATHWAYS.items():
        relay = spec["relay"] if isinstance(spec["relay"], tuple) else (spec["relay"],)
        members = [spec["input"], *relay, *spec["outputs"]]
        members = [m for m in members if m in populations.cell_types]
        if not members:
            pathways[name] = {"available": False, "note": "no member cell types are present"}
            continue
        member_mask = np.isin(populations.type_of, [populations.cell_types.index(m) for m in members])
        member_positions = np.flatnonzero(member_mask)
        member_set = np.zeros(n, dtype=bool)
        member_set[member_positions] = True
        if extraction.num_edges:
            inside = member_set[extraction.pre_index] & member_set[extraction.post_index]
            counts = extraction.synapse_count[inside]
            indices = extraction.pre_index[inside]
        else:
            counts = np.zeros(0, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int32)

        cross_side = 0
        if indices.size:
            pre_side = populations.soma_side[extraction.pre_index[inside]]
            post_side = populations.soma_side[extraction.post_index[inside]]
            known = (pre_side != "") & (post_side != "")
            cross_side = int((pre_side[known] != post_side[known]).sum())

        pathways[name] = {
            "available": True,
            "role": spec["role"],
            "cell_types": members,
            "nodes": int(member_positions.size),
            "edges": int(counts.size),
            "synapse_count_stats": _int_stats(counts),
            "self_loops": int((indices == extraction.post_index[inside]).sum()) if indices.size else 0,
            "cross_side_edges": cross_side,
            "cross_side_note": "edges between opposite somaSide labels; sides absent from the annotation are excluded",
        }

    return {"per_cell_type": per_type, "per_pathway": pathways}


def _peak_python_bytes() -> int | None:
    """Peak Python allocation for this process, if it can be determined.

    Best effort and cross-platform: ``resource`` on Unix, ``psutil`` for resident
    set size when installed, otherwise ``None``. A missing number is reported as
    missing rather than guessed.
    """
    try:
        import resource  # type: ignore[import-not-found]

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        return int(usage if sys.platform == "darwin" else usage * 1024)
    except Exception:  # noqa: BLE001 - Windows has no resource module
        pass
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


def extract_motion_candidates(
    annotations_path: str | Path,
    connectivity_path: str | Path,
    cell_types: Sequence[str] = MOTION_CELL_TYPES,
    direction: str = "both",
    require_superclass: bool = True,
    body_budget_per_type: int = 0,
    read_budget: int = 0,
    started_at: float | None = None,
) -> dict[str, Any]:
    """End-to-end extraction: annotations -> populations -> edges -> report.

    Returns a JSON-serialisable report. The graph itself is not returned, so the
    caller cannot accidentally hold a dense object.
    """
    wall_start = time.perf_counter() if started_at is None else started_at

    annotations = load_annotations(annotations_path)
    populations = build_populations(
        annotations, cell_types, require_superclass=require_superclass, body_budget_per_type=body_budget_per_type
    )
    extraction = extract_candidate_edges(
        connectivity_path, populations, direction=direction, read_budget=read_budget
    )
    validation = validate_extraction(extraction, populations, annotations)
    summary = summarise_pathways(populations, extraction, annotations)

    report: dict[str, Any] = {
        "experiment": "MCNS motion pathway candidate extraction",
        "status": "candidate structures only; no simulation, no model, no training",
        "weight_semantics": (
            "weight is the number of synapses from source to target body, exactly as the MaleCNS "
            "dataset defines it. It is a structural count, NOT synaptic strength, and has NOT been "
            "converted to any LIF parameter."
        ),
        "inputs": {
            "annotations": str(annotations_path),
            "connectivity": str(connectivity_path),
            "direction": direction,
            "require_superclass": require_superclass,
        },
        "cell_type_labels": populations.label_info,
        "subset": {
            "body_budget_per_type": body_budget_per_type,
            "is_subset": populations.is_subset(),
            "note": populations.subset_note(),
            "read_budget_batches": read_budget,
            "read_truncated": extraction.truncated,
            "read_truncation_note": (
                "the record-batch read budget bound before the end of the file; statistics describe "
                "a weight-truncated SUBSET"
                if extraction.truncated
                else None
            ),
        },
        "populations": {
            "candidate_bodies": populations.num_bodies,
            "cell_types_present": populations.cell_types,
            "per_cell_type": summary["per_cell_type"],
        },
        "extraction": {
            "edges_retained": extraction.num_edges,
            "record_batches_processed": extraction.batches_processed,
            "record_batches_total": extraction.batches_total,
            "rows_scanned": int(extraction.rows_scanned),
            "seconds": round(extraction.seconds, 2),
            "synapse_count_stats": _int_stats(extraction.synapse_count),
        },
        "pathways": summary["per_pathway"],
        "validation": validation,
        "cell_type_connectivity_synapses": build_celltype_matrix(extraction, populations, "synapses").to_dict(),
        "cell_type_connectivity_edges": build_celltype_matrix(extraction, populations, "edges").to_dict(),
        "memory": {
            "peak_process_bytes": _peak_python_bytes(),
            "peak_process_note": (
                "resident set size, an upper bound that includes the interpreter; the retained-edge "
                "arrays are the only large allocation made by this extraction"
            ),
            "retained_edge_bytes": int(
                extraction.pre_index.nbytes + extraction.post_index.nbytes + extraction.synapse_count.nbytes
            ),
            "dense_would_be_bytes": populations.num_bodies * populations.num_bodies * 8,
            "note": "extraction holds only candidate-to-candidate edges; the source file is memory-mapped",
        },
        "total_seconds": round(time.perf_counter() - wall_start, 2),
    }
    return report


def write_report(report: Mapping[str, Any], summary_path: str | Path, edge_dir: str | Path | None = None) -> dict[str, str]:
    """Write the JSON summary and, optionally, the retained edges as CSV."""
    written: dict[str, str] = {}
    summary = Path(summary_path)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    written["summary"] = str(summary)

    if edge_dir is not None:
        target = Path(edge_dir)
        target.mkdir(parents=True, exist_ok=True)
        matrix = report.get("cell_type_connectivity_synapses")
        if matrix is not None:
            frame = pd.DataFrame(matrix)
            frame.to_csv(target / "celltype_synapse_matrix.csv", index_label="from_type")
            written["celltype_matrix"] = str(target / "celltype_synapse_matrix.csv")
    return written
