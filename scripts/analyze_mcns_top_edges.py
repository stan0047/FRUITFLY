#!/usr/bin/env python
"""Bounded analysis of the strongest connections in a MaleCNS connectome table.

This is a **data-engineering feasibility experiment**, not biology. The output is
a *top-weight MCNS connectivity subset*: the N strongest ordered neuron pairs in
the file, by synapse count. It is **not** a visual circuit, not a selected
pathway, and not a subgraph of any biological system. No visual neuron is
identified here and none should be inferred from these numbers.

Why a bounded read is possible
------------------------------
The MaleCNS ``connectome-weights`` table is stored sorted by ``weight``
descending, in contiguous equal-weight blocks. The strongest N edges are
therefore the first N rows, so this script reads only as many Arrow record
batches as it needs and never touches the remaining 150 million rows. The file
is memory-mapped, so untouched batches are never paged in.

Nothing here builds a dense matrix, and nothing here removes a row silently:
self-loops and duplicate ordered pairs are measured and reported, not deleted.
Whether to keep them is a modelling decision for a later phase.

Usage
-----
    python scripts/analyze_mcns_top_edges.py --max-edges 1000000
    python scripts/analyze_mcns_top_edges.py --max-edges 200000 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.feather as feather  # noqa: E402

#: Documented in flybrain.data.mcns_provenance.MCNS_SEMANTICS.
EDGE_COLUMNS = ("body_pre", "body_post", "weight")
#: Annotation columns projected out of the 14 MB annotation table. Projecting
#: keeps the annotation side of the analysis small; the full table is 36 columns.
ANNOTATION_COLUMNS = ("bodyId", "type", "superclass", "class", "status", "somaSide", "somaNeuromere")

DATASET_LABEL = "MaleCNS v1.0"
BIOLOGICAL_LABEL = "Biological connectome data"
SUBSET_LABEL = "top-weight MCNS connectivity subset"

#: Below this many edges the dense N x N float32 matrix is still under ~1 GB.
#: Above it, a dense conversion is refused outright (see also BrainGraph guards).
_DENSE_REFUSE_NODES = 25_000


def _bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def read_top_edges(path: Path, max_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the first ``max_edges`` rows using as few record batches as possible.

    Returns the three column arrays plus read diagnostics (batches touched, bytes
    held, and whether the file was exhausted).
    """
    if not path.is_file():
        raise SystemExit(f"error: connectome table not found: {path}")

    source = pa.memory_map(str(path), "rb")
    reader = pa.ipc.open_file(source)
    total_batches = reader.num_record_batches

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
    held = int(pre.nbytes + post.nbytes + weight.nbytes)

    diagnostics = {
        "record_batches_in_file": total_batches,
        "record_batches_read": batches_used,
        "record_batches_untouched": total_batches - batches_used,
        "edges_held_in_memory_bytes": held,
        "file_size_bytes": path.stat().st_size,
        "file_total_rows": _file_total_rows(reader),
        "file_exhausted": read >= _file_total_rows(reader) if _file_total_rows(reader) > 0 else None,
    }
    return pre, post, weight, diagnostics


def _file_total_rows(reader: pa.ipc.RecordBatchFileReader) -> int:
    """Exact total row count from the Arrow footer, without reading any data."""
    metadata = reader.schema.metadata or {}
    if b"pandas" not in metadata:
        return -1
    try:
        pandas_metadata = json.loads(metadata[b"pandas"])
    except (json.JSONDecodeError, TypeError):
        return -1
    for column in pandas_metadata.get("index_columns", []) or []:
        if column.get("kind") == "range":
            return int(column["stop"])
    return -1


def load_annotations(path: Path) -> dict[str, np.ndarray]:
    """Project the annotation columns we need.

    Returns a dict keyed by column name, where ``bodyId`` holds the body
    identifiers themselves (the per-neuron annotations are aligned to that array
    by position) and every other key holds a parallel array of values.
    """
    if not path.is_file():
        raise SystemExit(f"error: annotation table not found: {path}")
    # Project only the columns we want. Each is attempted independently so a file
    # that lacks one of them degrades to "not available" instead of failing.
    projected: dict[str, Any] = {}
    for name in ANNOTATION_COLUMNS:
        try:
            projected[name] = feather.read_table(path, columns=[name])
        except (KeyError, pa.ArrowInvalid, ValueError):
            continue
    if "bodyId" not in projected:
        raise SystemExit(f"error: {path.name} has no bodyId column")

    frame = pd.DataFrame({name: table.column(0).to_numpy() for name, table in projected.items()})
    frame = frame.drop_duplicates(subset=["bodyId"]).reset_index(drop=True)

    out: dict[str, np.ndarray] = {"bodyId": frame["bodyId"].to_numpy()}
    for name in frame.columns:
        if name != "bodyId":
            out[name] = frame[name].to_numpy()
    return out


def _breakdown(
    all_ids: np.ndarray,
    annotations: dict[str, np.ndarray],
    field: str,
    annotated: np.ndarray,
    top: int,
) -> dict[str, Any]:
    """Counts per annotation value, with nulls reported explicitly.

    An absent annotation is reported as ``(unannotated)`` and never folded into a
    real category: a missing label means the label is unknown, not that the
    neuron is outside the category.
    """
    if field not in annotations:
        return {"available": False, "note": f"column {field!r} not present in the annotation table"}
    positions = np.clip(
        np.searchsorted(annotations["bodyId"], all_ids), 0, len(annotations["bodyId"]) - 1
    )
    matches = annotations["bodyId"][positions] == all_ids
    values = annotations[field][positions]
    not_null = np.asarray(pd.notna(pd.Series(values)), dtype=bool)
    present = annotated & matches & not_null
    counts = pd.Series(values[present]).astype(str).value_counts().head(top)
    return {
        "available": True,
        "annotated_neurons": int(present.sum()),
        "unannotated_neurons": int(len(all_ids) - present.sum()),
        "distinct_values": int(pd.Series(values).nunique(dropna=True)),
        "top": {str(k): int(v) for k, v in counts.items()},
    }


def analyse(max_edges: int, edges_path: Path, annotations_path: Path, top: int) -> dict[str, Any]:
    pre, post, weight, read_info = read_top_edges(edges_path, max_edges)
    if pre.size == 0:
        raise SystemExit("error: no edges read")

    source_ids = np.unique(pre)
    target_ids = np.unique(post)
    all_ids = np.union1d(source_ids, target_ids)

    # Duplicate ordered pairs and self-loops are measured, never removed.
    pairs = np.stack([pre, post], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    self_loops = int((pre == post).sum())
    self_loop_ids = np.unique(pre[pre == post]) if self_loops else np.zeros(0, dtype=np.int64)

    annotations = load_annotations(annotations_path)
    annotated = np.isin(all_ids, annotations["bodyId"])

    node_count = int(all_ids.size)
    dense_bytes = node_count * node_count * 4

    report: dict[str, Any] = {
        "dataset": DATASET_LABEL,
        "biological": True,
        "what_this_is": (
            f"{SUBSET_LABEL}: the {int(pre.size):,} strongest ordered neuron pairs by synapse count, "
            "read from the head of a descending-sorted table. This is a data-engineering feasibility "
            "experiment. It is NOT a visual circuit and no neuron here has been selected for being visual."
        ),
        "biological_data_status": {
            "provided_by_dataset": [
                "body_pre / body_post: segment (body) ids, directed, per official dataset documentation",
                "weight: segment-to-segment connection strength, integer synapse count",
                "node annotations: curated classes, types, sides, status",
            ],
            "derived_by_flybrain": [
                "the ordered-pair subset itself (a head-of-file read, not a dataset product)",
                "duplicate-pair and self-loop counts",
                "annotation coverage percentages",
                "per-category breakdowns of the subset",
            ],
            "assumed_by_flybrain": [
                "nothing: no weight is synthesised, and no neuron is assigned a category it lacks",
            ],
        },
        "edges": {
            "read": int(pre.size),
            "requested": int(max_edges),
            "file_total_rows": read_info["file_total_rows"],
            "fraction_of_file": (
                round(100.0 * pre.size / read_info["file_total_rows"], 4)
                if read_info["file_total_rows"] and read_info["file_total_rows"] > 0
                else None
            ),
            "unique_ordered_pairs": int(unique_pairs.shape[0]),
            "duplicate_ordered_pairs": int(pre.size - unique_pairs.shape[0]),
            "self_loops": self_loops,
            "self_loop_neurons": int(self_loop_ids.size),
            "self_loop_policy": "measured and retained; removal is a later modelling decision",
        },
        "nodes": {
            "unique_source_neurons": int(source_ids.size),
            "unique_target_neurons": int(target_ids.size),
            "unique_neurons_overall": node_count,
        },
        "weights": {
            "min": int(weight.min()),
            "max": int(weight.max()),
            "mean": float(weight.mean()),
            "median": float(np.median(weight)),
            "sum": int(weight.sum()),
        },
        "annotation_coverage": {
            "annotated_neurons": int(annotated.sum()),
            "unannotated_neurons": int((~annotated).sum()),
            "percent": round(100.0 * float(annotated.mean()), 2) if node_count else 0.0,
            "note": "an unannotated neuron has an unknown label; it is not evidence of non-membership",
        },
        "breakdowns": {
            field: _breakdown(all_ids, annotations, field, annotated, top)
            for field in ("class", "superclass", "status", "somaSide", "somaNeuromere", "type")
        },
        "read": {
            **{k: v for k, v in read_info.items() if k != "file_exhausted"},
            "file_exhausted": read_info["file_exhausted"],
        },
        "memory": {
            "edge_arrays_bytes": read_info["edges_held_in_memory_bytes"],
            "edge_arrays_human": _bytes(read_info["edges_held_in_memory_bytes"]),
            "dense_nxn_float32_bytes": int(dense_bytes),
            "dense_nxn_human": _bytes(int(dense_bytes)),
            "dense_policy": (
                f"a dense {node_count}x{node_count} float32 matrix would need {_bytes(int(dense_bytes))}; "
                "never built. Sparse CSR is used downstream."
            ),
        },
        "guard": {
            "dense_refuse_above_nodes": _DENSE_REFUSE_NODES,
            "this_subset_is_below_guard": node_count <= _DENSE_REFUSE_NODES,
        },
    }
    return report


def render(report: dict[str, Any], width: int = 78) -> str:
    lines = [
        "",
        "MCNS top-weight connectivity subset (bounded read)",
        "=" * width,
        f"Dataset      : {report['dataset']}",
        f"Biological   : {report['biological']}   [{BIOLOGICAL_LABEL}]",
        "",
        report["what_this_is"],
        "",
        "Read",
        "-" * width,
        f"edges read            : {report['edges']['read']:,} (requested {report['edges']['requested']:,})",
        f"record batches read   : {report['read']['record_batches_read']} of {report['read']['record_batches_in_file']}",
        f"record batches skipped: {report['read']['record_batches_untouched']:,}",
        f"edge arrays in memory : {report['memory']['edge_arrays_human']}",
        f"dense NxN would be    : {report['memory']['dense_nxn_human']}  (never built)",
        "",
        "Edges",
        "-" * width,
        f"unique ordered pairs  : {report['edges']['unique_ordered_pairs']:,}",
        f"duplicate pairs       : {report['edges']['duplicate_ordered_pairs']:,}",
        f"self-loops            : {report['edges']['self_loops']:,} "
        f"on {report['edges']['self_loop_neurons']:,} neuron(s)",
        f"self-loop policy      : {report['edges']['self_loop_policy']}",
        "",
        "Nodes",
        "-" * width,
        f"unique source neurons : {report['nodes']['unique_source_neurons']:,}",
        f"unique target neurons : {report['nodes']['unique_target_neurons']:,}",
        f"unique neurons overall: {report['nodes']['unique_neurons_overall']:,}",
        "",
        "Weights (synapse counts, as provided by the dataset)",
        "-" * width,
        f"min / max / mean      : {report['weights']['min']} / {report['weights']['max']} / {report['weights']['mean']:.3f}",
        f"median / sum          : {report['weights']['median']} / {report['weights']['sum']:,}",
        "",
        "Annotation coverage",
        "-" * width,
        f"annotated neurons     : {report['annotation_coverage']['annotated_neurons']:,}",
        f"unannotated neurons   : {report['annotation_coverage']['unannotated_neurons']:,}",
        f"coverage              : {report['annotation_coverage']['percent']:.2f}%",
        f"note                  : {report['annotation_coverage']['note']}",
        "",
        "Breakdowns over the subset's neurons",
        "-" * width,
    ]
    for field, block in report["breakdowns"].items():
        if not block.get("available"):
            lines.append(f"{field}: not available")
            continue
        lines.append(
            f"{field}: {block['annotated_neurons']:,} annotated, "
            f"{block['unannotated_neurons']:,} unannotated, {block['distinct_values']} distinct values"
        )
        for name, count in list(block["top"].items())[:8]:
            lines.append(f"    {name[:34]:<34} {count:>7,}")
        if block["unannotated_neurons"]:
            lines.append(
                "    (unannotated)                     "
                f"{block['unannotated_neurons']:>7,}   <- unknown label, not a category"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded analysis of the strongest connections in a MaleCNS connectome table. "
            "Data-engineering feasibility only; this is not a visual circuit."
        ),
    )
    parser.add_argument(
        "--edges",
        default="data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        help="path to the MaleCNS connectome-weights Feather file",
    )
    parser.add_argument(
        "--annotations",
        default="data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        help="path to the MaleCNS body-annotations Feather file",
    )
    parser.add_argument("--max-edges", type=int, default=1_000_000, help="rows to read from the head of the file")
    parser.add_argument("--top", type=int, default=10, help="values to list per breakdown")
    parser.add_argument("--json", action="store_true", help="also print the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_edges <= 0:
        raise SystemExit("error: --max-edges must be positive")

    report = analyse(args.max_edges, Path(args.edges), Path(args.annotations), args.top)
    print(render(report))
    if args.json:
        print()
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
