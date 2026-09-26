#!/usr/bin/env python
"""Extract the ON/OFF motion candidate pathways from local MaleCNS files.

Streams the connectivity table record batch by record batch and retains only
edges whose endpoints are both in the candidate cell populations. Nothing dense
is built, and nothing is truncated without saying so.

``weight`` is the **number of synapses** from source to target body. It is a
structural count, not synaptic strength, and is never rescaled here.

Examples
--------
    python scripts/extract_mcns_motion.py \\
        --annotations data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \\
        --connectivity data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \\
        --output-dir outputs/mcns_motion

    # deterministic per-cell-type body budget, for a quick pass
    python scripts/extract_mcns_motion.py --data-dir data/raw --body-budget 500

    # upstream only
    python scripts/extract_mcns_motion.py --data-dir data/raw --direction upstream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flybrain.data.motion_candidate import (  # noqa: E402
    DIRECTIONS,
    MOTION_CELL_TYPES,
    ExtractionError,
    extract_motion_candidates,
    write_report,
)

DEFAULT_DATA_DIR = "data/raw"
ANNOTATION_NAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CONNECTIVITY_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
_WIDTH = 78


def _resolve(data_dir: Path, annotations: str | None, connectivity: str | None) -> tuple[Path, Path]:
    annotation_path = Path(annotations) if annotations else data_dir / ANNOTATION_NAME
    connectivity_path = Path(connectivity) if connectivity else data_dir / CONNECTIVITY_NAME
    return annotation_path, connectivity_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded extraction of MCNS motion candidate pathways (ON: L1/Mi1/T4a-d, OFF: L2/Tm1/Tm2/T5a-d)."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="directory holding the two Feather files")
    parser.add_argument("--annotations", default=None, help="explicit annotation file path")
    parser.add_argument("--connectivity", default=None, help="explicit connectivity file path")
    parser.add_argument("--output-dir", default="outputs/mcns_motion", help="directory for artifacts")
    parser.add_argument(
        "--summary-name", default="mcns_motion_candidate_summary.json", help="summary filename"
    )
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="both")
    parser.add_argument(
        "--body-budget",
        type=int,
        default=0,
        help="deterministic per-cell-type body budget; 0 uses the complete annotated population",
    )
    parser.add_argument(
        "--read-budget",
        type=int,
        default=0,
        help="cap on record batches read from the connectivity file; 0 reads all",
    )
    parser.add_argument(
        "--keep-fragments",
        action="store_true",
        help="do not apply the superclass neuron predicate (diagnostic use only)",
    )
    parser.add_argument("--cell-types", nargs="*", default=list(MOTION_CELL_TYPES))
    parser.add_argument("--quiet", action="store_true")
    return parser


def render(report: dict) -> str:
    lines = [
        "",
        "MCNS motion pathway candidate extraction",
        "=" * _WIDTH,
        f"status  : {report['status']}",
        f"weight  : {report['weight_semantics']}",
        "",
        "Cell types",
        "-" * _WIDTH,
        f"{'type':<8}{'bodies':>8}{'side':>8}{'coords':>8}{'out':>10}{'in':>10}",
    ]
    for label, block in report["populations"]["per_cell_type"].items():
        lines.append(
            f"{label:<8}{block['bodies']:>8,}{block['with_somaSide']:>8,}"
            f"{block['with_coordinates']:>8,}{block['outgoing_edges']:>10,}{block['incoming_edges']:>10,}"
        )
    lines += [
        "",
        f"candidate bodies: {report['populations']['candidate_bodies']:,}",
        "",
        "Extraction",
        "-" * _WIDTH,
        f"direction                : {report['inputs']['direction']}",
        f"edges retained           : {report['extraction']['edges_retained']:,}",
        f"record batches           : {report['extraction']['record_batches_processed']} of {report['extraction']['record_batches_total']}",
        f"rows scanned             : {report['extraction']['rows_scanned']:,}",
        f"seconds                  : {report['extraction']['seconds']}",
        f"retained edge bytes      : {report['memory']['retained_edge_bytes']:,}",
        f"dense would have been    : {report['memory']['dense_would_be_bytes'] / 1e9:.2f} GB (never built)",
        "",
        "Pathways",
        "-" * _WIDTH,
    ]
    for name, block in report["pathways"].items():
        if not block.get("available"):
            lines.append(f"{name}: not available")
            continue
        stats = block["synapse_count_stats"]
        lines += [
            f"{name}  ({block['role']})",
            f"    types    : {', '.join(block['cell_types'])}",
            f"    nodes    : {block['nodes']:,}",
            f"    edges    : {block['edges']:,}",
            f"    synapses : min={stats['min']} median={stats['median']} max={stats['max']} total={stats['total']:,}",
            f"    self-loops: {block['self_loops']:,}   cross-side edges: {block['cross_side_edges']:,}",
        ]
    subset = report["subset"]
    if subset["is_subset"] or subset["read_truncated"]:
        lines += ["", "SUBSET", "-" * _WIDTH]
        if subset["note"]:
            lines.append(f"  {subset['note']}")
        if subset["read_truncation_note"]:
            lines.append(f"  {subset['read_truncation_note']}")
    lines += [
        "",
        "Validation",
        "-" * _WIDTH,
    ]
    for check in report["validation"]["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{mark}] {check['check']}: {check['detail']} (value={check['value']})")
    lines += [
        "",
        f"ALL CHECKS PASSED: {report['validation']['all_passed']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    annotation_path, connectivity_path = _resolve(data_dir, args.annotations, args.connectivity)

    try:
        report = extract_motion_candidates(
            annotation_path,
            connectivity_path,
            cell_types=args.cell_types,
            direction=args.direction,
            require_superclass=not args.keep_fragments,
            body_budget_per_type=args.body_budget,
            read_budget=args.read_budget,
        )
    except ExtractionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    written = write_report(report, output_dir / args.summary_name, output_dir)

    if not args.quiet:
        print(render(report))
        print("Artifacts")
        print("-" * _WIDTH)
        for key, value in written.items():
            print(f"  {key:<12} {value}")
    else:
        print(json.dumps({"edges": report["extraction"]["edges_retained"], "written": written}, default=str))

    return 0 if report["validation"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
