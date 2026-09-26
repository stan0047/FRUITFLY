#!/usr/bin/env python
"""Bounded seeded connectivity exploration over the local MaleCNS table.

Takes a set of seed body ids and expands their local connectivity to a bounded
neighborhood. The output is a **seeded connectivity neighborhood** — a
data-structure feasibility artifact. It is **not** the visual circuit, and
nothing here should be read as a claim about a biological pathway.

``weight`` is the **number of synapses** from pre to post, as provided by the
dataset. It is a structural count, not synaptic efficacy. ``--min-weight`` is a
**minimum synapse-count threshold**.

Because the connectome table is sorted by weight descending, this reads only a
prefix of the file. A prefix-limited neighborhood is therefore biased against
weak connections, and the truncation is reported explicitly.

Examples
--------
    # seeds from annotation cell types, 1 hop, small budgets
    python scripts/explore_seeded_neighborhood.py \\
        --seed-type L1 --seed-type L2 --seed-type L5 --seed-type Tm1 --seed-type Tm3 \\
        --hops 1 --max-nodes 20000 --max-edges 100000

    # seeds from a file, downstream only, minimum synapse count 5
    python scripts/explore_seeded_neighborhood.py \\
        --seed-file seeds.txt --hops 2 --direction downstream --min-weight 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flybrain.data.seeded_neighborhood import (  # noqa: E402
    DIRECTIONS,
    build_prefix_index,
    explore,
    load_annotation_table,
    load_seed_file,
    seeds_from_types,
)

DEFAULT_EDGES = "data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"

_WIDTH = 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded seeded connectivity neighborhood from the local MaleCNS table."
    )
    seeds = parser.add_argument_group("seeds (choose one)")
    seeds.add_argument("--seed-file", default=None, help="file of body ids (one per line, or a CSV id column)")
    seeds.add_argument(
        "--seed-type",
        action="append",
        default=[],
        help="annotation cell type to use as seeds; repeat for several types",
    )
    seeds.add_argument("--seed-column", default="type", help="annotation column the --seed-type values match")

    parser.add_argument("--edges", default=DEFAULT_EDGES, help="path to the connectome-weights Feather file")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, help="path to the body-annotations Feather file")
    parser.add_argument("--seed-id-column", default="bodyId", help="id column inside --seed-file")

    parser.add_argument("--hops", type=int, default=1, help="number of hops to expand (0 = seeds only)")
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="both")
    parser.add_argument(
        "--min-weight",
        type=int,
        default=1,
        help="minimum synapse-count threshold (NOT a synaptic strength threshold)",
    )
    parser.add_argument("--max-nodes", type=int, default=20_000, help="node budget")
    parser.add_argument("--max-edges", type=int, default=100_000, help="edge budget")
    parser.add_argument(
        "--prefix-edges",
        type=int,
        default=2_000_000,
        help="rows to read from the head of the weight-sorted file (bounds memory and time)",
    )
    parser.add_argument("--top", type=int, default=10, help="values to list per breakdown")
    parser.add_argument("--json", action="store_true", help="also print the result as JSON")
    return parser


def render(result) -> str:
    report = result.to_dict()
    lines = [
        "",
        "Seeded connectivity neighborhood (bounded, weight-truncated)",
        "=" * _WIDTH,
        report["what_this_is"],
        "",
        "Provided by the dataset",
        "-" * _WIDTH,
        "  body ids, connection direction, synapse counts, neuron annotations",
        "Derived by FlyBrain",
        "-" * _WIDTH,
        "  the read prefix, the neighborhood, budget truncation, coverage percentages",
        "Assumed by FlyBrain",
        "-" * _WIDTH,
        "  nothing: no weight is synthesised, no neuron is given a category it lacks",
        "",
        "Read budget",
        "-" * _WIDTH,
        f"rows read                 : {report['prefix']['edges_read']:,} of {report['prefix']['file_total_rows']:,}"
        f"  (truncated={report['prefix']['truncated']})",
        f"record batches read       : {report['prefix']['record_batches_read']} of {report['prefix']['record_batches_total']}",
        f"rows after min synapse ct : {report['prefix']['edges_after_minimum_synapse_count_threshold']:,}",
        f"nodes in prefix           : {report['prefix']['prefix_nodes']:,}",
        f"dense would have been     : {report['prefix']['dense_would_be_bytes'] / 1e9:.1f} GB  (never built)",
        "",
        "Seeds and budgets",
        "-" * _WIDTH,
        f"seed bodies               : {report['seeds']:,}",
        f"seeds present in prefix   : {report['seeds_present_in_prefix']:,}",
        f"hops / direction          : {report['hops']} / {report['direction']}",
        f"minimum synapse count     : >= {report['minimum_synapse_count_threshold']}",
        f"budgets                   : max_nodes={report['max_nodes']:,}  max_edges={report['max_edges']:,}",
        f"budget that bound the run : {report['budget_binding']}",
        "",
        "Growth",
        "-" * _WIDTH,
        f"{'stage':<20}{'new':>10}{'cumulative':>14}{'edges':>12}",
    ]
    for stage in report["stages"]:
        lines.append(
            f"{stage['stage']:<20}{stage['new_nodes']:>10,}"
            f"{stage['cumulative_nodes']:>14,}{stage['cumulative_edges']:>12,}"
        )
    lines += [
        "",
        f"final nodes {report['final_nodes']:,}   final discovery edges {report['final_edges']:,}",
        f"note: {report['edge_count_semantics']}",
        "",
    ]

    coverage = report.get("coverage") or {}
    if coverage:
        lines += [
            "Annotation coverage over the neighborhood",
            "-" * _WIDTH,
            f"in annotation table      : {coverage['in_annotation_table']:,}"
            f"  ({coverage['percent_in_annotation_table']:.2f}%)",
            f"with a type label        : {coverage['with_type_label']:,}"
            f"  ({coverage['percent_with_type_label']:.2f}%)",
            f"with a superclass label  : {coverage['with_superclass_label']:,}",
            f"note                     : {coverage['note']}",
            "",
        ]
    if report.get("node_superclass_top"):
        lines.append("superclass over the neighborhood (annotated only):")
        for name, count in report["node_superclass_top"].items():
            lines.append(f"    {name[:40]:<40} {count:>8,}")
        lines.append("")
    if report.get("node_type_top"):
        lines.append("cell type over the neighborhood (annotated only):")
        for name, count in report["node_type_top"].items():
            lines.append(f"    {name[:40]:<40} {count:>8,}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.seed_file and not args.seed_type:
        print("error: supply --seed-file or at least one --seed-type", file=sys.stderr)
        return 2

    if args.seed_file:
        seeds = load_seed_file(args.seed_file, column=args.seed_id_column)
    else:
        seeds = seeds_from_types(args.annotations, args.seed_type, column=args.seed_column)

    if len(seeds) == 0:
        print("error: no seeds resolved", file=sys.stderr)
        return 2

    index = build_prefix_index(args.edges, max_edges=args.prefix_edges, min_weight=args.min_weight)
    annotations = load_annotation_table(args.annotations, ("type", "superclass", "status"))
    result = explore(
        index,
        seeds,
        hops=args.hops,
        direction=args.direction,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        annotations=annotations,
        top=args.top,
    )

    print(render(result))
    if args.json:
        print()
        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
