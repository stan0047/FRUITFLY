#!/usr/bin/env python
"""Import a user-provided connectome file into FlyBrain's canonical format.

Pipeline: inspect -> resolve columns -> normalize -> validate -> write -> report.

FlyBrain never downloads data. Point ``--input`` at a file you obtained yourself
from the official provider.

Examples
--------
    # Minimal, with a node/edge budget so a huge export stays inspectable
    python scripts/import_connectome.py \\
        --input data/raw/my_export.csv \\
        --output data/processed/ \\
        --source-name flywire \\
        --dataset-name my-fafb-export \\
        --dataset-version 2024-01 \\
        --source-url <the page you downloaded from> \\
        --license CC-BY-4.0 \\
        --citation "..." \\
        --max-nodes 5000 --max-edges 50000 --on-limit subsample

    # When the column mapping is ambiguous, settle it explicitly
    python scripts/import_connectome.py --input x.csv --output out/ \\
        --source-column pre_id --target-column post_id --weight-column synapse_count

    # Re-report an already-imported dataset without rewriting it
    python scripts/import_connectome.py --input data/processed/my-fafb-export.csv --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flybrain.config import load_config  # noqa: E402
from flybrain.data.importer import (  # noqa: E402
    ConnectomeImportError,
    ImportOptions,
    import_connectome,
)
from flybrain.data.provenance import DataSource, Provenance  # noqa: E402
from flybrain.data.report import build_connectome_report  # noqa: E402
from flybrain.data.schema import SchemaError  # noqa: E402
from flybrain.data.validation import ValidationConfig, ValidationError  # noqa: E402

_ROLE_FLAGS = {
    "source": "source_column",
    "target": "target_column",
    "weight": "weight_column",
    "region": "region_column",
    "neuron_type": "neuron_type_column",
    "neuron_id": "neuron_id_column",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a local connectome file (no download) and print a dataset report.",
    )
    parser.add_argument("--input", "-i", required=True, help="path to a local connectome file")
    parser.add_argument("--output", "-o", default=None, help="directory for the processed table and sidecar")
    parser.add_argument("--config", default=None, help="YAML config to read defaults from")
    parser.add_argument("--report-only", action="store_true", help="load, report, and write nothing")

    provenance = parser.add_argument_group("provenance (nothing here is guessed)")
    provenance.add_argument("--source-name", required=False, help="who published the data, e.g. flywire")
    provenance.add_argument("--dataset-name", required=False, help="name of this specific dataset or export")
    provenance.add_argument("--source", choices=[s.value for s in DataSource], default=None)
    provenance.add_argument("--dataset-version", default=None, help="version or release date, if the source states one")
    provenance.add_argument("--source-url", default=None, help="page you downloaded the file from")
    provenance.add_argument("--license", default=None, help="licence identifier, e.g. CC-BY-4.0")
    provenance.add_argument("--citation", default=None, help="citation for the dataset")
    provenance.add_argument("--notes", default="", help="free-text notes carried into the report")
    provenance.add_argument(
        "--not-biological",
        action="store_true",
        help="declare that this file is NOT biological data (test fixtures, mock data)",
    )

    columns = parser.add_argument_group("explicit column mapping (overrides inference)")
    for role, flag in _ROLE_FLAGS.items():
        columns.add_argument(f"--{flag.replace('_', '-')}", dest=flag, default=None, help=f"{role} column")

    limits = parser.add_argument_group("validation and budget")
    limits.add_argument("--max-nodes", type=int, default=None, help="node budget (0 = unlimited)")
    limits.add_argument("--max-edges", type=int, default=None, help="edge budget (0 = unlimited)")
    limits.add_argument("--on-limit", choices=("error", "subsample"), default=None)
    limits.add_argument("--min-synapse-count", type=float, default=None, help="drop edges below this weight")
    limits.add_argument("--keep-self-loops", action="store_true", default=None)
    limits.add_argument("--no-aggregate-duplicates", action="store_true", help="drop duplicate edges instead of summing")
    limits.add_argument("--drop-null-rows", action="store_true", default=None)
    limits.add_argument("--allow-unknown-nodes", action="store_true", help="keep edges absent from the node table")
    limits.add_argument("--chunksize", type=int, default=None, help="rows per pandas chunk")
    limits.add_argument("--drop-extra-columns", action="store_true", help="do not carry unmapped source columns through")
    limits.add_argument("--weight-not-biological", action="store_true", help="weight column is a proxy, not a measurement")
    limits.add_argument("--node-table", default=None, help="optional neuron metadata table")
    limits.add_argument("--node-id-column", default=None, help="neuron id column in --node-table")
    limits.add_argument("--node-region-column", default=None, help="region column in --node-table")
    limits.add_argument("--node-type-column", default=None, help="neuron type column in --node-table")
    limits.add_argument("--seed", type=int, default=None, help="seed for reproducible subsampling")
    limits.add_argument("--json", action="store_true", help="also print the report as JSON")
    return parser


def _config_section(args: argparse.Namespace) -> dict:
    try:
        config = load_config(args.config)
    except Exception:  # a missing config must not block an explicit CLI import
        return {}
    return dict(config.get("connectome.import", {}) or {})


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def build_options(args: argparse.Namespace) -> ImportOptions:
    defaults = _config_section(args)
    overrides = {flag: getattr(args, flag) for flag in _ROLE_FLAGS.values() if getattr(args, flag)}
    validation = ValidationConfig(
        keep_self_loops=bool(_first(args.keep_self_loops, defaults.get("keep_self_loops"), False)),
        aggregate_duplicates=not bool(args.no_aggregate_duplicates) and bool(
            defaults.get("aggregate_duplicates", True)
        ),
        min_synapse_count=_first(args.min_synapse_count, defaults.get("min_synapse_count")),
        max_nodes=int(_first(args.max_nodes, defaults.get("max_nodes"), 0) or 0),
        max_edges=int(_first(args.max_edges, defaults.get("max_edges"), 0) or 0),
        on_limit=str(_first(args.on_limit, defaults.get("on_limit"), "error")),
        require_known_nodes=not bool(args.allow_unknown_nodes) and bool(defaults.get("require_known_nodes", True)),
        drop_null_rows=bool(_first(args.drop_null_rows, defaults.get("drop_null_rows"), False)),
        seed=int(_first(args.seed, defaults.get("seed"), 0)),
    )
    return ImportOptions(
        role_overrides=overrides,
        node_table_roles={
            role: value
            for role, value in {
                "neuron_id": args.node_id_column,
                "region": args.node_region_column,
                "neuron_type": args.node_type_column,
            }.items()
            if value
        },
        validation=validation,
        chunksize=_first(args.chunksize, defaults.get("chunksize")),
        keep_extra_columns=not bool(args.drop_extra_columns) and bool(defaults.get("keep_extra_columns", True)),
        weight_is_biological=not bool(args.weight_not_biological) and bool(defaults.get("weight_is_biological", True)),
        node_table_path=_first(args.node_table, defaults.get("node_table")),
    )


def build_provenance(args: argparse.Namespace, input_path: Path) -> Provenance:
    dataset_name = args.dataset_name or input_path.stem
    source_name = args.source_name or "user-provided"
    source = DataSource(args.source) if args.source else DataSource.EXTERNAL
    if args.not_biological and source.can_be_biological:
        source = DataSource.TEST_FIXTURE
    return Provenance(
        source_name=source_name,
        dataset_name=dataset_name,
        source=source,
        dataset_version=args.dataset_version,
        source_url=args.source_url,
        license=args.license,
        citation=args.citation,
        original_filename=input_path.name,
        biological_data=not bool(args.not_biological),
        notes=args.notes,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)

    if args.report_only and args.output:
        print("error: --report-only writes nothing; drop --output", file=sys.stderr)
        return 2

    provenance = build_provenance(args, input_path)
    options = build_options(args)

    try:
        result = import_connectome(
            input_path,
            provenance=provenance,
            options=options,
            output_dir=None if args.report_only else (args.output or "data/processed"),
        )
    except (SchemaError, ValidationError, ConnectomeImportError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = build_connectome_report(result.connectome, result.validation)
    print(report.render())
    print(result.validation.render())

    if result.schema.warnings:
        print()
        print("Schema warnings:")
        for warning in result.schema.warnings:
            print(f"  - {warning}")
    if result.schema.ambiguities:
        print()
        print("Ambiguous roles, settled automatically or left unset (never guessed):")
        for role, candidates in result.schema.ambiguities.items():
            print(f"  - {role}: {', '.join(candidates)}")

    print()
    print("Provenance:")
    print(f"  {result.provenance.warning()}")
    if result.provenance.missing_fields():
        print(f"  Not provided (not guessed): {', '.join(result.provenance.missing_fields())}")
    if result.output_path:
        print(f"  Wrote: {result.output_path}")
        print(f"  Wrote: {result.output_path.with_suffix('').with_suffix('.source.json')}")
    if result.node_table_path:
        print(f"  Wrote: {result.node_table_path}")
    if args.json:
        print()
        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
