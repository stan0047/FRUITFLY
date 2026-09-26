#!/usr/bin/env python
"""Inspect a user-provided connectome file without importing it.

Prints the file type, row count, columns, inferred dtypes, sample rows, and
FlyBrain's candidate column roles. When a role cannot be resolved with
confidence it is reported as unresolved rather than guessed, together with the
columns that could satisfy it.

This reads at most ``--nrows`` rows, so a multi-gigabyte export can be inspected
on a laptop. The full row count is obtained separately by streaming the file.

Examples
--------
    python scripts/inspect_connectome.py --input data/raw/my_export.csv
    python scripts/inspect_connectome.py --input export.parquet --nrows 5000
    python scripts/inspect_connectome.py --input export.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flybrain.data.schema import SchemaError, inspect_schema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the schema of a local connectome file (no download, no import).",
    )
    parser.add_argument("--input", "-i", required=True, help="path to a local CSV/TSV/Parquet/JSON file")
    parser.add_argument("--nrows", type=int, default=10_000, help="rows to read for profiling (default 10000)")
    parser.add_argument("--samples", type=int, default=5, help="sample rows to display (default 5)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON instead of text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        report = inspect_schema(args.input, nrows=args.nrows, n_samples=args.samples)
    except SchemaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.render())
        unresolved = report.required_roles()
        print()
        if unresolved:
            print("This file cannot be imported yet. Provide the missing column(s) explicitly:")
            for role in unresolved:
                candidates = report.role_candidates(role)
                suffix = f"   (candidates: {', '.join(candidates)})" if candidates else ""
                print(f"  --{role}-column <name>{suffix}")
        else:
            print("Schema looks usable. Next:")
            print(f"  python scripts/import_connectome.py --input {args.input} --output data/processed/")
            print("  (add --source-name/--dataset-name/--license/--citation so the report can state provenance)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
