#!/usr/bin/env python
"""Minimal neuPrint connectivity check for FlyBrain.

Verifies three things and nothing else:

1. the token in ``.env`` authenticates;
2. the dataset (default ``male-cns:v1.0``) is reachable;
3. a single harmless count query returns a sane response.

This script deliberately performs **one** tiny query. It is a connection test,
not a data pull. Bulk connectome data comes from the local Feather files; the
API is for targeted lookups only.

The token is never printed, logged, written to disk, or passed on a command line.
It is read from the environment (optionally seeded from a local ``.env``, which
is git-ignored) and handed straight to the client.

Usage
-----
    python scripts/test_neuprint.py
    python scripts/test_neuprint.py --dataset male-cns:v1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"
ENV_FILENAME = ".env"

SETUP_INSTRUCTIONS = f"""\
NEUPRINT_TOKEN is not set.

To set it up:
  1. Create a neuPrint account at {DEFAULT_SERVER}/ and sign in.
  2. Generate an API token from your account page
     (https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html#client-and-authorization-token).
  3. Create a local file named .env in the repository root (it is git-ignored):

         NEUPRINT_TOKEN=<paste your token here>
         NEUPRINT_SERVER={DEFAULT_SERVER}
         NEUPRINT_DATASET={DEFAULT_DATASET}

  4. Re-run:  python scripts/test_neuprint.py

The token is only needed for neuPrint queries. Everything FlyBrain does with the
local MaleCNS Feather files -- inspection, import, validation, sparse graph
construction, reports -- works with no token at all.
"""


def load_env(path: Path) -> None:
    """Seed the environment from a local .env without overwriting real vars.

    Implemented directly rather than via python-dotenv so this script has no
    hard third-party requirement, and so it is obvious that no value is ever
    echoed or persisted.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal authenticated neuPrint check (never prints the token).")
    parser.add_argument("--dataset", default=os.environ.get("NEUPRINT_DATASET", DEFAULT_DATASET))
    parser.add_argument("--server", default=os.environ.get("NEUPRINT_SERVER", DEFAULT_SERVER))
    parser.add_argument("--env-file", default=str(_ROOT / ENV_FILENAME))
    args = parser.parse_args(argv)

    load_env(Path(args.env_file))

    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        print(SETUP_INSTRUCTIONS)
        return 2

    print("neuPrint connectivity check")
    print("=" * 70)
    print(f"server  : {args.server}")
    print(f"dataset : {args.dataset}")
    print(f"token   : present ({len(token)} chars, value never printed)")
    print()

    try:
        import neuprint
    except ImportError:
        print("error: the neuprint client is not installed.")
        print("       install it with:  pip install neuprint-python")
        return 3

    try:
        client = neuprint.Client(args.server, dataset=args.dataset, token=token)
    except Exception as error:  # noqa: BLE001 - report whatever the client raised
        print(f"error: could not connect or authenticate: {type(error).__name__}: {error}")
        print("       check NEUPRINT_TOKEN and that the dataset name is correct.")
        return 1

    print("connection successful")
    print(f"dataset  : {client.dataset}")
    print(f"server   : {client.server}")

    try:
        # One count-only query. No rows are pulled, nothing is cached.
        row_count = client.execute("MATCH (n:Neuron) RETURN count(n)").data[0][0]
        print(f"neuron count reported by the server: {row_count:,}")
    except Exception as error:  # noqa: BLE001
        print(f"connected, but the count query failed: {type(error).__name__}: {error}")
        return 1

    print()
    print("OK: token valid and dataset reachable.")
    print("This was one count query. No data was downloaded, cached, or written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
