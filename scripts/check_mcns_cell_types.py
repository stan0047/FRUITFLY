#!/usr/bin/env python
"""Check whether cell-type labels used in the literature exist in the MCNS annotations.

This is a **label-availability audit**, not circuit extraction. It answers one
question per label: does this exact string exist in the MCNS ``type`` field, and
if so how many bodies carry it?

Three outcomes, never conflated:

``EXACT``
    The label is present verbatim.
``NOT_PRESENT``
    The label is absent. **No similar name is substituted.** A near-miss is
    reported separately as a suggestion for a human to adjudicate, never applied
    automatically.
``SET``
    The literature label is a group that MCNS splits into subtypes, e.g. the
    paper's ``T4`` versus MCNS ``T4a``/``T4b``/``T4c``/``T4d``. Reported as a
    set, because treating the group as one type would be a silent
    misrepresentation.

Usage
-----
    python scripts/check_mcns_cell_types.py
    python scripts/check_mcns_cell_types.py --labels T4a LC4 Mi1 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import pyarrow.feather as feather  # noqa: E402

DEFAULT_ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
COLUMNS = ("bodyId", "type", "superclass", "flywireType", "somaSide", "status")

#: Labels exactly as written in Hoeller et al. 2026 (Cell 189:5552-5570.e10).
#: Grouped by the paper's own vocabulary, not by any assumed function.
PAPER_LABELS: dict[str, list[str]] = {
    "visual_inputs": ["L1", "L2", "L3", "L4", "L5", "R7", "R8", "R7d", "R8d", "HB", "R1-R6"],
    "lamina_other": ["C2", "C3", "T1"],
    "motion_direction": ["T4", "T4a", "T4b", "T4c", "T4d", "T5", "T5a", "T5b", "T5c", "T5d", "VS"],
    "medulla_columnar": ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm20"],
    "lobula_dm_pm": ["Dm4", "Dm8a", "Dm8b", "Dm9", "Dm11", "Pm12"],
    "lobula_projection": ["LC4", "LC6", "LPLC1", "LPLC2", "LPLC4", "LPT23", "LC16", "LC22"],
    "central_projection": [
        "MeTu2a", "MeTu3b", "MeTu3c", "MeVP11", "MeVPMe12", "MeVP57", "MeVP64",
        "aMe12", "pC1_2a", "pC1_5b", "KCab", "Nod5", "ER2_c", "ER4d", "DNp03",
        "PVLP008_a", "PVLP133", "DNg13", "DNge104",
    ],
}

#: Literature labels that MCNS resolves into subtypes. Reported as sets.
SUBSET_GROUPS: dict[str, list[str]] = {
    "T4": ["T4a", "T4b", "T4c", "T4d"],
    "T5": ["T5a", "T5b", "T5c", "T5d"],
    "Tm1-Tm2": ["Tm1", "Tm2"],
}


def load_types(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"error: annotation file not found: {path}")
    return feather.read_table(path, columns=list(COLUMNS)).to_pandas()


def audit(frame: pd.DataFrame, labels: list[str], top_near: int = 3) -> list[dict]:
    """Audit each label. Near-misses are reported, never substituted."""
    all_types = set(frame["type"].dropna().astype(str))
    by_type = {name: group for name, group in frame.groupby(frame["type"].astype("string"), dropna=True)}

    results: list[dict] = []
    for label in labels:
        if label in SUBSET_GROUPS:
            members = [m for m in SUBSET_GROUPS[label] if m in all_types]
            total = int(frame["type"].isin(members).sum()) if members else 0
            results.append(
                {
                    "label": label,
                    "status": "SET",
                    "mcns_subtypes": members,
                    "bodies": total,
                    "note": (
                        f"MCNS resolves this group into {len(members)} subtypes; treating it as one "
                        "type would misrepresent the annotation"
                    ),
                }
            )
            continue

        if label not in all_types:
            near = sorted(t for t in all_types if label.lower() in t.lower() or t.lower() in label.lower())
            results.append(
                {
                    "label": label,
                    "status": "NOT_PRESENT",
                    "bodies": 0,
                    "near_miss_candidates": near[:top_near],
                    "note": "label absent from the MCNS type field; NOT substituted",
                }
            )
            continue

        group = by_type[label]
        superclass = group["superclass"].dropna().astype(str)
        superclass_top = superclass.value_counts().head(3).to_dict() if len(superclass) else {}
        results.append(
            {
                "label": label,
                "status": "EXACT",
                "bodies": int(len(group)),
                "distinct_bodies": int(group["bodyId"].nunique()),
                "superclass_top": {str(k): int(v) for k, v in superclass_top.items()},
                "has_flywireType": bool(group["flywireType"].notna().any()),
                "flywireType_nonnull_pct": round(100.0 * float(group["flywireType"].notna().mean()), 1),
                "has_somaSide": bool(group["somaSide"].notna().any()),
                "somaSide_top": {
                    str(k): int(v) for k, v in group["somaSide"].dropna().astype(str).value_counts().head(4).items()
                },
                "status_top": {
                    str(k): int(v) for k, v in group["status"].dropna().astype(str).value_counts().head(3).items()
                },
            }
        )
    return results


def render(audit_rows: list[dict], group_of: dict[str, str], width: int = 100) -> str:
    lines = [
        "",
        "MCNS cell-type label availability for Hoeller et al. 2026 labels",
        "=" * width,
        "EXACT        = the label exists verbatim in the MCNS type field",
        "SET          = MCNS splits the label into subtypes; reported as a set",
        "NOT_PRESENT  = the label is absent; no similar name is substituted",
        "",
        f"{'label':<14}{'status':<13}{'bodies':>9}  detail",
        "-" * width,
    ]
    for row in audit_rows:
        detail = ""
        if row["status"] == "EXACT":
            detail = (
                f"superclass={row['superclass_top']} somaSide={row['somaSide_top']} "
                f"flywireType={row['flywireType_nonnull_pct']}%"
            )
        elif row["status"] == "SET":
            detail = f"subtypes={row['mcns_subtypes']}"
        else:
            detail = f"near-miss (not applied)={row['near_miss_candidates']}"
        lines.append(f"{row['label']:<14}{row['status']:<13}{row['bodies']:>9,}  {detail}")

    lines += ["", "By literature group:"]
    for group, labels in PAPER_LABELS.items():
        rows = [r for r in audit_rows if r["label"] in labels]
        exact = [r["label"] for r in rows if r["status"] == "EXACT"]
        sets = [r["label"] for r in rows if r["status"] == "SET"]
        missing = [r["label"] for r in rows if r["status"] == "NOT_PRESENT"]
        bodies = sum(r["bodies"] for r in rows if r["status"] == "EXACT")
        lines += [
            "",
            f"{group}  (exact={len(exact)} set={len(sets)} absent={len(missing)}, bodies in exact={bodies:,})",
            f"    exact     : {', '.join(exact) if exact else '(none)'}",
            f"    as sets   : {', '.join(sets) if sets else '(none)'}",
            f"    absent    : {', '.join(missing) if missing else '(none)'}",
        ]
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit literature cell-type labels against MCNS annotations.")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labels", nargs="*", default=None, help="explicit labels instead of the default set")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_types(Path(args.annotations))

    if args.labels:
        rows = audit(frame, list(args.labels))
        print(render(rows, {}))
    else:
        ordered: list[str] = []
        group_of: dict[str, str] = {}
        for group, labels in PAPER_LABELS.items():
            for label in labels:
                if label not in group_of:
                    ordered.append(label)
                    group_of[label] = group
        rows = audit(frame, ordered)
        print(render(rows, group_of))

    print("Totals across the MCNS annotation table:")
    print(f"  bodies                : {len(frame):,}")
    print(f"  distinct type labels  : {frame['type'].nunique():,}")
    print(f"  bodies with a type    : {int(frame['type'].notna().sum()):,}")
    print(f"  bodies with superclass: {int(frame['superclass'].notna().sum()):,}")
    if args.json:
        print()
        print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
