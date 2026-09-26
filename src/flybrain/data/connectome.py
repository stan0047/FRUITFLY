"""Connectome data model, I/O and provenance.

The connectome is **biological data**: a published measurement of which neurons
are connected to which. It is *not* a neural network, and it contains no weights,
no activation functions, no dynamics and no learning rules. Anything this module
cannot obtain from a source is left ``None`` and reported as an author
assumption; fabricated values are never substituted.

Current status
--------------
**No real connectome is integrated.** The only loader implemented is
:func:`synthetic_connectome`, which generates a random graph explicitly flagged
as non-biological so that the simulation and tests have something to run on.
:func:`load_flywire_connectome` is an interface stub that raises.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "Connectome",
    "ConnectomeEdge",
    "ConnectomeError",
    "DataSource",
    "available_sources",
    "describe_schema",
    "is_source_available",
    "load_connectome",
    "load_flywire_connectome",
    "save_connectome",
    "synthetic_connectome",
]

#: Columns every processed connectome table must provide.
REQUIRED_COLUMNS: tuple[str, ...] = ("pre", "post", "weight")
#: Columns used when the upstream source reports them.
OPTIONAL_COLUMNS: tuple[str, ...] = ("synapse_type", "neurotransmitter", "neuropil")


class ConnectomeError(RuntimeError):
    """Raised for malformed connectome data or unavailable sources."""


class DataSource(str, Enum):
    """Known provenance labels.

    ``SYNTHETIC`` is the only one that is not biological. ``FLYWIRE`` is
    declared but not implemented; ``is_source_available`` reports ``False`` for
    it so callers can fail loudly instead of silently using a placeholder.
    """

    SYNTHETIC = "synthetic"
    FLYWIRE = "flywire"


#: Sources that are actually wired up, with the loader that serves them.
_SOURCE_LOADERS: dict[DataSource, str] = {DataSource.SYNTHETIC: "synthetic_connectome"}


@dataclass(frozen=True)
class ConnectomeEdge:
    """A single directed chemical synapse, as reported by the source.

    ``weight`` is biological **only if** the source reports synaptic strength.
    Otherwise it is an author-initialised value and ``weight_is_biological`` is
    ``False``.
    """

    pre: str
    post: str
    weight: float = 1.0
    synapse_type: str | None = None
    neurotransmitter: str | None = None
    neuropil: str | None = None
    weight_is_biological: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pre or not self.post:
            raise ConnectomeError("edge endpoints must be non-empty identifiers")
        if float(self.weight) < 0.0:
            raise ConnectomeError(f"negative synaptic weight for {self.pre} -> {self.post}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre": self.pre,
            "post": self.post,
            "weight": float(self.weight),
            "synapse_type": self.synapse_type,
            "neurotransmitter": self.neurotransmitter,
            "neuropil": self.neuropil,
            "weight_is_biological": bool(self.weight_is_biological),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConnectomeEdge":
        return cls(
            pre=str(payload["pre"]),
            post=str(payload["post"]),
            weight=float(payload.get("weight", 1.0)),
            synapse_type=_optional_str(payload.get("synapse_type")),
            neurotransmitter=_optional_str(payload.get("neurotransmitter")),
            neuropil=_optional_str(payload.get("neuropil")),
            weight_is_biological=bool(payload.get("weight_is_biological", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class Connectome:
    """A directed, weighted connectivity table plus its provenance.

    The container makes no claims about dynamics; it stores measured structure
    and nothing else.
    """

    name: str
    source: DataSource = DataSource.SYNTHETIC
    edges: list[ConnectomeEdge] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    version: str | None = None
    url: str | None = None
    license: str | None = None
    citation: str | None = None
    notes: str = ""
    _biological: bool | None = field(default=None, repr=False, compare=False)

    # --------------------------------------------------------- provenance

    @property
    def is_biological(self) -> bool:
        """Whether the topology originates from a published biological source.

        Derived from :attr:`source` unless explicitly overridden.
        """
        if self._biological is not None:
            return bool(self._biological)
        return self.source is not DataSource.SYNTHETIC

    def provenance(self) -> dict[str, Any]:
        """Summary suitable for logs, figures and reports."""
        return {
            "name": self.name,
            "source": self.source.value,
            "is_biological": self.is_biological,
            "version": self.version,
            "url": self.url,
            "license": self.license,
            "citation": self.citation,
            "num_edges": len(self.edges),
            "num_neurons": len(self.neurons),
            "notes": self.notes,
        }

    def warn_if_synthetic(self) -> str:
        """Return a one-line warning that must accompany synthetic results."""
        if self.is_biological:
            return ""
        return (
            f"[SIMULATION INPUT] '{self.name}' uses synthetic (non-biological) connectivity; "
            "results are not claims about the fruit-fly connectome."
        )

    # ----------------------------------------------------------- structure

    @property
    def neurons(self) -> list[str]:
        """Sorted unique neuron identifiers appearing as pre or post."""
        names: set[str] = set()
        for edge in self.edges:
            names.add(edge.pre)
            names.add(edge.post)
        return sorted(names)

    @property
    def num_neurons(self) -> int:
        return len(self.neurons)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def regions(self) -> dict[str, str]:
        """Map neuron -> neuropil, when the source reports neuropils."""
        mapping: dict[str, str] = {}
        for edge in self.edges:
            if edge.neuropil:
                mapping.setdefault(edge.pre, edge.neuropil)
                mapping.setdefault(edge.post, edge.neuropil)
        return mapping

    def __len__(self) -> int:
        return len(self.edges)

    def __iter__(self) -> Iterator[ConnectomeEdge]:
        return iter(self.edges)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy table, one row per synapse."""
        rows: list[dict[str, Any]] = []
        for edge in self.edges:
            row = edge.to_dict()
            row["metadata"] = json.dumps(dict(edge.metadata), sort_keys=True)
            rows.append(row)
        columns = ["pre", "post", "weight", "synapse_type", "neurotransmitter", "neuropil", "weight_is_biological", "metadata"]
        return pd.DataFrame(rows, columns=columns)

    def weight_matrix(self, node_order: Sequence[str] | None = None, dtype: Any = np.float32) -> np.ndarray:
        """Dense adjacency matrix ``W`` with ``W[i, j]`` the weight of ``i -> j``.

        Dense on purpose: connectome-scale graphs need sparse storage, which is
        a future concern once real data is loaded.
        """
        order = list(node_order) if node_order is not None else self.neurons
        index = {name: i for i, name in enumerate(order)}
        matrix = np.zeros((len(order), len(order)), dtype=dtype)
        for edge in self.edges:
            if edge.pre in index and edge.post in index:
                matrix[index[edge.pre], index[edge.post]] = float(edge.weight)
        return matrix

    def save(self, path: str | Path) -> Path:
        """Write ``<stem>.csv`` plus a ``<stem>.source.json`` provenance sidecar."""
        return save_connectome(self, path)

    # ------------------------------------------------------------ builders

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        name: str,
        source: DataSource = DataSource.SYNTHETIC,
        **metadata: Any,
    ) -> "Connectome":
        """Build a connectome from a table, validating the required columns."""
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ConnectomeError(f"missing required column(s): {', '.join(missing)}")
        edges = [
            ConnectomeEdge.from_dict(_row_to_edge_payload(row, source))
            for row in frame.to_dict(orient="records")
        ]
        return cls(name=name, source=source, edges=edges, **metadata)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "unknown", "?"}:
        return None
    return text


def _row_to_edge_payload(row: Mapping[str, Any], source: DataSource) -> dict[str, Any]:
    """Normalise one table row, restoring the JSON-encoded ``metadata`` column."""
    payload = dict(row)
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            payload["metadata"] = json.loads(raw_metadata)
        except json.JSONDecodeError:
            payload["metadata"] = {"raw": raw_metadata}
    weight = payload.get("weight")
    payload["weight"] = float(0.0 if weight is None or (isinstance(weight, float) and np.isnan(weight)) else weight)
    default_biological = bool(payload.pop("weight_is_biological", source is not DataSource.SYNTHETIC))
    return {**payload, "weight_is_biological": default_biological}


# --------------------------------------------------------------------- I/O


def describe_schema() -> dict[str, Any]:
    """Return the processed-table contract."""
    return {
        "required_columns": list(REQUIRED_COLUMNS),
        "optional_columns": list(OPTIONAL_COLUMNS),
        "granularity": "one row per synapse",
        "id_columns": ["pre", "post"],
        "weight_is_biological": (
            "True only if the upstream source reports synaptic strength. "
            "Initialised values must set it to False."
        ),
        "unknown_values": "Use null/NA rather than guessing. Unknown stays unknown.",
    }


def is_source_available(source: DataSource | str) -> bool:
    """Whether a loader exists for ``source``."""
    try:
        key = source if isinstance(source, DataSource) else DataSource(str(source))
    except ValueError:
        return False
    return key in _SOURCE_LOADERS


def available_sources() -> list[str]:
    """Sources with an implemented loader."""
    return [source.value for source in _SOURCE_LOADERS]


def load_connectome(path: str | Path, name: str | None = None) -> Connectome:
    """Load a processed connectome CSV plus its ``.source.json`` sidecar.

    The sidecar is mandatory in practice: without it there is no way to know
    whether the data is biological, so a missing file is an error rather than a
    silent default.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ConnectomeError(f"processed connectome not found: {csv_path}")

    sidecar_path = csv_path.with_suffix("").with_suffix(".source.json")
    if not sidecar_path.is_file():
        raise ConnectomeError(
            f"missing provenance sidecar {sidecar_path.name}; a source file is required so that "
            "biological and synthetic data can be distinguished"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    try:
        source = DataSource(str(sidecar.get("source", DataSource.SYNTHETIC.value)))
    except ValueError as exc:
        raise ConnectomeError(f"unknown source {sidecar.get('source')!r} in {sidecar_path}") from exc

    frame = pd.read_csv(csv_path)
    connectome = Connectome.from_dataframe(
        frame,
        name=name or str(sidecar.get("name", csv_path.stem)),
        source=source,
        version=sidecar.get("version"),
        url=sidecar.get("url"),
        license=sidecar.get("license"),
        citation=sidecar.get("citation"),
        notes=str(sidecar.get("notes", "")),
    )
    connectome.attributes.update({str(k): str(v) for k, v in (sidecar.get("attributes") or {}).items()})
    return connectome


def save_connectome(connectome: Connectome, path: str | Path) -> Path:
    """Write the connectome table and its provenance sidecar to disk."""
    target = Path(path)
    if target.suffix != ".csv":
        target = target.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    connectome.to_dataframe().to_csv(target, index=False)

    sidecar_path = target.with_suffix("").with_suffix(".source.json")
    sidecar = {
        "name": connectome.name,
        "source": connectome.source.value,
        "is_biological": connectome.is_biological,
        "version": connectome.version,
        "url": connectome.url,
        "license": connectome.license,
        "citation": connectome.citation,
        "attributes": connectome.attributes,
        "notes": connectome.notes,
        "schema": describe_schema(),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    return target


# ----------------------------------------------------------------- sources


def synthetic_connectome(
    num_neurons: int = 64,
    connectivity: float = 0.15,
    seed: int = 0,
    name: str = "synthetic-connectome",
) -> Connectome:
    """Generate a random directed graph for development and tests.

    .. warning::
       This is **not** biological data. It is a uniformly random graph with
       random weights, provided so that the simulation, the training loop and
       the tests have something to run on. It carries no anatomical meaning and
       must never be described as a fruit-fly connectome.

    Every neuron is guaranteed at least one outgoing and one incoming edge, so a
    leaky integrate-and-fire population driven by this graph actually fires.
    """
    if num_neurons < 2:
        raise ConnectomeError("synthetic_connectome needs at least 2 neurons")
    if not 0.0 < connectivity <= 1.0:
        raise ConnectomeError("connectivity must be in (0, 1]")

    rng = np.random.default_rng(seed)
    node_ids = [f"syn{index:04d}" for index in range(num_neurons)]
    edges: list[ConnectomeEdge] = []
    seen: set[tuple[str, str]] = set()

    for pre in node_ids:
        for post in node_ids:
            if pre == post:
                continue
            if rng.random() < connectivity:
                seen.add((pre, post))

    for node in node_ids:
        targets = {post for pre, post in seen if pre == node} or {
            other for other in node_ids if other != node and rng.random() < connectivity
        }
        for post in targets:
            seen.add((node, post))
    for node in node_ids:
        if any(post == node for _, post in seen):
            continue
        for candidate in rng.permutation(node_ids):
            candidate_id = str(candidate)
            if candidate_id != node:
                seen.add((candidate_id, node))
                break

    for pre, post in sorted(seen):
        edges.append(
            ConnectomeEdge(
                pre=pre,
                post=post,
                weight=float(rng.uniform(0.1, 1.0)),
                synapse_type="unspecified",
                neurotransmitter=None,
                neuropil=None,
                weight_is_biological=False,
                metadata={"generated": "synthetic_connectome", "seed": seed},
            )
        )

    return Connectome(
        name=name,
        source=DataSource.SYNTHETIC,
        edges=edges,
        attributes={"weight": "random (not biological)", "synapse_type": "placeholder"},
        notes="Random graph generated for testing. Contains no fruit-fly anatomy.",
    )


def load_flywire_connectome(*args: Any, **kwargs: Any) -> Connectome:
    """Placeholder for the real FlyWire connectome loader.

    Not implemented. No FlyWire data is bundled, downloaded, or approximated by
    this project, and no substitute values are invented.

    TODO(next milestone):
      1. Accept a local path to a *user-downloaded* FlyWire export
         (``neuroprops_mini``-style tables, one row per synapse, columns
         ``pre``/``post``/``weight`` where available). Never auto-download.
      2. Stream with ``pandas.read_csv(..., chunksize=...)``; the full FAFB
         table is millions of rows and must not be materialised eagerly.
      3. Populate :attr:`Connectome.version`, :attr:`Connectome.url`,
         :attr:`Connectome.license` and :attr:`Connectome.citation` from the
         export's own metadata, and require them.
      4. Set ``weight_is_biological=True`` only for columns the export documents
         as measured synaptic strength; initialise the rest and flag them.
      5. Emit a warning if a requested column is absent so the gap is visible
         rather than silently filled.
    """
    raise NotImplementedError(
        "The real FlyWire connectome is not integrated yet. FlyBrain currently has no "
        "published fruit-fly connectome data; it ships a synthetic graph for development only. "
        "See data/README.md for the intended schema and free sources."
    )


def build_connectome_from_edges(
    edges: Iterable[ConnectomeEdge],
    name: str = "connectome",
    source: DataSource = DataSource.SYNTHETIC,
    **metadata: Any,
) -> Connectome:
    """Assemble a connectome from an iterable of edges."""
    return Connectome(name=name, source=source, edges=list(edges), **metadata)
