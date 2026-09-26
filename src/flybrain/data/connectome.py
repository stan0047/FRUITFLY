"""Connectome data model, I/O and provenance.

The connectome is **biological data**: a published measurement of which neurons
are connected to which. It is *not* a neural network, and it contains no weights,
no activation functions, no dynamics and no learning rules. Anything this module
cannot obtain from a source is left ``None`` and reported as an author
assumption; fabricated values are never substituted.

Two storage paths
-----------------
``edges``
    A list of :class:`ConnectomeEdge` objects. Convenient, readable, and
    appropriate for the small synthetic graphs used in development. A Python
    object per synapse costs hundreds of bytes, so this path does not scale.

``frame``
    A canonical :class:`pandas.DataFrame` produced by
    :mod:`flybrain.data.importer`. One row per synapse, three required columns.
    This is the path real data takes, because it stays in vectorised typed
    memory and is never expanded into a dense ``nodes x nodes`` matrix.

:meth:`Connectome.sparse_matrix` is the only conversion to graph form, and it
always produces a :class:`scipy.sparse.csr_matrix`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybrain.data.provenance import DataSource, Provenance, ProvenanceError, biological_label

__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "Connectome",
    "ConnectomeEdge",
    "ConnectomeError",
    "DataSource",
    "Provenance",
    "available_sources",
    "describe_schema",
    "is_source_available",
    "load_connectome",
    "load_flywire_connectome",
    "save_connectome",
    "synthetic_connectome",
]

#: Columns the legacy ``pre``/``post`` edge format must provide.
REQUIRED_COLUMNS: tuple[str, ...] = ("pre", "post", "weight")
#: Columns used when the upstream source reports them.
OPTIONAL_COLUMNS: tuple[str, ...] = ("synapse_type", "neurotransmitter", "neuropil")

#: Canonical frame columns written by the importer.
CANONICAL_FRAME_COLUMNS: tuple[str, ...] = ("source_id", "target_id", "weight")

#: Guard against a dense matrix for a real-sized graph. 5000^2 float32 is 100 MB.
DEFAULT_MAX_DENSE_NODES = 5_000

#: Guard against materialising a huge graph in NetworkX.
DEFAULT_MAX_NETWORKX_NODES = 2_000


class ConnectomeError(RuntimeError):
    """Raised for malformed connectome data or unavailable sources."""


#: Sources that are actually wired up, with the loader that serves them.
_SOURCE_LOADERS: dict[DataSource, str] = {
    DataSource.SYNTHETIC: "synthetic_connectome",
    DataSource.EXTERNAL: "import_connectome",
    DataSource.FLYWIRE: "import_connectome",
}


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

    Biological status is **never** inferred from a filename or a directory name.
    It comes from :attr:`provenance_record`, and for the :attr:`edges` path from
    :attr:`source` combined with an explicit :attr:`biological_data` flag.
    """

    name: str
    source: DataSource = DataSource.SYNTHETIC
    edges: list[ConnectomeEdge] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    node_regions: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)
    version: str | None = None
    url: str | None = None
    license: str | None = None
    citation: str | None = None
    notes: str = ""
    provenance_record: Provenance | None = None
    biological_data: bool | None = None
    is_subset: bool = False
    subset_reason: str | None = None
    _biological: bool | None = field(default=None, repr=False, compare=False)
    _neuron_cache: pd.Index | None = field(default=None, repr=False, compare=False)

    # --------------------------------------------------------- provenance

    @property
    def is_biological(self) -> bool:
        """Whether the topology is declared to originate from a biological source.

        ``False`` for :attr:`DataSource.SYNTHETIC` and
        :attr:`DataSource.TEST_FIXTURE` under all circumstances, whatever the
        caller passed. A file called ``flywire.csv`` is therefore not biological
        data on the strength of its name.
        """
        if self.provenance_record is not None:
            return self.provenance_record.is_biological
        if not self.source.can_be_biological:
            return False
        if self.biological_data is not None:
            return bool(self.biological_data)
        if self._biological is not None:
            return bool(self._biological)
        # Conservative default: without an explicit declaration, nothing is
        # claimed to be biological. A filename is not a declaration.
        return False

    @property
    def label(self) -> str:
        """Banner for reports and figures."""
        if self.provenance_record is not None:
            return self.provenance_record.label
        return self.source.label if not self.source.can_be_biological else biological_label(self.is_biological)

    def provenance(self) -> dict[str, Any]:
        """Summary suitable for logs, figures and reports."""
        record = self.provenance_record
        return {
            "name": self.name,
            "source": self.source.value,
            "is_biological": self.is_biological,
            "label": self.label,
            "version": self.version or (record.dataset_version if record else None),
            "url": self.url or (record.source_url if record else None),
            "license": self.license or (record.license if record else None),
            "citation": self.citation or (record.citation if record else None),
            "source_name": record.source_name if record else self.source.value,
            "imported_at": record.imported_at if record else None,
            "original_filename": record.original_filename if record else None,
            "unknown_provenance_fields": record.missing_fields() if record else [],
            "num_edges": self.num_edges,
            "num_neurons": self.num_neurons,
            "scope": "subset" if self.is_subset else "complete",
            "notes": self.notes,
        }

    def warn_if_synthetic(self) -> str:
        """One-line caveat that must accompany any derived artefact."""
        if self.is_biological:
            record = self.provenance_record
            return record.warning() if record else f"[Biological connectome data] source='{self.source.value}'"
        if self.source is DataSource.TEST_FIXTURE:
            return (
                f"[TEST FIXTURE - NOT REAL BIOLOGICAL DATA] '{self.name}'; statistics are not anatomy."
            )
        return (
            f"[SIMULATION INPUT] '{self.name}' uses synthetic (non-biological) connectivity; "
            "results are not claims about the fruit-fly connectome."
        )

    # ----------------------------------------------------------- structure

    @property
    def edge_frame(self) -> pd.DataFrame:
        """Canonical edge table: one row per synapse, ``source_id``/``target_id``/``weight``.

        For the ``edges`` storage path this materialises the frame on first
        access and caches it; for the ``frame`` path it is returned directly.
        """
        if self.frame is None:
            if not self.edges:
                self.frame = pd.DataFrame(columns=list(CANONICAL_FRAME_COLUMNS))
            else:
                self.frame = pd.DataFrame(
                    {
                        "source_id": [edge.pre for edge in self.edges],
                        "target_id": [edge.post for edge in self.edges],
                        "weight": np.array([edge.weight for edge in self.edges], dtype="float64"),
                        "weight_is_biological": [bool(edge.weight_is_biological) for edge in self.edges],
                    }
                )
        return self.frame

    @property
    def weights_are_biological(self) -> bool:
        """Whether every edge weight came from a measured quantity."""
        frame = self.edge_frame
        if "weight_is_biological" not in frame.columns or frame.empty:
            return False
        return bool(frame["weight_is_biological"].fillna(False).astype(bool).all())

    @property
    def neuron_ids(self) -> pd.Index:
        """Unique neuron identifiers, cached."""
        if self._neuron_cache is None:
            frame = self.edge_frame
            if frame.empty:
                self._neuron_cache = pd.Index([], dtype="object")
            else:
                self._neuron_cache = pd.Index(
                    pd.unique(pd.concat([frame["source_id"], frame["target_id"]], ignore_index=True))
                )
        return self._neuron_cache

    @property
    def neurons(self) -> list[str]:
        """Sorted unique neuron identifiers. O(n) memory; prefer :attr:`neuron_ids`."""
        return sorted(str(node) for node in self.neuron_ids)

    @property
    def num_neurons(self) -> int:
        return int(len(self.neuron_ids))

    @property
    def num_edges(self) -> int:
        if self.frame is not None:
            return int(len(self.frame))
        return len(self.edges)

    def regions(self) -> dict[str, str]:
        """Neuron -> brain region, from the node table or the edge frame."""
        if self.node_regions:
            return dict(self.node_regions)
        frame = self.edge_frame
        mapping: dict[str, str] = {}
        if frame.empty:
            return mapping
        for id_column, region_column in (("source_id", "source_region"), ("target_id", "target_region")):
            if region_column not in frame.columns:
                continue
            pairs = frame[[id_column, region_column]].dropna()
            for node, region in zip(pairs[id_column], pairs[region_column], strict=True):
                mapping.setdefault(str(node), str(region))
        return mapping

    def region_counts(self, top_n: int = 10) -> dict[str, int]:
        """Number of neurons per region label, most common first."""
        counts: dict[str, int] = {}
        for region in self.regions().values():
            counts[region] = counts.get(region, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n])

    def __len__(self) -> int:
        return self.num_edges

    def __iter__(self) -> Iterator[ConnectomeEdge]:
        """Iterate edges. Materialises Python objects; avoid on large datasets."""
        if self.frame is None:
            return iter(self.edges)
        return iter(self._edges_from_frame())

    def _edges_from_frame(self) -> Iterator[ConnectomeEdge]:
        for row in self.edge_frame.itertuples(index=False):
            data = row._asdict()
            yield ConnectomeEdge.from_dict(
                {
                    "pre": str(data["source_id"]),
                    "post": str(data["target_id"]),
                    "weight": float(data.get("weight", 1.0) or 0.0),
                    "neuropil": data.get("target_region") or data.get("source_region"),
                    "weight_is_biological": bool(data.get("weight_is_biological", False)),
                }
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy table in the legacy ``pre``/``post`` shape, one row per synapse."""
        if self.frame is None:
            rows: list[dict[str, Any]] = []
            for edge in self.edges:
                row = edge.to_dict()
                row["metadata"] = json.dumps(dict(edge.metadata), sort_keys=True)
                rows.append(row)
            columns = [
                "pre",
                "post",
                "weight",
                "synapse_type",
                "neurotransmitter",
                "neuropil",
                "weight_is_biological",
                "metadata",
            ]
            return pd.DataFrame(rows, columns=columns)
        frame = self.edge_frame.copy(deep=False)
        return pd.DataFrame(
            {
                "pre": frame["source_id"],
                "post": frame["target_id"],
                "weight": frame["weight"],
                "synapse_type": frame["source_type"] if "source_type" in frame else None,
                "neurotransmitter": None,
                "neuropil": frame["target_region"] if "target_region" in frame else None,
                "weight_is_biological": frame.get("weight_is_biological", False),
                "metadata": "",
            }
        )

    # ------------------------------------------------------------ conversion

    def sparse_matrix(
        self,
        node_order: Sequence[str] | None = None,
        dtype: Any = np.float32,
    ) -> sp.csr_matrix:
        """Sparse adjacency matrix, ``W[i, j]`` the total weight of ``i -> j``.

        Always CSR and never dense. Duplicate ``(i, j)`` pairs are summed, which
        matches the duplicate-aggregation policy in
        :mod:`flybrain.data.validation`.
        """
        frame = self.edge_frame
        if frame.empty:
            size = len(node_order) if node_order is not None else 0
            return sp.csr_matrix((size, size), dtype=dtype)

        sources = frame["source_id"].to_numpy()
        targets = frame["target_id"].to_numpy()
        weights = frame["weight"].to_numpy(dtype="float64")
        codes, uniques = pd.factorize(np.concatenate([sources, targets]))
        half = len(frame)
        rows, cols = codes[:half], codes[half:]

        if node_order is not None:
            positions = pd.Index(uniques).get_indexer(pd.Index([str(n) for n in node_order]))
            keep = (positions[rows] >= 0) & (positions[cols] >= 0)
            rows, cols = positions[rows[keep]], positions[cols[keep]]
            size = len(node_order)
        else:
            size = len(uniques)

        return sp.coo_matrix(
            (weights.astype(dtype), (rows, cols)), shape=(size, size), dtype=dtype
        ).tocsr()

    def weight_matrix(
        self,
        node_order: Sequence[str] | None = None,
        dtype: Any = np.float32,
        max_dense_nodes: int = DEFAULT_MAX_DENSE_NODES,
    ) -> np.ndarray:
        """Dense adjacency matrix. Refuses for graphs too large to densify.

        Prefer :meth:`sparse_matrix`; the dense form exists for the small
        synthetic graphs the simulation uses.
        """
        order = list(node_order) if node_order is not None else self.neuron_ids
        if len(order) > max_dense_nodes:
            raise ConnectomeError(
                f"refusing to build a dense {len(order)}x{len(order)} matrix "
                f"(limit {max_dense_nodes}; that would need about "
                f"{len(order) * len(order) * 4 / 1e6:.0f} MB). Use sparse_matrix() instead, "
                "or extract a subgraph."
            )
        return np.asarray(self.sparse_matrix(node_order=order, dtype=dtype).todense())

    def to_networkx(
        self,
        max_nodes: int = DEFAULT_MAX_NETWORKX_NODES,
        node_order: Sequence[str] | None = None,
    ) -> Any:
        """Convert to a ``networkx.DiGraph``.

        Guarded, because NetworkX objects cost far more memory than the sparse
        matrix they come from. Extract a subgraph first for real-sized data.
        """
        import networkx as nx

        order = list(node_order) if node_order is not None else [str(n) for n in self.neuron_ids]
        if len(order) > max_nodes:
            raise ConnectomeError(
                f"refusing to build a NetworkX graph with {len(order)} nodes (limit {max_nodes}). "
                "NetworkX is intended for small extracted subgraphs only; use sparse_matrix() for "
                "the full connectome."
            )
        matrix = self.sparse_matrix(node_order=order)
        graph = nx.DiGraph()
        graph.add_nodes_from(order)
        coo = matrix.tocoo()
        for row, col, value in zip(coo.row, coo.col, coo.data, strict=True):
            graph.add_edge(order[int(row)], order[int(col)], weight=float(value))
        return graph

    def save(self, path: str | Path) -> Path:
        """Write the edge table plus a provenance sidecar next to it."""
        return save_connectome(self, path)

    # ------------------------------------------------------------ builders

    @classmethod
    def from_normalized(
        cls,
        frame: pd.DataFrame,
        name: str,
        provenance: Provenance | None = None,
        node_regions: Mapping[str, str] | None = None,
        is_subset: bool = False,
        subset_reason: str | None = None,
    ) -> "Connectome":
        """Build from a canonical edge frame produced by the importer.

        This is the path real data takes. The frame is kept as-is (no per-edge
        Python objects), so memory stays proportional to the edge count.
        """
        missing = [column for column in CANONICAL_FRAME_COLUMNS if column not in frame.columns]
        if missing:
            raise ConnectomeError(f"canonical frame is missing required column(s): {', '.join(missing)}")
        if frame.empty:
            raise ConnectomeError("cannot build a connectome from an empty edge table")

        source = provenance.source if provenance else DataSource.EXTERNAL
        return cls(
            name=name,
            source=source,
            frame=frame,
            node_regions=dict(node_regions or {}),
            version=provenance.dataset_version if provenance else None,
            url=provenance.source_url if provenance else None,
            license=provenance.license if provenance else None,
            citation=provenance.citation if provenance else None,
            notes=provenance.notes if provenance else "",
            provenance_record=provenance,
            biological_data=provenance.biological_data if provenance else None,
            is_subset=is_subset,
            subset_reason=subset_reason,
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        name: str,
        source: DataSource = DataSource.SYNTHETIC,
        **metadata: Any,
    ) -> "Connectome":
        """Build a connectome from a legacy ``pre``/``post`` table."""
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
    default_biological = bool(payload.pop("weight_is_biological", source.can_be_biological))
    return {**payload, "weight_is_biological": default_biological}


# --------------------------------------------------------------------- I/O


def describe_schema() -> dict[str, Any]:
    """Return the processed-table contract."""
    return {
        "legacy_columns": list(REQUIRED_COLUMNS),
        "canonical_columns": list(CANONICAL_FRAME_COLUMNS),
        "optional_columns": list(OPTIONAL_COLUMNS),
        "granularity": "one row per synapse",
        "storage": "pandas DataFrame; convert with sparse_matrix() (scipy CSR), never dense",
        "weight_is_biological": (
            "True only if the upstream source reports synaptic strength. "
            "Initialised or placeholder values must set it to False."
        ),
        "unknown_values": "Use null/NA rather than guessing. Unknown stays unknown.",
        "biological_status": "Declared explicitly via Provenance; never inferred from a filename.",
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


def load_connectome(path: str | Path, name: str | None = None) -> "Connectome":
    """Load a processed connectome table plus its ``.source.json`` sidecar.

    The sidecar is mandatory: without it there is no way to know whether the data
    is biological, so a missing file is an error rather than a silent default.
    """
    from flybrain.data.schema import detect_file_type, read_table

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

    frame = read_table(csv_path, file_type=detect_file_type(csv_path))
    if not isinstance(frame, pd.DataFrame):  # pragma: no cover
        raise ConnectomeError(f"could not read {csv_path.name} as a table")

    dataset_name = name or str(sidecar.get("name", csv_path.stem))
    if {"source_id", "target_id", "weight"} <= set(frame.columns):
        provenance = Provenance.from_dict(
            {
                "source_name": str(sidecar.get("source_name", source.value)),
                "dataset_name": dataset_name,
                "source": source.value,
                "dataset_version": sidecar.get("dataset_version", sidecar.get("version")),
                "source_url": sidecar.get("source_url", sidecar.get("url")),
                "license": sidecar.get("license"),
                "citation": sidecar.get("citation"),
                "imported_at": sidecar.get("imported_at"),
                "original_filename": sidecar.get("original_filename"),
                "biological_data": bool(sidecar.get("is_biological", False)),
                "notes": str(sidecar.get("notes", "")),
            }
        )
        connectome = Connectome.from_normalized(
            frame=frame,
            name=dataset_name,
            provenance=provenance,
            is_subset=bool(sidecar.get("is_subset", False)),
            subset_reason=sidecar.get("subset_reason"),
        )
    else:
        connectome = Connectome.from_dataframe(
            frame,
            name=dataset_name,
            source=source,
            version=sidecar.get("dataset_version", sidecar.get("version")),
            url=sidecar.get("source_url", sidecar.get("url")),
            license=sidecar.get("license"),
            citation=sidecar.get("citation"),
            notes=str(sidecar.get("notes", "")),
        )
    connectome.attributes.update({str(k): str(v) for k, v in (sidecar.get("attributes") or {}).items()})
    return connectome


def save_connectome(connectome: "Connectome", path: str | Path) -> Path:
    """Write the connectome table and its provenance sidecar to disk."""
    target = Path(path)
    if target.suffix not in {".csv", ".tsv", ".parquet", ".jsonl"}:
        target = target.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    frame = connectome.edge_frame
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix == ".jsonl":
        frame.to_json(target, orient="records", lines=True)
    else:
        frame.to_csv(target, index=False)

    sidecar_path = target.with_suffix("").with_suffix(".source.json")
    provenance = connectome.provenance_record
    sidecar = {
        "name": connectome.name,
        "source": connectome.source.value,
        "is_biological": connectome.is_biological,
        "label": connectome.label,
        "source_name": provenance.source_name if provenance else connectome.source.value,
        "dataset_name": provenance.dataset_name if provenance else connectome.name,
        "dataset_version": connectome.version,
        "version": connectome.version,
        "source_url": connectome.url,
        "url": connectome.url,
        "license": connectome.license,
        "citation": connectome.citation,
        "imported_at": provenance.imported_at if provenance else None,
        "original_filename": provenance.original_filename if provenance else None,
        "biological_data": provenance.biological_data if provenance else connectome.biological_data,
        "unknown_provenance_fields": provenance.missing_fields() if provenance else [],
        "is_subset": connectome.is_subset,
        "subset_reason": connectome.subset_reason,
        "weight_is_biological": connectome.weights_are_biological,
        "attributes": connectome.attributes,
        "notes": connectome.notes,
        "schema": describe_schema(),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str), encoding="utf-8")
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


def load_flywire_connectome(
    path: str | Path,
    provenance: Provenance | None = None,
    options: Any = None,
    output_dir: str | Path | None = None,
) -> Any:
    """Import a **locally downloaded** FlyWire export.

    This function does not download anything, does not talk to an API, does not
    authenticate, and contains no hardcoded dataset URL. You download the export
    yourself from the official FlyWire / Codex resources, then pass the local
    file path here.

    Parameters
    ----------
    path:
        Path to a local export: CSV, TSV, Parquet, JSON or JSON Lines.
    provenance:
        Attribution you supply. Anything you leave unset stays ``None`` and is
        reported as unverified; FlyBrain does not fill it in for you.
    options:
        Optional :class:`~flybrain.data.importer.ImportOptions`.
    output_dir:
        When given, the normalised table and provenance sidecar are written here.

    Returns
    -------
    ImportResult
        From :func:`flybrain.data.importer.import_connectome`, including the
        dataset report inputs.

    Notes
    -----
    FlyBrain has not been tested against a specific published export, because no
    export has been supplied yet. Column names differ between releases, which is
    why the importer inspects the schema and asks rather than assuming. Start
    with::

        python scripts/inspect_connectome.py --input <your-file>

    and pass ``--source-column``/``--target-column`` if the mapping is ambiguous.
    """
    from flybrain.data.importer import import_connectome

    # Check the raw string before Path() normalises it: on Windows
    # Path("https://host/x") becomes "https:\\host\\x" and the "://" check
    # would silently miss it.
    if "://" in str(path) or str(path).lower().startswith(("http:", "https:", "ftp:", "s3:", "gs:")):
        raise ConnectomeError(
            f"refusing to fetch '{path}'.\n"
            "FlyBrain never downloads connectome data. Download the export manually from the "
            "official FlyWire / Codex resources and pass the local path."
        )

    resolved = Path(path)
    if not resolved.is_file():
        raise ConnectomeError(
            f"file not found: {resolved}\n"
            "FlyBrain does not bundle or download a FlyWire dataset. Download it yourself from "
            "the official source and pass the local file path."
        )

    record = provenance or Provenance(
        source_name="flywire",
        dataset_name=resolved.stem,
        source=DataSource.FLYWIRE,
        original_filename=resolved.name,
        notes="User-supplied local export.",
    )
    if record.source is DataSource.SYNTHETIC:
        raise ProvenanceError(
            "a FlyWire import cannot be labelled synthetic; set Provenance.source explicitly"
        )
    return import_connectome(resolved, provenance=record, options=options, output_dir=output_dir)


def build_connectome_from_edges(
    edges: Iterable[ConnectomeEdge],
    name: str = "connectome",
    source: DataSource = DataSource.SYNTHETIC,
    **metadata: Any,
) -> Connectome:
    """Assemble a connectome from an iterable of edges."""
    return Connectome(name=name, source=source, edges=list(edges), **metadata)
