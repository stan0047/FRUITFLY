"""Connectivity graph container and generators.

A :class:`BrainGraph` is the boundary object between *data* and *simulation*.
It carries node identifiers, directed synaptic weights and region labels, and it
knows nothing about membrane potentials, rewards, or policies.

Provenance is preserved end to end: a graph built from a real connectome export
keeps ``source != "synthetic"`` and ``is_biological=True``, while the graph used
by the tests carries ``source="synthetic"`` and can never be mistaken for fly
anatomy.

Storage
-------
``weights`` may be a dense ``numpy`` array **or** a :class:`scipy.sparse.csr_matrix`.
Real connectomes are sparse by nature: a few hundred thousand neurons and tens of
millions of synapses, of which the overwhelming majority of ``nodes x nodes``
entries are zero. Densifying that is not an optimisation problem but an
impossibility, so :meth:`BrainGraph.from_connectome` builds a sparse graph by
default and the dense path is reserved for the small synthetic graphs the
simulation uses.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

from flybrain.data.connectome import Connectome, DataSource

__all__ = [
    "DEFAULT_MAX_NETWORKX_NODES",
    "SYNTHETIC_REGIONS",
    "BrainGraph",
    "BrainGraphError",
    "iter_regions",
    "synthetic_brain_graph",
]

#: Arbitrary labels for the synthetic graph. NOT neuropil names.
SYNTHETIC_REGIONS: tuple[str, ...] = ("visual_system", "central_brain", "descending_pathway")

#: Refuse to materialise a NetworkX graph above this many nodes.
DEFAULT_MAX_NETWORKX_NODES = 2_000


class BrainGraphError(RuntimeError):
    """Raised for inconsistent graphs or over-budget conversions."""


@dataclass
class BrainGraph:
    """A directed, weighted neuron-to-neuron graph.

    Attributes
    ----------
    node_ids:
        Stable ordering of node identifiers. The index of a node in this list is
        its index in every array the simulation produces.
    weights:
        ``weights[i, j]`` is the strength of the synapse ``node_ids[i] ->
        node_ids[j]``. Zero means no connection. Either a dense ``ndarray`` or a
        sparse CSR matrix; see :attr:`is_sparse`.
    regions:
        Optional node -> region label mapping used for region-level reporting.
    """

    node_ids: list[str]
    weights: Any
    regions: dict[str, str] = field(default_factory=dict)
    name: str = "graph"
    source: str = DataSource.SYNTHETIC.value
    is_biological: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ validation

    def __post_init__(self) -> None:
        self.node_ids = [str(node) for node in self.node_ids]
        if sp.issparse(self.weights):
            self.weights = sp.csr_matrix(self.weights, dtype=np.float32)
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32)

        expected = (len(self.node_ids), len(self.node_ids))
        if tuple(self.weights.shape) != expected:
            raise BrainGraphError(f"weights shape {self.weights.shape} does not match {expected}")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise BrainGraphError("node identifiers must be unique")
        unknown = set(self.regions) - set(self.node_ids)
        if unknown:
            raise BrainGraphError(f"region labels reference unknown nodes: {sorted(unknown)}")

    # -------------------------------------------------------------- storage

    @property
    def is_sparse(self) -> bool:
        """Whether weights are held sparsely."""
        return sp.issparse(self.weights)

    @property
    def nnz(self) -> int:
        """Number of stored non-zero weights."""
        return int(self.weights.nnz) if self.is_sparse else int(np.count_nonzero(self.weights))

    def to_dense(self) -> np.ndarray:
        """Densify. Refuses for graphs too large to hold in memory."""
        size = len(self.node_ids)
        if size > 5_000:
            raise BrainGraphError(
                f"refusing to densify a {size}x{size} matrix (that would need about "
                f"{size * size * 4 / 1e6:.0f} MB); extract a subgraph instead"
            )
        return np.asarray(self.weights.todense()) if self.is_sparse else self.weights

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.nnz

    @property
    def index(self) -> dict[str, int]:
        """Node identifier -> array index."""
        return {node: i for i, node in enumerate(self.node_ids)}

    def node_id(self, index: int) -> str:
        return self.node_ids[int(index)]

    def node_index(self, node_id: str) -> int:
        try:
            return self.index[node_id]
        except KeyError as exc:
            raise BrainGraphError(f"unknown node: {node_id!r}") from exc

    def region_of(self, node_id: str) -> str:
        """Region label for a node, or ``"unassigned"``."""
        return self.regions.get(str(node_id), "unassigned")

    def region_names(self) -> list[str]:
        seen = {self.region_of(node) for node in self.node_ids}
        return sorted(seen)

    def nodes_in_region(self, region: str) -> list[str]:
        return [node for node in self.node_ids if self.region_of(node) == region]

    def region_indices(self) -> dict[str, np.ndarray]:
        """Node indices grouped by region, for fast activity aggregation.

        Index arrays rather than boolean masks: a real connectome has far too
        many nodes for one mask per region to be free.
        """
        grouped: dict[str, list[int]] = {region: [] for region in self.region_names()}
        for position, node in enumerate(self.node_ids):
            grouped[self.region_of(node)].append(position)
        return {region: np.asarray(positions, dtype=np.int64) for region, positions in grouped.items()}

    def _degree_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """In- and out-degree per node, without densifying.

        Degree is the number of distinct connections, not the summed weight, so
        that dense and sparse graphs report the same quantity.
        """
        if self.is_sparse:
            binary = self.weights.copy()
            binary.data = np.ones_like(binary.data, dtype=np.int64)
            return (
                np.asarray(binary.sum(axis=0)).ravel().astype(np.int64),
                np.asarray(binary.sum(axis=1)).ravel().astype(np.int64),
            )
        size = self.num_nodes
        return (
            np.count_nonzero(self.weights, axis=0).astype(np.int64),
            np.count_nonzero(self.weights, axis=1).astype(np.int64),
        )

    def in_degree(self, node_id: str) -> int:
        return int(self._degree_vectors()[0][self.node_index(node_id)])

    def out_degree(self, node_id: str) -> int:
        return int(self._degree_vectors()[1][self.node_index(node_id)])

    def degrees(self) -> dict[str, np.ndarray]:
        """``{"in": ..., "out": ..., "total": ...}`` degree vectors."""
        in_degree, out_degree = self._degree_vectors()
        return {"in": in_degree, "out": out_degree, "total": in_degree + out_degree}

    def degree_summary(self) -> dict[str, float]:
        degrees = self.degrees()
        if self.num_nodes == 0:
            return {"mean_in": 0.0, "mean_out": 0.0, "max_in": 0, "max_out": 0, "mean_total": 0.0}
        return {
            "mean_in": float(degrees["in"].mean()),
            "mean_out": float(degrees["out"].mean()),
            "max_in": int(degrees["in"].max()),
            "max_out": int(degrees["out"].max()),
            "mean_total": float(degrees["total"].mean()),
        }

    def strongest_edges(self, k: int = 10) -> list[tuple[str, str, float]]:
        """The ``k`` heaviest synapses, for inspection and debugging."""
        if k <= 0 or self.num_nodes == 0:
            return []
        if self.is_sparse:
            matrix = self.weights.tocoo()
            if matrix.nnz == 0:
                return []
            picks = np.argpartition(matrix.data, -min(k, matrix.nnz))[-min(k, matrix.nnz) :]
            return [
                (
                    self.node_ids[int(matrix.row[position])],
                    self.node_ids[int(matrix.col[position])],
                    float(matrix.data[position]),
                )
                for position in picks[np.argsort(-matrix.data[picks])]
            ]
        flat = np.argsort(self.weights, axis=None)[::-1][:k]
        edges: list[tuple[str, str, float]] = []
        for position in flat:
            row, col = np.unravel_index(position, self.weights.shape)
            edges.append((self.node_ids[row], self.node_ids[col], float(self.weights[row, col])))
        return edges

    # --------------------------------------------------------- conversions

    @classmethod
    def from_connectome(
        cls,
        connectome: Connectome,
        name: str | None = None,
        sparse: bool = True,
        max_dense_nodes: int = 5_000,
    ) -> "BrainGraph":
        """Build a graph from a :class:`Connectome`, preserving provenance.

        Parameters
        ----------
        sparse:
            ``True`` (default) keeps the weights in a CSR matrix, which is the
            only viable option at connectome scale. ``False`` is available for
            small datasets and raises above ``max_dense_nodes``.
        """
        node_ids = [str(node) for node in connectome.neuron_ids]
        if not sparse and len(node_ids) > max_dense_nodes:
            raise BrainGraphError(
                f"refusing to build a dense graph for {len(node_ids)} nodes (limit {max_dense_nodes}). "
                "Use sparse=True, the default, or extract a subgraph first."
            )
        matrix = connectome.sparse_matrix(node_order=node_ids)
        weights = matrix if sparse else np.asarray(matrix.todense(), dtype=np.float32)
        return cls(
            node_ids=node_ids,
            weights=weights,
            regions=connectome.regions(),
            name=name or connectome.name,
            source=connectome.source.value,
            is_biological=connectome.is_biological,
            metadata={
                **connectome.provenance(),
                "storage": "sparse (scipy CSR)" if sparse else "dense (numpy)",
                "weights_are_biological": connectome.weights_are_biological,
            },
        )

    def to_networkx(
        self,
        max_nodes: int = DEFAULT_MAX_NETWORKX_NODES,
        allow_large: bool = False,
    ) -> nx.DiGraph:
        """Convert to a ``networkx.DiGraph`` for algorithms and plotting.

        Guarded by default: a NetworkX edge costs orders of magnitude more memory
        than a CSR entry, so the full connectome must never be converted. Pass
        ``allow_large=True`` only for graphs you know are small.
        """
        if not allow_large and self.num_nodes > max_nodes:
            raise BrainGraphError(
                f"refusing to build a NetworkX graph with {self.num_nodes} nodes (limit {max_nodes}). "
                "Extract a subgraph first, e.g. graph.subgraph(graph.top_nodes(500)), or pass "
                "allow_large=True if you really mean it."
            )
        graph = nx.DiGraph()
        for node in self.node_ids:
            graph.add_node(node, region=self.region_of(node))
        if self.is_sparse:
            coo = self.weights.tocoo()
            for row, col, value in zip(coo.row, coo.col, coo.data, strict=True):
                graph.add_edge(self.node_ids[int(row)], self.node_ids[int(col)], weight=float(value))
        else:
            rows, cols = np.nonzero(self.weights)
            for row, col in zip(rows, cols, strict=True):
                graph.add_edge(self.node_ids[int(row)], self.node_ids[int(col)], weight=float(self.weights[row, col]))
        return graph

    @classmethod
    def from_networkx(
        cls,
        graph: nx.DiGraph,
        name: str = "graph",
        regions: Mapping[str, str] | None = None,
    ) -> "BrainGraph":
        node_ids = list(graph.nodes())
        weights = nx.to_numpy_array(graph, nodelist=node_ids, dtype=np.float32)
        labels = dict(regions) if regions is not None else {n: str(graph.nodes[n].get("region", "unassigned")) for n in node_ids}
        return cls(node_ids=node_ids, weights=weights, regions=labels, name=name)

    def subgraph(self, node_ids: Sequence[str], name: str | None = None) -> "BrainGraph":
        """Restrict to a subset of nodes, keeping the induced connections.

        Stays sparse if the parent is sparse, so extracting a subgraph from a
        real connectome does not densify it.
        """
        keep = [node for node in node_ids if node in self.index]
        idx = [self.node_index(node) for node in keep]
        if self.is_sparse:
            sub_weights = sp.csr_matrix(self.weights[np.ix_(idx, idx)]) if idx else sp.csr_matrix((0, 0), dtype=np.float32)
        else:
            sub_weights = self.weights[np.ix_(idx, idx)] if idx else np.zeros((0, 0), dtype=np.float32)
        return BrainGraph(
            node_ids=keep,
            weights=sub_weights,
            regions={node: self.region_of(node) for node in keep},
            name=name or f"{self.name}:subgraph",
            source=self.source,
            is_biological=self.is_biological,
            metadata=dict(self.metadata),
        )

    def top_nodes(self, k: int = 100) -> list[str]:
        """The ``k`` most connected nodes, by total degree."""
        if self.num_nodes == 0 or k <= 0:
            return []
        total = self.degrees()["total"]
        picks = np.argpartition(total, -min(k, self.num_nodes))[-min(k, self.num_nodes) :]
        return [self.node_ids[int(position)] for position in picks[np.argsort(-total[picks])]]

    # ------------------------------------------------------------- (de)serial

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "is_biological": self.is_biological,
            "num_nodes": self.num_nodes,
            "node_ids": self.node_ids,
            "storage": "sparse" if self.is_sparse else "dense",
            "regions": self.regions,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Persist the graph next to ``path``.

        Weights go to ``<stem>.npz`` via SciPy's own sparse writer (NumPy's
        ``savez`` would coerce a CSR matrix into an object array), and node
        order, regions and provenance go to ``<stem>.json``.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        weights_path = target.with_suffix(".npz")
        if self.is_sparse:
            sp.save_npz(weights_path, sp.csr_matrix(self.weights))
        else:
            np.savez_compressed(weights_path, weights=self.weights)
        target.with_suffix(".json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BrainGraph":
        base = Path(path)
        payload = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        weights_path = base.with_suffix(".npz")
        if str(payload.get("storage", "dense")) == "sparse":
            weights: Any = sp.load_npz(weights_path)
        else:
            with np.load(weights_path) as archive:
                weights = archive["weights"]
        return cls(
            node_ids=[str(node) for node in payload.get("node_ids", [])],
            weights=weights,
            regions={str(k): str(v) for k, v in dict(payload.get("regions") or {}).items()},
            name=str(payload.get("name", base.stem)),
            source=str(payload.get("source", DataSource.SYNTHETIC.value)),
            is_biological=bool(payload.get("is_biological", False)),
            metadata=dict(payload.get("metadata") or {}),
        )

    # -------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Provenance, size and region breakdown."""
        region_sizes = {region: int(len(self.nodes_in_region(region))) for region in self.region_names()}
        return {
            "name": self.name,
            "source": self.source,
            "is_biological": self.is_biological,
            "storage": "sparse (scipy CSR)" if self.is_sparse else "dense (numpy)",
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "regions": region_sizes,
            **self.degree_summary(),
        }

    def provenance_warning(self) -> str:
        """Warning string when the graph is not real biological data."""
        if self.is_biological:
            return ""
        if self.source == DataSource.TEST_FIXTURE.value:
            return (
                f"[TEST FIXTURE - NOT REAL BIOLOGICAL DATA] graph '{self.name}'; "
                "its statistics are not anatomy."
            )
        return (
            f"[SIMULATION INPUT] graph '{self.name}' is synthetic (source='{self.source}'); "
            "its activity is a simulation and carries no anatomical claim."
        )


def synthetic_brain_graph(
    num_neurons: int = 64,
    connectivity: float = 0.15,
    regions: Iterable[str] = SYNTHETIC_REGIONS,
    seed: int = 0,
    name: str = "synthetic-brain",
) -> BrainGraph:
    """Build a small random directed graph for development and testing.

    .. warning::
       **Not biological.** This is a uniformly random graph with random weights
       and round-robin region labels. It exists so the simulation has a graph to
       run on at CPU speed. It is not a fruit-fly connectome and must not be
       described as one.

    Every neuron is guaranteed at least one incoming and one outgoing connection
    so that the LIF population actually fires under drive.
    """
    if num_neurons < 2:
        raise BrainGraphError("synthetic_brain_graph needs at least 2 neurons")
    if not 0.0 < connectivity <= 1.0:
        raise BrainGraphError("connectivity must be in (0, 1]")

    region_list = [str(region) for region in regions] or ["unassigned"]
    rng = np.random.default_rng(seed)
    node_ids = [f"n{index:04d}" for index in range(num_neurons)]
    weights = (rng.random((num_neurons, num_neurons)) < connectivity) & ~np.eye(num_neurons, dtype=bool)
    weights = weights.astype(np.float32)
    weights *= rng.uniform(0.1, 1.0, size=weights.shape).astype(np.float32)

    # Guarantee in- and out-degree >= 1 so the population cannot go silent.
    for index in range(num_neurons):
        if not weights[index].any():
            weights[index, (index + 1) % num_neurons] = 0.5
        if not weights[:, index].any():
            weights[(index - 1) % num_neurons, index] = 0.5

    labels = {node: region_list[i % len(region_list)] for i, node in enumerate(node_ids)}
    return BrainGraph(
        node_ids=node_ids,
        weights=weights,
        regions=labels,
        name=name,
        source=DataSource.SYNTHETIC.value,
        is_biological=False,
        metadata={
            "generator": "synthetic_brain_graph",
            "seed": seed,
            "connectivity": connectivity,
            "regions_are_arbitrary": True,
            "note": "Random graph for development only. No fruit-fly anatomy.",
        },
    )


def iter_regions(graph: BrainGraph) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(region, node_ids)`` pairs."""
    for region in graph.region_names():
        yield region, graph.nodes_in_region(region)
