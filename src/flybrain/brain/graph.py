"""Connectivity graph container and generators.

A :class:`BrainGraph` is the boundary object between *data* and *simulation*.
It carries node identifiers, directed synaptic weights and region labels, and it
knows nothing about membrane potentials, rewards, or policies.

Provenance is preserved end to end: a graph built from a real connectome export
keeps ``source != "synthetic"`` and ``is_biological=True``, while the graph used
by the tests carries ``source="synthetic"`` and can never be mistaken for fly
anatomy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from flybrain.data.connectome import Connectome, DataSource

__all__ = [
    "SYNTHETIC_REGIONS",
    "BrainGraph",
    "BrainGraphError",
    "synthetic_brain_graph",
]

#: Arbitrary labels for the synthetic graph. NOT neuropil names.
SYNTHETIC_REGIONS: tuple[str, ...] = ("visual_system", "central_brain", "descending_pathway")


class BrainGraphError(RuntimeError):
    """Raised for inconsistent or empty graphs."""


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
        node_ids[j]``. Zero means no connection.
    regions:
        Optional node -> region label mapping used for region-level reporting.
    """

    node_ids: list[str]
    weights: np.ndarray
    regions: dict[str, str] = field(default_factory=dict)
    name: str = "graph"
    source: str = DataSource.SYNTHETIC.value
    is_biological: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ validation

    def __post_init__(self) -> None:
        self.node_ids = [str(node) for node in self.node_ids]
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != (len(self.node_ids), len(self.node_ids)):
            raise BrainGraphError(
                f"weights shape {self.weights.shape} does not match {len(self.node_ids)} nodes"
            )
        if len(set(self.node_ids)) != len(self.node_ids):
            raise BrainGraphError("node identifiers must be unique")
        unknown = set(self.regions) - set(self.node_ids)
        if unknown:
            raise BrainGraphError(f"region labels reference unknown nodes: {sorted(unknown)}")

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(np.count_nonzero(self.weights))

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

    def region_matrix(self) -> dict[str, np.ndarray]:
        """Precomputed boolean masks per region, for fast activity aggregation."""
        masks: dict[str, np.ndarray] = {}
        for region in self.region_names():
            mask = np.zeros(self.num_nodes, dtype=bool)
            for node in self.nodes_in_region(region):
                mask[self.node_index(node)] = True
            masks[region] = mask
        return masks

    def in_degree(self, node_id: str) -> int:
        return int(np.count_nonzero(self.weights[:, self.node_index(node_id)]))

    def out_degree(self, node_id: str) -> int:
        return int(np.count_nonzero(self.weights[self.node_index(node_id), :]))

    def degree_summary(self) -> dict[str, float]:
        if self.num_nodes == 0:
            return {"mean_in": 0.0, "mean_out": 0.0, "max_in": 0, "max_out": 0}
        in_degrees = np.count_nonzero(self.weights, axis=0)
        out_degrees = np.count_nonzero(self.weights, axis=1)
        return {
            "mean_in": float(in_degrees.mean()),
            "mean_out": float(out_degrees.mean()),
            "max_in": int(in_degrees.max()),
            "max_out": int(out_degrees.max()),
        }

    def strongest_edges(self, k: int = 10) -> list[tuple[str, str, float]]:
        """The ``k`` heaviest synapses, for inspection and debugging."""
        flat = np.argsort(self.weights, axis=None)[::-1][: max(k, 0)]
        edges: list[tuple[str, str, float]] = []
        for position in flat:
            row, col = np.unravel_index(position, self.weights.shape)
            edges.append((self.node_ids[row], self.node_ids[col], float(self.weights[row, col])))
        return edges

    # --------------------------------------------------------- conversions

    @classmethod
    def from_connectome(cls, connectome: Connectome, name: str | None = None) -> "BrainGraph":
        """Build a graph from a :class:`Connectome`, preserving provenance."""
        node_ids = connectome.neurons
        weights = connectome.weight_matrix(node_order=node_ids)
        return cls(
            node_ids=node_ids,
            weights=weights,
            regions=connectome.regions(),
            name=name or connectome.name,
            source=connectome.source.value,
            is_biological=connectome.is_biological,
            metadata=connectome.provenance(),
        )

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a ``networkx.DiGraph`` for algorithms and plotting."""
        graph = nx.DiGraph()
        for node in self.node_ids:
            graph.add_node(node, region=self.region_of(node))
        rows, cols = np.nonzero(self.weights)
        for row, col in zip(rows, cols, strict=True):
            graph.add_edge(self.node_ids[row], self.node_ids[col], weight=float(self.weights[row, col]))
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
        """Restrict to a subset of nodes, keeping the induced connections."""
        keep = [node for node in node_ids if node in self.index]
        idx = [self.node_index(node) for node in keep]
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

    # ------------------------------------------------------------- (de)serial

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "is_biological": self.is_biological,
            "node_ids": self.node_ids,
            "regions": self.regions,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Persist node order, regions and provenance (weights go to .npz)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".npz"), weights=self.weights)
        target.with_suffix(".json").write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BrainGraph":
        base = Path(path)
        payload = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        weights = np.load(base.with_suffix(".npz"))["weights"]
        return cls(
            node_ids=list(payload["node_ids"]),
            weights=weights,
            regions=dict(payload.get("regions") or {}),
            name=str(payload.get("name", base.stem)),
            source=str(payload.get("source", DataSource.SYNTHETIC.value)),
            is_biological=bool(payload.get("is_biological", False)),
            metadata=dict(payload.get("metadata") or {}),
        )

    # -------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Provenance and size statistics, suitable for logs and figures."""
        return {
            "name": self.name,
            "source": self.source,
            "is_biological": self.is_biological,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "regions": {region: len(self.nodes_in_region(region)) for region in self.region_names()},
            **self.degree_summary(),
        }

    def provenance_warning(self) -> str:
        """Warning string when the graph is not real biological data."""
        if self.is_biological:
            return ""
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
