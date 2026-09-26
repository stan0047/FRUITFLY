"""Sparse representation of an extracted MCNS motion candidate circuit.

This module turns the Phase 3 extraction into a simulatable structure: a sparse
recurrent graph over MCNS bodies, with the biological metadata needed to read
results back to cell types and hemispheres.

Two claims are kept apart, deliberately and everywhere
------------------------------------------------------
``biological_synapse_count``
    The integer synapse count from the MaleCNS dataset. **Never modified,
    normalised, or reinterpreted here.** It is what the dataset measured.

``simulation_coupling``
    A dimensionless coefficient this project uses to drive a leaky
    integrate-and-fire neuron. It is **derived** from the synapse count by an
    explicit, documented modelling choice, and it is not a measurement of
    anything. See :meth:`MCNSCircuit.coupling_matrix`.

The synapse count is a structural count of chemical synapses. It is **not**
synaptic efficacy, not a conductance, and not a strength.

Two connectivity configurations
-------------------------------
``MODE_RECURRENT``
    Every candidate-to-candidate edge in the extraction. This is the measured
    connectivity, unfiltered.

``MODE_FEEDFORWARD``
    Only edges consistent with the two documented chains,
    ``L1 -> Mi1 -> T4a-d`` and ``L2 -> Tm1/Tm2 -> T5a-d``. This is a
    **simplification chosen by this project**, not a claim that the fly computes
    motion feedforward. The recurrent graph contains large feedback components
    that this configuration discards; the number discarded is reported.

Neither mode is "more correct" than the other. One is measured connectivity, the
other is a documented subset of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "CHAIN_EDGES",
    "FEEDFORWARD_CHAINS",
    "MCNSCircuit",
    "MCNSCircuitError",
    "MODE_FEEDFORWARD",
    "MODE_RECURRENT",
    "MODES",
    "candidate_metadata",
    "cell_type_summary",
]

MODE_RECURRENT = "recurrent"
MODE_FEEDFORWARD = "feedforward"
MODES = (MODE_RECURRENT, MODE_FEEDFORWARD)

#: The two documented candidate chains, as ordered (source, target) type pairs.
FEEDFORWARD_CHAINS: dict[str, tuple[tuple[str, str], ...]] = {
    "ON": (("L1", "Mi1"), ("Mi1", "T4a"), ("Mi1", "T4b"), ("Mi1", "T4c"), ("Mi1", "T4d")),
    "OFF": (
        ("L2", "Tm1"),
        ("L2", "Tm2"),
        ("Tm1", "T5a"),
        ("Tm1", "T5b"),
        ("Tm1", "T5c"),
        ("Tm1", "T5d"),
        ("Tm2", "T5a"),
        ("Tm2", "T5b"),
        ("Tm2", "T5c"),
        ("Tm2", "T5d"),
    ),
}
CHAIN_EDGES: frozenset[tuple[str, str]] = frozenset(
    edge for chain in FEEDFORWARD_CHAINS.values() for edge in chain
)

#: Densifying is refused once the dense form would exceed this many bytes.
#: A float64 dense matrix of the real 22,451-body candidate circuit would be
#: 22,451^2 * 8 = 4.03 GB, so the budget below refuses it by a wide margin.
DENSE_REFUSE_BYTES = 64 * 1024**2

#: Convenience node count equivalent to :data:`DENSE_REFUSE_BYTES` at float64.
DENSE_REFUSE_NODES = 2_896

#: LIF is only meaningful for a population; refuse absurd sizes rather than
#: allocating and then failing.
MAX_SIMULABLE_NODES = 200_000


class MCNSCircuitError(RuntimeError):
    """Raised when a circuit cannot be built or used as specified."""


# ------------------------------------------------------------- metadata


def candidate_metadata(annotations_path: Any, body_ids: Sequence[int]) -> pd.DataFrame:
    """Load per-body metadata for the candidate population only.

    Reads just the four needed columns, filters to the requested bodies inside
    Arrow, and only then converts to Python objects, so a large annotation table
    is never fully materialised as Python dicts for a small candidate set.
    """
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    from pathlib import Path

    resolved = Path(annotations_path)
    if not resolved.is_file():
        raise MCNSCircuitError(f"annotation file not found: {resolved}")

    table = feather.read_table(
        resolved, columns=["bodyId", "superclass", "somaSide", "somaLocation"]
    )
    wanted = pa_array(sorted({int(b) for b in body_ids}))
    filtered = table.filter(pc.is_in(table.column("bodyId"), value_set=wanted))
    if filtered.num_rows == 0:
        return pd.DataFrame(columns=["bodyId", "superclass", "somaSide", "soma_x", "soma_y", "soma_z"])

    records = filtered.to_pylist()
    rows: list[dict[str, Any]] = []
    for record in records:
        location = record.get("somaLocation")
        coords = (
            [int(v) for v in location]
            if isinstance(location, (list, tuple)) and len(location) == 3
            else None
        )
        rows.append(
            {
                "bodyId": int(record["bodyId"]),
                "superclass": None if record.get("superclass") is None else str(record["superclass"]),
                "somaSide": None if record.get("somaSide") is None else str(record["somaSide"]),
                "soma_x": coords[0] if coords else None,
                "soma_y": coords[1] if coords else None,
                "soma_z": coords[2] if coords else None,
            }
        )

    frame = pd.DataFrame(rows, columns=["bodyId", "superclass", "somaSide", "soma_x", "soma_y", "soma_z"])
    return frame.sort_values("bodyId").reset_index(drop=True)


def pa_array(values: Sequence[int]) -> Any:
    """int64 Arrow array, built without importing pyarrow at module import time."""
    import pyarrow as pa

    return pa.array(np.asarray(list(values), dtype=np.int64), type=pa.int64())


# --------------------------------------------------------------- circuit


@dataclass
class MCNSCircuit:
    """A sparse directed graph over MCNS bodies, with biological metadata.

    Attributes
    ----------
    body_ids:
        Ascending array of MCNS body ids. Position in this array is the
        simulation index.
    cell_type:
        Per-index cell-type label. Exact MCNS labels; subtypes are never merged.
    csr:
        Sparse adjacency of **biological synapse counts**, not coupling.
    """

    body_ids: np.ndarray
    cell_type: np.ndarray
    csr: sp.csr_matrix
    superclass: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    soma_side: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    soma_location: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    mode: str = MODE_RECURRENT
    cell_type_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.body_ids = np.asarray(self.body_ids, dtype=np.int64)
        self.cell_type = np.asarray(self.cell_type, dtype=object)
        if self.csr.shape != (self.body_ids.size, self.body_ids.size):
            raise MCNSCircuitError(
                f"csr shape {self.csr.shape} does not match {self.body_ids.size} bodies"
            )
        if self.cell_type.size != self.body_ids.size:
            raise MCNSCircuitError("cell_type must have one label per body")
        if self.body_ids.size != np.unique(self.body_ids).size:
            raise MCNSCircuitError("body ids must be unique")
        if np.any(np.diff(self.body_ids) <= 0):
            raise MCNSCircuitError("body_ids must be strictly ascending; index lookup relies on it")
        if self.mode not in MODES:
            raise MCNSCircuitError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not self.cell_type_names:
            seen = sorted({str(t) for t in self.cell_type})
            self.cell_type_names = seen
        self._index = {int(b): i for i, b in enumerate(self.body_ids)}

    # ------------------------------------------------------------ accessors

    @property
    def num_neurons(self) -> int:
        return int(self.body_ids.size)

    @property
    def num_edges(self) -> int:
        """Number of distinct ordered connections (non-zero entries)."""
        return int(self.csr.nnz)

    @property
    def index_of(self) -> dict[int, int]:
        """bodyId -> simulation index."""
        return dict(self._index)

    @property
    def body_of(self) -> dict[int, int]:
        """simulation index -> bodyId."""
        return {i: int(b) for i, b in enumerate(self.body_ids)}

    def to_index(self, body_id: int) -> int:
        """Simulation index for an MCNS body id."""
        try:
            return self._index[int(body_id)]
        except KeyError as exc:
            raise MCNSCircuitError(f"body id {body_id} is not in this circuit") from exc

    def to_body_id(self, index: int) -> int:
        """MCNS body id for a simulation index."""
        position = int(index)
        if not 0 <= position < self.body_ids.size:
            raise MCNSCircuitError(f"simulation index {position} out of range")
        return int(self.body_ids[position])

    def indices_of_type(self, cell_type: str) -> np.ndarray:
        """Simulation indices of one exact cell type."""
        if cell_type not in self.cell_type_names:
            raise KeyError(cell_type)
        return np.flatnonzero(self.cell_type == cell_type)

    def types_of(self, indices: np.ndarray) -> list[str]:
        return [str(self.cell_type[int(i)]) for i in np.asarray(indices, dtype=np.int64)]

    def cell_type_index(self) -> dict[str, np.ndarray]:
        return {name: self.indices_of_type(name) for name in self.cell_type_names}

    @property
    def self_loop_count(self) -> int:
        """Number of self-connections. Measured and retained, never silently dropped."""
        return int(self.csr.diagonal().astype(bool).sum())

    @property
    def total_synapse_count(self) -> int:
        """Sum of biological synapse counts. An exact integer count, not a strength."""
        return int(self.csr.data.sum())

    def has(self, cell_type: str) -> bool:
        return cell_type in self.cell_type_names

    # ---------------------------------------------------------- conversions

    def coupling_matrix(self, synapse_scale: float = 1.0, coupling: str = "linear") -> sp.csr_matrix:
        """Simulation coupling coefficients, derived from the synapse counts.

        **This is a modelling assumption, not a measurement.** The biological
        synapse count says how many chemical synapses connect two bodies. It does
        not say how much current each contributes to a postsynaptic membrane in a
        simplified model. The mapping from one to the other is chosen here.

        Parameters
        ----------
        synapse_scale:
            Overall gain. ``1.0`` means one coupling unit per synapse.
        coupling:
            ``"linear"`` (default) uses ``count * synapse_scale``.
            ``"log1p"`` uses ``log1p(count) * synapse_scale``, which compresses
            the observed ~100:1 dynamic range. Also a modelling choice; it is
            provided because a linear scale makes weak edges negligible next to
            strong ones, which is a conditioning problem, not a biological fact.
        """
        if synapse_scale < 0:
            raise MCNSCircuitError("synapse_scale must be >= 0")
        counts = self.csr
        if coupling == "linear":
            data = counts.data.astype(np.float64) * float(synapse_scale)
        elif coupling == "log1p":
            data = np.log1p(counts.data.astype(np.float64)) * float(synapse_scale)
        else:
            raise MCNSCircuitError(f"coupling must be 'linear' or 'log1p', got {coupling!r}")
        result = sp.csr_matrix((data, counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
        return result

    def propagation_matrix(self, synapse_scale: float = 1.0, coupling: str = "linear") -> sp.csr_matrix:
        """Coupling oriented for the recurrent update: **rows are targets**.

        ``self.csr`` is stored the natural reading way round, ``csr[source, target]``,
        so that in-degree and out-degree and :meth:`edge_type_matrix` all mean what
        they look like they mean. But ``matrix @ spikes`` computes row-by-column, so
        feeding that matrix a spike vector delivers each spike back to the *source's*
        row instead of to the target. That is silent and wrong: nothing crashes, the
        activity is just nonsense.

        This method returns the transposed matrix, so that::

            propagation_matrix(...) @ spikes

        gives, for each neuron, the total drive it receives from the neurons that
        spiked. Transpose once per run, outside the timestep loop.
        """
        return self.coupling_matrix(synapse_scale=synapse_scale, coupling=coupling).transpose().tocsr()

    def to_dense(self) -> np.ndarray:  # pragma: no cover - guard path
        """Refused once the dense form would be unreasonable.

        Present so the refusal is explicit rather than an accidental allocation.
        """
        dense_bytes = self.num_neurons**2 * 8
        if dense_bytes > DENSE_REFUSE_BYTES:
            raise MCNSCircuitError(
                f"refusing to densify a {self.num_neurons}x{self.num_neurons} matrix: the dense form "
                f"would need {dense_bytes / 1024**3:.2f} GB, over the "
                f"{DENSE_REFUSE_BYTES / 1024**2:.0f} MB budget. This circuit is sparse by nature "
                f"({self.num_edges:,} non-zeros); use the CSR matrix and `coupling_matrix()`."
            )
        return np.asarray(self.csr.todense())

    def sub_matrix(self, indices: Sequence[int]) -> sp.csr_matrix:
        """Sparse sub-matrix over a subset of simulation indices."""
        idx = np.asarray(sorted({int(i) for i in indices}), dtype=np.int64)
        return sp.csr_matrix(self.csr[np.ix_(idx, idx)])

    # ------------------------------------------------------------- builders

    @classmethod
    def from_extraction(
        cls,
        populations: Any,
        extraction: Any,
        annotations: Mapping[str, np.ndarray] | None = None,
        mode: str = MODE_RECURRENT,
    ) -> "MCNSCircuit":
        """Build a circuit from a Phase 3 extraction result.

        ``mode`` selects the connectivity configuration. Switching modes keeps
        every body in the population, so the node set and index mapping are
        identical between modes and results are directly comparable.
        """
        if mode not in MODES:
            raise MCNSCircuitError(f"mode must be one of {MODES}, got {mode!r}")

        n = populations.num_bodies
        pre = np.asarray(extraction.pre_index, dtype=np.int64)
        post = np.asarray(extraction.post_index, dtype=np.int64)
        # int64 so biological synapse counts stay exact integers in the data
        # layer. The float conversion happens later, in coupling_matrix().
        counts = np.asarray(extraction.synapse_count, dtype=np.int64)

        kept = np.ones(pre.size, dtype=bool)
        removed_by_reason: dict[str, int] = {}
        if mode == MODE_FEEDFORWARD:
            type_names = np.asarray(populations.cell_types, dtype=object)
            source_type = type_names[populations.type_of[pre]]
            target_type = type_names[populations.type_of[post]]
            allowed = np.zeros(pre.size, dtype=bool)
            for chain, edges in FEEDFORWARD_CHAINS.items():
                for source_label, target_label in edges:
                    allowed |= (source_type == source_label) & (target_type == target_label)
            removed_by_reason = {
                "not_on_a_documented_chain": int((~allowed).sum()),
            }
            kept = allowed

        rows, cols, values = pre[kept], post[kept], counts[kept]
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        matrix.sum_duplicates()

        superclass = np.full(n, "", dtype=object)
        soma_side = np.full(n, "", dtype=object)
        soma_location = np.full((n, 3), -1, dtype=np.int64)
        if annotations is not None:
            # get_indexer handles an unsorted table and returns -1 for bodies
            # that are absent, rather than silently clipping to a neighbour.
            body_index = pd.Index(np.asarray(annotations["bodyId"], dtype=np.int64))
            positions = body_index.get_indexer(populations.body_ids)
            present = positions >= 0
            if annotations.get("superclass") is not None:
                values = pd.Series(np.asarray(annotations["superclass"], dtype=object))
                superclass[:] = ""
                superclass[present] = np.asarray(
                    values.iloc[positions[present]].fillna("").to_numpy(), dtype=object
                )
            if annotations.get("somaSide") is not None:
                values = pd.Series(np.asarray(annotations["somaSide"], dtype=object))
                soma_side[:] = ""
                soma_side[present] = np.asarray(
                    values.iloc[positions[present]].fillna("").to_numpy(), dtype=object
                )

        return cls(
            body_ids=np.asarray(populations.body_ids, dtype=np.int64),
            cell_type=np.asarray(populations.cell_types, dtype=object)[populations.type_of],
            csr=matrix,
            superclass=superclass,
            soma_side=soma_side,
            soma_location=soma_location,
            mode=mode,
            metadata={
                "extraction_direction": getattr(extraction, "direction", "both"),
                "extraction_edges_in": int(extraction.num_edges),
                "edges_removed_for_mode": removed_by_reason,
                "body_budget_per_type": getattr(populations, "body_budget_per_type", 0),
            },
        )

    def with_soma_locations(self, frame: pd.DataFrame) -> "MCNSCircuit":
        """Return a copy carrying soma coordinates aligned to the body order.

        Bodies missing from ``frame`` keep ``-1`` in all three axes, so "no
        coordinate" is never confused with a real position at the origin.
        """
        if frame.empty or "bodyId" not in frame.columns:
            return self
        lookup = frame.drop_duplicates("bodyId").set_index("bodyId")
        aligned = lookup.reindex(self.body_ids)
        coords = np.full((self.num_neurons, 3), -1, dtype=np.int64)
        for axis, column in enumerate(("soma_x", "soma_y", "soma_z")):
            if column in aligned.columns:
                values = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
                coords[:, axis] = np.where(np.isnan(values), -1, values)
        self.soma_location = coords
        return self

    # ------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        """Structural summary. Every number here is measured, not modelled."""
        in_degree_all = np.diff(self.csr.tocsc().indptr)
        by_type: dict[str, dict[str, Any]] = {}
        for name in self.cell_type_names:
            idx = self.indices_of_type(name)
            sides = pd.Series(self.soma_side[idx]).replace("", pd.NA).dropna()
            by_type[name] = {
                "neurons": int(idx.size),
                "outgoing_edges": int(np.diff(self.csr.indptr)[idx].sum()),
                "incoming_edges": int(in_degree_all[idx].sum()),
                "somaSide_breakdown": {str(k): int(v) for k, v in sides.value_counts().items()},
                "with_soma_location": int((self.soma_location[idx, 0] >= 0).sum()),
            }
        return {
            "mode": self.mode,
            "neurons": self.num_neurons,
            "edges": self.num_edges,
            "self_loops": self.self_loop_count,
            "total_biological_synapse_count": self.total_synapse_count,
            "storage": "scipy CSR (sparse); dense construction refused",
            "cell_types": by_type,
            **self.metadata,
        }

    def edge_type_matrix(self) -> pd.DataFrame:
        """Total biological synapse count for every ordered cell-type pair."""
        names = self.cell_type_names
        position = {name: i for i, name in enumerate(names)}
        totals = np.zeros((len(names), len(names)), dtype=np.int64)
        coo = self.csr.tocoo()
        if coo.nnz:
            rows = np.fromiter(
                (position[str(t)] for t in self.cell_type[coo.row]), dtype=np.int64, count=coo.nnz
            )
            cols = np.fromiter(
                (position[str(t)] for t in self.cell_type[coo.col]), dtype=np.int64, count=coo.nnz
            )
            np.add.at(totals, (rows, cols), coo.data.astype(np.int64))
        frame = pd.DataFrame(totals, index=names, columns=names)
        frame.index.name = "from_type"
        frame.columns.name = "to_type"
        return frame


def cell_type_summary(circuit: MCNSCircuit, spikes: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """Per cell type: total spikes, firing rate, and mean membrane if present.

    ``spikes`` has shape ``(n_steps, num_neurons)`` and is boolean.
    """
    if spikes.size == 0:
        return pd.DataFrame(columns=["neurons", "spikes", "spike_rate_hz", "fraction_of_population"])
    counts = np.asarray(spikes, dtype=np.int64).sum(axis=0)
    duration = float(times[-1] - times[0]) if times.size > 1 else 0.0
    rows = []
    for name in circuit.cell_type_names:
        idx = circuit.indices_of_type(name)
        rows.append(
            {
                "cell_type": name,
                "neurons": int(idx.size),
                "spikes": int(counts[idx].sum()),
                "spike_rate_hz": float(counts[idx].sum() / duration) if duration > 0 else 0.0,
                "fraction_of_population": float(idx.size / circuit.num_neurons) if circuit.num_neurons else 0.0,
            }
        )
    frame = pd.DataFrame(rows).set_index("cell_type")
    frame["fraction_of_spikes"] = frame["spikes"] / max(frame["spikes"].sum(), 1)
    return frame
