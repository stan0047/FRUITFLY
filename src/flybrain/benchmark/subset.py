"""A bounded, deterministic, pathway-preserving sub-population of the MCNS circuit.

Why this exists
---------------
The full extracted candidate circuit is 22,451 bodies and 340,310 ordered edges.
Simulating it for every stimulus, trial, condition and coupling gain is a
whole-brain computation, and it is not what the Phase 4B question needs. The
question is whether the *wiring* transforms a controlled input into a
distinguishable representation. That can be answered on an explicitly bounded
population, provided the bound is deterministic, reported, and does not destroy
the pathway being studied.

Why a naive per-type sample does not work
-----------------------------------------
The obvious implementation — take the lowest-numbered 100 bodies of each of the
13 cell types — produces a population in which the documented chains are broken.
A T4a body picked by id has essentially no reason to be downstream of the Mi1
bodies that were also picked by id. Measured on the real data, that version of
this module selected 1,820 bodies and produced a ``T4+T5`` readout that was
**identically zero for every stimulus and every trial**: nothing reached
threshold, so the population carried no information at all, and a linear decoder
on the resulting all-zero features still reported accuracy 1.000. A bound that
silently deletes the pathway does not bound the experiment, it invalidates it.

The rule here instead
---------------------
Selection follows the pathway, in three deterministic phases, under one hard
per-type node budget.

1. **Input seeds.** The two documented input types, ``L1`` and ``L2``, take the
   lowest-numbered ``ceil(budget/2)`` body ids per hemisphere, using the existing
   already-tested rule in
   :func:`flybrain.data.motion_candidate._stratified_sample`.

2. **Downstream closure along the documented chains.** Breadth-first, level by
   level, over the *measured* edges that lie on ``L1 -> Mi1 -> T4a-d`` and
   ``L2 -> Tm1/Tm2 -> T5a-d``. At each level every cell type that has room in its
   budget is admitted at once, hemisphere-balanced, lowest body id first. Because
   the traversal follows the chains, the admitted Mi1 bodies are the ones the
   selected L1 bodies actually connect to, and the admitted T4a-d bodies are the
   ones the selected Mi1 bodies actually connect to.

3. **Top-up.** Any type the closure could not fill to its budget — because the
   measured graph has no further chain edges into it — is topped up from the
   remaining bodies by the same hemisphere-balanced lowest-id rule. These bodies
   may have no in-subset input. The count is reported per type as
   ``topped_up`` so the size of that caveat is never hidden.

Following the chains means the *same* selected population is pathway-connected
in both ``MCNS_FEEDFORWARD`` and ``MCNS_RECURRENT``; the recurrent condition
simply also keeps every other measured edge among those same bodies.

What is guaranteed
------------------
**Deterministic.** The same inputs always select the same bodies. No RNG is used
anywhere in selection. The only ordering is ascending body id, which is
meaningless with respect to how the source file happens to be laid out.

**Reported, never silent.** :meth:`BoundedPopulation.describe` states the total
selected against the total available, the rule in full, the per-type budget use,
how many selected bodies have an in-subset chain input, the top-up count, and the
number of measured edges the bound discards. A caller using this module cannot
produce a result that reads as a whole-brain result.

**Identical across conditions.** The selection depends only on the measured
connectivity, the annotations and the budget — never on a circuit mode, a
coupling gain, a stimulus or an RNG seed. :func:`restrict_circuit` is applied to
the feedforward and recurrent circuits alike.

**Sparse.** Nothing here allocates a dense adjacency. The full edge list is
scanned once to build offset-sorted adjacency lists, O(edges) time and memory.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from flybrain.brain.mcns_circuit import FEEDFORWARD_CHAINS, MCNSCircuit, MCNSCircuitError
from flybrain.data.motion_candidate import (
    MOTION_CELL_TYPES,
    MOTION_PATHWAYS,
    CandidatePopulations,
    _stratified_sample,
)

__all__ = [
    "SUBSET_DISCLAIMER",
    "BoundedPopulation",
    "bounded_population",
    "chain_edge_mask",
    "restrict_circuit",
]

SUBSET_DISCLAIMER = (
    "SUBSET MODE. The simulated population is a deterministic, pathway-following subset of the "
    "22,451-body MCNS candidate circuit, not the whole circuit. Neuron identities, connectivity "
    "and integer synapse counts within the subset are measured and unmodified; the act of selecting "
    "the subset is an assumption and is reported in full."
)

_SELECTION_RULE = (
    "One hard node budget per exact cell type (`max_bodies_per_type`). "
    "Phase 1, input seeds: the documented input types L1 and L2 take the lowest-numbered "
    "ceil(budget/2) body ids per hemisphere (somaSide L and R), then any remainder from the "
    "lowest-numbered remaining bodies. This is the existing hemisphere-balanced rule from "
    "flybrain.data.motion_candidate._stratified_sample. "
    "Phase 2, downstream closure: breadth-first level by level over the MEASURED edges that lie on "
    "the documented chains L1->Mi1->T4a-d and L2->Tm1/Tm2->T5a-d. At each level every cell type "
    "with remaining budget is admitted at once, taking the lowest-numbered ceil(remaining/2) body "
    "ids per hemisphere first. Following the chains means the admitted Mi1 bodies are the ones the "
    "selected L1 bodies actually connect to, and the admitted T4a-d bodies are the ones the "
    "selected Mi1 bodies actually connect to. "
    "Phase 3, top-up: any type the closure cannot fill to its budget is topped up from the "
    "remaining bodies by the same hemisphere-balanced lowest-id rule, and the count is reported as "
    "`topped_up`. "
    "No RNG is used. The selection is then applied identically to every circuit condition."
)

#: The two documented input types, taken from the pathway definition rather than
#: hard-coded here, so this module cannot drift from
#: :data:`flybrain.data.motion_candidate.MOTION_PATHWAYS`.
INPUT_TYPES: tuple[str, ...] = tuple(
    sorted({MOTION_PATHWAYS["ON"]["input"], MOTION_PATHWAYS["OFF"]["input"]})
)

#: Pairs whose measured edges the downstream closure is allowed to follow.
CHAIN_PAIRS: frozenset[tuple[str, str]] = frozenset(
    edge for chain in FEEDFORWARD_CHAINS.values() for edge in chain
)


@dataclass
class BoundedPopulation:
    """The outcome of :func:`bounded_population`, with its full audit trail."""

    #: Positions in the source circuit, ascending by body id.
    positions: np.ndarray
    #: Per exact cell type: seeds, closure admissions, and top-ups.
    seeds_per_type: dict[str, int]
    closure_per_type: dict[str, int]
    topped_up_per_type: dict[str, int]
    #: Per exact cell type: selected bodies holding at least one in-subset chain input.
    driven_per_type: dict[str, int]
    total_available: int
    total_available_edges: int
    edges_kept: int
    edges_omitted: int
    synapses_kept: int
    synapses_total: int
    hemisphere_balance: dict[str, dict[str, int]] = field(default_factory=dict)
    chain_connectivity: dict[str, Any] = field(default_factory=dict)
    budget_per_type: int = 0

    @property
    def is_subset(self) -> bool:
        return int(self.positions.size) < int(self.total_available)

    @property
    def num_neurons(self) -> int:
        return int(self.positions.size)

    def selected_per_type(self) -> dict[str, int]:
        names = set(self.seeds_per_type) | set(self.closure_per_type) | set(self.topped_up_per_type)
        return {
            name: int(
                self.seeds_per_type.get(name, 0)
                + self.closure_per_type.get(name, 0)
                + self.topped_up_per_type.get(name, 0)
            )
            for name in sorted(names)
        }

    def headline(self) -> str:
        """The one line that must appear in any report built on this selection.

        A full run says so in its own words, so a report can never be read as
        though a bound had been applied when none was.
        """
        if not self.is_subset:
            return (
                f"FULL POPULATION: all {self.total_available:,} annotated candidate bodies, "
                f"{self.total_available_edges:,} measured edges, nothing omitted"
            )
        return (
            f"SUBSET: {self.num_neurons:,} of {self.total_available:,} neurons "
            f"({self.edges_kept:,} of {self.total_available_edges:,} measured edges, "
            f"{self.edges_omitted:,} omitted by the bound)"
        )

    def describe(self) -> dict[str, Any]:
        selected = self.selected_per_type()
        return {
            "is_subset": self.is_subset,
            "selected_neurons": self.num_neurons,
            "available_neurons": int(self.total_available),
            "headline": self.headline(),
            "disclaimer": SUBSET_DISCLAIMER,
            "selection_rule": _SELECTION_RULE,
            "input_types": list(INPUT_TYPES),
            "chain_pairs_followed": [f"{a}->{b}" for a, b in sorted(CHAIN_PAIRS)],
            "node_budget_per_type": int(self.budget_per_type),
            "selected_per_type": selected,
            "input_seeds_per_type": dict(self.seeds_per_type),
            "closure_admitted_per_type": dict(self.closure_per_type),
            "topped_up_per_type": dict(self.topped_up_per_type),
            "with_in_subset_chain_input_per_type": dict(self.driven_per_type),
            "hemisphere_balance": dict(self.hemisphere_balance),
            "edges_kept": int(self.edges_kept),
            "edges_omitted": int(self.edges_omitted),
            "measured_edges_available": int(self.total_available_edges),
            "biological_synapses_kept": int(self.synapses_kept),
            "biological_synapses_total": int(self.synapses_total),
            "chain_connectivity": dict(self.chain_connectivity),
            "deterministic": True,
            "randomness_used_in_selection": False,
            "caveat": (
                "A `topped_up` body was added because the measured chain graph could not supply it, "
                "so it may have no in-subset input. Check `with_in_subset_chain_input_per_type` "
                "before reading a readout group as driven."
            ),
        }


def chain_edge_mask(body_types: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Boolean mask over edges: does this measured edge lie on a documented chain?

    ``body_types`` is one label per *body*, so it is indexed by the edge endpoints.
    """
    source = body_types[pre]
    target = body_types[post]
    mask = np.zeros(pre.size, dtype=bool)
    for source_label, target_label in CHAIN_PAIRS:
        mask |= (source == source_label) & (target == target_label)
    return mask


def _downstream_adjacency(num_nodes: int, pre: np.ndarray, post: np.ndarray) -> list[list[int]]:
    """Outgoing neighbour lists, in ascending target order. O(edges), never dense."""
    buckets: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for source, target in zip(pre.tolist(), post.tolist()):
        buckets[source].append(target)
    return [sorted(bucket) for bucket in buckets]


def _hemisphere_balanced_take(
    ids: np.ndarray, sides: np.ndarray, count: int
) -> np.ndarray:
    """Lowest-numbered ``count`` ids, ``ceil(count/2)`` per side before any remainder.

    Mirrors :func:`flybrain.data.motion_candidate._stratified_sample` so the two
    halves of the selection follow the same rule.
    """
    if count <= 0 or ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    if ids.size <= count:
        return np.sort(ids)
    per_side = int(np.ceil(count / 2))
    picked: list[np.ndarray] = []
    remaining = int(count)
    for pool in (ids[sides == "L"], ids[sides == "R"]):
        take = min(per_side, pool.size, remaining)
        if take > 0:
            picked.append(np.sort(pool)[:take])
            remaining -= take
    if remaining > 0:
        taken = np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)
        leftovers = np.setdiff1d(ids, taken, assume_unique=True)
        if leftovers.size:
            picked.append(np.sort(leftovers)[:remaining])
    return np.sort(np.unique(np.concatenate(picked))) if picked else np.zeros(0, dtype=np.int64)


def bounded_population(
    populations: CandidatePopulations,
    pre_index: np.ndarray,
    post_index: np.ndarray,
    synapse_count: np.ndarray | None = None,
    bodies_per_type: int = 100,
    cell_types: tuple[str, ...] = MOTION_CELL_TYPES,
) -> BoundedPopulation:
    """Select a deterministic, hemisphere-balanced, pathway-preserving sub-population.

    Parameters
    ----------
    synapse_count:
        Optional measured integer synapse count per edge, used only for
        reporting how much measured synaptic weight the bound retains. It is
        never modified.
    bodies_per_type:
        Hard node budget per exact cell type. ``0`` selects every available body,
        which makes this a no-op and is reported as a full-population run.
    cell_types:
        The exact cell types to cover. Types absent from the annotation table are
        skipped and reported as absent by absence from the per-type counts.
    """
    n = int(populations.num_bodies)
    pre = np.asarray(pre_index, dtype=np.int64)
    post = np.asarray(post_index, dtype=np.int64)
    if pre.size != post.size:
        raise MCNSCircuitError("pre_index and post_index must be the same length")
    if pre.size and (int(pre.max()) >= n or int(post.max()) >= n):
        raise MCNSCircuitError("edge indices fall outside the candidate population")
    if synapse_count is not None and np.asarray(synapse_count).size != pre.size:
        raise MCNSCircuitError("synapse_count must have one entry per edge")

    type_labels = np.asarray(populations.cell_types, dtype=object)
    type_of = np.asarray(populations.type_of, dtype=np.int64)
    body_types = type_labels[type_of]
    sides = np.asarray(populations.soma_side, dtype=object)
    body_ids = np.asarray(populations.body_ids, dtype=np.int64)
    available = {
        name: np.flatnonzero(type_of == populations.cell_types.index(name))
        for name in cell_types
        if name in populations.cell_types
    }

    if int(bodies_per_type) <= 0:
        return _full_population(populations, pre, post, synapse_count, body_types, sides)

    budget = int(bodies_per_type)
    selected = np.zeros(n, dtype=bool)
    chosen_ids: dict[str, set[int]] = defaultdict(set)
    seeds_per_type: dict[str, int] = {}
    closure_per_type: dict[str, int] = {}

    def _record(name: str, ids: np.ndarray) -> None:
        for value in ids.tolist():
            chosen_ids[name].add(int(value))

    # ---- phase 1: input seeds
    for name in INPUT_TYPES:
        positions = available.get(name)
        if positions is None or positions.size == 0:
            continue
        picked = _stratified_sample(body_ids[positions], sides[positions], budget)
        if picked.size:
            positions_picked = np.searchsorted(body_ids, picked)
            selected[positions_picked] = True
            seeds_per_type[name] = int(picked.size)
            _record(name, positions_picked)

    # ---- phase 2: downstream closure along the documented chains
    on_chain = chain_edge_mask(body_types, pre, post)
    downstream = _downstream_adjacency(n, pre[on_chain], post[on_chain])
    frontier = deque(int(i) for i in np.flatnonzero(selected))
    while frontier:
        candidates: dict[str, list[int]] = defaultdict(list)
        while frontier:
            node = frontier.popleft()
            for target in downstream[node]:
                if not selected[target]:
                    candidates[str(type_labels[int(type_of[target])])].append(int(target))
        if not candidates:
            break
        for name in sorted(candidates):
            taken_so_far = len(chosen_ids.get(name, ()))
            room = budget - taken_so_far
            if room <= 0:
                continue
            pool = np.unique(np.asarray(candidates[name], dtype=np.int64))
            pool = pool[~selected[pool]]
            if pool.size == 0:
                continue
            picked = _hemisphere_balanced_take(body_ids[pool], sides[pool], room)
            if picked.size == 0:
                continue
            positions_picked = np.searchsorted(body_ids, picked)
            selected[positions_picked] = True
            closure_per_type[name] = closure_per_type.get(name, 0) + int(picked.size)
            _record(name, positions_picked)
            frontier.extend(int(i) for i in positions_picked)

    # ---- phase 3: top up any type the closure could not fill
    topped_up_per_type: dict[str, int] = {}
    for name, positions in available.items():
        taken = len(chosen_ids.get(name, ()))
        room = budget - taken
        if room <= 0:
            continue
        remaining = positions[~selected[positions]]
        if remaining.size == 0:
            continue
        picked = _hemisphere_balanced_take(body_ids[remaining], sides[remaining], room)
        if picked.size == 0:
            continue
        picked_positions = np.searchsorted(body_ids, picked)
        selected[picked_positions] = True
        topped_up_per_type[name] = int(picked_positions.size)
        _record(name, picked_positions)

    positions = np.flatnonzero(selected)
    kept = selected[pre] & selected[post]
    weights = _weights(pre, synapse_count)
    kept_chain = kept & on_chain

    return BoundedPopulation(
        positions=positions,
        seeds_per_type=seeds_per_type,
        closure_per_type=closure_per_type,
        topped_up_per_type=topped_up_per_type,
        driven_per_type=_driven_per_type(body_types, pre, post, on_chain, selected),
        total_available=n,
        total_available_edges=int(pre.size),
        edges_kept=int(kept.sum()),
        edges_omitted=int(pre.size) - int(kept.sum()),
        synapses_kept=int(weights[kept].sum()),
        synapses_total=int(weights.sum()),
        hemisphere_balance=_hemisphere_balance(sides[positions], type_labels[type_of[positions]]),
        chain_connectivity=_chain_survival(
            body_types, pre, post, weights, kept, kept_chain, on_chain
        ),
        budget_per_type=budget,
    )


def _weights(pre: np.ndarray, synapse_count: np.ndarray | None) -> np.ndarray:
    """Measured synapse count per edge, or edge multiplicity when not supplied."""
    if synapse_count is None:
        return np.ones(int(pre.size), dtype=np.int64)
    return np.asarray(synapse_count, dtype=np.int64)


def _driven_per_type(
    body_types: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    on_chain: np.ndarray,
    selected: np.ndarray,
) -> dict[str, int]:
    """Per cell type, how many selected bodies hold an in-subset chain input.

    This is the number that says whether a readout group is actually driven by
    the bounded population, or padded with bodies that nothing connects to.
    """
    kept = on_chain & selected[pre] & selected[post]
    if not np.any(kept):
        return {}
    counts: dict[str, int] = defaultdict(int)
    for target in np.unique(post[kept]):
        counts[str(body_types[int(target)])] += 1
    return dict(counts)


def _chain_survival(
    body_types: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    weights: np.ndarray,
    kept: np.ndarray,
    kept_chain: np.ndarray,
    on_chain: np.ndarray,
) -> dict[str, Any]:
    """Measured synapses surviving on each documented chain, inside the selection."""
    source_type = body_types[pre[kept]]
    target_type = body_types[post[kept]]
    kept_weights = weights[kept]
    out: dict[str, Any] = {}
    for channel, pairs in FEEDFORWARD_CHAINS.items():
        per_pair: dict[str, dict[str, int]] = {}
        for source_label, target_label in pairs:
            mask = (source_type == source_label) & (target_type == target_label)
            key = f"{source_label}->{target_label}"
            per_pair[key] = {
                "measured_synapses": int(kept_weights[mask].sum()),
                "edges": int(np.count_nonzero(mask)),
            }
        present = [k for k, v in per_pair.items() if v["edges"] > 0]
        out[channel] = {
            "per_pair": per_pair,
            "pairs_present": len(present),
            "pairs_expected": len(pairs),
            "fully_connected": len(present) == len(pairs),
        }
    out["_totals"] = {
        "chain_edges_in_full_graph": int(on_chain.sum()),
        "chain_edges_kept": int(kept_chain.sum()),
        "all_edges_kept": int(kept.sum()),
    }
    return out


def _full_population(
    populations: CandidatePopulations,
    pre: np.ndarray,
    post: np.ndarray,
    synapse_count: np.ndarray | None,
    body_types: np.ndarray,
    sides: np.ndarray,
) -> BoundedPopulation:
    n = int(populations.num_bodies)
    type_of = np.asarray(populations.type_of, dtype=np.int64)
    per_type = {str(name): int((type_of == i).sum()) for i, name in enumerate(populations.cell_types)}
    weights = _weights(pre, synapse_count)
    selected = np.ones(n, dtype=bool)
    every_edge = np.ones(pre.size, dtype=bool)
    on_chain = chain_edge_mask(body_types, pre, post)
    return BoundedPopulation(
        positions=np.arange(n, dtype=np.int64),
        seeds_per_type=per_type,
        closure_per_type={name: 0 for name in per_type},
        topped_up_per_type={name: 0 for name in per_type},
        driven_per_type=_driven_per_type(body_types, pre, post, on_chain, selected),
        total_available=n,
        total_available_edges=int(pre.size),
        edges_kept=int(pre.size),
        edges_omitted=0,
        synapses_kept=int(weights.sum()),
        synapses_total=int(weights.sum()),
        hemisphere_balance=_hemisphere_balance(sides, body_types),
        chain_connectivity=_chain_survival(
            body_types, pre, post, weights, every_edge, on_chain, on_chain
        ),
        budget_per_type=0,
    )


def _hemisphere_balance(sides: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, int]]:
    """Per cell type, how the selection splits across hemispheres."""
    out: dict[str, dict[str, int]] = {}
    side_values = np.asarray(["" if s is None else str(s) for s in sides], dtype=object)
    for name in sorted(set(map(str, labels.tolist()))):
        mask = np.asarray([str(v) == name for v in labels], dtype=bool)
        picked = side_values[mask]
        out[name] = {
            "L": int(np.count_nonzero(picked == "L")),
            "R": int(np.count_nonzero(picked == "R")),
            "unlabelled": int(np.count_nonzero((picked != "L") & (picked != "R"))),
        }
    return out


def restrict_circuit(circuit: MCNSCircuit, positions: np.ndarray) -> MCNSCircuit:
    """Return the sub-circuit over ``positions``, with metadata re-aligned.

    ``positions`` are indices into ``circuit``. The result keeps only measured
    edges with **both** endpoints inside the selection, keeps exact cell-type
    labels, and keeps ``body_ids`` strictly ascending so index lookup still
    works. No weight, count or label is altered.
    """
    idx = np.asarray(positions, dtype=np.int64)
    if idx.size == 0:
        raise MCNSCircuitError("cannot build a circuit from an empty selection")
    if np.unique(idx).size != idx.size:
        raise MCNSCircuitError("selection positions must be unique")
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= circuit.num_neurons):
        raise MCNSCircuitError("selection positions fall outside the source circuit")

    # Ascending body id, which is the order the rest of the project relies on.
    order = np.argsort(circuit.body_ids[idx], kind="stable")
    idx = idx[order]

    sub = sp.csr_matrix(circuit.csr[idx][:, idx])
    sub.sum_duplicates()
    return MCNSCircuit(
        body_ids=circuit.body_ids[idx],
        cell_type=circuit.cell_type[idx],
        csr=sub,
        superclass=circuit.superclass[idx] if circuit.superclass.size else circuit.superclass,
        soma_side=circuit.soma_side[idx] if circuit.soma_side.size else circuit.soma_side,
        soma_location=(
            circuit.soma_location[idx] if circuit.soma_location.size else circuit.soma_location
        ),
        mode=circuit.mode,
        cell_type_names=list(circuit.cell_type_names),
        metadata={**circuit.metadata, "restricted_from_neurons": int(circuit.num_neurons)},
    )
