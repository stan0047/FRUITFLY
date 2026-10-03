"""A wiring control that preserves **per-neuron drive**, which the Phase 4C shuffle does not.

Why this module exists
----------------------
Phase 4C compared the measured recurrent graph against
:func:`~flybrain.benchmark.shuffled.shuffled_control`, a degree-preserving
configuration-model randomisation. That control preserves the *multiset*
invariants — 1300 neurons, 5605 edges, per-neuron in- and out-degree, the total
synapse count and the weight multiset — and it is honest about what it does not
preserve: its own metadata lists ``cell-type composition of edges`` under
``not_preserved``.

A read-only diagnostic on the frozen Phase 4C graph put numbers on that. Mean
total incoming drive (synapse count times the declared scale) moved between the
measured graph and the shuffle by:

============  ==========  ==========  ==============
cell type     measured    shuffled     ratio
============  ==========  ==========  ==============
L1            4.67        1.82        0.39
Mi1           19.02       3.78        0.20
Tm1           31.24       4.88        0.16
Tm2           25.53       6.35        0.25
T4a-d         4.61-5.74   8.71-13.74  1.63-2.50
T5a-d         5.48-6.59   14.86-16.95 2.53-2.75
============  ==========  ==========  ==============

with an overall per-neuron incoming-drive correlation of **-0.0055**. The two
arms were not receiving the same drive, and the drive that moved was moving
*into* the readout population. A recurrent-versus-control difference could
therefore never have been attributed to edge *arrangement* alone, and the Phase 4C
null is uninformative about wiring in both directions.

What this control is instead
----------------------------
A **strength-preserving rewiring**. The unit of change is the *target* of an
edge, never its weight:

1. Pick two edges of **identical** weight, ``(u -> v)`` and ``(u' -> v')``.
2. Re-point them at each other's targets: ``(u -> v')`` and ``(u' -> v)``.

Each weight stays attached to the same source, and because the two weights are
equal the exchange is weight-neutral for both targets. So, exactly:

* per-source **out-degree** and **outgoing weight** are preserved;
* per-target **in-degree** and **incoming weight** are preserved;
* every per-edge weight is preserved, so the weight multiset is preserved too;
* no weight is created, destroyed, or moved between sources.

The only thing that changes is *which target a given source drives* — which is
precisely the quantity the wiring question is about.

Why the equal-weight restriction, and what it costs
---------------------------------------------------
An unconstrained 2-switch that also swapped weights would preserve in/out weight
sums only in aggregate, not per neuron: target ``v'`` would receive ``w1``
instead of ``w2``, so its incoming total would change by ``w1 - w2``. Restricting
swaps to a single weight class is what makes the per-neuron invariants exact
rather than approximate.

The cost is the achievable rewiring fraction, and it is measured rather than
assumed. On the frozen Phase 4C subset, 5575 of 5605 edges (99.5%) sit in a
weight class with at least two members, and the weight-1 class alone holds 2211
edges, so the restriction is not binding here. On a graph with mostly distinct
weights it would be, and then
:func:`drive_matched_control` returns a graph whose
``edge_change_fraction`` is below the requested target and the Phase 5A gate
``G4`` declares the control arm **void** rather than reporting a comparison built
on a control that could not move.

This is a *different null from the existing shuffle, not a replacement for it*
and the two answer different questions. The existing control preserves the weight
multiset and destroys the strength profile; this one preserves the strength
profile and re-randomises the wiring. Phase 5A reports both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from flybrain.brain.mcns_circuit import MCNSCircuit, MCNSCircuitError

__all__ = [
    "DRIVE_MATCHED_DISCLAIMER",
    "DRIVE_MATCHED_PRESERVED",
    "DRIVE_MATCHED_NOT_PRESERVED",
    "drive_matched_control",
    "verify_drive_matching",
]

DRIVE_MATCHED_DISCLAIMER = (
    "DRIVE_MATCHED_CONTROL is a computational control, not a biological model. It is a "
    "strength-preserving rewiring of the measured MCNS candidate graph: per-neuron in-degree, "
    "out-degree, incoming weight and outgoing weight are all preserved exactly, and only the "
    "target each source drives is re-randomised. It is present to test whether any result "
    "depends on the measured wiring at all. It is not a model of any biological circuit."
)

#: The invariants this control is defined by. Every one is checked, not asserted.
DRIVE_MATCHED_PRESERVED: tuple[str, ...] = (
    "neuron count and body ids",
    "cell-type labels, soma side and soma location (so the stimulus geometry is identical)",
    "per-neuron in-degree (exact, per neuron, not just as a multiset)",
    "per-neuron out-degree (exact, per neuron, not just as a multiset)",
    "per-neuron incoming synaptic weight (exact) -- the property the Phase 4C shuffle did not preserve",
    "per-neuron outgoing synaptic weight (exact)",
    "every per-edge synapse count, unchanged, attached to the same source",
    "total synapse count and the edge-weight multiset",
)

#: What it explicitly does not preserve. Stated, because a control that preserved
#: everything would be a copy of the measured graph.
DRIVE_MATCHED_NOT_PRESERVED: tuple[str, ...] = (
    "which target a given source drives (this is the intervention)",
    "cell-type composition of edges, and therefore which cell type receives a given amount of drive",
    "soma-side composition of edges",
    "pathway structure, layer ordering, and any cluster or motif content",
)

#: Default requested fraction of edges to re-point. A control that barely moves
#: is not a control, so the request is high and the achieved value is reported.
DEFAULT_TARGET_CHANGE_FRACTION = 0.5


def _weight_classes(data: np.ndarray) -> dict[int, np.ndarray]:
    """Edge indices grouped by synapse count."""
    out: dict[int, np.ndarray] = {}
    for value in np.unique(data):
        out[int(value)] = np.flatnonzero(data == value)
    return out


def drive_matched_control(
    circuit: MCNSCircuit,
    seed: int = 0,
    target_change_fraction: float = DEFAULT_TARGET_CHANGE_FRACTION,
    max_attempts_factor: int = 200,
) -> MCNSCircuit:
    """Build a strength-preserving rewiring of ``circuit``.

    Repeatedly re-points two equal-weight edges at each other's targets, skipping
    any swap that would create a self-loop or a multi-edge, and stopping once
    ``target_change_fraction`` of edges have changed or the attempt budget is
    exhausted.

    The attempt budget is finite on purpose. A graph that cannot reach the
    requested fraction within it is returned with its achieved fraction recorded,
    and the caller decides what to do with that. Silently returning a
    near-copy of the measured graph and calling it a control would be the worst
    possible failure mode here.
    """
    coo = circuit.csr.tocoo()
    n = int(circuit.num_neurons)
    if coo.nnz == 0:
        raise MCNSCircuitError("cannot rewire a circuit with no edges")

    source = coo.row.astype(np.int64, copy=True)
    target = coo.col.astype(np.int64, copy=True)
    weight = coo.data.astype(np.int64, copy=True)
    if not np.array_equal(coo.row, source) or not np.array_equal(coo.col, target):
        raise MCNSCircuitError("edge list is not in COO order; refusing to rewire")

    total_edges = int(source.size)
    fraction = float(np.clip(target_change_fraction, 0.0, 1.0))
    wanted = int(np.ceil(fraction * total_edges))
    classes = _weight_classes(weight)
    present = set((source * np.int64(n) + target).tolist())
    rng = np.random.default_rng(int(seed))

    original_target = target.copy()
    changed = np.zeros(total_edges, dtype=bool)
    budget = int(max_attempts_factor * total_edges)

    for _ in range(budget):
        if int(changed.sum()) >= wanted:
            break
        usable = [idx for idx in classes.values() if idx.size >= 2]
        if not usable:
            break
        cls = usable[int(rng.integers(0, len(usable)))]
        i, j = (int(v) for v in rng.choice(cls, size=2, replace=False))
        u, u2 = int(source[i]), int(source[j])
        v, v2 = int(target[i]), int(target[j])
        if v == v2:
            continue                      # swapping would change nothing
        if u == v2 or u2 == v:
            continue                      # would create a self-loop
        key_a = u * np.int64(n) + v2
        key_b = u2 * np.int64(n) + v
        if key_a in present or key_b in present:
            continue                      # would create a multi-edge
        present.discard(u * np.int64(n) + v)
        present.discard(u2 * np.int64(n) + v2)
        present.add(int(key_a))
        present.add(int(key_b))
        target[i], target[j] = v2, v
        changed[i] = True
        changed[j] = True

    achieved = float(changed.sum() / total_edges)
    matrix = sp.coo_matrix((weight, (source, target)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    if matrix.nnz != total_edges:
        # A multi-edge slipped through. Refuse rather than emit a graph whose
        # invariants do not hold.
        raise MCNSCircuitError(
            f"rewiring produced {matrix.nnz} distinct edges from {total_edges}; "
            "the strength-preserving invariant would be violated"
        )

    return MCNSCircuit(
        body_ids=circuit.body_ids.copy(),
        cell_type=circuit.cell_type.copy(),
        csr=matrix,
        superclass=circuit.superclass.copy(),
        soma_side=circuit.soma_side.copy(),
        soma_location=circuit.soma_location.copy(),
        mode=circuit.mode,
        cell_type_names=list(circuit.cell_type_names),
        metadata={
            **circuit.metadata,
            "condition": "DRIVE_MATCHED_CONTROL",
            "is_measured": False,
            "disclaimer": DRIVE_MATCHED_DISCLAIMER,
            "rewire_seed": int(seed),
            "preserved": list(DRIVE_MATCHED_PRESERVED),
            "not_preserved": list(DRIVE_MATCHED_NOT_PRESERVED),
            "target_change_fraction": fraction,
            "edge_change_fraction": achieved,
            "attempt_budget": budget,
            "n_edges_changed": int(changed.sum()),
            "distinct_from_original": bool(np.array_equal(original_target, target) is False),
        },
    )


def verify_drive_matching(original: MCNSCircuit, matched: MCNSCircuit) -> dict[str, Any]:
    """Check every invariant :func:`drive_matched_control` claims to preserve.

    All of them, exactly, per neuron. The Phase 4C failure was a control that
    looked matched on the aggregate and was not matched per neuron, so the
    per-neuron quantities are the ones compared here and the aggregate totals are
    reported alongside only as context.
    """
    if original.num_neurons != matched.num_neurons:
        return {
            "compatible": False,
            "reason": f"neuron counts differ: {original.num_neurons} vs {matched.num_neurons}",
        }

    a_csc, b_csc = original.csr.tocsc(), matched.csr.tocsc()
    in_degree_a = np.diff(a_csc.indptr)
    in_degree_b = np.diff(b_csc.indptr)
    out_degree_a = np.diff(original.csr.indptr)
    out_degree_b = np.diff(matched.csr.indptr)

    # Per-neuron weight totals. Densely summing a 1300x1300 matrix is fine at this
    # scale and exact; the graph is never densified beyond one sparse product.
    in_weight_a = np.asarray(a_csc.sum(axis=0)).ravel()
    in_weight_b = np.asarray(b_csc.sum(axis=0)).ravel()
    out_weight_a = np.asarray(original.csr.sum(axis=1)).ravel()
    out_weight_b = np.asarray(matched.csr.sum(axis=1)).ravel()

    edges_a = set(zip(*original.csr.nonzero()))
    edges_b = set(zip(*matched.csr.nonzero()))
    overlap = len(edges_a & edges_b)

    return {
        "compatible": True,
        "neurons_equal": original.num_neurons == matched.num_neurons,
        "body_ids_equal": bool(np.array_equal(original.body_ids, matched.body_ids)),
        "cell_types_equal": bool(np.array_equal(original.cell_type, matched.cell_type)),
        "soma_side_equal": bool(np.array_equal(original.soma_side, matched.soma_side)),
        "soma_location_equal": bool(
            np.array_equal(np.asarray(original.soma_location), np.asarray(matched.soma_location))
        ),
        "edges_equal": original.num_edges == matched.num_edges,
        "in_degree_per_neuron_equal": bool(np.array_equal(in_degree_a, in_degree_b)),
        "out_degree_per_neuron_equal": bool(np.array_equal(out_degree_a, out_degree_b)),
        "in_degree_mismatches": int(np.count_nonzero(in_degree_a != in_degree_b)),
        "out_degree_mismatches": int(np.count_nonzero(out_degree_a != out_degree_b)),
        "incoming_weight_per_neuron_equal": bool(np.array_equal(in_weight_a, in_weight_b)),
        "outgoing_weight_per_neuron_equal": bool(np.array_equal(out_weight_a, out_weight_b)),
        "incoming_weight_max_abs_diff": int(np.abs(in_weight_a - in_weight_b).max())
        if in_weight_a.size else 0,
        "outgoing_weight_max_abs_diff": int(np.abs(out_weight_a - out_weight_b).max())
        if out_weight_a.size else 0,
        "edge_weight_multiset_equal": bool(
            np.array_equal(np.sort(original.csr.data), np.sort(matched.csr.data))
        ),
        "total_synapses_equal": original.total_synapse_count == matched.total_synapse_count,
        "shared_edges": overlap,
        "edge_change_fraction": float(1.0 - overlap / original.num_edges)
        if original.num_edges else 0.0,
        "self_loops": int(matched.self_loop_count),
        "topology_changed": bool(edges_a != edges_b),
        "metadata_edge_change_fraction": matched.metadata.get("edge_change_fraction"),
    }
