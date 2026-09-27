"""Degree-preserving randomised connectivity, as a computational control.

This is **not** a biological model and not a claim about any real circuit. It is
a null: same number of neurons, same number of edges, same in- and out-degree
sequences, same synapse-count distribution, different wiring.

Why not just permute the targets
--------------------------------
Permuting post-synaptic indices preserves out-degrees but scrambles the
in-degree sequence, so the control would differ from the real graph in a way that
has nothing to do with topology. Instead this matches *both* degree sequences
exactly, using a stub-matching (configuration-model) construction followed by
repair of self-loops and multi-edges. The degree distribution is then preserved
exactly rather than approximately, which is strictly stronger than required and
removes degree as a confound.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from flybrain.brain.mcns_circuit import MCNSCircuit, MCNSCircuitError

__all__ = ["SHUFFLED_DISCLAIMER", "shuffled_control", "verify_degree_preservation"]

SHUFFLED_DISCLAIMER = (
    "SHUFFLED_CONTROL is a computational control, not a biological model. It is a degree-preserving "
    "randomisation of the MCNS candidate graph, present to test whether any measured separability "
    "depends on the real wiring at all. It is not a model of any biological circuit."
)


def shuffled_control(
    circuit: MCNSCircuit,
    seed: int = 0,
    max_repair_passes: int = 64,
) -> MCNSCircuit:
    """Build a degree-preserving randomisation of ``circuit``.

    Preserved exactly: neuron count, edge count, every neuron's in-degree and
    out-degree, and the multiset of synapse counts. Not preserved: which neuron
    connects to which.

    Deterministic: the same ``(circuit, seed)`` always yields the same graph.
    """
    coo = circuit.csr.tocoo()
    n = circuit.num_neurons
    if coo.nnz == 0:
        raise MCNSCircuitError("cannot shuffle a circuit with no edges")

    out_degree = np.bincount(coo.row, minlength=n).astype(np.int64)
    in_degree = np.bincount(coo.col, minlength=n).astype(np.int64)

    rng = np.random.default_rng(seed)

    # Stub matching. Source stubs stay in ascending order so out-degrees are
    # preserved by construction; destination stubs are shuffled so the in-degree
    # multiset is redistributed at random.
    source = np.repeat(np.arange(n, dtype=np.int64), out_degree)
    target = np.repeat(np.arange(n, dtype=np.int64), in_degree)
    rng.shuffle(target)

    source, target = _repair(source, target, n, rng, max_repair_passes)

    # Synapse counts keep their multiset but are re-assigned independently, so
    # count is no longer correlated with the (now random) topology.
    counts = rng.permutation(coo.data.astype(np.int64))

    matrix = sp.coo_matrix((counts, (source, target)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()

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
            "condition": "SHUFFLED_CONTROL",
            "is_measured": False,
            "disclaimer": SHUFFLED_DISCLAIMER,
            "shuffle_seed": int(seed),
            "preserved": [
                "neuron count",
                "edge count",
                "in-degree sequence (exact)",
                "out-degree sequence (exact)",
                "synapse-count multiset (exact)",
            ],
            "not_preserved": [
                "which neuron connects to which",
                "cell-type composition of edges",
                "soma-side composition of edges",
                "topology, clustering, and pathway structure",
            ],
        },
    )


def _repair(
    source: np.ndarray,
    target: np.ndarray,
    n: int,
    rng: np.random.Generator,
    max_passes: int,
    attempts_per_offender: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove self-loops and multi-edges while holding both degree sequences fixed.

    The only repair operation is ``target[i] <-> target[j]``. That permutes the
    target multiset, so every neuron's in-degree is unchanged, and it never
    touches a source, so every out-degree is unchanged. Both invariants survive
    any number of swaps.

    Swapping sources as well would also preserve the multisets, but it can move
    a target onto its own source and *create* a self-loop, so it is not used.
    """
    source = source.copy()
    target = target.copy()
    total = int(target.size)
    every_index = np.arange(total, dtype=np.int64)
    present = set((source.astype(np.int64) * n + target).tolist())

    for _ in range(max_passes):
        # Which keys appear more than once? Every copy of a repeated key is a
        # multi-edge and every copy must be repaired, not just the extras.
        key = source * np.int64(n) + target
        _, inverse, occurrences = np.unique(key, return_inverse=True, return_counts=True)
        bad = occurrences[inverse] > 1
        bad |= source == target
        offenders = every_index[bad]
        healthy = every_index[~bad]
        if offenders.size == 0:
            return source, target
        if healthy.size == 0:  # pragma: no cover - pathological input
            raise MCNSCircuitError("could not repair the randomised graph")

        order_here = rng.permutation(healthy.size)
        cursor = 0
        for offender in offenders.tolist():
            old_key = int(source[offender]) * n + int(target[offender])
            present.discard(old_key)
            fixed = False
            for _ in range(attempts_per_offender):
                if cursor >= order_here.size:
                    order_here = rng.permutation(healthy.size)
                    cursor = 0
                partner = int(healthy[order_here[cursor]])
                cursor += 1
                if partner == offender:
                    continue
                candidate = int(target[partner])
                if candidate == int(source[offender]):
                    continue
                new_offender = int(source[offender]) * n + candidate
                new_partner = int(source[partner]) * n + int(target[offender])
                if new_offender in present or new_partner in present:
                    continue
                present.discard(int(source[partner]) * n + int(target[partner]))
                target[offender], target[partner] = target[partner], target[offender]
                present.add(new_offender)
                present.add(new_partner)
                fixed = True
                break
            if not fixed:
                present.add(old_key)

    raise MCNSCircuitError(  # pragma: no cover - needs a pathological input
        f"randomised graph still had self-loops or multi-edges after {max_passes} passes"
    )


def verify_degree_preservation(original: MCNSCircuit, shuffled: MCNSCircuit) -> dict[str, Any]:
    """Compare the invariants that make the control a fair comparison."""
    out_a, out_b = np.diff(original.csr.indptr), np.diff(shuffled.csr.indptr)
    in_a, in_b = np.diff(original.csr.tocsc().indptr), np.diff(shuffled.csr.tocsc().indptr)
    return {
        "neurons_equal": original.num_neurons == shuffled.num_neurons,
        "edges_equal": original.num_edges == shuffled.num_edges,
        "out_degree_sequence_equal": bool(np.array_equal(np.sort(out_a), np.sort(out_b))),
        "in_degree_sequence_equal": bool(np.array_equal(np.sort(in_a), np.sort(in_b))),
        "synapse_count_total_equal": original.total_synapse_count == shuffled.total_synapse_count,
        "synapse_count_distribution_equal": bool(
            np.array_equal(
                np.sort(original.csr.data), np.sort(shuffled.csr.data)
            )
        ),
        "self_loops": int(shuffled.self_loop_count),
        "topology_changed": not _same_edges(original, shuffled),
        "out_degree_mean_original": float(out_a.mean()),
        "out_degree_mean_shuffled": float(out_b.mean()),
        "in_degree_mean_original": float(in_a.mean()),
        "in_degree_mean_shuffled": float(in_b.mean()),
    }


def _same_edges(a: MCNSCircuit, b: MCNSCircuit) -> bool:
    if a.csr.nnz != b.csr.nnz:
        return False
    left = set(zip(*a.csr.nonzero()))
    right = set(zip(*b.csr.nonzero()))
    return left == right
