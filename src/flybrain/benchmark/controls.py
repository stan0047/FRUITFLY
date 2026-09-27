"""The controls that decide whether a benchmark result means anything.

The problem these solve
-----------------------
A linear decoder will separate any two conditions from any set of numbers. That
is not a bug, it is what a decoder is. The only way to know whether a reported
accuracy says something about the **circuit** is to measure, from the same run,
three other things:

``INPUT_INSTANT``
    What an instantaneous scalar summary of the drive can do. This is the
    **acceptance gate**. If a two-number summary of the injected current already
    separates the conditions above chance, then the conditions differ in
    something other than their spatiotemporal structure, and no downstream
    number can be attributed to MCNS. The run is flagged invalid rather than
    reported as a result.

``INPUT_SPATIAL``
    What the *drive pattern* contains, with the same lateral resolution as the
    population readout: mean drive per lateral bin per time bin. This is the
    spatial relay bound — a circuit that passed its input through unchanged would
    score exactly this.

``INPUT_TEMPORAL``
    What the *ordering* of the drive contains, and nothing about where it is: the
    normalised time at which each lateral bin is driven, one number per bin. For a
    rightward sweep the bins are visited left to right, so the time centroids
    increase with bin index; for a leftward sweep they decrease. That slope is the
    direction, recoverable from the input alone, and it is the smallest motion
    statistic that exists. A common random-number start offset shifts every centroid
    equally and leaves the slope intact, which is why randomised starts are safe.

    A lagged cross-product of the two channel *totals* was tried first and is
    useless here: the spot drives exactly ``n_units`` units at a fixed amplitude at
    every step, so each channel's population mean is constant in time and any lag
    product of it is a constant. The direction is in the ordering, not the totals.

``MAGNITUDE_ONLY``
    Total activity as a single feature. If this alone separates, the reported
    accuracy is about drive level and nothing else.

None of these is derived from the circuit, so they are computed once from the
injected current and are identical across circuit conditions. They are still
reported per condition, because in the recurrent graph the input neurons receive
feedback and their spike counts are no longer stimulus-determined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "ACCEPTANCE_GATE_DISCLAIMER",
    "CONTROL_GROUPS",
    "INPUT_INSTANT",
    "INPUT_SPATIAL",
    "INPUT_TEMPORAL",
    "acceptance_gate",
    "input_control_features",
    "input_instant_identity",
]

#: Acceptance gate: the input must not already contain the answer.
INPUT_INSTANT = "INPUT_INSTANT"
#: Relay bound with the same spatial resolution as the population readout.
INPUT_SPATIAL = "INPUT_SPATIAL"
#: Relay bound from temporal structure only, no spatial detail.
INPUT_TEMPORAL = "INPUT_TEMPORAL"

#: The stimulus-side controls, in reporting order.
CONTROL_GROUPS: tuple[str, ...] = (INPUT_INSTANT, INPUT_SPATIAL, INPUT_TEMPORAL)

ACCEPTANCE_GATE_DISCLAIMER = (
    "The acceptance gate asks whether an instantaneous two-number summary of the injected current "
    "already distinguishes the conditions. It must not. If it does, the conditions differ in "
    "something other than spatiotemporal structure and the run is invalid for downstream "
    "interpretation."
)


def _channel_indices(circuit, on_types: tuple[str, ...], off_types: tuple[str, ...]):
    on = np.concatenate(
        [circuit.indices_of_type(t) for t in on_types if circuit.has(t)]
    ).astype(np.int64)
    off = np.concatenate(
        [circuit.indices_of_type(t) for t in off_types if circuit.has(t)]
    ).astype(np.int64)
    if on.size == 0 or off.size == 0:
        raise ValueError("the stimulus channels matched no neurons")
    return on, off


def _bin_edges(n_steps: int, bin_steps: int, skip_bins: int) -> np.ndarray:
    edges = np.arange(0, n_steps + 1, bin_steps)
    if edges[-1] < n_steps:
        edges = np.append(edges, n_steps)
    if skip_bins:
        edges = edges[skip_bins:]
    return edges


def _binned_series(tensor: np.ndarray, indices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean drive per neuron of ``indices``, one value per time bin."""
    block = tensor[:, indices].astype(np.float64)
    binned = np.add.reduceat(block, edges[:-1], axis=0)
    binned[-1] = block[edges[-2] :, :].mean(axis=0)
    return binned.mean(axis=1)


def _bin_matrix(per_step: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Reduce a ``(n_steps, k)`` array to ``(n_bins, k)`` by mean over each bin."""
    binned = np.add.reduceat(per_step, edges[:-1], axis=0)
    widths = np.diff(edges).astype(np.float64)
    return binned / np.maximum(widths, 1.0)[:, None]


def input_control_features(
    tensor: np.ndarray,
    circuit,
    lateral_bin: np.ndarray,
    on_types: tuple[str, ...] = ("L1",),
    off_types: tuple[str, ...] = ("L2",),
    bin_steps: int = 25,
    skip_bins: int = 0,
    n_lags: int = 3,
    normalise: bool = False,
) -> dict[str, np.ndarray]:
    """Build every stimulus-side control from one injected-current tensor.

    `tensor` is `(n_steps, n_neurons)`. Pass the **noise-free drive**, not the
    noisy injected current. The relay bound is meant to be the most favourable
    assumption available to a pure relay, and the additive noise is common across
    conditions by construction, so including it can only degrade the bound without
    adding information. Measured on the real subset, including it pushes the drive
    below the noise floor in every lateral bin and the temporal statistic becomes
    noise-driven.

    Shapes
    ------
    ``INPUT_INSTANT``  ``(1, 2)``
        Whole-run mean drive of the ON and OFF channels. One row, because a
        whole-run mean is one number per channel by construction.
    ``INPUT_SPATIAL``  ``(n_bins, 2 * n_lateral_bins)``
        Mean drive per channel per lateral bin, per time bin.
    ``INPUT_TEMPORAL`` ``(1, n_lateral_bins + 1)``
        Normalised time centroid of the drive in each lateral bin, then the
        signed mean lateral velocity. The ordering, with no magnitude.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 2:
        raise ValueError(f"expected a (n_steps, n_neurons) tensor, got {tensor.shape}")
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")
    on, off = _channel_indices(circuit, on_types, off_types)
    edges = _bin_edges(tensor.shape[0], bin_steps, skip_bins)
    if edges.size < 2:
        raise ValueError("no time bins survive bin_steps and skip_bins")

    on_series = _binned_series(tensor, on, edges)
    off_series = _binned_series(tensor, off, edges)

    out: dict[str, np.ndarray] = {}
    out[INPUT_INSTANT] = np.array([[on_series.mean(), off_series.mean()]], dtype=np.float64)

    n_lateral = int(lateral_bin.max()) + 1
    lateral = np.asarray(lateral_bin, dtype=np.int64)

    spatial_columns = []
    for indices in (on, off):
        for bin_index in range(n_lateral):
            chosen = indices[lateral[indices] == bin_index]
            if chosen.size == 0:
                spatial_columns.append(np.zeros(edges.size - 1, dtype=np.float64))
            else:
                spatial_columns.append(_binned_series(tensor, chosen, edges))
    out[INPUT_SPATIAL] = np.column_stack(spatial_columns)

    # Where the drive is, per timestep, on the ON channel: a 1-D spatial profile.
    per_step = np.zeros((tensor.shape[0], n_lateral), dtype=np.float64)
    for bin_index in range(n_lateral):
        chosen = on[lateral[on] == bin_index]
        if chosen.size:
            per_step[:, bin_index] = np.abs(tensor[:, chosen]).mean(axis=1)
    profile = _bin_matrix(per_step, edges)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    centroids = np.zeros(n_lateral, dtype=np.float64)
    span = float(bin_centres[-1] - bin_centres[0]) or 1.0
    for bin_index in range(n_lateral):
        column = profile[:, bin_index]
        total = float(column.sum())
        if total > 0:
            centroids[bin_index] = float((bin_centres * column).sum() / total)
    centroids = (centroids - bin_centres[0]) / span
    # Signed mean lateral velocity, from consecutive profile centroids.
    spatial_centre = profile @ np.arange(n_lateral, dtype=np.float64)
    weight = profile.sum(axis=1)
    valid = weight > 0
    velocity = 0.0
    if valid.sum() >= 2:
        steps = np.diff(spatial_centre[valid])
        velocity = float(np.mean(steps) / max(n_lateral - 1, 1))
    out[INPUT_TEMPORAL] = np.concatenate([centroids, [velocity]])[None, :]

    if normalise:
        out = {name: _row_normalise(value) for name, value in out.items()}
    return out


def _gate_vector(entry: Any) -> np.ndarray:
    """Pull the gate feature vector out of whatever the caller passed.

    Accepts a raw array, a single record, or a list of records, so the identity
    check can be run on the feature dicts directly or on the very record lists
    the decoder is handed. The second is what the benchmark uses: it then
    verifies the numbers that were actually decoded, not a parallel copy.
    """
    if entry is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(entry, np.ndarray):
        return np.asarray(entry, dtype=np.float64).ravel()
    if hasattr(entry, "features"):
        return np.asarray(entry.features, dtype=np.float64).ravel()
    if isinstance(entry, (list, tuple)):
        parts = [_gate_vector(item) for item in entry]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.zeros(0, dtype=np.float64)
        widths = {p.size for p in parts}
        if len(widths) != 1:
            raise ValueError(f"gate records have inconsistent widths: {sorted(widths)}")
        return np.vstack(parts).ravel()
    raise TypeError(f"cannot read a gate feature from {type(entry)!r}")


def _polarity_of(name: str) -> str:
    return "REVERSED" if str(name).endswith("_POLARITY_REVERSED") else "NORMAL"


def input_instant_identity(
    features: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[str],
    polarity_of: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """How far apart the gate features are, **within a trial and polarity**, before any decoder.

    Two deliberate restrictions, each for a reason:

    *Within a trial.* The per-trial amplitude perturbation is keyed on
    ``(seed, trial)`` and shared by every condition, so the gate features are
    *expected* to differ from one trial to the next; that variation is the noise
    the permutation null is built from. What must not happen is two conditions
    differing within a trial, because that would mean they carry different total
    drive.

    *Within a polarity.* A polarity twin deliberately drives the OFF channel with
    the opposite sign, so its OFF mean is negative by construction. Comparing it
    against a normal-polarity condition would report a large spread that is the
    control working as designed, not a confound.

    Where the energy match is exact this returns ``max_spread == 0.0``, and the
    gate's pass becomes a positive statement about the stimulus design rather than
    a decoder happening to fail to find signal.
    """
    labels = {str(c): str((polarity_of or {}).get(c) or _polarity_of(c)) for c in conditions}
    per_trial: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for name in conditions:
        for rec in features.get(str(name), {}).get(INPUT_INSTANT, []) or []:
            key = (labels[str(name)], int(getattr(rec, "trial", 0)))
            per_trial.setdefault(key, {})[str(name)] = _gate_vector(rec)

    spreads: list[float] = []
    per_polarity: dict[str, list[float]] = {}
    widths: set[int] = set()
    examples: dict[str, list[float]] = {}
    for (_polarity, _trial), by_condition in sorted(per_trial.items()):
        usable = {k: v for k, v in by_condition.items() if v.size}
        if len(usable) < 2:
            continue
        widths.update(v.size for v in usable.values())
        stack = np.vstack(list(usable.values()))
        spread = float(np.abs(stack - stack[0]).max())
        spreads.append(spread)
        per_polarity.setdefault(_polarity, []).append(spread)
        examples = {k: [float(x) for x in v] for k, v in usable.items()}

    if not spreads:
        return {
            "trials_compared": 0, "max_spread": float("nan"), "identical": False,
            "identical_to_1e_12": False, "per_polarity_max_spread": {}, "values": {},
            "note": "need at least two conditions in at least one trial to compare",
        }
    worst = float(max(spreads))
    return {
        "trials_compared": len(spreads),
        "feature_width": int(max(widths)) if len(widths) == 1 else -1,
        "max_spread": worst,
        "mean_spread": float(np.mean(spreads)),
        "per_polarity_max_spread": {k: float(max(v)) for k, v in per_polarity.items()},
        "identical": bool(worst == 0.0),
        "identical_to_1e_12": bool(worst < 1e-12),
        "values": examples,
        "note": (
            "max_spread is the largest absolute difference between any two conditions' gate "
            "features WITHIN a trial and WITHIN a polarity. Zero means the drive totals are "
            "exactly equal, so the gate passes by construction. Trial-to-trial variation is "
            "expected and is what the permutation null is built from. A polarity twin is "
            "excluded because its OFF channel sign is reversed on purpose."
        ),
    }


def _row_normalise(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def acceptance_gate(
    results: dict[str, dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Decide whether the run may be interpreted downstream at all.

    ``results`` maps a condition name to its ``INPUT_INSTANT`` payload, as
    returned by :func:`~flybrain.benchmark.decoder.cross_validate` with the
    permutation null attached.

    The gate passes when the observed cross-validated accuracy for
    ``INPUT_INSTANT`` does **not** exceed the 95th percentile of its own
    label-permutation null, in every condition. If it does, the injected current
    already distinguishes the conditions on its own and the benchmark is flagged
    invalid rather than quietly reported.
    """
    per_condition: dict[str, Any] = {}
    failed: list[str] = []
    undecidable: list[str] = []
    for condition, payload in results.items():
        null = payload.get("permutation_null", {})
        observed = payload.get("mean_test_accuracy")
        entry: dict[str, Any] = {
            "observed_mean_test_accuracy": observed,
            "null_status": null.get("status"),
            "null_mean": null.get("null_mean"),
            "null_p95": null.get("null_p95"),
            "z_score": null.get("z_score"),
        }
        if null.get("status") != "ok" or observed is None:
            entry["verdict"] = "undecidable"
            entry["reason"] = (
                "no usable permutation null, so the gate cannot be decided from this run"
            )
            undecidable.append(condition)
        elif bool(null.get("exceeds_null_p95")):
            entry["verdict"] = "fail"
            entry["reason"] = (
                "the injected current separates the conditions on its own, above the 95th "
                "percentile of its own label-permutation null"
            )
            failed.append(condition)
        else:
            entry["verdict"] = "pass"
            entry["reason"] = (
                "an instantaneous two-number summary of the drive does not separate the "
                "conditions above chance"
            )
        per_condition[condition] = entry

    passed = not failed
    undecidable_all = bool(results) and len(undecidable) == len(results)
    if failed:
        status = "fail"
    elif undecidable_all:
        # No condition had a usable null, so the gate was never actually tested.
        # Reporting this as a pass would be a decoder's failure to find signal
        # dressed up as a positive result.
        status = "undecidable"
    elif undecidable:
        status = "pass_with_undecidable"
    else:
        status = "pass"
    return {
        "gate": INPUT_INSTANT,
        "status": status,
        "disclaimer": ACCEPTANCE_GATE_DISCLAIMER,
        "alpha": float(alpha),
        "passed": bool(passed),
        "interpretation_allowed": bool(passed and not undecidable_all),
        "failed_conditions": failed,
        "undecidable_conditions": undecidable,
        "per_condition": per_condition,
        "instruction": (
            "The gate PASSED. Downstream groups may be interpreted, always relative to "
            "INPUT_SPATIAL, INPUT_TEMPORAL and SHUFFLED_CONTROL."
            if passed and not undecidable_all
            else "The gate FAILED. The input already distinguishes the conditions, so no T4/T5 "
                 "number may be attributed to MCNS processing. This is a property of the stimulus "
                 "and readout configuration, not evidence about the circuit, and it must not be "
                 "reported as a downstream result."
            if failed
            else "The gate is UNDECIDABLE: no usable permutation null, so nothing was actually "
                 "tested. Run with --permutations above zero before interpreting any downstream "
                 "group. Note that the gate's features are exactly equal across conditions by "
                 "construction, which is a stronger statement than a decoder failing to find "
                 "signal; check gate_feature_max_spread in the summary."
        ),
    }
