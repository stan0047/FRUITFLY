"""A lateral activity-profile readout, and the calibration that proves it measures structure.

Why the Phase 4C readout had to be replaced
-------------------------------------------
The Phase 4C primary readout was ``subtype_lateral`` on ``T4+T5``: 8 exact
subtypes x 4 lateral bins, unit-normalised per time bin, scored by a
leave-one-trial-out linear decoder. Three measured properties made it unable to
answer the Phase 5 question:

1. **It was not resolving structure.** ``MAGNITUDE_ONLY`` -- a *single* feature,
   total T4+T5 activity -- scored 0.5750 against the 32-feature readout's 0.5687.
   A one-number control matched a 32-column representation.
2. **It was quantised far below the effect.** Each fold scored 8 held-out rows
   (2 sweep directions x 4 surviving time bins), so fold accuracy moved only in
   steps of 1/8 and the 20-fold mean was a multiple of 1/160. The two arms tied at
   exactly 91/160, an integer coincidence rather than a measurement.
3. **It conflated two things.** Eight subtypes crossed with four lateral bins in
   one 32-vector, so a lateral effect could be expressed as a subtype effect and
   the two could not be separated.

What this module does instead
-----------------------------
It works on **per-neuron spike counts over the analysed window** -- the highest
information-per-parameter representation available, and one with no 1/8
quantisation -- and reduces them to a **lateral activity profile**: mean activity
per lateral bin, for one cell group. From that it takes a **centroid**, the
normalised mean lateral position of the activity.

The centroid is a continuous statistic with a smooth null. It is also the smallest
thing that answers the Phase 5 question, which is about *where* drive arrives
rather than how much.

The count-preserving scramble: calibrating the statistic, not the data
---------------------------------------------------------------------
A centroid can be moved by more spikes, or by the same spikes landing somewhere
else. Those are different claims and the report must not confuse them. So the
statistic is **calibrated** before it is interpreted: within each trial, the
per-neuron spike counts are permuted across neurons. That preserves

* the total number of spikes in the trial, exactly;
* the multiset of per-neuron spike counts, exactly;
* the number of neurons, their lateral bins and their identities;

while destroying the association between *which* neuron fired and *where* it sits.

If the observed centroid shift is reproduced by that scramble, the statistic is
measuring activity, not spatial structure, and acceptance gate ``G5`` in
:mod:`flybrain.benchmark.phase5a` fails. This is a property of the statistic, so
it is reported as a gate rather than as a p-value, and it is deliberately not
folded into the inferential null.

Direction selectivity is out of scope
------------------------------------
Nothing here tests or can test it. A centroid says where activity is, not which
direction a neuron prefers. The MCNS connectivity table carries
``body_pre``/``body_post``/``weight`` and **no synaptic sign**, so the model can
sum and never subtract, and no amount of tuning turns an unsigned sum into a
direction-selective response. See :mod:`flybrain.benchmark.phase5a`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "LATERAL_READOUT_DISCLAIMER",
    "calibration_shift_null",
    "lateral_profile",
    "profile_centroid",
    "profile_shape_distance",
    "scramble_counts_preserving_total",
]


LATERAL_READOUT_DISCLAIMER = (
    "A lateral activity-profile readout: mean per-neuron spike count per lateral bin, reduced to a "
    "normalised centroid. It reports where activity is, not how much, and not which direction a "
    "neuron prefers. Direction selectivity is out of scope and is not testable on unsigned "
    "connectivity."
)


def _group_mask(
    cell_type_index: Mapping[str, np.ndarray], group: Sequence[str]
) -> np.ndarray | None:
    """Boolean mask over neurons for a group, or ``None`` when a type is absent.

    Returns ``None`` rather than an all-false mask when the whole group is
    missing, so a caller can report "group absent" instead of silently analysing
    zeros.
    """
    parts = []
    for name in group:
        indices = cell_type_index.get(name)
        if indices is not None and np.asarray(indices).size:
            parts.append(np.asarray(indices, dtype=np.int64))
    if not parts:
        return None
    mask = np.zeros(int(max(int(p.max()) for p in parts) + 1), dtype=bool)
    for part in parts:
        mask[part] = True
    return mask


def lateral_profile(
    counts: np.ndarray,
    lateral_bin: np.ndarray,
    cell_type_index: Mapping[str, np.ndarray],
    group: Sequence[str],
) -> np.ndarray:
    """Mean per-neuron spike count in each lateral bin, for one cell group.

    ``counts`` is ``(n_neurons,)`` -- total spikes for one trial over the analysed
    window. ``lateral_bin`` is the per-neuron bin index from
    :func:`~flybrain.benchmark.stimulus.lateral_coordinates`, using the *same*
    geometry the stimulus is built from, so "bin 0" means the same place in the
    readout as it does in the drive.

    A bin with no neurons in the group keeps its slot at zero rather than being
    dropped, so a profile is always ``n_bins`` long and two conditions are always
    directly comparable bin by bin. Group membership is a *mask*, so a cell type
    absent from the circuit cannot silently shift the bin numbering.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    lateral = np.asarray(lateral_bin, dtype=np.int64).ravel()
    if counts.size != lateral.size:
        raise ValueError(f"counts ({counts.size}) and lateral_bin ({lateral.size}) disagree")
    n_bins = int(lateral.max()) + 1 if lateral.size else 0
    if n_bins < 1:
        raise ValueError("lateral_bin must contain at least one bin")

    mask = _group_mask(cell_type_index, group)
    if mask is None:
        return np.zeros(n_bins, dtype=np.float64)
    if mask.size < counts.size:
        padded = np.zeros(counts.size, dtype=bool)
        padded[: mask.size] = mask
        mask = padded

    profile = np.zeros(n_bins, dtype=np.float64)
    for bin_index in range(n_bins):
        chosen = mask & (lateral == bin_index)
        if chosen.any():
            profile[bin_index] = float(counts[chosen].mean())
    return profile


def profile_centroid(profile: np.ndarray, bin_centres: np.ndarray | None = None) -> float:
    """Normalised mean lateral position of a profile's activity.

    With ``n_bins`` bins the result is 0 for all activity in bin 0 and 1 for all
    activity in the last bin, so profiles of different widths are comparable. A
    profile with no activity anywhere returns ``nan`` rather than 0, because 0
    would be a real position and would silently average into a mean.
    """
    profile = np.asarray(profile, dtype=np.float64).ravel()
    if profile.size == 0:
        return float("nan")
    if bin_centres is None:
        # Normalised bin *centres*: 0 in the first bin, 1 in the last. Using raw
        # indices instead would report 3.0 for the last of 4 bins while claiming
        # to be normalised, and every threshold expressed as a fraction of the
        # lateral extent would then be wrong by a factor of n_bins - 1.
        centres = (
            np.arange(profile.size, dtype=np.float64) / (profile.size - 1)
            if profile.size > 1
            else np.zeros(1, dtype=np.float64)
        )
    else:
        centres = np.asarray(bin_centres, dtype=np.float64).ravel()
        if centres.size != profile.size:
            raise ValueError("bin_centres and profile must have the same length")
    total = float(profile.sum())
    if total <= 0.0:
        return float("nan")
    return float((profile * centres).sum() / total)


def profile_shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance ``1 - cos(a, b)`` between two profiles.

    ``nan`` when either profile is all zeros, which is reported rather than
    coerced: a zero profile has no direction to compare.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - float(np.dot(x, y) / denominator))


def scramble_counts_preserving_total(
    counts: np.ndarray, seed: int | Sequence[int], group: Sequence[int] | None = None
) -> np.ndarray:
    """Permute per-neuron spike counts across neurons, exactly preserving the multiset.

    This is the count-preserving lateral calibration. Total spikes, the per-neuron
    count multiset, the neuron set and the lateral-bin assignment are all
    unchanged; only *which* neuron carries which count changes, so the association
    between firing and lateral position is destroyed while activity is not.

    ``group`` restricts the permutation to the indices of one cell group, so a
    T4 profile can be calibrated without moving T4 counts onto T5 neurons.

    ``seed`` is passed to :func:`numpy.random.default_rng` unchanged, so a seed
    *sequence* is honoured. ``calibration_shift_null`` keys its scrambles on
    ``(seed, group, stimulus, trial)``; collapsing that to an int would make every
    replicate of every trial draw the same permutation, and the null would be one
    permutation repeated rather than an independent one.
    """
    counts = np.asarray(counts, dtype=np.int64).ravel()
    out = counts.copy()
    indices = np.arange(counts.size, dtype=np.int64) if group is None else np.asarray(
        group, dtype=np.int64
    )
    if indices.size < 2:
        return out
    out[indices] = counts[np.random.default_rng(seed).permutation(indices)]
    return out


def calibration_shift_null(
    counts_by_key: Mapping[tuple[str, int], np.ndarray],
    lateral_bin: np.ndarray,
    cell_type_index: Mapping[str, np.ndarray],
    group: Sequence[str],
    left: str,
    right: str,
    trials: int,
    n_scrambles: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Centroid and shape separation under the count-preserving scramble.

    ``counts_by_key`` maps ``(stimulus, trial)`` to a per-neuron count vector. The
    statistic is computed on the real data and on ``n_scrambles`` independent
    within-trial permutations of the counts, and both are returned so the report
    can state how far the observed separation sits above the activity-preserving
    floor.

    The seed is derived from ``(seed, stimulus, trial, replicate)`` so a scramble
    is reproducible on its own terms and does not depend on the order replicates
    happen to be generated in.
    """
    masks = _group_mask(cell_type_index, group)
    if masks is None:
        return {"status": "group_absent", "group": list(group)}
    group_indices = np.flatnonzero(masks)

    def _pairs() -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for trial in range(int(trials)):
            a = counts_by_key.get((left, trial))
            b = counts_by_key.get((right, trial))
            if a is not None and b is not None:
                out.append((np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)))
        return out

    pairs = _pairs()
    if not pairs:
        return {"status": "no_paired_trials", "group": list(group)}

    observed_shifts: list[float] = []
    observed_shapes: list[float] = []
    null_shifts: list[float] = []
    null_shapes: list[float] = []
    for trial, (a, b) in enumerate(pairs):
        pa = lateral_profile(a, lateral_bin, cell_type_index, group)
        pb = lateral_profile(b, lateral_bin, cell_type_index, group)
        observed_shifts.append(
            profile_centroid(pa) - profile_centroid(pb)
        )
        observed_shapes.append(profile_shape_distance(pa, pb))
        for replicate in range(int(n_scrambles)):
            sa = scramble_counts_preserving_total(
                a, seed=[int(seed), 1, trial, replicate], group=group_indices
            )
            sb = scramble_counts_preserving_total(
                b, seed=[int(seed), 2, trial, replicate], group=group_indices
            )
            qa = lateral_profile(sa, lateral_bin, cell_type_index, group)
            qb = lateral_profile(sb, lateral_bin, cell_type_index, group)
            null_shifts.append(profile_centroid(qa) - profile_centroid(qb))
            null_shapes.append(profile_shape_distance(qa, qb))

    def _summary(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if array.size == 0:
            return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                    "p95": float("nan"), "max_abs": float("nan")}
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "p95": float(np.percentile(array, 95.0)),
            "max_abs": float(np.abs(array).max()),
        }

    observed = np.asarray([v for v in observed_shifts if np.isfinite(v)], dtype=np.float64)
    null = np.asarray([v for v in null_shifts if np.isfinite(v)], dtype=np.float64)
    observed_abs = np.abs(observed)
    null_abs = np.abs(null)
    null_abs_p95 = float(np.percentile(null_abs, 95.0)) if null_abs.size else float("nan")
    # Planned Phase 5A choice: UPPER_TAIL. The gate passes only when the largest
    # observed centroid shift exceeds the 95th percentile of the count-preserving
    # null shift magnitudes. This is deliberately the upper tail of the null, not
    # the null maximum: an observed value tied with the largest null value no
    # longer counts as independent of the activity floor.
    exceeds = bool(null_abs.size and observed_abs.size and observed_abs.max() > null_abs_p95)
    return {
        "status": "ok",
        "group": list(group),
        "n_trials": int(observed.size),
        "n_scrambles": int(n_scrambles),
        "seed": int(seed),
        "null_of": (
            "per-neuron spike counts permuted across neurons within a trial, preserving the total "
            "spike count and the per-neuron count multiset exactly while destroying the "
            "association between which neuron fired and where it sits laterally"
        ),
        "observed_centroid_shift": [float(v) for v in observed_shifts],
        "observed_shape_distance": [float(v) for v in observed_shapes],
        "observed_shift_summary": _summary(observed_shifts),
        "null_shift_summary": _summary(null_shifts),
        "observed_shape_summary": _summary(observed_shapes),
        "null_shape_summary": _summary(null_shapes),
        "observed_max_abs_shift": float(np.abs(observed).max()) if observed.size else float("nan"),
        "null_max_abs_shift": float(np.abs(null).max()) if null.size else float("nan"),
        "null_abs_p95": null_abs_p95,
        "observed_exceeds_calibration": exceeds,
        "reading": (
            "the observed lateral separation is larger than the 95th percentile of a count-preserving "
            "scramble, so the statistic is carrying spatial structure rather than activity alone"
            if exceeds else
            "a count-preserving scramble's upper notch includes the observed lateral separation, so "
            "the statistic is measuring activity and must not be read as spatial structure"
        ),
    }
