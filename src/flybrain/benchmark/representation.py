"""Population representations and the distance metrics used in the benchmark.

A *population vector* here is the mean spike count per neuron of one cell type,
within one time bin. So a vector for ``T4a`` has one column, and a vector for
``T4`` has four columns, one per exact subtype. Subtype identity is never
collapsed: the ``T4`` group is a convenience for reporting and always carries
the four subtypes separately alongside it.

Readout resolution
------------------
:meth:`population_features` has three documented readouts, and the choice is
reported rather than implied.

``"subtype_lateral"`` (default, 32 columns for ``T4+T5``)
    One column per **exact cell type per lateral bin**: 8 subtypes x 4 bins of the
    ``[-1, 1]`` field, using the same geometry the stimulus is built from. This
    is the primary representation because a direction cue for a moving stimulus
    lives in *which part of the field* responded, not in the population total. A
    population mean throws exactly that away.

``"cell_type"``
    One column per exact cell type: the mean spikes per neuron of that type in the
    bin, 8 columns for ``T4+T5``. Retained as a secondary comparison. It cannot
    in principle represent a moving spot's direction, and it is reported so that
    the difference between it and ``subtype_lateral`` is visible.

``"per_neuron"``
    One column per neuron. Available for inspection. With ``m`` trials the sample
    count is ``trials * bins`` while the column count is the population size, so
    any decoder fitted on it is overparameterised by construction and every
    reported distance grows with population size rather than with information.

Distances are reported, not scores
----------------------------------
These are distances between activity patterns. None of them is an accuracy, and
none of them is evidence that the circuit computes anything. A distance can only
be interpreted next to the within-stimulus variability of the same measure, which
is why :func:`separability` always reports both.

Memory
------
Every pairwise routine here is **row-wise**: the ``(n, n, d)`` broadcast tensor
that a naive implementation builds is never allocated. For the real candidate
population that tensor is 60 x 60 x 13,580 float64 = 391 MB per stimulus per
permutation, which is what made the first benchmark attempt take hours. The
row-wise form is exact, allocates ``O(n d)``, and returns identical values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.benchmark.stimulus import lateral_coordinates

__all__ = [
    "ABBREVIATIONS",
    "AGGREGATE_GROUPS",
    "ALL_GROUPS",
    "INPUT_GROUP",
    "LATREADOUT_BINS",
    "MAGNITUDE_GROUP",
    "READOUTS",
    "SUBTYPE_GROUPS",
    "PopulationRecord",
    "activity_descriptor",
    "cosine_distance",
    "cross_condition_distance",
    "euclidean_distance",
    "fast_separation_ratio",
    "lateral_coordinates",
    "magnitude_features",
    "pairwise_metric_table",
    "pearson_correlation",
    "population_features",
    "separation_ratio_null",
    "separability",
    "stimulus_centroids",
    "unit_normalise",
    "within_between",
]

#: Exact cell types recorded individually, per the benchmark specification.
SUBTYPE_GROUPS: tuple[str, ...] = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)

#: Aggregations for reporting. These never replace the subtype data.
AGGREGATE_GROUPS: tuple[str, ...] = ("T4", "T5", "T4+T5")

#: The input control: the stimulus as the circuit actually receives it.
INPUT_GROUP = "INPUT"

#: Control that can only sort stimuli by overall activity, not by pattern.
MAGNITUDE_GROUP = "MAGNITUDE_ONLY"

ALL_GROUPS: tuple[str, ...] = (INPUT_GROUP,) + AGGREGATE_GROUPS + SUBTYPE_GROUPS

#: Column labels for each group, in order.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "T4": ("T4a", "T4b", "T4c", "T4d"),
    "T5": ("T5a", "T5b", "T5c", "T5d"),
    "T4+T5": ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"),
    INPUT_GROUP: ("L1", "L2"),
}
ABBREVIATIONS.update({name: (name,) for name in SUBTYPE_GROUPS})
# The magnitude control has a single derived column rather than cell types.
ABBREVIATIONS[MAGNITUDE_GROUP] = ("T4+T5 total activity",)

#: Readout resolutions accepted by :func:`population_features`.
READOUTS: tuple[str, ...] = ("subtype_lateral", "cell_type", "per_neuron")

#: Default number of lateral bins for the primary readout. 8 subtypes x 4 bins = 32.
LATREADOUT_BINS = 4

#: The primary readout is the only one that resolves space as well as identity.
PRIMARY_READOUT = "subtype_lateral"


@dataclass
class PopulationRecord:
    """Binned population activity for one stimulus presentation.

    ``features`` has shape ``(n_bins, n_columns)``. The actual width is reported
    by :meth:`summary` so the readout and the cell-type list can never be
    confused: ``cell_type`` gives one column per cell type, ``per_neuron`` one
    per neuron, and ``subtype_lateral`` one per cell type per lateral bin.
    """

    stimulus: str
    trial: int
    group: str
    features: np.ndarray
    bin_centres: np.ndarray
    readout: str = PRIMARY_READOUT
    lateral_bins: int = 0

    @property
    def n_bins(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def summary(self) -> dict[str, Any]:
        cell_types = list(ABBREVIATIONS[self.group])
        return {
            "stimulus": self.stimulus,
            "trial": self.trial,
            "group": self.group,
            "cell_types": cell_types,
            "bins": self.n_bins,
            "feature_columns": self.n_columns,
            "readout": self.readout,
            "lateral_bins": int(self.lateral_bins),
            "one_column_per_cell_type": self.n_columns == len(cell_types),
            "one_column_per_cell_type_per_lateral_bin": bool(
                self.lateral_bins and self.n_columns == len(cell_types) * self.lateral_bins
            ),
            "total_spikes_per_column": [float(v) for v in self.features.sum(axis=0)],
        }


def unit_normalise(features: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm, so only the *pattern* across subtypes remains.

    Without this, a linear decoder can separate two stimuli purely because one
    produces more spikes than the other, which says nothing about whether the
    circuit represents them differently. Normalising removes that shortcut. Rows
    that are exactly zero are left as zero rather than turned into NaN, and the
    number of such rows is reported by the caller.
    """
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = norms > 0
    return np.divide(x, norms, out=np.zeros_like(x), where=safe)


def population_features(
    spikes: np.ndarray,
    cell_type_index: Mapping[str, np.ndarray],
    group: str,
    bin_steps: int,
    skip_bins: int = 0,
    normalise: bool = True,
    readout: str = PRIMARY_READOUT,
    lateral_bin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a spike matrix into population vectors.

    ``spikes`` is ``(n_steps, n_neurons)`` and boolean. Mean per neuron, not raw
    counts, so that a column is comparable across cell types with different
    population sizes.

    ``readout`` selects the column resolution; see :data:`READOUTS` and the
    module docstring. ``subtype_lateral`` needs ``lateral_bin``, a per-neuron bin
    index from :func:`~flybrain.benchmark.stimulus.lateral_coordinates`, and
    produces ``len(ABBREVIATIONS[group]) * n_lateral_bins`` columns.

    With ``normalise`` the returned rows are unit-norm, so the features describe
    the *relative* activity across columns and carry no overall magnitude
    information. That is the default because a magnitude difference is a
    confound, not a finding.

    ``skip_bins`` drops the leading bins, which is where the network transient
    lives. It is a reported parameter, not a hidden trim.
    """
    if group not in ABBREVIATIONS:
        raise KeyError(f"unknown group {group!r}")
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    n_steps = spikes.shape[0]
    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")

    n_lateral = 0
    if readout == "subtype_lateral":
        if lateral_bin is None:
            raise ValueError(
                "readout='subtype_lateral' needs lateral_bin, the per-neuron bin index from "
                "lateral_coordinates(circuit, n_bins)"
            )
        lateral_bin = np.asarray(lateral_bin, dtype=np.int64)
        n_lateral = int(lateral_bin.max()) + 1 if lateral_bin.size else 0
        if n_lateral < 1:
            raise ValueError("lateral_bin must contain at least one bin")

    edges = np.arange(0, n_steps + 1, bin_steps)
    if edges[-1] < n_steps:
        edges = np.append(edges, n_steps)
    centres = 0.5 * (edges[:-1] + edges[1:]) / max(n_steps, 1)

    columns = []
    for name in ABBREVIATIONS[group]:
        indices = cell_type_index.get(name)
        indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        if readout == "subtype_lateral":
            for bin_index in range(n_lateral):
                if indices is None or indices.size == 0:
                    columns.append(np.zeros(centres.size, dtype=np.float64))
                    continue
                chosen = indices[lateral_bin[indices] == bin_index]
                if chosen.size == 0:
                    # An empty cell keeps its column, at zero, so subtypes and
                    # bins stay aligned and comparable across conditions.
                    columns.append(np.zeros(centres.size, dtype=np.float64))
                    continue
                columns.append(_binned_mean(spikes, chosen, edges))
            continue
        if indices is None or indices.size == 0:
            columns.append(np.zeros(centres.size, dtype=np.float64))
            continue
        if readout == "cell_type":
            # One column per exact cell type: the mean over its neurons.
            columns.append(_binned_total(spikes, indices, edges) / float(indices.size))
        else:
            # One column per neuron: a full population vector, shape
            # (n_bins, len(indices)). Overparameterised for a decoder at this
            # trial count, and reported as such.
            columns.append(_binned_rows(spikes, indices, edges) / float(indices.size))

    features = np.column_stack(columns)
    if skip_bins:
        features = features[skip_bins:]
        centres = centres[skip_bins:]
    if normalise:
        features = unit_normalise(features)
    return features, centres


def _binned_rows(spikes: np.ndarray, indices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin-edge totals for one neuron set, shape ``(n_bins, len(indices))``."""
    block = spikes[:, indices].astype(np.float64)
    binned = np.add.reduceat(block, edges[:-1], axis=0)
    # reduceat mis-handles a final partial bin; recompute that row safely.
    binned[-1] = block[edges[-2] :, :].sum(axis=0)
    return binned


def _binned_mean(spikes: np.ndarray, indices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean spikes per neuron over time, shape ``(n_bins,)``."""
    return _binned_rows(spikes, indices, edges).mean(axis=1)


def _binned_total(spikes: np.ndarray, indices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Total spikes over the neuron set, shape ``(n_bins,)``."""
    return _binned_rows(spikes, indices, edges).sum(axis=1)


def activity_descriptor(
    spikes: np.ndarray,
    cell_type_index: Mapping[str, np.ndarray],
    group: str = "T4+T5",
) -> dict[str, Any]:
    """Is the readout population actually firing?

    A decoder run on a silent population reports a number, and that number is
    noise. This descriptor is what lets a report say "T4/T5 were silent at this
    coupling gain" instead of leaving the reader to infer it from a feature width.
    """
    indices = np.concatenate(
        [np.asarray(cell_type_index[name], dtype=np.int64)
         for name in ABBREVIATIONS[group]
         if name in cell_type_index and np.asarray(cell_type_index[name]).size]
    ) if group in ABBREVIATIONS else np.zeros(0, dtype=np.int64)
    if indices.size == 0:
        return {"group": group, "neurons": 0, "spikes": 0, "active": False}
    counts = spikes[:, indices].sum(axis=0)
    per_neuron = counts / max(int(spikes.shape[0]), 1)
    return {
        "group": group,
        "neurons": int(indices.size),
        "spikes": int(counts.sum()),
        "neurons_firing": int(np.count_nonzero(counts)),
        "fraction_neurons_firing": float(np.count_nonzero(counts) / indices.size),
        "max_spikes_per_neuron": int(counts.max()),
        "mean_spikes_per_neuron": float(counts.mean()),
        "mean_rate_per_neuron_per_step": float(per_neuron.mean()),
        "active": bool(counts.sum() > 0),
    }


def magnitude_features(features: np.ndarray) -> np.ndarray:
    """Collapse a population vector to its total activity, as a single feature.

    Used for the ``MAGNITUDE_ONLY`` control: a single-feature decoder that can
    only sort stimuli by how much activity they produce. If that control is
    already accurate, the multi-feature accuracy says nothing about pattern.
    """
    return np.asarray(features, dtype=np.float64).sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ metrics


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - cos(a, b)``. NaN when either vector has zero norm.

    Zero-norm vectors are common in sparse binned activity, so the count of NaNs
    is reported rather than quietly coerced to 0, which would fake a perfect
    match.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denominator)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r. NaN when either vector is constant, which is reported not hidden."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.linalg.norm(a_centred) * np.linalg.norm(b_centred))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a_centred, b_centred) / denominator)


METRICS = {
    "cosine_distance": cosine_distance,
    "euclidean_distance": euclidean_distance,
    "pearson_correlation": pearson_correlation,
}


def _upper_triangle_pairs(x: np.ndarray, metric_name: str) -> np.ndarray:
    """Every distinct-pair metric value for one block of vectors, row-wise.

    Returns the strict upper triangle in row-major order, which is the order the
    equivalent double loop would produce, so means agree. The naive
    ``x[:, None, :] - x[None, :, :]`` broadcast is deliberately **not** used: for
    60 vectors of 13,580 columns that single expression allocates 391 MB, and
    the null test calls it thousands of times. This form allocates one row at a
    time.

    NaN semantics match the scalar functions: ``cosine_distance`` is NaN when
    either vector has zero norm, ``pearson_correlation`` when either is
    constant, ``euclidean_distance`` is never NaN.
    """
    block = np.asarray(x, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError("expected a 2-D block of vectors")
    m = block.shape[0]
    if m < 2:
        return np.zeros(0, dtype=np.float64)

    if metric_name == "pearson_correlation":
        centred = block - block.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1)
        scale = np.where(norms > 0, norms, np.nan)
    elif metric_name == "cosine_distance":
        centred = block
        norms = np.linalg.norm(block, axis=1)
        scale = np.where(norms > 0, norms, np.nan)
    else:
        centred = block
        scale = None

    pieces: list[np.ndarray] = []
    for i in range(m - 1):
        tail = centred[i + 1 :]
        if scale is None:
            difference = centred[i] - tail
            pieces.append(np.sqrt(np.einsum("ij,ij->i", difference, difference)))
            continue
        # Both cosine and Pearson are a normalised inner product, so the row is
        # one matrix-vector product rather than a difference and a square root.
        # They differ only in what they do with it: cosine_distance is
        # 1 - cos, pearson_correlation is r itself.
        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = (tail @ centred[i]) / (scale[i] * scale[i + 1 :])
        pieces.append(1.0 - normalised if metric_name == "cosine_distance" else normalised)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def _mean_pairwise(vectors: Sequence[np.ndarray], metric_name: str) -> tuple[float, int, float]:
    """Mean finite distinct-pair distance over a set of vectors.

    Returns ``(mean, n_finite_pairs, sd)``. The finite filter is applied before
    averaging so a zero-norm pair cannot quietly become a perfect match.
    """
    if len(vectors) == 0:
        return float("nan"), 0, float("nan")
    block = np.vstack([np.asarray(v, dtype=np.float64).reshape(1, -1) for v in vectors])
    values = _upper_triangle_pairs(block, metric_name)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), 0, float("nan")
    return float(finite.mean()), int(finite.size), float(finite.std())


def pairwise_metric_table(
    records: Sequence[PopulationRecord],
    stimuli: Sequence[str],
) -> dict[str, Any]:
    """Centroid-to-centroid distance for every stimulus pair.

    The centroid is the mean population vector across trials and bins, which
    removes trial-to-trial noise from the between-stimulus comparison. That noise
    is quantified separately by :func:`within_between`.
    """
    centroids: dict[str, np.ndarray] = {}
    for stimulus in stimuli:
        stacked = [r.features for r in records if r.stimulus == stimulus]
        if not stacked:
            continue
        centroids[stimulus] = np.mean(np.concatenate(stacked, axis=0), axis=0)

    out: dict[str, Any] = {}
    for i, first in enumerate(stimuli):
        for second in stimuli[i + 1 :]:
            if first not in centroids or second not in centroids:
                continue
            a, b = centroids[first], centroids[second]
            out[f"{first}|{second}"] = {
                name: float(fn(a, b)) for name, fn in METRICS.items()
            }
    return out


def within_between(
    records: Sequence[PopulationRecord],
    stimuli: Sequence[str],
) -> dict[str, Any]:
    """Within-stimulus spread versus between-stimulus separation.

    ``separation_ratio`` is between divided by within. Above 1 means the stimuli
    are further apart from each other than repeated presentations of the same
    stimulus are. It is a descriptive ratio, not an accuracy.

    The within-stimulus term pools every bin of every trial of every stimulus
    into one block, so it measures trial-to-trial *and* bin-to-bin spread with
    equal weight per stimulus. The between-stimulus term is the mean distance
    between stimulus centroids. The null in :func:`separation_ratio_null`
    permutes whole trials and is evaluated by :func:`fast_separation_ratio`,
    which computes the identical euclidean quantity with the identical pooling.
    """
    centroids: dict[str, np.ndarray] = {}
    pooled: dict[str, list[np.ndarray]] = {}
    for stimulus in stimuli:
        stacked = [r.features for r in records if r.stimulus == stimulus]
        if not stacked:
            continue
        pooled[stimulus] = [block for record in stacked for block in record]
        centroids[stimulus] = np.mean(np.concatenate(stacked, axis=0), axis=0)

    summary: dict[str, Any] = {}
    keys = list(centroids)
    for metric_name in METRICS:
        within_mean, n_within, within_sd = _mean_pairwise(
            [v for stimulus in keys for v in pooled[stimulus]], metric_name
        )
        between_mean, n_between, between_sd = _mean_pairwise(
            [centroids[key] for key in keys], metric_name
        )
        summary[metric_name] = {
            "within_stimulus_mean": within_mean,
            "within_stimulus_sd": within_sd,
            "within_stimulus_pairs": n_within,
            "between_stimulus_mean": between_mean,
            "between_stimulus_sd": between_sd,
            "between_stimulus_pairs": n_between,
            "separation_ratio": (
                float(between_mean / within_mean)
                if np.isfinite(within_mean)
                and within_mean not in (0.0,)
                and np.isfinite(between_mean)
                else float("nan")
            ),
        }
    return summary


def _pairwise_mean(values: np.ndarray) -> float:
    """Mean of the strictly upper triangle of a square distance matrix."""
    if values.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices(values.shape[0], k=1)
    return float(values[iu].mean())


def fast_separation_ratio(
    features: Sequence[np.ndarray], labels: Sequence[str], stimuli: Sequence[str]
) -> float:
    """Vectorised equivalent of the ``euclidean_distance`` separation ratio.

    The null in :func:`separation_ratio_null` needs this statistic hundreds of
    times per group per condition. It pools every bin of every trial of one
    stimulus into a single block, takes the mean centroid distance between
    stimuli, and divides by the mean distance between all distinct pairs of
    pooled vectors — the identical quantity the pairwise loop in
    :func:`within_between` computes for ``euclidean_distance``, so an observed
    value and a permuted value are the same statistic by construction.

    The ``(n, n, d)`` broadcast tensor that a naive implementation builds is
    never allocated; see :func:`_upper_triangle_pairs`.
    """
    blocks: list[np.ndarray] = []
    centroids: dict[str, np.ndarray] = {}
    for stimulus in stimuli:
        stacked = [f for f, label in zip(features, labels) if label == stimulus]
        if not stacked:
            continue
        block = np.concatenate(stacked, axis=0)
        centroids[stimulus] = block.mean(axis=0)
        blocks.append(block)

    keys = list(centroids)
    if len(blocks) == 0 or len(keys) < 2:
        return float("nan")
    pooled = np.vstack(blocks)
    if pooled.shape[0] < 2:
        return float("nan")
    values = _upper_triangle_pairs(pooled, "euclidean_distance")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    within = float(finite.mean())
    between_terms = [
        float(np.linalg.norm(centroids[a] - centroids[b]))
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    ]
    if not between_terms:
        return float("nan")
    between = float(np.mean(between_terms))
    if within <= 0.0:
        return float("nan")
    return between / within


def separation_ratio_null(
    records: Sequence[PopulationRecord],
    stimuli: Sequence[str],
    metric_name: str = "euclidean_distance",
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Null distribution of the separation ratio under random group assignment.

    This is the statistically valid null for "are the stimulus groups more
    distinct than arbitrary groups of the same size?", and it is not saturated:
    it is a continuous statistic.

    It matters because the *decoder* permutation null is uninformative when the
    classes form tight, well-separated clusters. Permuting labels across trials
    destroys the label mapping, but the geometry survives, so a linear model
    still recovers the true grouping and scores 1.000. That equality of accuracy
    and null is evidence of a saturated benchmark, not of a real effect.

    Permuting whole trials keeps each trial's internal structure intact and only
    reassigns which stimulus a trial is claimed to be.

    The observed statistic is computed by the *same* routine as every
    permutation, so the comparison is exact rather than approximate.
    """
    if metric_name != "euclidean_distance":
        raise ValueError("the vectorised null currently supports euclidean_distance only")
    features = [r.features for r in records]
    labels = [r.stimulus for r in records]
    if len({*labels}) < 2 or not all(labels.count(s) for s in stimuli):
        return {"status": "insufficient_groups"}

    descriptive = within_between(records, stimuli)[metric_name]
    observed = fast_separation_ratio(features, labels, stimuli)
    if not np.isfinite(observed):
        return {
            "status": "non_finite_observed",
            "observed_separation_ratio": observed,
            "note": (
                "the observed separation ratio is not finite, so there is nothing to compare a "
                "null against; the population vectors are degenerate"
            ),
        }

    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels, dtype=object)
    ratios: list[float] = []
    for _ in range(int(n_permutations)):
        shuffled = list(rng.permutation(label_array))
        value = fast_separation_ratio(features, shuffled, stimuli)
        if np.isfinite(value):
            ratios.append(float(value))

    if not ratios:
        return {
            "status": "no_finite_permutations",
            "observed_separation_ratio": observed,
        }
    array = np.asarray(ratios)
    return {
        "status": "ok",
        "metric": metric_name,
        "observed_separation_ratio": observed,
        "descriptive_within_mean": descriptive["within_stimulus_mean"],
        "descriptive_between_mean": descriptive["between_stimulus_mean"],
        "descriptive_separation_ratio": descriptive["separation_ratio"],
        "observed_within_mean": descriptive["within_stimulus_mean"],
        "observed_between_mean": descriptive["between_stimulus_mean"],
        "n_permutations": len(ratios),
        "seed": int(seed),
        "null_mean": float(array.mean()),
        "null_sd": float(array.std()),
        "null_p95": float(np.percentile(array, 95)),
        "z_score": (
            float((observed - array.mean()) / array.std())
            if array.std() > 0
            else float("nan")
        ),
        "exceeds_null_p95": bool(observed > np.percentile(array, 95)),
    }


def cross_condition_distance(
    centroids: Mapping[str, Mapping[str, np.ndarray]],
    metric_name: str = "cosine_distance",
) -> dict[str, Any]:
    """Distance between the representations two conditions produce for the same stimuli.

    A continuous, non-saturated comparison. Every condition can be perfectly good
    at separating stimuli, so decoder accuracy cannot tell them apart; this asks
    whether they produce the *same* representation, which they should not if the
    wiring matters at all.

    ``centroids`` maps condition name -> stimulus -> centroid vector.
    """
    metric = METRICS[metric_name]
    conditions = sorted(centroids)
    out: dict[str, Any] = {}
    for i, first in enumerate(conditions):
        for second in conditions[i + 1 :]:
            per_stimulus = {}
            for stimulus in centroids[first]:
                if stimulus not in centroids[second]:
                    continue
                value = metric(centroids[first][stimulus], centroids[second][stimulus])
                per_stimulus[stimulus] = float(value)
            finite = [v for v in per_stimulus.values() if np.isfinite(v)]
            out[f"{first}|{second}"] = {
                "per_stimulus": per_stimulus,
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "note": (
                    "distance between the two conditions' population representations of the same "
                    "stimulus; a large value means the wiring changed the representation"
                ),
            }
    return out


def stimulus_centroids(
    records: Sequence[PopulationRecord], stimuli: Sequence[str]
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for stimulus in stimuli:
        stacked = [r.features for r in records if r.stimulus == stimulus]
        if stacked:
            centroids[stimulus] = np.mean(np.concatenate(stacked, axis=0), axis=0)
    return centroids


def separability(records: Sequence[PopulationRecord], stimuli: Sequence[str]) -> dict[str, Any]:
    """Everything the report needs about one group's representational geometry."""
    return {
        "group": records[0].group if records else None,
        "columns": list(ABBREVIATIONS[records[0].group]) if records else [],
        "pairwise_centroid_distances": pairwise_metric_table(records, stimuli),
        "variability": within_between(records, stimuli),
        "n_records": len(records),
    }
