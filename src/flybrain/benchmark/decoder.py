"""A deliberately small linear decoder, and the split protocol that keeps it honest.

What it is for
--------------
One question only: *does this population activity pattern contain information that
distinguishes two stimulus conditions?* A multinomial logistic regression is about
the smallest model that answers that. It is linear in the activity features, has
no hidden state, and can be read end to end.

It is **not** a policy, not a navigation module, and not a classifier we would
trust to infer behaviour. High accuracy means the condition identity is linearly
recoverable from these features. It does not mean the circuit performs any
computation, and it certainly does not mean anything about a fly.

Trial identity: the bug this module used to have
------------------------------------------------
An earlier version indexed records as ``{record.trial: record}``. Trial indices run
0..N-1 **within each stimulus**, so that dict silently discarded every stimulus
except the last one. Every group then reported accuracy 1.000 with a confusion
matrix of ``[[0, 0], [0, n]]`` — a constant prediction against a single remaining
class, which is not a discrimination at all. It was read as "the benchmark is
saturated", and that misreading cost a whole experimental phase.

Two properties make it impossible to recur:

* :func:`_trial_table` keys on ``(stimulus, trial)``, never on trial alone.
* :func:`cross_validate` refuses to score a problem that does not actually contain
  both classes in the training folds, and asserts the expected test-sample count
  rather than trusting it.

Deliberate choices
------------------
* Implemented in NumPy rather than scikit-learn, so the benchmark adds no
  dependency and the whole optimiser can be audited.
* Folds are fit **as a batch** with ``bmm``, so leave-one-trial-out costs one
  NumPy call per iteration rather than one per fold per iteration. That is what
  makes a few hundred permutation nulls affordable.
* Standardisation uses training-fold statistics only, per fold.
* Zero initialisation, no shuffling, no early stopping on the held-out fold, so a
  run is bit-for-bit reproducible and the reported number is not a lucky seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DECODER_DISCLAIMER",
    "LinearDecoder",
    "confusion_matrix",
    "cv_permutation_null",
    "cross_validate",
    "decode_population",
    "permutation_null",
]

DECODER_DISCLAIMER = (
    "A linear decoder over simulated population activity. It measures whether stimulus identity is "
    "linearly recoverable from these features, nothing more. It is not a navigation policy, not a "
    "model of fly behaviour, and not evidence of direction selectivity."
)

#: Minimum trials before leave-one-trial-out is meaningful.
MIN_TRIALS_FOR_CV = 3


@dataclass
class LinearDecoder:
    """Multinomial logistic regression, full batch, zero init.

    Retained as the single-fold reference implementation. :func:`cross_validate`
    uses a batched equivalent and a test asserts the two agree.

    Parameters
    ----------
    n_features, n_classes:
        Shape of the problem.
    learning_rate, n_iterations, l2:
        Optimiser settings. Fixed, not tuned per condition, so that conditions
        are compared under identical decoder settings.
    """

    n_features: int
    n_classes: int
    learning_rate: float = 0.5
    n_iterations: int = 400
    l2: float = 1e-2
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    final_loss: float = field(init=False, default=float("nan"))

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.n_features, self.n_classes), dtype=np.float64)
        self.bias = np.zeros(self.n_classes, dtype=np.float64)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearDecoder":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected features of shape (n, {self.n_features}), got {x.shape}")
        if y.shape[0] != x.shape[0]:
            raise ValueError("features and labels disagree on trial count")
        if np.unique(y).size < 2:
            raise ValueError("refusing to fit: the training data contains fewer than two classes")

        onehot = np.zeros((y.shape[0], self.n_classes), dtype=np.float64)
        onehot[np.arange(y.shape[0]), y] = 1.0
        scale = 1.0 / max(x.shape[0], 1)
        for _ in range(self.n_iterations):
            logits = x @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            error = (probabilities - onehot) * scale
            grad_w = x.T @ error + self.l2 * self.weights
            grad_b = error.sum(axis=0)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        self.final_loss = self._loss(x, onehot)
        return self

    def _loss(self, x: np.ndarray, onehot: np.ndarray) -> float:
        logits = x @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(logits).sum(axis=1))
        return float(
            -(onehot * (logits - log_partition[:, None])).sum() / max(x.shape[0], 1)
            + 0.5 * self.l2 * float((self.weights**2).sum())
        )

    def decision(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision(features), axis=1).astype(np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.decision(features)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ indexing


def _trial_table(records: Sequence[Any], stimuli: Sequence[str]) -> dict[tuple[str, int], Any]:
    """``(stimulus, trial) -> record``.

    Keying on the **pair** is the whole point. Keying on trial alone silently
    keeps one stimulus and discards the rest, which is exactly the failure this
    module exists to prevent.
    """
    table: dict[tuple[str, int], Any] = {}
    for record in records:
        key = (str(record.stimulus), int(record.trial))
        if key in table:
            raise ValueError(
                f"duplicate record for stimulus={key[0]!r} trial={key[1]}: trial indices must be "
                f"unique within a stimulus. Two records share one key, so a split could not "
                f"hold them apart."
            )
        table[key] = record
    present = {stimulus for stimulus, _ in table}
    missing = [str(s) for s in stimuli if str(s) not in present]
    if missing:
        raise ValueError(f"no records for stimuli {missing}; present: {sorted(present)}")
    return table


def _trial_blocks(
    records: Sequence[Any],
    stimuli: Sequence[str],
    labels_per_trial: np.ndarray | None = None,
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    """One ``(features, labels)`` block per trial index, both stimuli concatenated.

    ``labels_per_trial`` overrides the stimulus recorded on each block. It exists
    only for the permutation null, where the trial-to-condition mapping is
    deliberately destroyed while the trial structure is preserved.
    """
    table = _trial_table(records, stimuli)
    trials = sorted({trial for _, trial in table})
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for position, trial in enumerate(trials):
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for stimulus in stimuli:
            record = table.get((str(stimulus), trial))
            if record is None:
                continue
            block = np.asarray(record.features, dtype=np.float64)
            if block.ndim != 2:
                raise ValueError(f"record features must be 2-D, got shape {block.shape}")
            label = (
                int(labels_per_trial[position])
                if labels_per_trial is not None
                else list(stimuli).index(str(stimulus))
            )
            features.append(block)
            labels.append(np.full(block.shape[0], label, dtype=np.int64))
        if not features:
            blocks.append((np.zeros((0, 0)), np.zeros(0, dtype=np.int64)))
        else:
            blocks.append((np.concatenate(features, axis=0), np.concatenate(labels)))
    return trials, blocks


def confusion_matrix(
    true_labels: Sequence[int], predicted: Sequence[int], n_classes: int
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(np.asarray(true_labels, dtype=np.int64), np.asarray(predicted, dtype=np.int64)):
        if 0 <= truth < n_classes and 0 <= guess < n_classes:
            matrix[truth, guess] += 1
    return matrix


# ------------------------------------------------------------ batched solver


def _fit_batch(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_classes: int,
    n_iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one multinomial logistic model per batch element.

    ``x`` is ``(K, N, d)``, ``y`` is ``(K, N)`` and ``mask`` is ``(K, N, 1)``
    marking the **real** rows. Batches are rectangular, so short folds are
    zero-padded; without the mask those padding rows would enter the softmax and
    be counted as class 0, quietly biasing the fit. Masked rows contribute to
    neither the probabilities' gradient nor the normalisation.

    Mathematically identical to :meth:`LinearDecoder.fit` run ``K`` times on the
    unmasked rows; a test asserts the two agree to floating-point tolerance.
    """
    k, n, d = x.shape
    weights = np.zeros((k, d, n_classes), dtype=np.float64)
    bias = np.zeros((k, n_classes), dtype=np.float64)
    onehot = np.zeros((k, n, n_classes), dtype=np.float64)
    np.put_along_axis(onehot, y[:, :, None], 1.0, axis=2)
    counts = np.maximum(mask.sum(axis=1), 1.0)  # (K, 1)
    scale = (mask / counts[:, :, None]).astype(np.float64)  # (K, N, 1)
    for _ in range(int(n_iterations)):
        logits = x @ weights + bias[:, None, :]
        logits -= logits.max(axis=2, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / exp.sum(axis=2, keepdims=True)
        error = (probabilities - onehot) * scale
        weights -= learning_rate * (np.swapaxes(x, 1, 2) @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=1)
    return weights, bias


def _standardise_batch(
    train: np.ndarray, test: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-fold standardisation from *masked* training statistics only.

    The mask matters: the per-fold rows are zero-padded, and an unmasked mean
    would be dragged toward zero in proportion to how much padding a fold has.

    ``constant_features`` is a **whole-design** count, not a per-fold one. It was
    originally read from ``constant[0, 0, :]``, i.e. fold 0's training block only,
    and reported as if it described the design. That made a single fold's dead
    columns stand in for all ``n_folds``, which matters here because the number
    differs sharply between conditions (13 of 32 for the measured graph against 24
    of 32 for the randomised one at 20 trials) and a reader would reasonably take
    it as an averaged property.

    The count is now the number of feature columns that are constant in **at least
    one** fold's training block, taken as a ``max`` over folds. That is
    deterministic, order-independent, computable without a second pass, and it is
    the count a decoder actually has to work around: a column constant in any
    single training fold is uninformative for that fold's fit. The
    ``constant_features_always`` companion counts columns constant in *every*
    fold, which is the stricter and more interpretable "dead for the whole
    design" figure, and both are reported so the difference is visible rather than
    guessed at.
    """
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    mean = (train * mask).sum(axis=1, keepdims=True) / counts
    centred = (train - mean) * mask
    variance = (centred**2).sum(axis=1, keepdims=True) / counts
    scale = np.sqrt(variance)
    constant = scale < 1e-12
    scale = np.where(constant, 1.0, scale)
    per_fold = constant[:, 0, :]                      # (K, d), training block only
    info = {
        "n_features": int(train.shape[2]),
        "n_folds": int(train.shape[0]),
        "constant_features": int(np.count_nonzero(per_fold.any(axis=0))),
        "constant_features_every_fold": int(np.count_nonzero(per_fold.all(axis=0))),
        "constant_features_definition": (
            "columns constant (train sd < 1e-12) in at least one fold's training block; "
            "the _every_fold companion counts columns constant in all folds"
        ),
    }
    return (train - mean) / scale, (test - mean) / scale, info


def _loo_design(
    blocks: list[tuple[np.ndarray, np.ndarray]], n_classes: int
) -> dict[str, Any]:
    """Rectangular leave-one-trial-out design: standardise once, reuse for every fit.

    Each fold holds out one trial index across **all** conditions, so a held-out
    fold always contains every class and no fold can be advantaged by which
    condition happened to be excluded.

    Returning the design rather than the score is what makes a permutation null
    affordable: permuting the trial-to-condition mapping changes only the
    **labels**, never a single feature value, so the standardised tensors, the
    masks and the fold structure are built once and reused.
    """
    usable = [i for i, (x, _) in enumerate(blocks) if x.shape[0] > 0]
    if len(usable) < MIN_TRIALS_FOR_CV:
        raise ValueError(
            f"leave-one-trial-out needs at least {MIN_TRIALS_FOR_CV} usable trials, got {len(usable)}"
        )
    widths = {blocks[i][0].shape[1] for i in usable}
    if len(widths) != 1:
        raise ValueError(f"records have inconsistent feature widths: {sorted(widths)}")
    width = widths.pop()

    train_parts, test_parts, train_labels, test_labels = [], [], [], []
    train_groups: list[list[tuple[int, int]]] = []
    test_trial: list[int] = []
    for held in usable:
        others = [i for i in usable if i != held]
        train_parts.append(np.concatenate([blocks[i][0] for i in others], axis=0))
        train_labels.append(np.concatenate([blocks[i][1] for i in others]))
        test_parts.append(blocks[held][0])
        test_labels.append(blocks[held][1])
        # Exact row layout: which trial each contiguous block of training rows is.
        # Recorded because a trial can contribute a different number of rows in
        # different folds when a condition is missing for it, and a permutation
        # must relabel whole trials rather than rows.
        train_groups.append([(int(i), int(blocks[i][0].shape[0])) for i in others])
        test_trial.append(int(usable.index(held)))

    n_folds = len(usable)
    max_train = max(p.shape[0] for p in train_parts)
    max_test = max(p.shape[0] for p in test_parts)
    train_x = np.zeros((n_folds, max_train, width), dtype=np.float64)
    train_y = np.zeros((n_folds, max_train), dtype=np.int64)
    test_x = np.zeros((n_folds, max_test, width), dtype=np.float64)
    test_y = np.zeros((n_folds, max_test), dtype=np.int64)
    train_mask = np.zeros((n_folds, max_train, 1), dtype=np.float64)
    test_mask = np.zeros((n_folds, max_test, 1), dtype=np.float64)
    for position in range(n_folds):
        a, b = train_parts[position], test_parts[position]
        train_x[position, : a.shape[0]] = a
        train_y[position, : a.shape[0]] = train_labels[position]
        train_mask[position, : a.shape[0], 0] = 1.0
        test_x[position, : b.shape[0]] = b
        test_y[position, : b.shape[0]] = test_labels[position]
        test_mask[position, : b.shape[0], 0] = 1.0

    for position in range(n_folds):
        rows = train_y[position][train_mask[position, :, 0] > 0]
        if np.unique(rows).size < 2:
            raise ValueError(
                f"training fold {position + 1} of {n_folds} contains fewer than two classes; "
                f"this problem cannot be scored for discrimination"
            )

    train_x, test_x, info = _standardise_batch(train_x, test_x, train_mask)
    return {
        "n_folds": n_folds,
        "width": width,
        "train_x": train_x,
        "train_y": train_y,
        "train_mask": train_mask,
        "train_counts": train_mask[:, :, 0].sum(axis=1).astype(np.int64),
        "train_groups": train_groups,
        "test_trial": test_trial,
        "test_x": test_x,
        "test_y": test_y,
        "test_mask": test_mask,
        "test_counts": test_mask[:, :, 0].sum(axis=1).astype(np.int64),
        "info": info,
    }


def _score_design(
    design: dict[str, Any],
    n_classes: int,
    n_iterations: int,
    learning_rate: float,
    l2: float,
    train_y: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit and score a design. ``train_y`` overrides the labels (permutations)."""
    labels = design["train_y"] if train_y is None else train_y
    weights, bias = _fit_batch(
        design["train_x"], labels, design["train_mask"],
        n_classes, n_iterations, learning_rate, l2,
    )
    logits = design["test_x"] @ weights + bias[:, None, :]
    logits -= logits.max(axis=2, keepdims=True)
    exp = np.exp(logits)
    predicted = np.argmax(exp / exp.sum(axis=2, keepdims=True), axis=2)
    valid = design["test_mask"][..., 0] > 0
    correct = (predicted == design["test_y"]) & valid
    fold = correct.sum(axis=1) / np.maximum(design["test_counts"], 1)
    return {"fold_accuracies": fold, "predicted": predicted, "correct": correct, "valid": valid}


def _pooled_confusion(
    predicted: np.ndarray, valid: np.ndarray, test_y: np.ndarray, n_classes: int
) -> np.ndarray:
    """Confusion matrix over **every** held-out sample, correct and incorrect.

    Counting only the correct rows would produce a diagonal matrix and hide
    precisely the errors a confusion matrix exists to show.
    """
    pooled = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(test_y[valid].tolist(), predicted[valid].tolist()):
        pooled[truth, guess] += 1
    return pooled


# -------------------------------------------------------------- public entry


def cross_validate(
    records: Sequence[Any],
    stimuli: Sequence[str],
    learning_rate: float = 0.5,
    n_iterations: int = 400,
    l2: float = 1e-2,
) -> dict[str, Any]:
    """Leave-one-trial-out cross-validated decode of a two-or-more-way problem.

    Returns mean and standard deviation of the per-fold accuracy, the number of
    folds, the pooled confusion matrix, and the sample counts. A single
    train/test split is deliberately **not** used: with a handful of trials one
    split is one draw, and reporting it without its spread is how an
    over-parameterised readout ends up looking like a result.
    """
    stimuli = [str(s) for s in stimuli]
    if len(stimuli) < 2:
        raise ValueError(f"need at least two stimuli to decode, got {stimuli}")
    try:
        trials, blocks = _trial_blocks(records, stimuli)
    except ValueError as exc:
        return {"status": "invalid_records", "reason": str(exc)}
    if len(trials) < MIN_TRIALS_FOR_CV:
        return {
            "status": "insufficient_trials",
            "n_trials": len(trials),
            "min_trials_required": MIN_TRIALS_FOR_CV,
            "reason": (
                "leave-one-trial-out needs at least three trials; with fewer, the training "
                "folds are too small to fit a decoder and any accuracy would be meaningless"
            ),
        }

    try:
        design = _loo_design(blocks, len(stimuli))
        solved = _score_design(
            design, len(stimuli), n_iterations, learning_rate, l2
        )
    except ValueError as exc:
        return {"status": "not_discriminable", "reason": str(exc)}

    fold = solved["fold_accuracies"]
    n_test = int(design["test_counts"].sum())
    expected = sum(
        int(getattr(r, "n_bins", 0)) for r in records if str(r.stimulus) in stimuli
    )
    pooled = _pooled_confusion(
        solved["predicted"], solved["valid"], design["test_y"], len(stimuli)
    )
    return {
        "status": "ok",
        "protocol": "leave_one_trial_out",
        "stimuli": list(stimuli),
        "n_trials": len(trials),
        "n_folds": int(design["n_folds"]),
        "n_features": int(design["info"]["n_features"]),
        "constant_features": int(design["info"]["constant_features"]),
        "constant_features_every_fold": int(design["info"]["constant_features_every_fold"]),
        "constant_features_definition": design["info"]["constant_features_definition"],
        "mean_test_accuracy": float(fold.mean()),
        "sd_test_accuracy": float(fold.std()),
        "min_fold_accuracy": float(fold.min()),
        "max_fold_accuracy": float(fold.max()),
        "fold_accuracies": [float(v) for v in fold],
        "chance_accuracy": float(1.0 / len(stimuli)),
        "n_test_samples": n_test,
        "n_test_samples_expected": int(expected),
        "sample_count_matches": bool(n_test == expected),
        "confusion_matrix": pooled.tolist(),
        "n_iterations": int(n_iterations),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "disclaimer": DECODER_DISCLAIMER,
    }


def cv_permutation_null(
    records: Sequence[Any],
    stimuli: Sequence[str],
    n_permutations: int = 200,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Null distribution of the cross-validated accuracy under relabelled trials.

    Permutes the **trial-to-condition mapping**, keeping each trial's own feature
    block and the fold structure intact. That destroys exactly the thing the
    benchmark measures and nothing else, so the null answers "what would this
    score look like if the condition labels carried no information about the
    trials?"

    Because a permutation changes only the labels, the leave-one-trial-out design
    and its standardisation are built **once** and every permutation is scored in
    a single batched fit. The observed score is produced by the identical code
    path, so observed and null are directly comparable.
    """
    stimuli = [str(s) for s in stimuli]
    n_classes = len(stimuli)
    try:
        trials, blocks = _trial_blocks(records, stimuli)
    except ValueError as exc:
        return {"status": "invalid_records", "reason": str(exc)}
    if int(n_permutations) <= 0:
        return {"status": "disabled", "n_permutations": 0}
    if len(trials) < MIN_TRIALS_FOR_CV:
        return {"status": "insufficient_trials", "n_trials": len(trials)}

    observed = cross_validate(records, stimuli, **kwargs)
    if observed.get("status") != "ok":
        return {
            "status": "not_scored",
            "observed_status": observed.get("status"),
            "reason": observed.get("reason", ""),
        }

    n_iterations = int(kwargs.get("n_iterations", 400))
    learning_rate = float(kwargs.get("learning_rate", 0.5))
    l2 = float(kwargs.get("l2", 1e-2))
    try:
        design = _loo_design(blocks, n_classes)
    except ValueError as exc:
        return {"status": "not_scored", "reason": str(exc)}

    rng = np.random.default_rng(seed)
    n_folds = int(design["n_folds"])
    n_train = int(design["train_counts"].max())
    mask = design["train_mask"]
    train_x = design["train_x"]
    test_x = design["test_x"]
    test_y = design["test_y"]
    test_mask = design["test_mask"]
    test_counts = np.maximum(design["test_counts"], 1)

    scores: list[float] = []
    skipped = 0
    # Tile the design across permutations so every permutation is one batched fit.
    # Memory is (n_permutations * n_folds, n_train, width) float64; for 32
    # permutations, 8 folds, 56 rows and 32 columns that is under 4 MB.
    for start in range(0, int(n_permutations), 32):
        take = min(32, int(n_permutations) - start)
        labels = rng.integers(0, n_classes, size=(take, len(trials)))
        good = [p for p in range(take) if np.unique(labels[p]).size >= 2]
        skipped += take - len(good)
        if not good:
            continue
        labels = labels[good]
        take = labels.shape[0]
        # Relabel whole trials: each fold's training rows inherit the permuted
        # label of the trial they came from, and padding rows are masked out.
        expanded = np.zeros((take, n_folds, n_train), dtype=np.int64)
        for fold in range(n_folds):
            cursor = 0
            for trial_position, n_rows in design["train_groups"][fold]:
                expanded[:, fold, cursor : cursor + n_rows] = labels[:, trial_position][:, None]
                cursor += n_rows
        batch_x = np.repeat(train_x[None, ...], take, axis=0).reshape(
            take * n_folds, n_train, design["width"]
        )
        batch_mask = np.repeat(mask[None, ...], take, axis=0).reshape(take * n_folds, n_train, 1)
        weights, bias = _fit_batch(
            batch_x, expanded.reshape(take * n_folds, n_train), batch_mask,
            n_classes, n_iterations, learning_rate, l2,
        )
        batch_test_x = np.repeat(test_x[None, ...], take, axis=0).reshape(
            take * n_folds, int(test_x.shape[1]), design["width"]
        )
        logits = batch_test_x @ weights + bias[:, None, :]
        logits -= logits.max(axis=2, keepdims=True)
        exp = np.exp(logits)
        predicted = np.argmax(exp / exp.sum(axis=2, keepdims=True), axis=2)
        batch_test_y = np.repeat(test_y[None, ...], take, axis=0).reshape(
            take * n_folds, int(test_x.shape[1])
        )
        batch_valid = np.repeat(test_mask[None, ..., 0], take, axis=0).reshape(
            take * n_folds, int(test_x.shape[1])
        ) > 0
        correct = (predicted == batch_test_y) & batch_valid
        counts = np.tile(test_counts, take)
        fold_scores = correct.sum(axis=1) / np.maximum(counts, 1)
        scores.extend(fold_scores.reshape(take, n_folds).mean(axis=1).tolist())

    if not scores:
        return {"status": "no_usable_permutations", "n_skipped_degenerate": skipped}
    array = np.asarray(scores)
    observed_score = float(observed["mean_test_accuracy"])
    p95 = float(np.percentile(array, 95))
    return {
        "status": "ok",
        "protocol": "leave_one_trial_out",
        "null_of": "trial_to_condition_mapping",
        "n_permutations": int(array.size),
        "n_skipped_degenerate": int(skipped),
        "seed": int(seed),
        "observed_mean_test_accuracy": observed_score,
        "null_mean": float(array.mean()),
        "null_sd": float(array.std()),
        "null_p95": p95,
        "null_max": float(array.max()),
        "z_score": (
            float((observed_score - array.mean()) / array.std()) if array.std() > 0 else float("nan")
        ),
        "exceeds_null_p95": bool(observed_score > p95),
        "note": (
            "labels permuted across whole trials, fold structure and feature blocks preserved; "
            "a real effect should sit above this band"
        ),
    }


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Standardise using training statistics only."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    info = {
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "n_features": int(train.shape[1]),
        "constant_features": int(np.count_nonzero(train.std(axis=0) < 1e-12)),
    }
    return (train - mean) / scale, (test - mean) / scale, info


def decode_population(
    records: Sequence[Any],
    train_trials: Sequence[int],
    test_trials: Sequence[int],
    stimuli: Sequence[str],
    learning_rate: float = 0.5,
    n_iterations: int = 400,
    l2: float = 1e-2,
) -> dict[str, Any]:
    """Single fixed-split decode. Correct, but a single split is one draw.

    Retained because a fixed split is easy to reason about and is used as the
    cross-check for the batched solver. **Prefer :func:`cross_validate` for any
    reported number**: with a handful of trials, one split carries no information
    about its own uncertainty.
    """
    stimuli = [str(s) for s in stimuli]
    if len(stimuli) < 2:
        raise ValueError(f"need at least two stimuli to decode, got {stimuli}")
    try:
        table = _trial_table(records, stimuli)
    except ValueError as exc:
        # Refuse rather than raise: a missing condition must not abort a run.
        return {"status": "invalid_records", "reason": str(exc)}
    train_trials = [
        t for t in sorted({int(t) for t in train_trials})
        if any((s, t) in table for s in stimuli)
    ]
    test_trials = [
        t for t in sorted({int(t) for t in test_trials})
        if any((s, t) in table for s in stimuli)
    ]
    if not train_trials or not test_trials:
        return {
            "status": "insufficient_trials",
            "train_trials": train_trials,
            "test_trials": test_trials,
        }

    def build(trials: list[int]) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for trial in trials:
            for stimulus in stimuli:
                record = table.get((stimulus, trial))
                if record is None:
                    continue
                features.append(np.asarray(record.features, dtype=np.float64))
                labels.append(np.full(record.features.shape[0], stimuli.index(stimulus), dtype=np.int64))
        if not features:
            return np.zeros((0, 0)), np.zeros(0, dtype=np.int64)
        return np.concatenate(features, axis=0), np.concatenate(labels)

    train_x, train_y = build(train_trials)
    test_x, test_y = build(test_trials)
    if np.unique(train_y).size < 2:
        return {
            "status": "not_discriminable",
            "reason": (
                "the training split contains fewer than two conditions, so any accuracy would be "
                "a constant prediction rather than a discrimination"
            ),
        }
    train_x, test_x, info = _standardise(train_x, test_x)

    decoder = LinearDecoder(
        n_features=train_x.shape[1],
        n_classes=len(stimuli),
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        l2=l2,
    )
    decoder.fit(train_x, train_y)
    train_pred = decoder.predict(train_x)
    test_pred = decoder.predict(test_x)

    return {
        "status": "ok",
        "protocol": "fixed_split",
        **info,
        "train_trials": train_trials,
        "test_trials": test_trials,
        "stimuli": list(stimuli),
        "train_accuracy": float((train_pred == train_y).mean()),
        "test_accuracy": float((test_pred == test_y).mean()),
        "n_test_samples": int(test_y.shape[0]),
        "chance_accuracy": float(1.0 / max(len(stimuli), 1)),
        "confusion_matrix": confusion_matrix(test_y, test_pred, len(stimuli)).tolist(),
        "final_loss": float(decoder.final_loss),
        "disclaimer": DECODER_DISCLAIMER,
    }


def permutation_null(
    records: Sequence[Any],
    train_trials: Sequence[int],
    test_trials: Sequence[int],
    stimuli: Sequence[str],
    n_permutations: int = 20,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Refit the fixed-split decoder on relabelled trials, as a chance reference.

    Labels are permuted **across trials**, keeping each trial's block intact.
    Prefer :func:`cv_permutation_null`, which scores the same statistic the
    benchmark reports.
    """
    if int(n_permutations) <= 0:
        return {"status": "disabled", "n_permutations": 0}
    stimuli = [str(s) for s in stimuli]
    try:
        table = _trial_table(records, stimuli)
    except ValueError as exc:
        return {"status": "invalid_records", "reason": str(exc)}
    train_trials = [
        t for t in sorted({int(t) for t in train_trials})
        if any((s, t) in table for s in stimuli)
    ]
    test_trials = [
        t for t in sorted({int(t) for t in test_trials})
        if any((s, t) in table for s in stimuli)
    ]
    if not train_trials or not test_trials:
        return {"status": "insufficient_trials"}

    def collect(trials: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
        """One block per (trial, stimulus) that exists, plus its condition label."""
        blocks: list[np.ndarray] = []
        labels: list[int] = []
        for trial in trials:
            for stimulus in stimuli:
                record = table.get((stimulus, trial))
                if record is None:
                    continue
                blocks.append(np.asarray(record.features, dtype=np.float64))
                labels.append(stimuli.index(stimulus))
        return blocks, np.asarray(labels, dtype=np.int64)

    train_blocks, train_labels = collect(train_trials)
    test_blocks, test_labels = collect(test_trials)
    if not train_blocks or not test_blocks:
        return {"status": "insufficient_trials"}
    if np.unique(train_labels).size < 2:
        return {
            "status": "not_discriminable",
            "reason": "the training split contains fewer than two conditions",
        }

    train_x, test_x, _ = _standardise(
        np.concatenate(train_blocks, axis=0), np.concatenate(test_blocks, axis=0)
    )
    # One label per trial, so a permutation moves whole trials and never splits one.
    per_trial = train_labels.reshape(len(train_trials), -1)[:, 0]
    test_true = np.concatenate(
        [np.full(block.shape[0], label, dtype=np.int64)
         for block, label in zip(test_blocks, test_labels)]
    )

    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    for _ in range(int(n_permutations)):
        shuffled = np.repeat(rng.permutation(per_trial), train_labels.reshape(len(train_trials), -1).shape[1])
        decoder = LinearDecoder(
            n_features=train_x.shape[1],
            n_classes=len(stimuli),
            learning_rate=float(kwargs.get("learning_rate", 0.5)),
            n_iterations=int(kwargs.get("n_iterations", 400)),
            l2=float(kwargs.get("l2", 1e-2)),
        )
        try:
            decoder.fit(train_x, shuffled)
        except ValueError:
            continue
        accuracies.append(float((decoder.predict(test_x) == test_true).mean()))

    if not accuracies:
        return {"status": "no_usable_permutations"}
    return {
        "status": "ok",
        "protocol": "fixed_split",
        "n_permutations": len(accuracies),
        "seed": int(seed),
        "mean_test_accuracy": float(np.mean(accuracies)),
        "sd_test_accuracy": float(np.std(accuracies)),
        "max_test_accuracy": float(np.max(accuracies)),
        "note": (
            "stimulus labels shuffled across trials, block structure kept; a real decode should sit "
            "above this band"
        ),
    }
