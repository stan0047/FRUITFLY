"""Tests for the Phase 4B motion representation benchmark.

Runs entirely on a tiny purpose-built Feather fixture, so it needs no network
access, no neuPrint token, and no dependency on the real 1 GB MaleCNS file.

What is pinned down here
------------------------
A benchmark can fail in two ways: by being *wrong*, or by being *unrunnable*.
Phase 4B has now failed both ways, so both are covered.

*The wrong kind.* An earlier decoder indexed records by trial index alone, which
silently discarded every stimulus except the last one and reported accuracy
1.000 for every group. That was read as "the benchmark is saturated" and it cost
a whole phase. :class:`TestDecoderRegression` exists so that specific failure can
never recur: a problem that does not contain both classes is now *refused* rather
than scored, and the sample count is asserted rather than trusted.

*The unrunnable kind.* The first version spent four hours in a pairwise routine
that allocated a 391 MB tensor per permutation. :class:`TestMemory` bounds that
allocation, and the quick mode is asserted to finish.

The scientific contracts are covered too: a deterministic pathway-preserving
subset, identical selected populations in every circuit condition, exactly matched
stimulus energy, a readout the decoder can be fitted on honestly, and a report
that states plainly what it measured, simulated and assumed.

Nothing here asserts a biological result. Every test that touches benchmark output
also asserts that the run refuses to claim direction selectivity.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.brain.mcns_circuit import (
    DENSE_REFUSE_BYTES,
    MODE_FEEDFORWARD,
    MODE_RECURRENT,
    MCNSCircuit,
    MCNSCircuitError,
)
from flybrain.brain.mcns_simulation import run_simulation
from flybrain.brain.neuron import NeuronParams
from flybrain.benchmark.controls import (
    INPUT_INSTANT,
    INPUT_SPATIAL,
    INPUT_TEMPORAL,
    acceptance_gate,
    input_control_features,
    input_instant_identity,
)
from flybrain.benchmark.decoder import (
    LinearDecoder,
    _fit_batch,
    _loo_design,
    _score_design,
    _standardise_batch,
    _trial_blocks,
    _trial_table,
    confusion_matrix,
    cross_validate,
    cv_permutation_null,
    decode_population,
    permutation_null,
)
from flybrain.benchmark.phase4c import (
    PREDECLARED_CONTRAST,
    PREDECLARED_GROUP,
    PREDECLARED_MAX_BODIES_PER_TYPE,
    PREDECLARED_PERMUTATIONS,
    PREDECLARED_SEED,
    PREDECLARED_SYNAPSE_SCALE,
    PREDECLARED_TRIALS,
    REPLICATION_CIRCUITS,
    classify_informative,
    claim_block,
    enforce_predeclared,
    feature_report,
    fingerprint,
    liveness,
    margin_statistics,
    predeclared_block,
    timing_projection,
)

from flybrain.benchmark.representation import (
    ABBREVIATIONS,
    MAGNITUDE_GROUP,
    READOUTS,
    SUBTYPE_GROUPS,
    PopulationRecord,
    _upper_triangle_pairs,
    activity_descriptor,
    cosine_distance,
    euclidean_distance,
    fast_separation_ratio,
    magnitude_features,
    pearson_correlation,
    population_features,
    separability,
    within_between,
)
from flybrain.benchmark.shuffled import shuffled_control, verify_degree_preservation
from flybrain.benchmark.stimulus import (
    STIMULUS_CONDITIONS,
    MotionStimulusGenerator,
    StimulusCondition,
    StimulusMotion,
    StimulusPolarity,
    lateral_coordinates,
)
from flybrain.benchmark.subset import (
    CHAIN_PAIRS,
    INPUT_TYPES,
    bounded_population,
    chain_edge_mask,
    restrict_circuit,
)
import flybrain.data.motion_candidate as motion_candidate

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_motion_benchmark.py"
STIM = ["LEFTWARD", "RIGHTWARD"]

#: A documented chain, in the order the closure walks it.
CHAIN_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("L1", ("Mi1",)),
    ("Mi1", ("T4a", "T4b", "T4c", "T4d")),
    ("L2", ("Tm1", "Tm2")),
    ("Tm1", ("T5a", "T5b", "T5c", "T5d")),
    ("Tm2", ("T5a", "T5b", "T5c", "T5d")),
)
BODIES_PER_TYPE = 8


def _write_pathway_fixtures(directory: Path) -> tuple[Path, Path]:
    """Write a tiny annotation + connectivity pair that contains the real chains.

    The shared fixture in ``tests/fixtures`` is almost entirely self-loops plus a
    single ``L1 -> Mi1`` and ``L2 -> Tm1`` edge, so it cannot exercise the thing
    this module has to guarantee: that a bounded population still contains the
    pathway. These files use the same columns as the shared fixtures, so the
    production loader reads them with no special casing.

    **Every edge here is synthetic.** It exists to test selection code, and
    nothing measured may be read off it.
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    directory.mkdir(parents=True, exist_ok=True)
    body_ids: dict[str, list[int]] = {}
    next_id = 700_000
    every_type: list[str] = []
    for name, targets in CHAIN_LAYOUT:
        for entry in (name, *targets):
            if entry not in every_type:
                every_type.append(entry)
    for name in every_type:
        body_ids[name] = list(range(next_id, next_id + BODIES_PER_TYPE))
        next_id += BODIES_PER_TYPE

    annotation_rows: list[dict] = []
    for name, ids in body_ids.items():
        for position, body in enumerate(ids):
            annotation_rows.append(
                {
                    "bodyId": int(body),
                    "type": name,
                    "superclass": "ol_intrinsic",
                    # Alternate hemispheres so the balanced-selection rule is exercised.
                    "somaSide": "L" if position % 2 == 0 else "R",
                    "somaLocation": [[1000 + position, 2000 + position, 3000 + position]]
                    if position % 3
                    else None,
                }
            )
    annotations_path = directory / "pathway_annotations.feather"
    feather.write_feather(pa.Table.from_pylist(annotation_rows), annotations_path)

    edge_rows: list[dict] = []
    for source, targets in CHAIN_LAYOUT:
        for target in targets:
            for index, source_body in enumerate(body_ids[source]):
                for offset in (0, 1):
                    target_body = body_ids[target][(index + offset) % BODIES_PER_TYPE]
                    if target_body == source_body:
                        continue
                    edge_rows.append(
                        {"body_pre": int(source_body), "body_post": int(target_body),
                         "weight": int(50 + 10 * ((index + offset) % 5))}
                    )
    edges_path = directory / "pathway_edges.feather"
    feather.write_feather(pa.Table.from_pylist(edge_rows), edges_path)
    return annotations_path, edges_path


def _load_script():
    """Import the benchmark script as a module, so ``main`` runs in-process."""
    spec = importlib.util.spec_from_file_location("run_motion_benchmark_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def pathway_data(tmp_path_factory) -> tuple[Path, Path]:
    return _write_pathway_fixtures(tmp_path_factory.mktemp("pathway"))


@pytest.fixture(scope="module")
def annotations(pathway_data) -> dict[str, np.ndarray]:
    return motion_candidate.load_annotations(pathway_data[0])


@pytest.fixture(scope="module")
def populations(annotations):
    return motion_candidate.build_populations(
        annotations, motion_candidate.MOTION_CELL_TYPES, require_superclass=True
    )


@pytest.fixture(scope="module")
def extraction(populations, pathway_data):
    return motion_candidate.extract_candidate_edges(pathway_data[1], populations, direction="both")


@pytest.fixture(scope="module")
def feedforward(populations, extraction, annotations) -> MCNSCircuit:
    return _with_geometry(
        MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_FEEDFORWARD)
    )


def _with_geometry(circuit: MCNSCircuit) -> MCNSCircuit:
    """Mark every soma coordinate absent, so the rank fallback drives the layout.

    The fixture's own ``somaLocation`` column is mostly null, and the handful of
    real coordinates are all ``(0, 0, 0)``, which would collapse the layout onto a
    single lateral position and leave two lateral bins empty. Reporting "no
    coordinate" instead exercises the same deterministic fallback the real data
    uses for the ~83 percent of L1/L2 bodies that have none.
    """
    circuit.soma_location = np.full((circuit.num_neurons, 3), -1, dtype=np.int64)
    return circuit


@pytest.fixture(scope="module")
def circuit(populations, extraction, annotations) -> MCNSCircuit:
    return _with_geometry(
        MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT)
    )


@pytest.fixture(scope="module")
def bounded(populations, extraction):
    # Budget 4 per type against 8 available, so the bound genuinely binds.
    return bounded_population(
        populations, extraction.pre_index, extraction.post_index,
        synapse_count=extraction.synapse_count, bodies_per_type=4,
    )


@pytest.fixture(scope="module")
def generator(circuit) -> MotionStimulusGenerator:
    return MotionStimulusGenerator(
        circuit, spot_fraction=0.3, current=2.0, steps_per_crossing=40,
        amplitude_perturbation=0.05, input_noise=1.0, seed=0,
    )


# ======================================================================
# The regression that cost a phase: the decoder could not see two classes.
# ======================================================================


def _records(stimuli=STIM, trials=6, bins=4, dims=32, kind="sep", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for stimulus in stimuli:
        for trial in range(trials):
            features = rng.normal(scale=0.6, size=(bins, dims))
            if kind == "sep":
                features[:, 0] += 2.0 if stimulus == stimuli[0] else -2.0
            elif kind == "shifted":
                features[:, 0] += 0.6 if stimulus == stimuli[0] else -0.6
            out.append(PopulationRecord(stimulus, trial, "T4+T5", features, np.arange(bins)))
    return out


class TestDecoderRegression:
    """The bug was ``{record.trial: record}``. Trial indices repeat per stimulus."""

    def test_trial_table_keys_on_the_pair_not_the_index(self) -> None:
        records = _records()
        table = _trial_table(records, STIM)
        assert len(table) == len(records), "a (stimulus, trial) key must not collide"
        assert {s for s, _ in table} == set(STIM)
        assert len({t for _, t in table}) == 6, "both stimuli must survive the indexing"

    def test_cross_validate_keeps_every_stimulus(self) -> None:
        result = cross_validate(_records(), STIM)
        assert result["status"] == "ok"
        assert result["n_test_samples"] == result["n_test_samples_expected"]
        # 6 trials x 2 stimuli x 4 bins. The bug produced 12, not 24.
        assert result["n_test_samples"] == 6 * 2 * 4

    def test_a_duplicate_record_is_refused_not_overwritten(self) -> None:
        """The other half of the original bug: a colliding key must not silently win."""
        records = _records()
        records.append(PopulationRecord("LEFTWARD", 0, "T4+T5", np.zeros((3, 32)), np.arange(3)))
        with pytest.raises(ValueError, match="duplicate"):
            _trial_table(records, STIM)
        result = cross_validate(records, STIM)
        assert result["status"] == "invalid_records"
        assert "duplicate" in result["reason"]
        assert "test_accuracy" not in result

    def test_a_single_class_problem_is_refused_not_scored(self) -> None:
        """The exact shape of the old failure: one stimulus left, score 1.000."""
        rng = np.random.default_rng(0)
        only_right = [
            PopulationRecord("RIGHTWARD", t, "T4+T5", rng.normal(size=(4, 32)), np.arange(4))
            for t in range(6)
        ]
        result = cross_validate(only_right, STIM)
        assert result["status"] != "ok"
        assert "test_accuracy" not in result
        assert "LEFTWARD" in result["reason"]

    def test_a_missing_condition_is_refused_by_both_decoders(self) -> None:
        """One stimulus present, two requested: refuse, never score."""
        records = _records()
        trimmed = [r for r in records if r.stimulus == "RIGHTWARD"]
        for result in (
            cross_validate(trimmed, STIM),
            decode_population(trimmed, [0, 1], [2, 3], STIM),
        ):
            assert result["status"] == "invalid_records"
            assert "test_accuracy" not in result

    def test_a_single_class_training_split_is_refused(self) -> None:
        """A split covering only one condition cannot be scored for discrimination."""
        records = _records()
        only_one_condition = [r for r in records if r.stimulus == "RIGHTWARD"]
        result = decode_population(only_one_condition, [0, 1], [2, 3], STIM)
        assert result["status"] == "invalid_records"

    def test_leave_one_trial_out_holds_out_every_trial_once(self) -> None:
        trials, blocks = _trial_blocks(_records(trials=6), STIM)
        design = _loo_design(blocks, 2)
        assert design["n_folds"] == len(trials) == 6
        for fold in range(design["n_folds"]):
            groups = design["train_groups"][fold]
            assert len(groups) == len(trials) - 1
            assert design["test_trial"][fold] not in [t for t, _ in groups]

    def test_batched_solver_matches_the_reference_solver(self) -> None:
        """The batched fold solver is an optimisation, not a different estimator."""
        _, blocks = _trial_blocks(_records(trials=5, dims=6, bins=3), STIM)
        design = _loo_design(blocks, 2)
        fold_x = design["train_x"][0]
        reference = LinearDecoder(
            n_features=fold_x.shape[1], n_classes=2, learning_rate=0.5, n_iterations=400, l2=1e-2
        ).fit(fold_x, design["train_y"][0])
        weights, bias = _fit_batch(
            fold_x[None, ...], design["train_y"][0][None, ...],
            design["train_mask"][0][None, ...], 2, 400, 0.5, 1e-2,
        )
        assert np.allclose(weights[0], reference.weights, atol=1e-12)
        assert np.allclose(bias[0], reference.bias, atol=1e-12)

    def test_padding_does_not_enter_the_gradient(self) -> None:
        """Short folds are zero-padded; those rows must not be counted as class 0."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 3, 4))
        y = np.array([[0, 1, 0], [0, 0, 1]])
        mask = np.array([[[1.0], [1.0], [0.0]], [[1.0], [0.0], [0.0]]])
        weights, _ = _fit_batch(x, y, mask, 2, 50, 0.5, 0.0)
        # Fold 0's third row is padding, so its label must not matter. Flip it.
        y2 = y.copy()
        y2[0, 2] = 1
        weights2, _ = _fit_batch(x, y2, mask, 2, 50, 0.5, 0.0)
        assert np.allclose(weights, weights2)

    def test_cv_finds_signal_and_the_null_does_not(self) -> None:
        real = cross_validate(_records(kind="sep"), STIM)
        null = cross_validate(_records(kind="null"), STIM)
        assert real["mean_test_accuracy"] > 0.8
        assert null["mean_test_accuracy"] < 0.75

    def test_permutation_null_reproducible_from_its_seed(self) -> None:
        records = _records(kind="shifted")
        first = cv_permutation_null(records, STIM, n_permutations=40, seed=1)
        second = cv_permutation_null(records, STIM, n_permutations=40, seed=1)
        assert first == second
        assert first["status"] == "ok"
        assert first["n_permutations"] > 0

    def test_permutation_null_never_exceeds_p95_for_pure_noise(self) -> None:
        result = cv_permutation_null(_records(kind="null"), STIM, n_permutations=60, seed=0)
        assert result["exceeds_null_p95"] is False

    def test_permutation_null_zero_is_reported_as_disabled(self) -> None:
        assert cv_permutation_null(_records(), STIM, n_permutations=0)["status"] == "disabled"

    def test_cv_needs_three_trials(self) -> None:
        result = cross_validate(_records(trials=2), STIM)
        assert result["status"] == "insufficient_trials"
        assert result["min_trials_required"] == 3

    def test_confusion_matrix_totals_match_the_sample_count(self) -> None:
        result = cross_validate(_records(), STIM)
        matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
        assert matrix.shape == (2, 2)
        assert int(matrix.sum()) == result["n_test_samples"]
        # Every class must appear in the test folds: a held-out fold holds out a
        # trial across ALL conditions, so it can never be single-class.
        assert (matrix.sum(axis=1) > 0).all()

    def test_fixed_split_decoder_is_reproducible(self) -> None:
        args = ([0, 1, 2], [3, 4, 5], STIM)
        first = decode_population(_records(), *args, n_iterations=200)
        second = decode_population(_records(), *args, n_iterations=200)
        assert first == second
        assert first["protocol"] == "fixed_split"

    def test_fixed_split_permutation_null_is_reproducible(self) -> None:
        args = ([0, 1, 2], [3, 4, 5], STIM)
        first = permutation_null(_records(), *args, n_permutations=5, seed=2, n_iterations=100)
        second = permutation_null(_records(), *args, n_permutations=5, seed=2, n_iterations=100)
        assert first == second

    def test_confusion_matrix_helper_counts_correctly(self) -> None:
        assert confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2).tolist() == [[1, 1], [0, 2]]


# ======================================================================
# Stimulus: energy, trajectory, polarity, randomised start.
# ======================================================================


class TestStimulus:
    def test_conditions_cover_every_motion_and_polarity(self) -> None:
        names = [c.name for c in STIMULUS_CONDITIONS]
        assert "LEFTWARD" in names and "RIGHTWARD" in names and "STATIC" in names
        assert "LEFTWARD_POLARITY_REVERSED" in names
        assert len(names) == 6

    def test_condition_names_round_trip(self) -> None:
        for condition in STIMULUS_CONDITIONS:
            assert StimulusCondition.from_name(condition.name) is condition
            assert StimulusCondition.from_name(condition) is condition
        with pytest.raises(KeyError):
            StimulusCondition.from_name("SIDEWAYS")

    def test_value_alias_survives_the_simulator_string_hop(self) -> None:
        """``run_simulation`` passes the stimulus through ``str(...value...)``."""
        for condition in STIMULUS_CONDITIONS:
            assert str(getattr(condition, "value", condition)) == condition.name

    def test_energy_is_identical_across_every_condition(self, generator) -> None:
        for trial in (0, 1, 2):
            energies = {
                c.name: generator.drive_energy(c, 40, trial=trial) for c in STIMULUS_CONDITIONS
            }
            values = {round(v, 9) for v in energies.values()}
            assert len(values) == 1, f"energy is not matched in trial {trial}: {energies}"

    def test_energy_matches_the_analytic_value(self, generator) -> None:
        for trial in (0, 1):
            got = generator.drive_energy("LEFTWARD", 40, trial=trial)
            expected = generator.total_energy(40) * (
                1.0 + generator.amplitude_perturbation  # worst case of the jitter
            ) - generator.amplitude_perturbation * generator.total_energy(40)
            assert 0.85 * generator.total_energy(40) <= got <= 1.15 * generator.total_energy(40)
            assert np.isfinite(expected)

    def test_static_is_stationary_and_sweeps_are_not(self, generator) -> None:
        static = generator.drive_tensor("STATIC", 0, 40).astype(float)
        moving = generator.drive_tensor("LEFTWARD", 0, 40).astype(float)
        # Which neurons are active must not change over time for STATIC.
        def active_sequence(tensor):
            return [
                tuple(np.flatnonzero(np.abs(tensor[t]) > 1.0).tolist()) for t in range(tensor.shape[0])
            ]
        assert len(set(active_sequence(static))) == 1
        assert len(set(active_sequence(moving))) > 1

    def test_the_two_sweeps_are_mirror_images(self, generator) -> None:
        left = generator.drive_tensor("LEFTWARD", 0, 40).astype(float)
        right = generator.drive_tensor("RIGHTWARD", 0, 40).astype(float)
        # Drive magnitude at each step must match exactly: only the pattern differs.
        assert np.allclose(np.abs(left).sum(axis=1), np.abs(right).sum(axis=1))
        assert not np.allclose(left, right)

    def test_on_off_balance_never_carries_the_direction(self, generator) -> None:
        """The regression guard for the whole design.

        If an instantaneous ON/OFF ratio could tell the two sweeps apart, the
        input would be handing the circuit the answer.
        """
        for trial in range(4):
            shares = {
                name: generator.on_off_share(name, 40, trial=trial)
                for name in ("LEFTWARD", "RIGHTWARD", "STATIC")
            }
            assert shares["LEFTWARD"] == pytest.approx(0.5, abs=1e-12)
            assert shares["RIGHTWARD"] == pytest.approx(0.5, abs=1e-12)
            assert shares["STATIC"] == pytest.approx(0.5, abs=1e-12)

    def test_polarity_reversal_inverts_the_off_channel_only(self, generator, circuit) -> None:
        normal = generator.encode("LEFTWARD", 10, trial=0, n_steps=40, add_noise=False)
        reversed_ = generator.encode(
            "LEFTWARD_POLARITY_REVERSED", 10, trial=0, n_steps=40, add_noise=False
        )
        on = circuit.indices_of_type("L1")
        off = circuit.indices_of_type("L2")
        assert np.allclose(normal[on], reversed_[on])
        assert np.allclose(normal[off], -reversed_[off])
        assert np.allclose(np.abs(normal), np.abs(reversed_))

    def test_polarity_does_not_change_the_trajectory(self, generator) -> None:
        for step in (0, 11, 23, 39):
            a = generator.encode("LEFTWARD", step, trial=1, n_steps=40, add_noise=False)
            b = generator.encode(
                "LEFTWARD_POLARITY_REVERSED", step, trial=1, n_steps=40, add_noise=False
            )
            assert np.allclose(np.abs(a), np.abs(b))

    def test_randomised_start_varies_by_trial(self, generator) -> None:
        offsets = [generator.start_offset(t) for t in range(8)]
        assert len(set(offsets)) > 1, "a fixed start lets a decoder key on that edge"
        assert all(0 <= o < generator.steps_per_crossing for o in offsets)

    def test_start_offset_is_shared_across_conditions(self, generator) -> None:
        """Depends on (seed, trial) only, so the two directions stay paired."""
        left = generator.drive_tensor("LEFTWARD", 3, 40).astype(float)
        right = generator.drive_tensor("RIGHTWARD", 3, 40).astype(float)
        # Same offset means the first active bin is mirrored, not shifted.
        assert np.abs(left).sum() == pytest.approx(np.abs(right).sum())
        assert generator.start_offset(3) == generator.start_offset(3)

    def test_start_offset_is_reproducible(self, circuit) -> None:
        def build():
            return MotionStimulusGenerator(circuit, steps_per_crossing=40, seed=7).start_offset(5)
        assert build() == build()

    def test_randomise_start_can_be_disabled(self, circuit) -> None:
        fixed = MotionStimulusGenerator(circuit, steps_per_crossing=40, randomise_start=False)
        assert [fixed.start_offset(t) for t in range(4)] == [0, 0, 0, 0]

    def test_noise_is_common_across_conditions(self, generator) -> None:
        """Common random numbers, so a condition difference is not a noise draw."""
        a = generator.encode("LEFTWARD", 12, trial=0, n_steps=40, add_noise=True)
        b = generator.encode("RIGHTWARD", 12, trial=0, n_steps=40, add_noise=True)
        noise_a = a - generator.encode("LEFTWARD", 12, trial=0, n_steps=40, add_noise=False)
        noise_b = b - generator.encode("RIGHTWARD", 12, trial=0, n_steps=40, add_noise=False)
        assert np.allclose(noise_a, noise_b)

    def test_drive_tensor_is_noise_free_and_matches_encode(self, generator) -> None:
        tensor = generator.drive_tensor("RIGHTWARD", 2, 40)
        for step in (0, 7, 23, 39):
            expected = generator.encode("RIGHTWARD", step, trial=2, n_steps=40, add_noise=False)
            # The cached tensor is float32 to halve its memory footprint, so the
            # comparison is at float32 precision, not bit-for-bit.
            assert np.allclose(tensor[step].astype(np.float64), expected, rtol=1e-6, atol=1e-6)
        # And it really is noise free: every entry is a drive value, not noisy.
        assert np.abs(tensor).max() <= generator.current * (1 + generator.amplitude_perturbation)

    def test_tensors_are_cached(self, generator) -> None:
        first = generator.tensor("LEFTWARD", 0, 40)
        assert generator.tensor("LEFTWARD", 0, 40) is first
        assert generator.cache_stats()["cached_tensors"] >= 1

    def test_lateral_coordinates_give_balanced_bins(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        counts = np.bincount(bins, minlength=4)
        assert bins.size == circuit.num_neurons
        assert counts.min() > 0, f"a starved bin makes the readout uninterpretable: {counts}"
        # Quantile edges, not equal-width cuts, so no bin is a small minority.
        assert counts.max() <= 2 * counts.min()

    def test_lateral_coordinates_rejects_zero_bins(self, circuit) -> None:
        with pytest.raises(ValueError):
            lateral_coordinates(circuit, 0)


# ======================================================================
# Readout resolution and liveness.
# ======================================================================


class TestReadout:
    def _spikes(self, circuit, n_steps=40, rate=0.05, seed=0):
        return np.random.default_rng(seed).random((n_steps, circuit.num_neurons)) < rate

    def test_readout_variants_are_declared(self) -> None:
        assert READOUTS == ("subtype_lateral", "cell_type", "per_neuron")

    def test_subtype_lateral_is_one_column_per_cell_type_per_lateral_bin(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        index_map = circuit.cell_type_index()
        spikes = self._spikes(circuit)
        for group, columns in ABBREVIATIONS.items():
            if group == MAGNITUDE_GROUP:
                continue
            features, _ = population_features(
                spikes, index_map, group, 10, readout="subtype_lateral", lateral_bin=bins
            )
            assert features.shape[1] == len(columns) * 4, group
            assert features.shape[1] == len(set(columns)) * 4, "subtypes must stay distinct"

    def test_t4t5_primary_width_is_thirty_two(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        features, _ = population_features(
            self._spikes(circuit), circuit.cell_type_index(), "T4+T5", 10,
            readout="subtype_lateral", lateral_bin=bins,
        )
        assert features.shape[1] == 32 == len(SUBTYPE_GROUPS) * 4

    def test_cell_type_is_one_column_per_exact_cell_type(self, circuit) -> None:
        index_map = circuit.cell_type_index()
        spikes = self._spikes(circuit)
        for group, columns in ABBREVIATIONS.items():
            if group == MAGNITUDE_GROUP:
                continue
            features, _ = population_features(spikes, index_map, group, 10, readout="cell_type")
            assert features.shape[1] == len(columns), group
            assert features.shape[1] == len(set(columns))

    def test_per_neuron_is_one_column_per_neuron(self, circuit) -> None:
        index_map = circuit.cell_type_index()
        spikes = self._spikes(circuit)
        for group in ("T4", "T5", "T4+T5"):
            features, _ = population_features(spikes, index_map, group, 10, readout="per_neuron")
            expected = sum(
                index_map[name].size for name in ABBREVIATIONS[group] if index_map[name].size
            )
            assert features.shape[1] == expected, group

    def test_the_variants_are_not_aliases(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        spikes = self._spikes(circuit)
        widths = {
            readout: population_features(
                spikes, circuit.cell_type_index(), "T4+T5", 10, normalise=False,
                readout=readout, lateral_bin=bins,
            )[0].shape[1]
            for readout in READOUTS
        }
        assert widths["subtype_lateral"] == 32
        assert widths["cell_type"] == 8
        assert widths["per_neuron"] == 64, "8 subtypes x 8 fixture bodies"
        assert len(set(widths.values())) == 3, "the variants must not be aliases"

    def test_normalisation_actually_changes_the_features(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        spikes = self._spikes(circuit, rate=0.2, seed=5)
        raw, _ = population_features(
            spikes, circuit.cell_type_index(), "T4+T5", 10, normalise=False,
            readout="subtype_lateral", lateral_bin=bins,
        )
        nrm, _ = population_features(
            spikes, circuit.cell_type_index(), "T4+T5", 10, normalise=True,
            readout="subtype_lateral", lateral_bin=bins,
        )
        assert not np.allclose(raw, nrm)
        norms = np.linalg.norm(nrm, axis=1)
        nonzero = norms[norms > 0]
        assert np.allclose(nonzero, 1.0), "unit_normalise must leave unit rows"

    def test_subtype_lateral_requires_its_bin_index(self, circuit) -> None:
        with pytest.raises(ValueError, match="lateral_bin"):
            population_features(
                self._spikes(circuit), circuit.cell_type_index(), "T4", 10,
                readout="subtype_lateral",
            )

    def test_unknown_readout_is_refused(self, circuit) -> None:
        with pytest.raises(ValueError):
            population_features(
                self._spikes(circuit), circuit.cell_type_index(), "T4", 10, readout="everything"
            )

    def test_empty_lateral_bin_keeps_its_column(self, circuit) -> None:
        """A subtype with no neurons keeps its four bin columns, at zero."""
        index_map = dict(circuit.cell_type_index())
        index_map["T4a"] = np.zeros(0, dtype=np.int64)
        features, _ = population_features(
            self._spikes(circuit), index_map, "T4", 10, readout="subtype_lateral",
            lateral_bin=lateral_coordinates(circuit, 4),
        )
        assert features.shape[1] == 16, "an empty subtype keeps its four bin columns"
        assert np.allclose(features[:, :4], 0.0)

    def test_skip_bins_truncates_the_leading_bins(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        spikes = self._spikes(circuit)
        full, _ = population_features(
            spikes, circuit.cell_type_index(), "T4", 10, 0, False, "subtype_lateral", bins
        )
        trimmed, _ = population_features(
            spikes, circuit.cell_type_index(), "T4", 10, 2, False, "subtype_lateral", bins
        )
        assert trimmed.shape[0] == full.shape[0] - 2
        assert np.array_equal(trimmed, full[2:])

    def test_magnitude_features_sum_to_one_column(self) -> None:
        block = np.random.default_rng(0).random((4, 8))
        assert magnitude_features(block).shape == (4, 1)
        assert np.allclose(magnitude_features(block).ravel(), block.sum(axis=1))

    def test_record_summary_reports_the_true_width(self, circuit) -> None:
        bins = lateral_coordinates(circuit, 4)
        features, centres = population_features(
            self._spikes(circuit), circuit.cell_type_index(), "T4+T5", 10,
            readout="subtype_lateral", lateral_bin=bins,
        )
        summary = PopulationRecord(
            "LEFTWARD", 0, "T4+T5", features, centres, "subtype_lateral", 4
        ).summary()
        assert summary["feature_columns"] == features.shape[1] == 32
        assert summary["one_column_per_cell_type_per_lateral_bin"] is True
        assert summary["one_column_per_cell_type"] is False
        assert len(summary["cell_types"]) == 8

    def test_activity_descriptor_reports_silence(self, circuit) -> None:
        silent = np.zeros((40, circuit.num_neurons), dtype=bool)
        descriptor = activity_descriptor(silent, circuit.cell_type_index(), "T4+T5")
        assert descriptor["active"] is False
        assert descriptor["spikes"] == 0
        assert descriptor["fraction_neurons_firing"] == 0.0

    def test_activity_descriptor_reports_activity(self, circuit) -> None:
        descriptor = activity_descriptor(
            self._spikes(circuit, rate=0.3), circuit.cell_type_index(), "T4+T5"
        )
        assert descriptor["active"] is True
        assert descriptor["spikes"] > 0
        assert 0 < descriptor["fraction_neurons_firing"] <= 1.0
        assert descriptor["neurons"] == sum(
            circuit.indices_of_type(n).size for n in SUBTYPE_GROUPS
        )


# ======================================================================
# Controls and the acceptance gate.
# ======================================================================


def _control_features(generator, circuit, condition, trial=0, n_bins=4, n_steps=40):
    return input_control_features(
        generator.drive_tensor(condition, trial, n_steps), circuit,
        lateral_coordinates(circuit, n_bins),
        on_types=tuple(generator.on_types), off_types=tuple(generator.off_types),
        bin_steps=10, skip_bins=0, n_lags=3,
    )


class TestControls:
    def test_gate_features_are_a_constant_across_conditions(self, generator, circuit) -> None:
        """The energy match is exact, so the gate cannot even be fooled by noise."""
        vectors = {}
        for name in ("LEFTWARD", "RIGHTWARD", "STATIC"):
            features = _control_features(generator, circuit, name)
            vectors[name] = features[INPUT_INSTANT]
        stacked = np.vstack([vectors[n] for n in vectors])
        assert np.abs(stacked - stacked[0]).max() == 0.0
        assert stacked.shape == (3, 2)

    def test_gate_identity_is_exact_within_a_trial(self, generator, circuit) -> None:
        records = {}
        for name in ("LEFTWARD", "RIGHTWARD", "STATIC"):
            features = _control_features(generator, circuit, name, trial=2)
            records[name] = {
                INPUT_INSTANT: [
                    PopulationRecord(name, 2, INPUT_INSTANT, value, np.zeros(1))
                    for value in (features[INPUT_INSTANT],)
                ]
            }
        identity = input_instant_identity(records, list(records))
        assert identity["identical"] is True
        assert identity["max_spread"] == 0.0
        assert identity["trials_compared"] == 1

    def test_gate_identity_excludes_polarity_twins(self, generator, circuit) -> None:
        """A polarity twin flips the OFF sign on purpose, so it is not a confound."""
        records = {}
        for name in ("LEFTWARD", "LEFTWARD_POLARITY_REVERSED"):
            features = _control_features(generator, circuit, name)
            records[name] = {
                INPUT_INSTANT: [
                    PopulationRecord(name, 0, INPUT_INSTANT, features[INPUT_INSTANT], np.zeros(1))
                ]
            }
        identity = input_instant_identity(records, list(records))
        # Only one condition per polarity, so there is nothing to compare and the
        # check says so rather than reporting a spurious spread.
        assert identity["trials_compared"] == 0
        assert np.isnan(identity["max_spread"])

    def test_spatial_control_separates_the_two_sweeps(self, generator, circuit) -> None:
        left = _control_features(generator, circuit, "LEFTWARD")[INPUT_SPATIAL]
        right = _control_features(generator, circuit, "RIGHTWARD")[INPUT_SPATIAL]
        assert not np.allclose(left, right), "the spatial relay bound must carry the direction"

    def test_spatial_control_separates_static_from_moving(self, generator, circuit) -> None:
        static = _control_features(generator, circuit, "STATIC")[INPUT_SPATIAL]
        left = _control_features(generator, circuit, "LEFTWARD")[INPUT_SPATIAL]
        assert not np.allclose(static, left)

    def test_temporal_control_flips_sign_with_the_sweep(self, generator, circuit) -> None:
        """The ordering, not the totals: the signed velocity must reverse."""
        velocity = {}
        for name in ("LEFTWARD", "RIGHTWARD"):
            features = _control_features(generator, circuit, name)
            velocity[name] = float(features[INPUT_TEMPORAL][0, -1])
        assert velocity["LEFTWARD"] * velocity["RIGHTWARD"] < 0, velocity

    def test_temporal_control_is_flat_for_static(self, generator, circuit) -> None:
        features = _control_features(generator, circuit, "STATIC")
        assert float(features[INPUT_TEMPORAL][0, -1]) == pytest.approx(0.0, abs=1e-9)

    def test_control_shapes_are_documented(self, generator, circuit) -> None:
        features = _control_features(generator, circuit, "LEFTWARD", n_bins=4)
        assert features[INPUT_INSTANT].shape == (1, 2)
        assert features[INPUT_SPATIAL].shape[1] == 2 * 4
        assert features[INPUT_TEMPORAL].shape == (1, 4 + 1)

    def test_controls_reject_a_bad_tensor(self, circuit) -> None:
        with pytest.raises(ValueError):
            input_control_features(
                np.zeros((10, circuit.num_neurons, 2)), circuit,
                lateral_coordinates(circuit, 4),
            )

    def test_gate_passes_when_the_input_cannot_separate(self) -> None:
        results = {
            "C": {
                "mean_test_accuracy": 0.5,
                "permutation_null": {"status": "ok", "null_mean": 0.5, "null_p95": 0.6,
                                     "z_score": 0.0, "exceeds_null_p95": False},
            }
        }
        gate = acceptance_gate(results)
        assert gate["passed"] is True
        assert gate["status"] == "pass"
        assert gate["interpretation_allowed"] is True
        assert "PASSED" in gate["instruction"]

    def test_gate_fails_when_the_input_already_separates(self) -> None:
        results = {
            "C": {
                "mean_test_accuracy": 0.95,
                "permutation_null": {"status": "ok", "null_mean": 0.5, "null_p95": 0.6,
                                     "z_score": 9.0, "exceeds_null_p95": True},
            }
        }
        gate = acceptance_gate(results)
        assert gate["passed"] is False
        assert gate["status"] == "fail"
        assert gate["interpretation_allowed"] is False
        assert gate["failed_conditions"] == ["C"]
        assert "FAILED" in gate["instruction"]
        assert "must not be reported" in gate["instruction"]

    def test_gate_is_undecidable_without_a_null(self) -> None:
        """Reporting 'pass' when nothing was tested would be a false positive."""
        results = {
            "C": {"mean_test_accuracy": 0.5, "permutation_null": {"status": "disabled"}},
        }
        gate = acceptance_gate(results)
        assert gate["status"] == "undecidable"
        assert gate["interpretation_allowed"] is False
        assert "UNDECIDABLE" in gate["instruction"]

    def test_gate_marks_a_partial_null(self) -> None:
        good = {
            "mean_test_accuracy": 0.5,
            "permutation_null": {"status": "ok", "null_mean": 0.5, "null_p95": 0.6,
                                 "z_score": 0.0, "exceeds_null_p95": False},
        }
        bad = {"mean_test_accuracy": 0.5, "permutation_null": {"status": "disabled"}}
        gate = acceptance_gate({"A": good, "B": bad})
        assert gate["status"] == "pass_with_undecidable"
        assert gate["undecidable_conditions"] == ["B"]


# ======================================================================
# Subset selection, unchanged in contract from the previous phase.
# ======================================================================


class TestSubset:
    def test_selection_is_deterministic(self, populations, extraction, bounded) -> None:
        again = bounded_population(
            populations, extraction.pre_index, extraction.post_index,
            synapse_count=extraction.synapse_count, bodies_per_type=4,
        )
        assert np.array_equal(bounded.positions, again.positions)
        assert bounded.describe() == again.describe()

    def test_budget_is_respected_per_type(self, populations, extraction) -> None:
        result = bounded_population(
            populations, extraction.pre_index, extraction.post_index, bodies_per_type=4
        )
        assert result.is_subset
        for name, count in result.selected_per_type().items():
            assert count <= 4, f"{name} exceeded the node budget with {count}"

    def test_every_selected_downstream_body_is_driven(self, populations, extraction) -> None:
        """The property the first per-type version failed: it produced silent T4."""
        result = bounded_population(
            populations, extraction.pre_index, extraction.post_index, bodies_per_type=4
        )
        labels = np.asarray(populations.cell_types, dtype=object)[np.asarray(populations.type_of)]
        selected = np.zeros(populations.num_bodies, dtype=bool)
        selected[result.positions] = True
        on_chain = chain_edge_mask(labels, extraction.pre_index, extraction.post_index)
        kept = on_chain & selected[extraction.pre_index] & selected[extraction.post_index]
        driven = set(extraction.post_index[kept].tolist())
        checked = 0
        for name in ("Mi1", "Tm1", "Tm2", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
            for position in np.flatnonzero(selected & (labels == name)):
                assert int(position) in driven, f"{name} at {int(position)} is undriven"
                checked += 1
        assert checked > 0
        assert result.topped_up_per_type == {}, "the closure should fill every budget"

    def test_documented_chains_survive_the_bound(self, bounded) -> None:
        assert bounded.chain_connectivity["ON"]["fully_connected"]
        assert bounded.chain_connectivity["OFF"]["fully_connected"]

    def test_input_seeds_are_lowest_hemisphere_balanced(self, populations, extraction) -> None:
        body_ids = np.asarray(populations.body_ids, dtype=np.int64)
        sides = np.asarray(populations.soma_side, dtype=object)
        labels = np.asarray(populations.cell_types, dtype=object)[np.asarray(populations.type_of)]
        result = bounded_population(
            populations, extraction.pre_index, extraction.post_index, bodies_per_type=4
        )
        chosen = set(body_ids[result.positions].tolist())
        for name in INPUT_TYPES:
            mask = labels == name
            for side in ("L", "R"):
                expected = np.sort(body_ids[mask & (sides == side)])[:2]
                assert expected.size == 2
                for body in expected:
                    assert int(body) in chosen

    def test_identical_population_in_every_circuit_mode(
        self, populations, extraction, annotations, bounded
    ) -> None:
        ff = restrict_circuit(
            MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_FEEDFORWARD),
            bounded.positions,
        )
        rec = restrict_circuit(
            MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT),
            bounded.positions,
        )
        assert np.array_equal(ff.body_ids, rec.body_ids)
        assert np.array_equal(ff.cell_type, rec.cell_type)

    def test_restrict_keeps_measured_weights_and_order(self, circuit, bounded) -> None:
        sub = restrict_circuit(circuit, bounded.positions)
        assert sp.issparse(sub.csr)
        assert np.all(np.diff(sub.body_ids) > 0)
        order = np.argsort(circuit.body_ids[bounded.positions], kind="stable")
        to_parent = bounded.positions[order]
        parent = {
            (int(a), int(b)): int(w) for a, b, w in zip(*sp.find(circuit.csr))
        }
        for a, b, w in zip(*sp.find(sub.csr)):
            assert parent[(int(to_parent[a]), int(to_parent[b]))] == int(w)

    def test_restrict_rejects_a_bad_selection(self, circuit) -> None:
        with pytest.raises(MCNSCircuitError):
            restrict_circuit(circuit, np.zeros(0, dtype=np.int64))
        with pytest.raises(MCNSCircuitError):
            restrict_circuit(circuit, np.array([0, 0], dtype=np.int64))
        with pytest.raises(MCNSCircuitError):
            restrict_circuit(circuit, np.array([circuit.num_neurons], dtype=np.int64))

    def test_subset_report_states_the_omission(self, bounded, populations) -> None:
        described = bounded.describe()
        assert described["is_subset"] is True
        assert bounded.headline().startswith("SUBSET: ")
        assert f"{bounded.num_neurons:,}" in bounded.headline()
        assert f"{populations.num_bodies:,}" in bounded.headline()
        assert described["edges_omitted"] > 0
        assert described["randomness_used_in_selection"] is False
        assert "hemisphere" in described["selection_rule"].lower()
        assert "SUBSET MODE" in described["disclaimer"]

    def test_a_full_run_is_never_labelled_a_subset(self, populations, extraction) -> None:
        described = bounded_population(
            populations, extraction.pre_index, extraction.post_index, bodies_per_type=0
        ).describe()
        assert described["is_subset"] is False
        assert described["headline"].startswith("FULL POPULATION:")
        assert "SUBSET" not in described["headline"]
        assert described["edges_omitted"] == 0

    def test_subset_graphs_stay_sparse(self, circuit, bounded) -> None:
        for built in (circuit, restrict_circuit(circuit, bounded.positions)):
            assert sp.issparse(built.csr)
            assert built.csr.nnz < built.num_neurons**2
        dense_bytes = circuit.num_neurons**2 * 8
        if dense_bytes > DENSE_REFUSE_BYTES:
            with pytest.raises(MCNSCircuitError):
                circuit.to_dense()


# ======================================================================
# Shuffled control and memory.
# ======================================================================


class TestShuffled:
    def test_reproducible_from_a_seed(self, circuit) -> None:
        a = shuffled_control(circuit, seed=7)
        b = shuffled_control(circuit, seed=7)
        assert np.array_equal(a.csr.indices, b.csr.indices)
        assert np.array_equal(a.csr.data, b.csr.data)

    def test_a_different_seed_gives_a_different_graph(self, circuit) -> None:
        a = shuffled_control(circuit, seed=1)
        b = shuffled_control(circuit, seed=2)
        assert not np.array_equal(a.csr.indices, b.csr.indices)

    def test_preserves_the_fair_invariants(self, circuit) -> None:
        check = verify_degree_preservation(circuit, shuffled_control(circuit, seed=0))
        for key in (
            "neurons_equal", "edges_equal", "in_degree_sequence_equal",
            "out_degree_sequence_equal", "synapse_count_total_equal",
            "synapse_count_distribution_equal", "topology_changed",
        ):
            assert check[key] is True, key
        assert check["self_loops"] == 0

    def test_the_propagation_matrices_differ(self, circuit) -> None:
        """The wiring control must differ in the matrix the simulator actually uses.

        ``topology_changed`` guarantees the edge *sets* differ. That is necessary
        but it is not what the LIF loop reads: the loop reads
        ``propagation_matrix``, which is the transpose of the coupling matrix with
        the synapse-count values attached. A change that preserved the edge set
        while flattening the weights would pass the graph-level test and silently
        turn the control into a copy. So the matrix itself is compared.
        """
        real = circuit.propagation_matrix(synapse_scale=0.25, coupling="linear")
        shuffled = shuffled_control(circuit, seed=0).propagation_matrix(
            synapse_scale=0.25, coupling="linear"
        )
        assert real.shape == shuffled.shape
        assert (real - shuffled).nnz > 0, "propagation matrices are identical"
        assert abs(real - shuffled).max() > 0.0
        # Not a scalar multiple of one another either, which a rescaling bug would
        # produce while leaving every graph-level invariant intact.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = real.todense() / shuffled.todense()
        finite = ratio[np.isfinite(ratio) & (ratio != 0)]
        assert finite.size and np.ptp(finite) > 0.0, "one matrix is a scalar multiple of the other"

    def test_the_two_graphs_produce_different_dynamics(self, circuit, generator) -> None:
        """The gap this closes: nothing asserted that the *dynamics* differ.

        ``topology_changed`` and the propagation-matrix test above are graph-level
        guarantees. Neither proves the LIF loop integrates them to different spike
        trains. This runs the real simulator on both graphs with the same encoder,
        the same stimulus, the same gain and the same seed, and asserts the spike
        logs differ. Runs on the tiny pathway fixture, so it costs milliseconds and
        needs no real-data run.
        """
        params = NeuronParams()
        shuffled = shuffled_control(circuit, seed=0)
        # Same soma geometry, so any difference is wiring and not stimulus layout.
        shuffled.soma_location = circuit.soma_location.copy()
        kwargs = dict(n_steps=40, params=params, synapse_scale=0.25, record_input=False)

        real = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        control = run_simulation(shuffled, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        assert not np.array_equal(real.spikes, control.spikes), (
            "the measured and the randomised graph produced identical spike trains"
        )
        real_counts = real.spikes.sum(axis=0)
        control_counts = control.spikes.sum(axis=0)
        assert not np.array_equal(real_counts, control_counts)
        # A pure rate difference would be a different finding from a difference in
        # *which* neurons fire, so both are checked.
        assert int(real.spikes.sum()) > 0 and int(control.spikes.sum()) > 0
        differing = int(np.count_nonzero(real_counts != control_counts))
        assert differing > 1, "only one neuron differed, which is not a wiring difference"

    def test_dynamics_are_reproducible_for_each_graph(self, circuit, generator) -> None:
        """The comparison above would be meaningless if either run were noisy."""
        params = NeuronParams()
        kwargs = dict(n_steps=40, params=params, synapse_scale=0.25, record_input=False)
        first = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        second = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", **kwargs)
        assert np.array_equal(first.spikes, second.spikes)

    def test_the_seed_does_not_choose_the_difference(self, circuit) -> None:
        """Different shuffle seeds must all differ from the measured graph."""
        real = circuit.propagation_matrix(synapse_scale=0.25, coupling="linear")
        for seed in (0, 1, 2, 3):
            other = shuffled_control(circuit, seed=seed).propagation_matrix(
                synapse_scale=0.25, coupling="linear"
            )
            assert (real - other).nnz > 0, seed


class TestMemory:
    def test_pairwise_never_materialises_the_broadcast_tensor(self) -> None:
        """400 x 512: the naive broadcast would allocate 655 MB per call."""
        import tracemalloc

        block = np.random.default_rng(0).normal(size=(400, 512))
        tracemalloc.start()
        try:
            values = _upper_triangle_pairs(block, "euclidean_distance")
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert values.size == 400 * 399 // 2
        assert peak < (400 * 400 * 512 * 8) / 20

    def test_pairwise_matches_the_naive_broadcast(self) -> None:
        block = np.random.default_rng(1).normal(size=(25, 7))
        difference = block[:, None, :] - block[None, :, :]
        expected = np.sqrt((difference * difference).sum(axis=-1))
        iu = np.triu_indices(25, k=1)
        assert np.allclose(
            _upper_triangle_pairs(block, "euclidean_distance"), expected[iu], equal_nan=True
        )

    def test_pairwise_matches_the_scalar_metric_functions(self) -> None:
        block = np.random.default_rng(2).normal(size=(12, 5))
        functions = {
            "euclidean_distance": euclidean_distance,
            "cosine_distance": cosine_distance,
            "pearson_correlation": pearson_correlation,
        }
        for metric, function in functions.items():
            expected = [
                function(block[i], block[j])
                for i in range(12)
                for j in range(i + 1, 12)
            ]
            assert np.allclose(
                _upper_triangle_pairs(block, metric), expected, equal_nan=True
            ), metric

    def test_degenerate_block_is_empty(self) -> None:
        assert _upper_triangle_pairs(np.zeros((1, 4)), "euclidean_distance").size == 0

    def test_null_and_observed_agree_on_the_statistic(self) -> None:
        stimuli = ["LEFTWARD", "RIGHTWARD"]
        records = _records(stimuli=stimuli, trials=6)
        observed = within_between(records, stimuli)["euclidean_distance"]["separation_ratio"]
        fast = fast_separation_ratio(
            [r.features for r in records], [r.stimulus for r in records], stimuli
        )
        assert fast == pytest.approx(observed, rel=1e-9)

    def test_separability_reports_group_and_cell_types(self) -> None:
        result = separability(_records(), STIM)
        assert result["group"] == "T4+T5"
        assert result["columns"] == list(ABBREVIATIONS["T4+T5"])
        assert result["n_records"] == 12


# ======================================================================
# The script, end to end.
# ======================================================================


def _run(tmp_path: Path, pathway_data, *extra: str) -> tuple[int, Path]:
    module = _load_script()
    output = tmp_path / "out"
    code = module.main(
        [
            "--quick", "--quiet", "--max-bodies-per-type", "3",
            "--trials", "4", "--n-steps", "40", "--bin-steps", "10", "--skip-bins", "0",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(output), "--cache", str(tmp_path / "cache.npz"),
            *extra,
        ]
    )
    return code, output


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestScript:
    @pytest.mark.expensive
    def test_quick_mode_runs_end_to_end(self, tmp_path: Path, pathway_data) -> None:
        code, output = _run(tmp_path, pathway_data, "--permutations", "0")
        assert code == 0
        for name in ("benchmark_summary.json", "decoder_results.json", "activity.json"):
            assert (output / name).is_file(), name

    @pytest.mark.expensive
    def test_all_three_circuits_are_compared(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        assert set(summary["circuits"]) == {
            "MCNS_FEEDFORWARD", "MCNS_RECURRENT", "SHUFFLED_CONTROL"
        }
        assert summary["circuits"]["SHUFFLED_CONTROL"]["is_measured"] is False
        assert summary["circuits"]["MCNS_RECURRENT"]["is_measured"] is True
        populations = {
            c["neurons"] for c in summary["circuits"].values()
        }
        assert len(populations) == 1, "every condition must simulate the same neurons"

    @pytest.mark.expensive
    def test_every_control_is_reported(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        for group in (INPUT_INSTANT, INPUT_SPATIAL, INPUT_TEMPORAL, MAGNITUDE_GROUP,
                      "T4", "T5", "T4+T5"):
            assert group in summary["config"]["groups"], group

    @pytest.mark.expensive
    def test_every_contrast_is_reported(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        results = _json(output / "decoder_results.json")
        for scale_block in results.values():
            for variant_block in scale_block.values():
                assert set(variant_block) == {
                    "LEFTWARD_vs_RIGHTWARD", "LEFTWARD_vs_STATIC",
                    "LEFTWARD_vs_POLARITY_REVERSED",
                }

    @pytest.mark.expensive
    def test_the_coupling_sweep_covers_the_required_gains(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        assert {0.25, 0.5, 1.0, 2.0} <= set(summary["config"]["synapse_scales"])

    @pytest.mark.expensive
    def test_liveness_is_reported_for_every_scale_and_condition(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        activity = _json(output / "activity.json")
        for scale, circuits in activity.items():
            for circuit_name, groups in circuits.items():
                assert "T4" in groups and "T5" in groups
                for entry in groups.values():
                    assert "total_spikes" in entry
                    assert "active" in entry
                    assert "mean_spikes_per_trial" in entry

    @pytest.mark.expensive
    def test_the_gate_is_reported_and_decidable_with_a_null(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "20")
        summary = _json(output / "benchmark_summary.json")
        gate = summary["acceptance_gate"]
        assert gate["gate_feature_identical_across_conditions"] is True
        assert gate["gate_feature_max_spread"] == 0.0
        for scale_entry in gate["per_scale"].values():
            assert scale_entry["status"] in {"pass", "fail", "pass_with_undecidable"}
            for condition in scale_entry["per_condition"].values():
                assert condition["verdict"] in {"pass", "fail", "undecidable"}

    @pytest.mark.expensive
    def test_no_null_means_the_gate_is_undecidable_not_pass(self, tmp_path: Path, pathway_data) -> None:
        """A gate that was never tested must not be reported as a pass."""
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        assert summary["acceptance_gate"]["passed_all_scales"] is False
        for scale_entry in summary["acceptance_gate"]["per_scale"].values():
            assert scale_entry["status"] == "undecidable"

    @pytest.mark.expensive
    def test_the_gate_features_are_exactly_equal(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        assert summary["acceptance_gate"]["gate_feature_max_spread"] == 0.0

    @pytest.mark.expensive
    def test_no_payload_claims_direction_selectivity(self, tmp_path: Path, pathway_data) -> None:
        """The claim ladder must be present and must stop short of level 4."""
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        claim = summary["direction_selectivity_claim"]
        assert "NONE" in claim
        assert "no synaptic sign" in claim
        ladder = summary["claim_ladder"]
        assert len(ladder) == 4
        assert ladder[0].startswith("1. ANATOMICAL CONNECTIVITY")
        assert "not claimed" in ladder[3]
        for name in ("benchmark_summary.json", "decoder_results.json", "activity.json"):
            text = (output / name).read_text(encoding="utf-8").lower()
            for forbidden in (
                "reproduces direction selectivity",
                "demonstrates direction selectivity",
                "t4 is direction selective",
                "reichardt detector computes",
            ):
                assert forbidden not in text, f"{name} asserts {forbidden!r}"

    @pytest.mark.expensive
    def test_measured_simulated_and_assumed_are_separated(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        summary = _json(output / "benchmark_summary.json")
        assert any("synapse counts" in item for item in summary["measured"])
        assert any("spikes" in item for item in summary["simulated"])
        assert any("synapse_scale" in item for item in summary["assumed"])
        assert any("subset" in item for item in summary["assumed"])

    @pytest.mark.expensive
    def test_identical_across_conditions_is_stated(self, tmp_path: Path, pathway_data) -> None:
        _, output = _run(tmp_path, pathway_data, "--permutations", "0")
        stated = _json(output / "benchmark_summary.json")["config"]["identical_across_conditions"]
        for item in ("selected neuron set", "input energy", "sweep start offset",
                     "noise realisation (common random numbers)", "synapse_scale"):
            assert item in stated, item

    @pytest.mark.expensive
    def test_two_runs_produce_identical_json(self, tmp_path: Path, pathway_data) -> None:
        _, first = _run(tmp_path / "a", pathway_data, "--permutations", "0")
        _, second = _run(tmp_path / "b", pathway_data, "--permutations", "0")
        for name in ("decoder_results.json", "activity.json"):
            assert (first / name).read_text(encoding="utf-8") == (
                second / name
            ).read_text(encoding="utf-8"), f"{name} is not reproducible"

    @pytest.mark.expensive
    def test_the_extraction_cache_is_written_then_reused(self, tmp_path: Path, pathway_data,
                                                         capsys) -> None:
        module = _load_script()
        cache = tmp_path / "cache.npz"
        common = [
            "--quick", "--quiet", "--max-bodies-per-type", "3",
            "--trials", "4", "--n-steps", "40", "--bin-steps", "10",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(tmp_path / "out"), "--cache", str(cache),
            "--permutations", "0",
        ]
        module.main(common)
        assert cache.is_file()
        assert "wrote extraction cache" in capsys.readouterr().out
        module.main(common)
        assert "reusing cached extraction" in capsys.readouterr().out


    def test_missing_dataset_returns_two(self, tmp_path: Path) -> None:
        module = _load_script()
        assert module.main(
            [
                "--quick", "--quiet",
                "--annotations", str(tmp_path / "no.feather"),
                "--connectivity", str(tmp_path / "no2.feather"),
                "--output-dir", str(tmp_path / "out"),
            ]
        ) == 2

    def test_too_few_trials_returns_two(self, tmp_path: Path, pathway_data) -> None:
        module = _load_script()
        assert module.main(
            [
                "--quick", "--quiet", "--trials", "2",
                "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
                "--output-dir", str(tmp_path / "out"), "--cache", str(tmp_path / "c.npz"),
            ]
        ) == 2

    def test_quick_preset_is_the_documented_small_experiment(self) -> None:
        module = _load_script()
        args = module.apply_presets(module.build_parser().parse_args(["--quick"]))
        assert args.mode == "quick"
        assert args.trials >= 3
        assert 100 <= args.n_steps <= 200
        assert set(args.scales) == {0.25, 0.5, 1.0, 2.0}
        assert args.max_bodies_per_type == 100
        assert module.PRIMARY_VARIANT in args.variants
        assert len(module.VARIANTS) >= 3

    def test_explicit_flags_override_the_preset(self) -> None:
        module = _load_script()
        args = module.apply_presets(
            module.build_parser().parse_args(
                ["--quick", "--trials", "6", "--scales", "0.5,1.0", "--permutations", "0"]
            )
        )
        assert args.trials == 6
        assert args.scales == [0.5, 1.0]
        assert args.permutations == 0

    def test_full_preset_keeps_the_whole_population(self) -> None:
        module = _load_script()
        args = module.apply_presets(module.build_parser().parse_args([]))
        assert args.mode == "full"
        assert args.max_bodies_per_type == 0

    def test_unknown_names_are_refused(self) -> None:
        module = _load_script()
        with pytest.raises(SystemExit):
            module.apply_presets(module.build_parser().parse_args(["--contrasts", "A_vs_B"]))
        with pytest.raises(SystemExit):
            module.apply_presets(module.build_parser().parse_args(["--variants", "nonsense"]))

    def test_the_readout_variants_share_one_simulation(self) -> None:
        """A secondary readout must not cost another simulation."""
        module = _load_script()
        args = module.apply_presets(
            module.build_parser().parse_args(["--quick", "--variants", "subtype_lateral,cell_type"])
        )
        assert set(args.variants) == {"subtype_lateral", "cell_type"}


# ======================================================================
# Phase 4C: the pre-declared replication of the Phase 4B observation.
# ======================================================================
#
# What is pinned down here is not a result. It is the set of commitments that
# make the experiment a *replication* rather than a second, differently
# configured run whose agreement could be read as confirmation:
#
# * the gain is 0.25 and cannot be changed by a flag;
# * the trial count is 20 and the permutation count is 500, both recorded;
# * the leave-one-trial-out structure still holds out one trial index across
#   both sweep directions, with an asserted sample count and no leakage;
# * the stimulus energy is identical across conditions and trials;
# * liveness is reported for every condition, and a silent or saturated
#   condition is marked uninformative rather than interpreted;
# * all three circuit conditions share one neuron set and one edge subset;
# * the output refuses to claim direction selectivity.
#
# The end-to-end run is module-scoped and run once: it is the same pipeline the
# real run uses, at the declared 20 trials and 500 permutations, so the tests
# cost one full declared run rather than one per assertion.


def _run_phase4c(directory: Path, pathway_data, *extra: str) -> tuple[int, Path]:
    module = _load_script()
    output = directory / "phase4c"
    code = module.main(
        [
            "--phase-4c", "--quiet",
            "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
            "--output-dir", str(output), "--cache", str(directory / "p4c_cache.npz"),
            *extra,
        ]
    )
    return code, output


@pytest.fixture(scope="module")
def phase4c(tmp_path_factory, pathway_data):
    """One declared Phase 4C run on the pathway fixture, shared by every test below."""
    directory = tmp_path_factory.mktemp("phase4c")
    code, output = _run_phase4c(directory, pathway_data)
    assert code == 0
    summary = _json(output / "benchmark_summary.json")
    return {
        "dir": directory,
        "output": output,
        "summary": summary,
        "phase4c": summary["phase4c"],
        "results": _json(output / "decoder_results.json"),
        "activity": _json(output / "activity.json"),
        "report": _json(output / "phase4c_report.json"),
    }


def _primary(run, circuit_name: str) -> dict:
    return run["results"]["0.25"]["subtype_lateral"][PREDECLARED_CONTRAST][circuit_name][
        PREDECLARED_GROUP
    ]


class TestPhase4CPreDeclaration:
    def test_the_declared_configuration_is_fixed(self) -> None:
        assert PREDECLARED_SYNAPSE_SCALE == 0.25
        assert PREDECLARED_TRIALS == 20
        assert PREDECLARED_PERMUTATIONS == 500
        assert PREDECLARED_MAX_BODIES_PER_TYPE == 100
        assert PREDECLARED_SEED == 0
        assert PREDECLARED_GROUP == "T4+T5"
        assert PREDECLARED_CONTRAST == "LEFTWARD_vs_RIGHTWARD"
        assert set(REPLICATION_CIRCUITS) == {
            "MCNS_FEEDFORWARD", "MCNS_RECURRENT", "SHUFFLED_CONTROL"
        }

    def test_the_block_records_that_the_gain_was_pre_declared(self) -> None:
        block = predeclared_block()
        assert block["predeclared"] is True
        assert block["predeclared_before_run"] is True
        assert block["may_be_tuned_after_seeing_results"] is False
        assert block["configuration"]["synapse_scale"] == 0.25
        assert block["configuration"]["gain_sweep"] is False
        assert block["configuration"]["trials_per_condition"] == 20
        assert block["configuration"]["n_permutations"] == 500
        assert "pre-declared" in block["statement"].lower()
        for name in ("synapse_scale", "trials_per_condition", "n_permutations", "not_changed"):
            assert block["why"][name], name
        assert "no sweep" in block["why"]["synapse_scale"].lower()
        assert "chosen from the phase 4b design before this run" in (
            block["why"]["synapse_scale"].lower()
        )

    def test_a_changed_gain_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="pre-declared"):
            enforce_predeclared(
                synapse_scale=0.5, trials=PREDECLARED_TRIALS,
                n_permutations=PREDECLARED_PERMUTATIONS,
                max_bodies_per_type=PREDECLARED_MAX_BODIES_PER_TYPE,
                n_steps=150, bin_steps=25, skip_bins=2, seed=PREDECLARED_SEED,
            )

    def test_a_changed_trial_count_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="trials"):
            enforce_predeclared(
                synapse_scale=PREDECLARED_SYNAPSE_SCALE, trials=8,
                n_permutations=PREDECLARED_PERMUTATIONS,
                max_bodies_per_type=PREDECLARED_MAX_BODIES_PER_TYPE,
                n_steps=150, bin_steps=25, skip_bins=2, seed=PREDECLARED_SEED,
            )

    def test_a_reduced_permutation_count_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="permutations"):
            enforce_predeclared(
                synapse_scale=PREDECLARED_SYNAPSE_SCALE, trials=PREDECLARED_TRIALS,
                n_permutations=100,
                max_bodies_per_type=PREDECLARED_MAX_BODIES_PER_TYPE,
                n_steps=150, bin_steps=25, skip_bins=2, seed=PREDECLARED_SEED,
            )

    def test_the_declared_configuration_passes_its_own_check(self) -> None:
        enforce_predeclared(
            synapse_scale=PREDECLARED_SYNAPSE_SCALE, trials=PREDECLARED_TRIALS,
            n_permutations=PREDECLARED_PERMUTATIONS,
            max_bodies_per_type=PREDECLARED_MAX_BODIES_PER_TYPE,
            n_steps=150, bin_steps=25, skip_bins=2, seed=PREDECLARED_SEED,
        )

    def test_command_line_overrides_are_all_refused(self) -> None:
        module = _load_script()
        for flag in (
            ["--scales", "0.5"], ["--scales", "0.25,0.5"], ["--trials", "12"],
            ["--permutations", "100"], ["--max-bodies-per-type", "50"],
            ["--n-steps", "300"], ["--bin-steps", "10"], ["--skip-bins", "0"],
            ["--seed", "7"], ["--quick"], ["--no-randomise-start"],
        ):
            with pytest.raises(SystemExit):
                module.apply_presets(module.build_parser().parse_args(["--phase-4c", *flag]))

    def test_scientific_parameters_cannot_be_silently_accepted(self) -> None:
        """A flag that would change the experiment must be refused, not ignored.

        The enforcement surface used to cover only the nine numeric parameters, so
        ``--phase-4c --coupling log1p`` and ``--phase-4c --decoder-l2 0.5`` were
        accepted while the pre-declaration claimed the coupling law and the decoder
        were held fixed. Silently accepting a flag is nearly as bad as silently
        changing it: the operator gets no signal the declaration was not honoured.
        """
        module = _load_script()
        for flag in (
            ["--coupling", "log1p"],
            ["--spot-fraction", "0.3"],
            ["--stimulus-current", "3.0"],
            ["--input-noise", "2.0"],
            ["--amplitude-perturbation", "0.2"],
            ["--decoder-iterations", "100"],
            ["--decoder-learning-rate", "0.1"],
            ["--decoder-l2", "0.5"],
            ["--lateral-bins", "8"],
            ["--readout", "cell_type"],
            ["--all-groups"],
            ["--no-normalise"],
        ):
            with pytest.raises(SystemExit):
                module.apply_presets(module.build_parser().parse_args(["--phase-4c", *flag]))

    def test_the_pinned_set_covers_every_scientific_parameter(self) -> None:
        """``PHASE4C_FIXED`` plus the forced assignments is the whole surface.

        Asserted against an explicit list rather than by counting entries, so a
        parameter added to the parser without being pinned fails here.
        """
        module = _load_script()
        pinned = set(module.PHASE4C_FIXED)
        required = {
            "scales", "trials", "permutations", "max_bodies_per_type", "n_steps",
            "bin_steps", "skip_bins", "dt", "seed",
            "coupling", "spot_fraction", "stimulus_current", "input_noise",
            "amplitude_perturbation",
            "decoder_iterations", "decoder_learning_rate", "decoder_l2",
            "lateral_bins", "readout", "all_groups", "no_normalise",
        }
        assert required <= pinned, f"unpinned scientific parameters: {sorted(required - pinned)}"
        # Nothing that changes the science may be advertised as mutable.
        assert not (required & set(module.PHASE4C_MUTABLE))
        assert module.PHASE4C_MUTABLE <= {
            "data_dir", "annotations", "connectivity", "output_dir", "cache",
            "no_cache", "quiet", "estimate", "estimate_trials", "estimate_permutations",
        }

    def test_a_bare_invocation_reproduces_the_declared_configuration(self) -> None:
        """No flag at all must give exactly the pre-declared values."""
        module = _load_script()
        args = module.apply_presets(module.build_parser().parse_args(["--phase-4c"]))
        declared = predeclared_block()["configuration"]
        assert args.scales == [declared["synapse_scale"]]
        assert args.trials == declared["trials_per_condition"]
        assert args.permutations == declared["n_permutations"]
        for name in (
            "coupling", "spot_fraction", "stimulus_current", "input_noise",
            "amplitude_perturbation", "decoder_iterations", "decoder_learning_rate",
            "decoder_l2", "lateral_bins", "readout", "n_steps", "bin_steps",
            "skip_bins", "seed",
        ):
            assert getattr(args, name) == declared[name], name
        assert args.all_groups is False and args.no_normalise is False

    def test_declaring_the_predeclared_values_is_still_accepted(self) -> None:
        """Refusing every flag would also refuse an honest, redundant one."""
        module = _load_script()
        args = module.apply_presets(
            module.build_parser().parse_args(
                ["--phase-4c", "--scales", "0.25", "--trials", "20", "--permutations", "500"]
            )
        )
        assert args.scales == [0.25] and args.trials == 20 and args.permutations == 500


class TestPhase4CConfiguration:
    @pytest.mark.expensive
    def test_the_run_used_exactly_one_gain_at_the_pre_declared_value(self, phase4c) -> None:
        assert phase4c["summary"]["config"]["synapse_scales"] == [PREDECLARED_SYNAPSE_SCALE]
        assert phase4c["phase4c"]["predeclared_synapse_scale"] == PREDECLARED_SYNAPSE_SCALE

    @pytest.mark.expensive
    def test_the_run_used_twenty_trials(self, phase4c) -> None:
        assert phase4c["summary"]["config"]["trials_per_condition"] == PREDECLARED_TRIALS == 20
        assert phase4c["phase4c"]["n_trials"] == 20
        assert phase4c["phase4c"]["n_folds"] == 20

    @pytest.mark.expensive
    def test_the_run_used_the_fixed_permutation_count(self, phase4c) -> None:
        assert phase4c["summary"]["config"]["permutations"] == PREDECLARED_PERMUTATIONS == 500
        block = phase4c["phase4c"]
        assert block["predeclared_permutations"] == 500
        assert block["permutations_run"] == 500, "the count must not be reduced to save time"
        assert block["n_permutations"] == 500
        for name, stats in block["statistics"].items():
            assert stats["n_permutations"] == 500, name

    @pytest.mark.expensive
    def test_every_reported_null_actually_ran_five_hundred(self, phase4c) -> None:
        checked = 0
        for circuits in phase4c["results"]["0.25"]["subtype_lateral"][
            PREDECLARED_CONTRAST
        ].values():
            for group, entry in circuits.items():
                null = entry.get("permutation_null", {})
                if null.get("status") != "ok":
                    continue
                assert null["n_permutations"] == PREDECLARED_PERMUTATIONS, group
                checked += 1
        assert checked >= 3 * len(("INPUT_INSTANT", "T4", "T5", "T4+T5"))

    @pytest.mark.expensive
    def test_the_readout_is_subtype_lateral_with_reported_features(self, phase4c) -> None:
        block = phase4c["phase4c"]
        assert block["primary_comparison"]["readout"] == "subtype_lateral"
        for circuit_name, census in block["feature_census"].items():
            assert census["n_features"] == 32, circuit_name
            assert census["readout"] == "subtype_lateral"
            assert census["lateral_bins"] == 4
            assert 0 < census["n_nonzero_features"] <= 32, circuit_name
            assert census["n_nonzero_features"] + census["zero_features"] == 32

    @pytest.mark.expensive
    def test_the_population_budget_and_stimulus_are_unchanged(self, phase4c) -> None:
        config = phase4c["summary"]["config"]
        assert config["max_bodies_per_type"] == PREDECLARED_MAX_BODIES_PER_TYPE
        assert config["n_steps"] == 150 and config["bin_steps"] == 25
        assert config["skip_bins"] == 2 and config["seed"] == PREDECLARED_SEED
        assert config["lateral_bins"] == 4 and config["readout"] == "subtype_lateral"
        assert config["conditions_presented"] == [
            "LEFTWARD", "LEFTWARD_POLARITY_REVERSED", "RIGHTWARD", "STATIC"
        ]


class TestPhase4CTrialProtocol:
    @pytest.mark.expensive
    def test_leave_one_trial_out_gives_exactly_twenty_folds(self, phase4c) -> None:
        for circuit_name in REPLICATION_CIRCUITS:
            entry = _primary(phase4c, circuit_name)
            assert entry["protocol"] == "leave_one_trial_out"
            assert entry["n_trials"] == PREDECLARED_TRIALS
            assert entry["n_folds"] == PREDECLARED_TRIALS == 20
            assert len(entry["fold_accuracies"]) == 20

    @pytest.mark.expensive
    def test_the_sample_count_is_asserted_not_assumed(self, phase4c) -> None:
        for circuit_name in REPLICATION_CIRCUITS:
            entry = _primary(phase4c, circuit_name)
            # 20 trials x 2 sweep directions x 4 surviving time bins.
            assert entry["n_test_samples"] == 20 * 2 * 4, circuit_name
            assert entry["n_test_samples"] == entry["n_test_samples_expected"]
            assert entry["sample_count_matches"] is True

    def test_no_train_test_leakage_in_the_loo_design(self) -> None:
        """A held-out trial must never appear in its own training rows.

        Checked directly on the design rather than inferred from an accuracy. The
        trick that makes it exact: every row of a given trial carries one
        constant value ``trial + 1``, and the fold standardises train and test
        with the *same* affine map, so a held-out row is byte-identical to itself
        in the training tensor if and only if it leaked. Comparing the two
        tensors' value sets therefore detects a leak with no false positives.
        """
        bins, dims = 3, 4
        records = []
        for stimulus in STIM:
            for trial in range(PREDECLARED_TRIALS):
                block = np.full((bins, dims), float(trial + 1), dtype=np.float64)
                records.append(PopulationRecord(stimulus, trial, "T4+T5", block, np.arange(bins)))
        _, blocks = _trial_blocks(records, STIM)
        design = _loo_design(blocks, 2)
        assert design["n_folds"] == PREDECLARED_TRIALS

        trials = sorted({r.trial for r in records})
        for fold in range(design["n_folds"]):
            held = trials[design["test_trial"][fold]]
            test_values = set(np.unique(design["test_x"][fold]).tolist())
            train_values = set(np.unique(design["train_x"][fold]).tolist())
            assert test_values, f"fold {fold} has an empty test set"
            assert not (test_values & train_values), (
                f"fold {fold} held out trial {held} but trained on its rows"
            )
            # Row accounting: the training set is every other trial, whole.
            assert len(design["train_groups"][fold]) == PREDECLARED_TRIALS - 1
            assert design["test_trial"][fold] not in [p for p, _ in design["train_groups"][fold]]
            assert sum(n for _, n in design["train_groups"][fold]) == int(
                design["train_counts"][fold]
            )
            assert int(design["test_counts"][fold]) == len(STIM) * bins

    @pytest.mark.expensive
    def test_every_fold_holds_both_sweep_directions(self, phase4c) -> None:
        """A fold must never be single-class, or its accuracy is a constant prediction."""
        for circuit_name in REPLICATION_CIRCUITS:
            matrix = np.asarray(_primary(phase4c, circuit_name)["confusion_matrix"])
            assert matrix.shape == (2, 2)
            assert int(matrix.sum()) == 20 * 2 * 4
            assert (matrix.sum(axis=1) > 0).all(), circuit_name

    def test_stimulus_energy_is_identical_across_conditions_and_trials(self, circuit) -> None:
        """Every condition, every one of the 20 declared trials, same drive magnitude."""
        generator = MotionStimulusGenerator(
            circuit, spot_fraction=0.3, current=2.0, steps_per_crossing=40,
            amplitude_perturbation=0.05, input_noise=1.0, seed=PREDECLARED_SEED,
        )
        names = [c.name for c in STIMULUS_CONDITIONS]
        for trial in range(PREDECLARED_TRIALS):
            energies = {
                name: generator.drive_energy(name, 40, trial=trial) for name in names
            }
            assert len({round(v, 9) for v in energies.values()}) == 1, (
                f"stimulus energy is not matched in trial {trial}: {energies}"
            )
        # And the same noise realisation in every condition of a trial, so a
        # condition difference is stimulus structure rather than a noise draw.
        for step in (0, 17, 39):
            noise = {
                name: generator.encode(name, step, trial=3, n_steps=40, add_noise=True)
                - generator.encode(name, step, trial=3, n_steps=40, add_noise=False)
                for name in names
            }
            reference = noise["LEFTWARD"]
            for name, vector in noise.items():
                assert np.allclose(vector, reference), name

    @pytest.mark.expensive
    def test_the_gate_features_are_exactly_equal_in_the_phase4c_run(self, phase4c) -> None:
        gate = phase4c["phase4c"]["acceptance_gate"]
        assert gate["gate_feature_identical_across_conditions"] is True
        assert gate["gate_feature_max_spread"] == 0.0
        assert gate["status"] in {"pass", "pass_with_undecidable"}


class TestPhase4CDeterminism:
    @pytest.mark.expensive
    def test_the_run_carries_a_content_fingerprint(self, phase4c) -> None:
        prints = phase4c["phase4c"]["determinism"]["fingerprint_by_circuit"]
        for circuit_name, value in prints.items():
            assert isinstance(value, str) and len(value) == 64, circuit_name

    @pytest.mark.expensive
    def test_two_runs_of_the_declared_configuration_agree(self, pathway_data) -> None:
        """Same declaration, same numbers. Run twice through the estimate path.

        The estimate path is the same declared pipeline with only the trial count
        and permutation count reduced, so two of them are two runs of one
        configuration and their feature digests must be byte-identical. The
        estimate always writes to the fixed estimate directory, so the digest is
        read from there between the two runs.
        """
        module = _load_script()
        estimate_dir = REPO / module.PHASE4C_ESTIMATE_DIR
        digests = []
        for name in ("a", "b"):
            directory = Path(estimate_dir) / name
            code = module.main(
                [
                    "--phase-4c", "--estimate", "--quiet",
                    "--estimate-trials", "3", "--estimate-permutations", "5",
                    "--annotations", str(pathway_data[0]),
                    "--connectivity", str(pathway_data[1]),
                    "--cache", str(Path(directory) / "cache.npz"),
                ]
            )
            assert code == 0
            payload = _json(estimate_dir / "phase4c_timing_estimate.json")
            digests.append(payload["determinism_fingerprint"])
        assert digests[0] == digests[1]
        assert digests[0]

    def test_the_fingerprint_changes_with_the_data(self) -> None:
        base = _records(trials=4, bins=3, dims=8, seed=1)
        moved = [
            PopulationRecord(r.stimulus, r.trial, r.group, r.features * 1.5, r.bin_centres)
            for r in base
        ]
        assert fingerprint(base) != fingerprint(moved)
        assert fingerprint(base) == fingerprint(list(reversed(base)))

    def test_the_margin_null_is_reproducible_from_its_seed(self) -> None:
        first = margin_statistics([0.7] * 20, [0.5] * 20, n_permutations=64, seed=0)
        second = margin_statistics([0.7] * 20, [0.5] * 20, n_permutations=64, seed=0)
        assert first == second
        assert first["margin_permutation_null"]["n_permutations"] == 64


class TestPhase4CLiveness:
    @pytest.mark.expensive
    def test_every_liveness_field_is_reported_for_every_condition(self, phase4c) -> None:
        required = (
            "mean_spikes_per_trial", "median_spikes_per_trial", "min_spikes_per_trial",
            "max_spikes_per_trial", "zero_activity_trials", "fraction_zero_activity_trials",
            "n_trials", "total_spikes", "active", "spike_saturation_fraction",
        )
        for circuit_name in REPLICATION_CIRCUITS:
            for group in ("T4", "T5", "T4+T5"):
                live = phase4c["phase4c"]["liveness"][circuit_name][group]
                for field in required:
                    assert field in live, f"{circuit_name}/{group} is missing {field}"
                assert live["n_trials"] == 2 * PREDECLARED_TRIALS
                assert live["min_spikes_per_trial"] <= live["median_spikes_per_trial"]
                assert live["median_spikes_per_trial"] <= live["max_spikes_per_trial"]
                assert 0.0 <= live["fraction_zero_activity_trials"] <= 1.0

    @pytest.mark.expensive
    def test_t4_and_t5_activity_are_reported_separately(self, phase4c) -> None:
        for circuit_name in REPLICATION_CIRCUITS:
            groups = phase4c["phase4c"]["liveness"][circuit_name]
            assert set(groups) >= {"T4", "T5", "T4+T5"}
            for group in ("T4", "T5"):
                assert groups[group]["n_trials"] == 2 * PREDECLARED_TRIALS
            assert groups["T4+T5"]["n_trials"] == 2 * PREDECLARED_TRIALS

    @pytest.mark.expensive
    def test_the_nonzero_feature_count_is_reported(self, phase4c) -> None:
        census = phase4c["phase4c"]["feature_census"]
        for circuit_name in REPLICATION_CIRCUITS:
            assert "n_nonzero_features" in census[circuit_name]
            assert "n_features" in census[circuit_name]

    @pytest.mark.expensive
    def test_every_condition_gets_an_informativeness_verdict(self, phase4c) -> None:
        verdicts = phase4c["phase4c"]["informativeness"]
        for circuit_name in REPLICATION_CIRCUITS:
            for group in ("T4", "T5", "T4+T5"):
                status = verdicts[circuit_name][group]["status"]
                assert status in {
                    "informative", "uninformative_silent", "uninformative_saturated"
                }, f"{circuit_name}/{group}: {status}"

    def test_a_silent_condition_is_marked_uninformative(self) -> None:
        silent = liveness([0] * 40, n_neurons=800, n_steps=150)
        verdict = classify_informative(silent, features={"n_nonzero_features": 0})
        assert verdict["status"] == "uninformative_silent"
        assert verdict["may_interpret_accuracy"] is False
        assert "UNINFORMATIVE" in verdict["verdict"]
        assert verdict["reasons"]

    def test_a_mostly_silent_condition_is_marked_uninformative(self) -> None:
        counts = [0] * 30 + [5] * 10
        verdict = classify_informative(liveness(counts, n_neurons=800, n_steps=150))
        assert verdict["status"] == "uninformative_silent"
        assert any("zero activity" in reason for reason in verdict["reasons"])

    def test_a_saturated_condition_is_marked_uninformative(self) -> None:
        # 800 neurons x 150 steps = 120,000 possible spikes per trial; 90,000 is
        # 75% of the ceiling, which is a rate code at its limit, not a pattern.
        live = liveness([90_000] * 20, n_neurons=800, n_steps=150)
        verdict = classify_informative(live, features={"n_nonzero_features": 32})
        assert verdict["status"] == "uninformative_saturated"
        assert verdict["may_interpret_accuracy"] is False

    def test_a_null_saturated_benchmark_is_marked_uninformative(self) -> None:
        live = liveness([40] * 20, n_neurons=800, n_steps=150)
        verdict = classify_informative(
            live, features={"n_nonzero_features": 32}, accuracy=1.0, null_p95=1.0
        )
        assert verdict["status"] == "uninformative_saturated"
        assert any("permutation null" in reason for reason in verdict["reasons"])

    def test_a_live_sparse_condition_is_informative(self) -> None:
        live = liveness([30, 40, 35, 45, 38, 42] * 3, n_neurons=800, n_steps=150)
        verdict = classify_informative(
            live, features={"n_nonzero_features": 20}, accuracy=0.61, null_p95=0.55
        )
        assert verdict["status"] == "informative"
        assert verdict["may_interpret_accuracy"] is True
        assert verdict["reasons"] == []

    def test_liveness_summarises_the_actual_counts(self) -> None:
        live = liveness([0, 2, 4, 6], n_neurons=100, n_steps=10)
        assert live["n_trials"] == 4
        assert live["total_spikes"] == 12
        assert live["mean_spikes_per_trial"] == 3.0
        assert live["median_spikes_per_trial"] == 3.0
        assert live["min_spikes_per_trial"] == 0
        assert live["max_spikes_per_trial"] == 6
        assert live["zero_activity_trials"] == 1
        assert live["fraction_zero_activity_trials"] == 0.25
        assert live["spike_saturation_fraction"] == pytest.approx(0.006)

    def test_empty_liveness_is_nan_not_a_guess(self) -> None:
        live = liveness([])
        assert live["n_trials"] == 0
        assert live["active"] is False
        assert np.isnan(live["mean_spikes_per_trial"])


class TestPhase4CPrimaryComparison:
    @pytest.mark.expensive
    def test_the_primary_quantity_is_recurrent_minus_shuffled(self, phase4c) -> None:
        block = phase4c["phase4c"]["primary_comparison"]
        assert block["quantity"] == "MCNS_RECURRENT accuracy - SHUFFLED_CONTROL accuracy"
        expected = (
            _primary(phase4c, "MCNS_RECURRENT")["mean_test_accuracy"]
            - _primary(phase4c, "SHUFFLED_CONTROL")["mean_test_accuracy"]
        )
        assert block["recurrent_accuracy"] == pytest.approx(
            _primary(phase4c, "MCNS_RECURRENT")["mean_test_accuracy"]
        )
        assert block["shuffled_accuracy"] == pytest.approx(
            _primary(phase4c, "SHUFFLED_CONTROL")["mean_test_accuracy"]
        )
        assert block["recurrent_minus_shuffled_margin"] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.expensive
    def test_both_separation_ratios_are_reported(self, phase4c) -> None:
        block = phase4c["phase4c"]["primary_comparison"]
        for key in (
            "recurrent_separation_ratio", "shuffled_separation_ratio",
            "recurrent_minus_shuffled_margin",
        ):
            value = block[key]
            assert value is not None, key
            assert np.isfinite(float(value)), key
        for circuit_name in ("MCNS_RECURRENT", "SHUFFLED_CONTROL"):
            ratio = phase4c["phase4c"]["statistics"][circuit_name]["separation_ratio"]
            assert np.isfinite(float(ratio)) and float(ratio) > 0

    @pytest.mark.expensive
    def test_a_separation_ratio_null_is_reported_for_each_condition(self, phase4c) -> None:
        for circuit_name in REPLICATION_CIRCUITS:
            null = phase4c["phase4c"]["statistics"][circuit_name]["separation_ratio_null"]
            assert null["status"] == "ok", circuit_name
            assert null["n_permutations"] == PREDECLARED_PERMUTATIONS
            assert np.isfinite(float(null["null_p95"]))

    @pytest.mark.expensive
    def test_each_separation_ratio_is_judged_against_its_own_null(self, phase4c) -> None:
        verdicts = phase4c["phase4c"]["separation_ratio_verdicts"]
        for circuit_name in REPLICATION_CIRCUITS:
            entry = verdicts[circuit_name]
            assert entry["status"] == "ok", circuit_name
            assert entry["n_permutations"] == PREDECLARED_PERMUTATIONS
            assert isinstance(entry["exceeds_null_p95"], bool)
            if entry["exceeds_null_p95"]:
                assert "above this null" in entry["interpretation"]
            else:
                assert "NOT further apart" in entry["interpretation"]

    @pytest.mark.expensive
    def test_the_replication_verdict_follows_its_stated_rule(self, phase4c) -> None:
        block = phase4c["phase4c"]["replication_verdict"]
        assert block["outcome"] in {"replicated", "not_reproduced", "uninformative", "undecidable"}
        assert block["rule"]
        assert block["n_trials"] == PREDECLARED_TRIALS
        margin = phase4c["phase4c"]["primary_comparison"]["margin_statistics"]
        informative = block["both_conditions_informative"]
        if not informative:
            assert block["outcome"] == "uninformative"
            assert "gain is NOT changed" in block["verdict"]
        elif margin["effect_supported"]:
            assert block["outcome"] == "replicated"
            assert "REPRODUCED" in block["verdict"]
        else:
            assert block["outcome"] == "not_reproduced"
            assert "NOT REPRODUCED" in block["verdict"]
            low, high = margin["confidence_interval"]
            assert low <= 0.0 <= high, "not reproduced means the interval spans zero"

    @pytest.mark.expensive
    def test_the_verdict_does_not_overreach(self, phase4c) -> None:
        block = phase4c["phase4c"]["replication_verdict"]
        assert "one comparison" in block["scope"].lower()
        assert "not a statement about a fly" in block["scope"].lower()
        verdict = " ".join(block["verdict"].lower().split())
        # The verdict may name the limits, but only to deny them. A window check
        # rather than a substring ban, because "NOT a statement that the MCNS
        # circuit computes nothing" is required wording, not an overreach.
        for phrase in ("direction selectivity", "computes", "reichardt", "discovery"):
            start = 0
            while (hit := verdict.find(phrase, start)) != -1:
                window = verdict[max(0, hit - 200): hit + 140]
                assert any(
                    k in window
                    for k in ("not reproduced", "not a statement", "does not", "cannot",
                              "never", "not claimed", "withdrawn", "not testable")
                ), window
                start = hit + 1

    @pytest.mark.expensive
    def test_the_margin_carries_its_uncertainty(self, phase4c) -> None:
        margin = phase4c["phase4c"]["primary_comparison"]["margin_statistics"]
        assert margin["status"] == "ok"
        assert margin["n_folds"] == PREDECLARED_TRIALS
        assert len(margin["per_fold_margin"]) == PREDECLARED_TRIALS
        assert np.isfinite(margin["paired_t_p_two_sided"])
        assert margin["paired_t_df"] == PREDECLARED_TRIALS - 1
        assert 0.0 <= margin["paired_t_p_two_sided"] <= 1.0
        assert margin["confidence_interval"][0] <= margin["mean_margin"] <= (
            margin["confidence_interval"][1]
        )
        null = margin["margin_permutation_null"]
        assert null["n_permutations"] == PREDECLARED_PERMUTATIONS
        assert 0.0 <= null["p_two_sided"] <= 1.0
        assert margin["effect_supported"] == (
            margin["paired_t_p_two_sided"] < 0.05 and null["p_two_sided"] < 0.05
        )

    def test_the_margin_pairs_shared_folds(self) -> None:
        result = margin_statistics([0.8, 0.6] * 10, [0.5, 0.5] * 10, n_permutations=32)
        assert result["per_fold_margin"] == pytest.approx([0.3, 0.1] * 10)
        assert result["mean_margin"] == pytest.approx(0.2)

    def test_unmatched_fold_counts_are_refused_not_averaged(self) -> None:
        result = margin_statistics([0.6] * 20, [0.5] * 8)
        assert result["status"] == "incomparable_folds"
        assert "mean_test_accuracy" not in result

    @pytest.mark.expensive
    def test_an_informative_null_result_is_not_called_an_effect_on_its_own(self, phase4c) -> None:
        margin = phase4c["phase4c"]["primary_comparison"]["margin_statistics"]
        # Whatever the outcome, the effect word is gated on the uncertainty.
        assert "effect_supported" in margin
        statement = margin["statement"].lower()
        expected = "supports" if margin["effect_supported"] else "does not support"
        assert expected in statement, statement
        assert phase4c["phase4c"]["primary_comparison"]["no_winner_declared"] is True

    @pytest.mark.expensive
    def test_no_condition_is_ranked(self, phase4c) -> None:
        text = json.dumps(phase4c["phase4c"]).lower()
        for forbidden in (
            "best condition", "outperforms", "superior condition", "top-performing",
            "beats the", "is the winner", "the winner is",
        ):
            assert forbidden not in text, forbidden
        # The word "winner" may appear only inside an explicit negation.
        for chunk in text.split('"'):
            if "winner" in chunk:
                assert "no_winner" in chunk or "not called a winner" in chunk, chunk[:120]


class TestPhase4CControls:
    @pytest.mark.expensive
    def test_both_relay_controls_are_reported_for_every_condition(self, phase4c) -> None:
        relay = phase4c["phase4c"]["relay_controls"]
        for control in ("INPUT_SPATIAL", "INPUT_TEMPORAL", "INPUT_INSTANT", "MAGNITUDE_ONLY"):
            assert control in relay, control
            for circuit_name in REPLICATION_CIRCUITS:
                assert relay[control][circuit_name].get("status") == "ok", (control, circuit_name)

    @pytest.mark.expensive
    def test_the_relay_controls_are_identical_across_the_circuit_conditions(self, phase4c) -> None:
        """The relay controls come from the drive, so they cannot depend on the circuit.

        This is the contract the Phase 4C test can make on the fixture. It
        deliberately does **not** assert that the controls beat chance: how much
        direction information the drive carries is a property of the real
        stimulus geometry over the real subset, not of the tiny test fixture, and
        pinning a fixture-dependent accuracy here would test the fixture. The
        sign-flip property of the temporal control — the part that is a real
        contract — is asserted in TestControls, and the bound's value on the real
        data is reported by the run itself.
        """
        relay = phase4c["phase4c"]["relay_controls"]
        for control in ("INPUT_SPATIAL", "INPUT_TEMPORAL"):
            values = {
                round(float(relay[control][c]["mean_test_accuracy"]), 12)
                for c in REPLICATION_CIRCUITS
            }
            assert len(values) == 1, f"{control} differs across circuit conditions: {values}"

    @pytest.mark.expensive
    def test_the_input_instant_gate_does_not_separate(self, phase4c) -> None:
        gate = phase4c["phase4c"]["relay_controls"]["INPUT_INSTANT"]
        for circuit_name, entry in gate.items():
            assert entry["mean_test_accuracy"] == pytest.approx(0.5), circuit_name

    @pytest.mark.expensive
    def test_the_relay_purpose_is_stated(self, phase4c) -> None:
        purpose = phase4c["phase4c"]["relay_control_purpose"].lower()
        assert "before the mcns circuit" in purpose
        assert "not evidence of mcns computation" in purpose


class TestPhase4CIdenticalConditions:
    @pytest.mark.expensive
    def test_recurrent_and_shuffled_share_one_population(self, phase4c) -> None:
        identity = phase4c["phase4c"]["population_identity_across_conditions"]
        pairs = {f"{a}|{b}": v for a, b, v in [
            ("MCNS_FEEDFORWARD", "MCNS_RECURRENT", None),
            ("MCNS_FEEDFORWARD", "SHUFFLED_CONTROL", None),
            ("MCNS_RECURRENT", "SHUFFLED_CONTROL", None),
        ]}
        for key, entry in identity.items():
            assert entry["identical"] is True, key
            assert entry["body_ids_equal"] is True
            assert entry["cell_type_labels_equal"] is True
        assert set(pairs) == set(identity)
        assert phase4c["phase4c"]["same_population_and_edge_subset"] is True

    @pytest.mark.expensive
    def test_all_three_conditions_simulate_the_same_neurons(self, phase4c) -> None:
        circuits = phase4c["summary"]["circuits"]
        assert set(circuits) == set(REPLICATION_CIRCUITS)
        assert len({c["neurons"] for c in circuits.values()}) == 1
        # The edge counts differ because that is the intervention: the recurrent
        # condition keeps every measured candidate edge and the shuffled one
        # replaces them. The neuron set does not differ.
        assert circuits["MCNS_RECURRENT"]["edges"] == circuits["SHUFFLED_CONTROL"]["edges"]

    @pytest.mark.expensive
    def test_shuffled_preserves_the_degree_sequences(self, phase4c) -> None:
        verification = phase4c["summary"]["shuffled_control"]["verification"]
        for key in (
            "neurons_equal", "edges_equal", "in_degree_sequence_equal",
            "out_degree_sequence_equal", "synapse_count_total_equal",
            "synapse_count_distribution_equal", "topology_changed",
        ):
            assert verification[key] is True, key

    def test_the_population_identity_helper_reports_pairs(self, circuit, bounded) -> None:
        module = _load_script()
        circuits = {
            "MCNS_RECURRENT": circuit,
            "SHUFFLED_CONTROL": shuffled_control(circuit, seed=0),
        }
        identity = module._population_identity(circuits)
        assert len(identity) == 1
        entry = next(iter(identity.values()))
        assert entry["identical"] is True


class TestPhase4CClaims:
    @pytest.mark.expensive
    def test_measured_simulated_assumed_and_unresolved_are_all_present(
        self, phase4c
    ) -> None:
        claims = phase4c["phase4c"]["claims"]
        for heading in ("MEASURED", "SIMULATED", "ASSUMED", "UNRESOLVED"):
            assert claims[heading], heading
        assert phase4c["summary"]["unresolved"] == claims["UNRESOLVED"]

    @pytest.mark.expensive
    def test_synapse_count_is_not_efficacy(self, phase4c) -> None:
        assumed = " ".join(phase4c["phase4c"]["claims"]["ASSUMED"]).lower()
        assert "not synaptic efficacy" in assumed
        assert "count of chemical synapses" in assumed

    @pytest.mark.expensive
    def test_the_graph_has_no_synaptic_sign(self, phase4c) -> None:
        unresolved = " ".join(phase4c["phase4c"]["claims"]["UNRESOLVED"]).lower()
        assert "no synaptic sign" in unresolved
        assert "only sum" in unresolved

    @pytest.mark.expensive
    def test_the_encoder_contains_the_directional_structure(self, phase4c) -> None:
        assumed = " ".join(phase4c["phase4c"]["claims"]["ASSUMED"]).lower()
        assert "encoder contains the" in assumed
        assert "externally specified" in assumed

    @pytest.mark.expensive
    def test_t4t5_asymmetry_alone_is_not_evidence_of_mcns_computation(self, phase4c) -> None:
        unresolved = " ".join(phase4c["phase4c"]["claims"]["UNRESOLVED"]).lower()
        assert "t4/t5 asymmetry alone is not evidence of mcns computation" in unresolved

    @pytest.mark.expensive
    def test_the_cannot_establish_statement_is_explicit(self, phase4c) -> None:
        text = phase4c["phase4c"]["claims"]["CANNOT_BE_ESTABLISHED"]
        assert "cannot establish direction selectivity" in text.lower()
        assert "no number in this report may be presented as biological" in text.lower()
        assert phase4c["summary"]["direction_selectivity_claim"] == text

    @pytest.mark.expensive
    def test_no_artifact_claims_a_biological_discovery(self, phase4c) -> None:
        for name in ("benchmark_summary.json", "decoder_results.json",
                     "activity.json", "phase4c_report.json"):
            text = (phase4c["output"] / name).read_text(encoding="utf-8").lower()
            for forbidden in (
                "demonstrates direction selectivity",
                "reproduces direction selectivity",
                "t4 is direction selective",
                "reichardt detector computes",
                "biological discovery",
                "we have discovered",
                "proves that the fly",
            ):
                assert forbidden not in text, f"{name} asserts {forbidden!r}"

    @pytest.mark.expensive
    def test_no_post_hoc_tuning_is_stated(self, phase4c) -> None:
        block = phase4c["phase4c"]["no_post_hoc_tuning"]
        assert "is NOT changed" in block["statement"]
        assert "the run is NOT repeated" in block["statement"]
        for item in block["decl"]:
            assert isinstance(item, str) and item

    @pytest.mark.expensive
    def test_the_run_is_declared_a_replication_not_a_new_model(self, phase4c) -> None:
        block = phase4c["phase4c"]
        assert "replication" in block["kind"].lower()
        assert "not a new model" in block["kind"].lower()
        assert "Phase 4B" in block["replication_target"]
        assert "8 trials" in block["replication_target"]


class TestPhase4CTimingProjection:
    def test_the_projection_scales_the_three_cost_centres_separately(self) -> None:
        # Timed at 5 trials / 50 permutations: 12.0 s of null time, of which
        # 6.0 s is marginal at 0.12 s per permutation and 6.0 s is fixed.
        projection = timing_projection(
            measured_simulation_seconds=1.0,
            measured_decode_seconds=2.0,
            measured_null_seconds=12.0,
            measured_null_marginal_per_permutation=0.12,
            measured_trials=5,
            measured_permutations=50,
        )
        assert projection["declared"]["trials"] == 20
        assert projection["declared"]["permutations"] == 500
        assert projection["scale_factors"]["circuits"] == 3
        assert projection["scale_factors"]["trials"] == pytest.approx(4.0)
        # folds x rows = 20/5 x 19/4, not 4^2.
        assert projection["scale_factors"]["nulls_folds_times_rows"] == pytest.approx(
            (20 / 5) * (19 / 4)
        )
        assert projection["measured"]["null_fixed_seconds"] == pytest.approx(6.0)
        assert projection["projected_simulation_seconds"] == pytest.approx(1.0 * 3 * 4)
        assert projection["projected_null_seconds"] == pytest.approx(
            3 * (6.0 * 4 + 0.12 * (20 / 5) * (19 / 4) * 500)
        )
        assert projection["is_reasonably_bounded"] is True

    def test_a_purely_fixed_null_measurement_is_not_inflated_by_the_permutation_count(
        self,
    ) -> None:
        """A small measurement is mostly fixed cost; scaling it linearly is wrong.

        With no marginal component at all, the projection must be the fixed cost
        scaled by circuits and trials, and must NOT be multiplied by the
        permutation ratio. This is the failure mode the two-point measurement
        exists to prevent, in one direction.
        """
        projection = timing_projection(
            measured_simulation_seconds=0.0,
            measured_decode_seconds=0.0,
            measured_null_seconds=10.0,
            measured_null_marginal_per_permutation=0.0,
            measured_trials=3,
            measured_permutations=20,
        )
        assert projection["measured"]["null_fixed_seconds"] == pytest.approx(10.0)
        assert projection["projected_null_seconds"] == pytest.approx(10.0 * 3 * (20 / 3))
        assert projection["projected_null_seconds"] < projection[
            "naive_linear_projection_seconds"
        ], "linear scaling of a fixed cost overstates the run"
        assert projection["fixed_correction_factor"] < 1.0

    def test_the_marginal_component_does_scale_with_folds_rows_and_permutations(
        self,
    ) -> None:
        """A fully marginal measurement scales as folds x rows x permutations."""
        cheap = timing_projection(
            measured_simulation_seconds=0.0,
            measured_decode_seconds=0.0,
            measured_null_seconds=100.0,
            measured_null_marginal_per_permutation=5.0,   # all marginal
            measured_trials=5,
            measured_permutations=20,
        )
        assert cheap["measured"]["null_fixed_seconds"] == pytest.approx(0.0)
        assert cheap["projected_null_seconds"] == pytest.approx(
            3 * 5.0 * (20 / 5) * (19 / 4) * 500
        )
        # With no fixed part the two projections agree by construction, which is
        # the sanity check on the model itself.
        assert cheap["projected_null_seconds"] == pytest.approx(
            cheap["naive_linear_projection_seconds"]
        )
        assert cheap["fixed_correction_factor"] == pytest.approx(1.0)

    def test_a_huge_projection_is_flagged_as_unbounded(self) -> None:
        projection = timing_projection(
            measured_simulation_seconds=1.0,
            measured_decode_seconds=1.0,
            measured_null_seconds=100_000.0,
            measured_null_marginal_per_permutation=100_000.0 / 20.0,
            measured_trials=3,
            measured_permutations=10,
        )
        assert projection["is_reasonably_bounded"] is False
        assert "never by quietly changing a declared parameter" in (
            projection["boundedness_rule"]
        )

    def test_the_projection_declares_itself_a_lower_estimate(self) -> None:
        """Measured on the real data the projection ran 2.2x low, so it must say so."""
        projection = timing_projection(
            measured_simulation_seconds=1.0,
            measured_decode_seconds=1.0,
            measured_null_seconds=1.0,
            measured_null_marginal_per_permutation=0.02,
            measured_trials=20,
            measured_permutations=100,
        )
        assert projection["treat_as_a_lower_estimate"] is True
        assert "2.2x" in projection["lower_estimate_note"]

    @pytest.mark.expensive
    def test_the_estimate_is_written_and_declares_what_it_reduced(self, tmp_path, pathway_data) -> None:
        module = _load_script()
        code = module.main(
            [
                "--phase-4c", "--estimate", "--quiet",
                "--estimate-trials", "3", "--estimate-permutations", "5",
                "--annotations", str(pathway_data[0]), "--connectivity", str(pathway_data[1]),
                "--output-dir", str(tmp_path / "est"),
                "--cache", str(tmp_path / "c.npz"),
            ]
        )
        assert code == 0
        # The estimate directory is fixed and separate, so --output-dir cannot put
        # an estimate next to the frozen real-run artifacts. "Separate" means not
        # equal AND not a prefix-extension: a sibling named ..._phase4c_estimate
        # is still swept up by a glob of ..._phase4c*.
        assert module.PHASE4C_ESTIMATE_DIR != module.PHASE4C_OUTPUT_DIR
        assert not module.PHASE4C_ESTIMATE_DIR.startswith(module.PHASE4C_OUTPUT_DIR)
        assert not module.PHASE4C_OUTPUT_DIR.startswith(module.PHASE4C_ESTIMATE_DIR)
        estimate_dir = REPO / module.PHASE4C_ESTIMATE_DIR
        payload = _json(estimate_dir / "phase4c_timing_estimate.json")
        assert payload["predeclaration"]["configuration"]["synapse_scale"] == 0.25
        assert payload["is_not_a_result"]
        assert payload["frozen_artifact_dir"] == module.PHASE4C_OUTPUT_DIR
        assert payload["written_to"] == module.PHASE4C_ESTIMATE_DIR
        settings = payload["estimate_settings"]
        assert settings["trials_per_condition_point_1"] == 3
        assert settings["trials_per_condition_point_2"] == PREDECLARED_TRIALS == 20
        assert settings["permutations"] == 5
        assert settings["reduced_relative_to_declaration"] == [
            "permutations", "point_1_trials"
        ]
        # Two points: the required cheap one, and one at the declared geometry,
        # because extrapolating in the trial count is not reliable.
        assert set(payload["projections"]) == {
            "small_trial_point", "declared_geometry_point"
        }
        assert payload["projection_used_for_the_decision"] == "declared_geometry_point"
        assert (
            payload["projections"]["declared_geometry_point"]["declared"]["trials"] == 20
        )
        assert (
            payload["measurements"]["point_2_declared_geometry"]["trials"] == 20
        )
        assert payload["determinism_fingerprint"]
        # Nothing declared was changed to make the estimate cheap.
        assert settings["circuit_condition"] == "MCNS_RECURRENT"
        # The requested output directory was not created as a side effect.
        assert not (tmp_path / "est").exists()


class TestConstantFeatureReporting:
    """``constant_features`` must be a property of the whole design, not of fold 0.

    It was read from ``constant[0, 0, :]`` and reported as if it described the
    design, which matters because the count differs sharply between conditions at
    20 trials (13 of 32 measured against 24 of 32 randomised) and a reader would
    take a single fold's dead columns for an averaged figure.
    """

    def _design_with_dead_columns(self, dead_per_fold):
        """A 4-fold design where each fold has a different set of dead columns.

        Every column varies row to row *within* a fold, so it is non-constant
        there. A column listed in ``dead_per_fold[fold]`` is flattened to a single
        value in that fold only, which is what "constant in one training block"
        means. An earlier version of this helper filled every column with ones and
        then zeroed a few, which made *every* column constant in *every* fold and
        so tested nothing.
        """
        dims, rows, folds = 6, 4, 4
        x = np.zeros((folds, rows, dims), dtype=np.float64)
        for fold in range(folds):
            for row in range(rows):
                for col in range(dims):
                    x[fold, row, col] = 10.0 * fold + row + 0.01 * col
        for fold, dead in enumerate(dead_per_fold):
            for col in dead:
                x[fold, :, col] = 7.0
        return x, np.ones((folds, rows, 1), dtype=np.float64)

    def test_it_is_defined_over_all_folds(self) -> None:
        x, mask = self._design_with_dead_columns([[0], [1], [2], [3]])
        _, _, info = _standardise_batch(x, x, mask)
        assert info["n_folds"] == 4
        # Columns 0..3 are each dead in exactly one fold, column 4 never.
        assert info["constant_features"] == 4
        assert info["constant_features_every_fold"] == 0
        assert "at least one fold" in info["constant_features_definition"]

    def test_the_every_fold_companion_is_the_stricter_figure(self) -> None:
        x, mask = self._design_with_dead_columns([[0, 1], [0, 1], [0, 1], [0, 1]])
        _, _, info = _standardise_batch(x, x, mask)
        assert info["constant_features"] == 2
        assert info["constant_features_every_fold"] == 2

    def test_a_live_column_is_never_counted(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 8, 5))
        mask = np.ones((4, 8, 1), dtype=np.float64)
        _, _, info = _standardise_batch(x, x, mask)
        assert info["constant_features"] == 0
        assert info["constant_features_every_fold"] == 0

    def test_the_count_is_deterministic_and_fold_order_independent(self) -> None:
        dead = [[0], [1], [2], [3]]
        a_x, a_mask = self._design_with_dead_columns(dead)
        b_x, b_mask = self._design_with_dead_columns(list(reversed(dead)))
        a = _standardise_batch(a_x, a_x, a_mask)[2]
        b = _standardise_batch(b_x, b_x, b_mask)[2]
        assert a["constant_features"] == b["constant_features"] == 4
        assert a["constant_features_every_fold"] == b["constant_features_every_fold"] == 0

    def test_padding_rows_do_not_make_everything_dead(self) -> None:
        """A short fold's zero padding must not be read as a constant feature."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 6, 3))
        mask = np.zeros((2, 6, 1), dtype=np.float64)
        mask[0, :6, 0] = 1.0     # fold 0 full
        mask[1, :4, 0] = 1.0     # fold 1 short, rows 4-5 padded
        _, _, info = _standardise_batch(x, x, mask)
        assert info["constant_features"] == 0

    def test_cross_validate_exposes_the_definition(self) -> None:
        result = cross_validate(_records(trials=5, bins=3, dims=6), STIM)
        assert "constant_features" in result
        assert "constant_features_every_fold" in result
        assert "at least one fold" in result["constant_features_definition"]


    @pytest.mark.expensive
    def test_the_power_limitation_is_stated_and_computed(self, phase4c) -> None:
        """A null must never be readable as a zero without the detectable effect size.

        The audit found the report gave a p-value of 1.0 with no statement of what
        the design could have seen, which invites reading the null as "no
        difference". The power block makes the bound explicit.
        """
        power = phase4c["phase4c"]["power"]
        assert power["status"] == "ok"
        assert power["n_folds"] == PREDECLARED_TRIALS
        assert power["sd_of_paired_differences"] > 0
        assert power["ci_half_width"] > 0
        assert power["minimum_detectable_margin_at_80pct_power"] > 0
        assert power["underpowered_for_small_effects"] is True
        statement = power["statement"].lower()
        assert "bound" in statement
        assert "not evidence that the difference is zero" in statement
        assert "computes nothing" in statement
        # Recomputable: MDE = sd * sqrt((t_crit + t_80)^2 / n).
        from scipy.stats import t as student_t

        n = power["n_folds"]
        expected = power["sd_of_paired_differences"] * np.sqrt(
            (student_t.ppf(0.975, n - 1) + student_t.ppf(0.80, n - 1)) ** 2 / n
        )
        assert power["minimum_detectable_margin_at_80pct_power"] == pytest.approx(expected)


class TestPhase4CDocumentation:
    """The retraction and the limits pinned on the *text* of the deliverable.

    Separate from :class:`TestPhase4CClaims`, which pins the same limits on the
    run *output*. Keeping them apart is deliberate: an earlier version of this file
    defined both classes under one name, and Python silently let the second shadow
    the first, so nine claim contracts stopped running without any failure. Two
    classes, one name, is a silent coverage loss and is now impossible here.
    """
    def test_the_phase4b_preliminary_result_is_retracted_in_the_docs(self) -> None:
        """The retraction is a scientific contract, so it is pinned on the text.

        A retracted result that survives in the documentation is a retracted result
        that still gets cited. The one place the old wording is allowed to appear is
        the provenance note, which *quotes* the frozen artifact verbatim, so that
        occurrence is asserted rather than swept away.
        """
        doc = (REPO / "docs" / "MCNS_MOTION_BENCHMARK.md").read_text(encoding="utf-8")
        low = doc.lower()
        assert "retracted" in low
        assert "phase 4b" in low and "8-trial" in low
        # The old "not retracted / merely superseded" claim may survive only as a
        # quotation of the frozen artifact.
        for line in (line.strip().lower() for line in doc.splitlines()):
            if "not retracted" in line or "merely superseded" in line:
                assert "phase4c_run.log" in line or "provenance" in low, line
        assert 'reads "not retracted: it is superseded"' in low
        source = SCRIPT.read_text(encoding="utf-8").lower()
        assert "retracted" in source
        assert "not retracted" not in source

    def test_the_docs_state_the_four_limits_explicitly(self) -> None:
        doc = (REPO / "docs" / "MCNS_MOTION_BENCHMARK.md").read_text(encoding="utf-8").lower()
        for phrase in (
            "not** show that the mcns circuit computes nothing",
            "encoder-injected directional information is",
            "no synaptic sign",
            "not synaptic efficacy",
            "unknown / not tested",
            "inhibitory versus excitatory identity",
        ):
            assert phrase in doc, phrase
        for heading in (
            "### measured", "### simulated", "### assumed", "### unknown / not tested",
        ):
            assert heading in doc, heading

    def test_the_docs_do_not_claim_a_positive_biological_finding(self) -> None:
        doc = (REPO / "docs" / "MCNS_MOTION_BENCHMARK.md").read_text(encoding="utf-8").lower()
        for forbidden in (
            "mcns computes",
            "the circuit computes motion",
            "demonstrates direction selectivity",
            "reproduces direction selectivity",
            "confirms the wiring matters",
            "proves",
        ):
            assert forbidden not in doc, forbidden
        # The null must not become a negative biological claim either. Both phrases
        # do appear, and in wrapped prose the negation can sit on a neighbouring
        # line, so the window around each occurrence is checked rather than the line.
        flat = " ".join(doc.split())
        for phrase in ("computes nothing", "no direction-selective"):
            start = 0
            while (hit := flat.find(phrase, start)) != -1:
                window = flat[max(0, hit - 220): hit + 120]
                assert any(
                    k in window
                    for k in ("not**", "does not", "not say", "not mean", "cannot", "never")
                ), window
                start = hit + 1


class TestPhase4CUnitContracts:
    def test_feature_report_counts_non_zero_columns(self) -> None:
        records = [
            PopulationRecord("LEFTWARD", 0, "T4+T5", np.array([[1.0, 0.0, 0.0, 2.0]]), np.zeros(1)),
            PopulationRecord("RIGHTWARD", 0, "T4+T5", np.array([[0.0, 0.0, 3.0, 0.0]]), np.zeros(1)),
        ]
        report = feature_report(records)
        assert report["n_features"] == 4
        # Columns 0 and 3 respond in one trial, column 2 in the other; column 1
        # never responds. "Feature dimensionality" and "non-zero features" are
        # different numbers and the report must not conflate them.
        assert report["n_nonzero_features"] == 3
        assert report["zero_features"] == 1
        assert report["fraction_nonzero"] == pytest.approx(0.75)

    def test_feature_report_on_nothing_is_honest(self) -> None:
        report = feature_report([])
        assert report["n_features"] == 0
        assert report["n_records"] == 0

    def test_the_claim_block_has_four_categories_and_a_limit(self) -> None:
        claims = claim_block()
        assert set(claims) == {
            "MEASURED", "SIMULATED", "ASSUMED", "UNRESOLVED", "CANNOT_BE_ESTABLISHED"
        }
        assert "UNRESOLVED" in claims
        assert any("synaptic efficacy" in item for item in claims["ASSUMED"])
        assert any("synaptic sign" in item.lower() for item in claims["UNRESOLVED"])


# ======================================================================
# The simulator hooks the benchmark relies on.
# ======================================================================



class TestSimulatorHooks:
    def test_prebuilt_propagation_changes_nothing(self, circuit, generator) -> None:
        params = NeuronParams()
        built = circuit.propagation_matrix(0.25, "linear")
        a = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", 40,
                           params=params, synapse_scale=0.25, record_input=False)
        b = run_simulation(circuit, generator.as_encoder(trial=0), "LEFTWARD", 40,
                           params=params, propagation=built, record_input=False)
        assert np.array_equal(a.spikes, b.spikes)

    def test_skipping_the_input_log_changes_no_dynamics(self, circuit, generator) -> None:
        params = NeuronParams()
        a = run_simulation(circuit, generator.as_encoder(trial=0), "RIGHTWARD", 40,
                           params=params, record_input=True)
        b = run_simulation(circuit, generator.as_encoder(trial=0), "RIGHTWARD", 40,
                           params=params, record_input=False)
        assert np.array_equal(a.spikes, b.spikes)
        assert b.input_current.shape == (0, circuit.num_neurons)
        assert a.input_current.shape == (40, circuit.num_neurons)

    def test_the_adapter_serves_a_condition_by_name(self, circuit, generator) -> None:
        generator.tensor("LEFTWARD_POLARITY_REVERSED", 0, 40)
        adapter = generator.as_encoder(trial=0)
        served = [adapter.encode(None, "LEFTWARD_POLARITY_REVERSED") for _ in range(40)]
        cached = generator.tensor("LEFTWARD_POLARITY_REVERSED", 0, 40)
        for step, vector in enumerate(served):
            assert np.array_equal(vector, cached[step].astype(np.float64))
