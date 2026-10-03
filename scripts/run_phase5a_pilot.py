#!/usr/bin/env python
"""Phase 5A pilot: does the measured wiring deliver spatially offset drive to T4/T5?

This is a **new and smaller question**, not an extension of the Phase 4B/4C
benchmark. Phase 4C established a null for the recurrent-versus-shuffled contrast
at 20 trials and retracted the Phase 4B preliminary signal; that result says
nothing about whether the circuit computes anything, and it is underpowered below
roughly |margin| = 0.104.

Phase 5A asks what the unsigned MCNS wiring *can* answer:

> **Does the measured, unsigned MCNS wiring deliver spatially offset,
> subtype-differentiated drive to T4/T5, relative to a strength-preserving
> rewiring of the same graph?**

Three arms, and the existing control is **retained**:

``MEASURED``
    the measured recurrent graph, synapse counts unmodified
``DRIVE_MATCHED_CONTROL``
    the primary control. A rewiring that preserves per-neuron in-degree,
    out-degree, incoming weight and outgoing weight exactly, and re-randomises
    only which target each source drives. This is the fix for the defect the
    Phase 4C audit found in the old control, where per-cell-type incoming drive
    moved by 0.16x to 2.75x.
``EXISTING_SHUFFLE``
    the Phase 4C degree-preserving configuration-model randomisation, kept as a
    **secondary** control. It preserves the weight multiset and does not preserve
    per-cell-type drive, so it is labelled drive-mismatched wherever it appears.

Direction selectivity is out of scope. See
:data:`flybrain.benchmark.phase5a.DIRECTION_SELECTIVITY_OUT_OF_SCOPE`.

Usage
-----
    python scripts/run_phase5a_pilot.py --variance-estimation   # stage 1: sd only
    python scripts/run_phase5a_pilot.py --trials N --sd S       # stage 2: the pilot

The variance-estimation stage reports a variance and nothing else. It exists so
the trial count is fixed **before** the main run, and it must be run first: the
pilot refuses to start without an explicit ``--sd``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from flybrain.benchmark.lateral import (  # noqa: E402
    LATERAL_READOUT_DISCLAIMER,
    calibration_shift_null,
    lateral_profile,
    profile_centroid,
)
from flybrain.benchmark.matched import (  # noqa: E402
    DRIVE_MATCHED_DISCLAIMER,
    drive_matched_control,
    verify_drive_matching,
)
from flybrain.benchmark.liveness_gated import (  # noqa: E402
    ATTEMPT_SEED_POLICY,
    CONTROL_NAME,
    MAX_ATTEMPTS,
    STRUCTURAL_CONSTRAINTS,
    select_liveness_gated_control,
)
from flybrain.benchmark.phase5a import (  # noqa: E402
    ARMS,
    DIRECTION_SELECTIVITY_OUT_OF_SCOPE,
    GATE_ORDER,
    PREDECLARED,
    PREDECLARED_PHASE,
    PRIMARY_ARM,
    PRIMARY_CONTROL,
    SECONDARY_CONTROL,
    arms,
    claim_block,
    evaluate_gates,
    power_requirement,
    predeclared_block,
)
from flybrain.benchmark.phase4c import liveness  # noqa: E402
from flybrain.benchmark.phase4c import margin_statistics  # noqa: E402
from flybrain.benchmark.shuffled import (  # noqa: E402
    SHUFFLED_DISCLAIMER,
    shuffled_control,
    verify_degree_preservation,
)
from flybrain.benchmark.stimulus import (  # noqa: E402
    MotionStimulusGenerator,
    StimulusCondition,
    lateral_coordinates,
)
from flybrain.benchmark.subset import (  # noqa: E402
    SUBSET_DISCLAIMER,
    bounded_population,
    restrict_circuit,
)
from flybrain.brain.mcns_circuit import (  # noqa: E402
    MODE_FEEDFORWARD,
    MODE_RECURRENT,
    MCNSCircuit,
    candidate_metadata,
)
from flybrain.brain.mcns_simulation import run_simulation  # noqa: E402
from flybrain.brain.neuron import NeuronParams  # noqa: E402
from flybrain.data.motion_candidate import (  # noqa: E402
    MOTION_CELL_TYPES,
    EdgeExtraction,
    build_populations,
    extract_candidate_edges,
    load_annotations,
)

DEFAULT_DATA_DIR = "data/raw"
ANNOTATION_NAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CONNECTIVITY_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_OUTPUT = "outputs/mcns_phase5a_pilot"
VARIANCE_OUTPUT = "outputs/mcns_phase5a_variance"

#: Readout groups. T4 and T5 are separate so a lateral effect cannot be expressed
#: as a T4-versus-T5 effect, which is exactly the confound the Phase 4C
#: subtype x bin product could not separate.
READOUT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("T4", ("T4a", "T4b", "T4c", "T4d")),
    ("T5", ("T5a", "T5b", "T5c", "T5d")),
)

#: Per-subtype centroids, reported as secondary.
SUBTYPE_GROUPS: tuple[tuple[str, tuple[str]], ...] = (
    (name, (name,)) for name in
    ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
)

PRIMARY_CONTRAST = ("LEFTWARD", "RIGHTWARD")
STATIC_CONTRAST = ("LEFTWARD", "STATIC")
SUBTYPE_PRIMARY = "T4"
_W = 78


class Progress:
    """Stage counter with flushed output, so a long run is never opaque."""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = int(total)
        self.enabled = bool(enabled)
        self.index = 0
        self._started = time.perf_counter()

    def stage(self, label: str) -> None:
        self.index += 1
        if self.enabled:
            print(f"\n[{self.index}/{self.total}] {label}", flush=True)

    def note(self, text: str) -> None:
        if self.enabled:
            print(f"    {text}", flush=True)

    def total_seconds(self) -> float:
        return time.perf_counter() - self._started


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _load_extraction(annotations_path: str, connectivity_path: str, cache: Path) -> EdgeExtraction:
    """Reuse the Phase 4C extraction cache when it matches the population.

    Keyed on the full candidate body list, never on a subset, so subset mode never
    invalidates it and never re-reads the 1 GB connectivity table.
    """
    populations = build_populations(
        load_annotations(annotations_path), MOTION_CELL_TYPES, require_superclass=True
    )
    if cache.is_file():
        blob = np.load(cache, allow_pickle=False)
        if int(blob["n_bodies"]) == int(populations.num_bodies) and np.array_equal(
            blob["body_ids"], np.asarray(populations.body_ids, dtype=np.int64)
        ):
            print(f"  reusing cached extraction: {cache}")
            return EdgeExtraction(
                pre_index=blob["pre_index"].astype(np.int64),
                post_index=blob["post_index"].astype(np.int64),
                synapse_count=blob["synapse_count"].astype(np.int64),
                batches_processed=int(blob["batches_processed"]),
                batches_total=int(blob["batches_total"]),
                rows_scanned=int(blob["rows_scanned"]),
                read_budget=int(blob["read_budget"]),
                truncated=bool(blob["truncated"]),
                direction=str(blob["direction"]),
                seconds=0.0,
            )
    extraction = extract_candidate_edges(connectivity_path, populations, direction="both")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        pre_index=np.asarray(extraction.pre_index, dtype=np.int64),
        post_index=np.asarray(extraction.post_index, dtype=np.int64),
        synapse_count=np.asarray(extraction.synapse_count, dtype=np.int64),
        batches_processed=extraction.batches_processed,
        batches_total=extraction.batches_total,
        rows_scanned=extraction.rows_scanned,
        read_budget=extraction.read_budget,
        truncated=extraction.truncated,
        direction=str(extraction.direction),
        n_bodies=int(populations.num_bodies),
        body_ids=np.asarray(populations.body_ids, dtype=np.int64),
    )
    print(f"  wrote extraction cache: {cache}")
    return extraction


def build_arms(
    populations, extraction, annotations, annotations_path: str,
    bodies_per_type: int, seed: int,
) -> tuple[dict[str, MCNSCircuit], Any, Any]:
    """Build the population and all three arms, asserting a shared neuron set.

    ``EXISTING_SHUFFLE`` is built and kept, not replaced. Reporting only the new
    control would be a choice of convenience; the two answer different questions.

    ``annotations`` is the loaded table for building the circuits, but
    ``candidate_metadata`` re-reads the Feather file by path to filter inside Arrow,
    so both are needed. Passing the loaded dict instead raises ``TypeError`` from
    ``pathlib``, which is how this signature got its path argument.
    """
    full = MCNSCircuit.from_extraction(
        populations, extraction, annotations, mode=MODE_RECURRENT
    )
    subset = bounded_population(
        populations, extraction.pre_index, extraction.post_index,
        synapse_count=extraction.synapse_count, bodies_per_type=int(bodies_per_type),
    )
    measured = restrict_circuit(full, subset.positions)
    meta = candidate_metadata(annotations_path, measured.body_ids)
    measured.with_soma_locations(meta)

    existing = shuffled_control(measured, seed=int(seed))
    existing.soma_location = measured.soma_location.copy()
    matched = drive_matched_control(measured, seed=int(seed))

    circuits = {
        PRIMARY_ARM: measured,
        PRIMARY_CONTROL: matched,
        SECONDARY_CONTROL: existing,
    }
    names = list(circuits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = circuits[first], circuits[second]
            if not (np.array_equal(a.body_ids, b.body_ids)
                    and np.array_equal(a.cell_type, b.cell_type)):
                raise RuntimeError(
                    f"arms {first} and {second} do not share one neuron set, so the comparison "
                    "is not a contrast about wiring"
                )
    return circuits, subset, full


def _counts_for_trials(
    circuit: MCNSCircuit,
    index_map,
    propagation,
    generator: MotionStimulusGenerator,
    conditions: Sequence[str],
    trials: int,
    n_steps: int,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    """Simulate one arm and return per-trial per-neuron counts plus a liveness record.

    Per-neuron counts, not a binned population vector: the readout needs the
    highest information-per-parameter representation available, and a 1/8-quantised
    fold accuracy is what Phase 4C found to be uninformative.

    Every condition in ``conditions`` is simulated, and the liveness record is
    keyed by condition *and* ``"ALL"`` -- the per-trial counts pooled over all of
    them. G3 gates the pooled record, because a group that fires on only one of the
    two contrast directions has a spatial offset that is a property of that
    direction alone, and a gate computed from a single condition would pass it.
    Pooling re-runs :func:`liveness` on the concatenated per-trial counts rather
    than averaging the per-condition summaries, so a mean driven by a handful of
    active trials cannot hide behind an inactive majority.
    """
    params = NeuronParams()
    counts: dict[tuple[str, int], np.ndarray] = {}
    per_condition_counts: dict[str, dict[str, list[int]]] = {
        str(c): {name: [] for name, _ in groups} for c in conditions
    }
    neurons: dict[str, int] = {name: 0 for name, _ in groups}
    for condition in conditions:
        spec = StimulusCondition.from_name(condition)
        for trial in range(int(trials)):
            result = run_simulation(
                circuit, generator.as_encoder(trial=trial), spec,
                n_steps=int(n_steps), params=params,
                synapse_scale=float(PREDECLARED["synapse_scale"]),
                coupling="linear", trace_steps=0, propagation=propagation, record_input=False,
            )
            per_neuron = result.spikes.sum(axis=0).astype(np.int64)
            counts[(str(spec.name), trial)] = per_neuron
            for name, members in groups:
                indices = np.concatenate(
                    [np.asarray(index_map[m], dtype=np.int64) for m in members if m in index_map]
                )
                per_condition_counts[str(spec.name)][name].append(int(per_neuron[indices].sum()))
                neurons[name] = int(indices.size)
    live: dict[str, Any] = {}
    for name, _ in groups:
        pooled_spikes = [
            v for c in conditions for v in per_condition_counts[str(c)][name]
        ]
        ceiling = int(n_steps) * max(1, len(conditions)) * int(trials)
        live[name] = liveness(
            pooled_spikes, n_neurons=neurons[name], n_steps=ceiling
        )
        live[name]["_per_condition"] = {
            str(c): liveness(
                per_condition_counts[str(c)][name], n_neurons=neurons[name], n_steps=int(n_steps)
            )
            for c in conditions
        }
        live[name]["_neurons"] = neurons[name]
    return counts, live


def _drive_profile_variance_fraction(
    generator: MotionStimulusGenerator, lateral_bin: np.ndarray, group: tuple[str, ...],
    n_steps: int,
) -> dict[str, Any]:
    """How spatially non-uniform is the injected drive? Gate G1's input.

    Gate G1 must judge the *input pathway's* injected drive. This implementation
    therefore uses the declared input channels (the L1/L2 boxes the stimulus is
    built from) filtered by ``lateral`` so all input neurons carry it. Passing the
    readout group (T4/T5) samples those neurons' direct external drive, but T4/T5
    are downstream targets in the chain and normally receive no direct injected
    current; binning their injected drive always loses the spatial signal G1 is
    supposed to detect. Do not substitute a downstream activity proxy here.
    """
    indices = np.concatenate(
        [generator.circuit.indices_of_type(t) for t in group if generator.circuit.has(t)]
    ).astype(np.int64)
    lateral = np.asarray(lateral_bin, dtype=np.int64)
    n_bins = int(lateral.max()) + 1
    profiles = []
    for trial in range(3):
        drive = generator.drive_tensor(
            StimulusCondition.from_name(PRIMARY_CONTRAST[0]), trial, int(n_steps)
        ).astype(np.float64)
        profile = np.zeros(n_bins, dtype=np.float64)
        for bin_index in range(n_bins):
            chosen = indices[lateral[indices] == bin_index]
            profile[bin_index] = float(np.abs(drive[:, chosen]).sum()) if chosen.size else 0.0
        profiles.append(profile)
    stacked = np.vstack(profiles)
    mean_profile = stacked.mean(axis=0)
    total = float(np.abs(mean_profile).sum())
    variance_fraction = (
        float(np.abs(mean_profile - mean_profile.mean()).sum() / total) if total > 0 else 0.0
    )
    return {
        "mean_drive_profile": [float(v) for v in mean_profile],
        "variance_fraction": variance_fraction,
        "is_flat": bool(variance_fraction < 1e-12),
        "note": (
            "sum of |deviation from the mean| over the total, on the noise-free drive, aggregated "
            "into the declared input population's lateral bins"
        ),
    }


def _input_identity(
    generator: MotionStimulusGenerator,
    circuits: dict[str, MCNSCircuit],
    conditions: Sequence[str],
    trials: int,
    n_steps: int,
    n_bins: int,
) -> dict[str, Any]:
    """G2: the primary arms are driven by bitwise identical input, on equal energy.

    G2 exists to rule out the failure where a difference between the arms comes
    from the stimulus rather than from the wiring. That is established by
    *measurement* here, not by assertion. For every condition and trial the
    noise-free drive tensor is regenerated and the two primary arms are compared
    bitwise, and the ON/OFF energy split is compared between the two sweep
    directions.

    This is falsifiable, unlike the earlier version of this gate, which called
    ``input_instant_identity({}, [])`` and so compared nothing and reported a
    maximum spread of exactly 0 for every input. A generator rebuilt per arm, a
    control restricted to a different neuron count, a changed neuron order, or a
    different lateral-bin assignment all make the tensors differ in shape or
    content and fail the gate.

    It is a *design* check and is reported as such. It does not replace the Phase 4
    acceptance gate, which asks whether the stimulus already separates the
    conditions before the circuit sees them; that gate is unchanged and still
    required, and a passing G2 says nothing about it.
    """
    left, right = PRIMARY_CONTRAST
    max_spread = 0.0
    compared = 0
    order_identical = True
    lateral_identical = True
    reference = circuits[PRIMARY_ARM]
    reference_lateral = lateral_coordinates(reference, int(n_bins))
    for name, circuit in circuits.items():
        order_identical = order_identical and np.array_equal(
            circuit.body_ids, reference.body_ids
        ) and np.array_equal(circuit.cell_type, reference.cell_type)
        lateral_identical = lateral_identical and np.array_equal(
            lateral_coordinates(circuit, int(n_bins)), reference_lateral
        )
    for condition in conditions:
        for trial in range(int(trials)):
            arms = [
                generator.drive_tensor(StimulusCondition.from_name(condition), trial, int(n_steps))
                for circuit in circuits.values()
            ]
            if len({tuple(a.shape) for a in arms}) > 1:
                return {
                    "max_spread_across_arms": float("inf"),
                    "arms_compared": compared,
                    "order_identical": order_identical,
                    "lateral_identical": lateral_identical,
                    "direction_energy_equal": False,
                    "checked": True,
                    "note": "arms disagree on drive shape, so the input cannot be identical",
                }
            reference_drive = arms[0].astype(np.float64)
            for other in arms[1:]:
                max_spread = max(
                    max_spread, float(np.abs(reference_drive - other.astype(np.float64)).max())
                )
            compared += 1
    energies = {
        str(c): float(generator.drive_energy(StimulusCondition.from_name(c), int(n_steps)))
        for c in (left, right)
    }
    spread = max(energies.values()) - min(energies.values())
    return {
        "max_spread_across_arms": max_spread,
        "arms_compared": compared,
        "order_identical": order_identical,
        "lateral_identical": lateral_identical,
        "direction_energy": energies,
        "direction_energy_spread": spread,
        "direction_energy_equal": bool(spread <= 1e-9 * max(1.0, max(energies.values()))),
        "checked": True,
        "note": (
            "noise-free drive regenerated per arm and compared bitwise, and the two sweep "
            "directions compared on ON/OFF energy; a design check on the input, not a "
            "substitute for the Phase 4 acceptance gate"
        ),
    }


def _phase5a_paired_margin(
        primary_shift: Sequence[float], control_shift: Sequence[float],
        n_permutations: int, seed: int,
) -> dict[str, Any]:
    """Report NaN centroids as a blocker, not as a finite margin.

    A silent/empty readout yields an undefined centroid and therefore a NaN in
    the shift. Those trials are real trials; counting them as zero, dropping them
    silently, or letting sign-flip indices happen to be all False would manufacture
    an apparently supported effect. If any fold is non-finite the pilot's primary
    margin is explicit and undecidable: no paired-t CI, no z-score, no claim.
    The raw per-trial shifts are retained for provenance.
    """
    primary = np.asarray(list(primary_shift), dtype=np.float64)
    control = np.asarray(list(control_shift), dtype=np.float64)
    fold_margin = primary - control
    nan_count = int(np.count_nonzero(~np.isfinite(fold_margin)))
    finite_count = int(np.count_nonzero(np.isfinite(fold_margin)))
    if primary.size != control.size:
        return {
            "status": "incomparable_folds",
            "n_folds_primary": int(primary.size),
            "n_folds_control": int(control.size),
            "reason": "the two primary statistics do not share one fold length",
            "n_nonfinite_folds": nan_count,
            "effect_supported": False,
        }
    if finite_count < 2:
        return {
            "status": "insufficient_finite_folds",
            "n_folds": int(fold_margin.size),
            "n_finite_folds": finite_count,
            "n_nonfinite_folds": nan_count,
            "reason": "at least two finite paired fold margins are required; the rest are the undefined centroids reported below",
            "per_fold_margin": [float(v) for v in fold_margin],
            "effect_supported": False,
        }
    if nan_count > 0:
        return {
            "status": "nan_folds_present",
            "n_folds": int(fold_margin.size),
            "n_finite_folds": finite_count,
            "n_nonfinite_folds": nan_count,
            "reason": "one or more per-trial paired fold margins are NaN because a centroid was undefined on a silent/empty readout; no finite effect is manufactured",
            "per_fold_margin": [float(v) for v in fold_margin],
            "effect_supported": False,
        }
    result = margin_statistics(primary, control, n_permutations=n_permutations, seed=seed)
    result["nan_audit"] = {"n_nonfinite_folds": 0, "n_finite_folds": finite_count}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variance-estimation", action="store_true",
                        help="stage 1: report the sd of the primary statistic and nothing else")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--sd", type=float, default=None,
                        help="sd from stage 1; REQUIRED for the pilot and frozen into the config")
    parser.add_argument("--delta", type=float, default=None,
                        help="minimum detectable centroid difference; default 0.05")
    parser.add_argument("--n-steps", type=int, default=int(PREDECLARED.get("n_steps", 150) or 150))
    parser.add_argument("--max-bodies-per-type", type=int, default=100)
    parser.add_argument("--lateral-bins", type=int, default=int(PREDECLARED["lateral_bins"]))
    parser.add_argument("--scrambles", type=int, default=int(PREDECLARED["calibration_scrambles"]))
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--connectivity", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n_steps is None:
        args.n_steps = 150
    delta = 0.05 if args.delta is None else float(args.delta)
    estimating = bool(args.variance_estimation)
    if not estimating and args.sd is None:
        print(
            "error: the pilot requires --sd from the predeclared variance-estimation stage.\n"
            "       Run:  python scripts/run_phase5a_pilot.py --variance-estimation\n"
            "       That stage reports a variance and nothing else, so the trial count is fixed\n"
            "       before the main run and cannot be revised after an effect is seen.",
            file=sys.stderr,
        )
        return 2

    data_dir = Path(args.data_dir)
    annotations_path = args.annotations or str(data_dir / ANNOTATION_NAME)
    connectivity_path = args.connectivity or str(data_dir / CONNECTIVITY_NAME)
    for path in (Path(annotations_path), Path(connectivity_path)):
        if not Path(path).is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    output_dir = Path(args.output_dir or (VARIANCE_OUTPUT if estimating else DEFAULT_OUTPUT))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else output_dir / "extraction_cache.npz"
    trials = int(args.trials or (5 if estimating else 20))
    progress = Progress(total=5, enabled=not args.quiet)

    print("=" * _W)
    print(f"MCNS Phase {PREDECLARED_PHASE} pilot  [{'variance estimation' if estimating else 'pilot'}]")
    print("=" * _W)
    for line in _wrap(PREDECLARED["primary_question"], _W - 2):
        print(f"  {line}")
    print()
    print(f"  arms          : {', '.join(ARMS)}")
    print(f"  synapse_scale : {PREDECLARED['synapse_scale']}  PRE-DECLARED, not swept")
    print(f"  readout       : {PREDECLARED['readout']}  ({args.lateral_bins} lateral bins)")
    print(f"  gates         : {', '.join(GATE_ORDER)}")
    print()

    progress.stage("Load annotations, population, extraction cache")
    load_started = time.perf_counter()
    annotations = load_annotations(annotations_path)
    populations = build_populations(annotations, MOTION_CELL_TYPES, require_superclass=True)
    extraction = _load_extraction(annotations_path, connectivity_path, cache)
    circuits, subset, _ = build_arms(
        populations, extraction, annotations, annotations_path,
        int(args.max_bodies_per_type), int(args.seed),
    )
    graph_seconds = time.perf_counter() - load_started
    print(f"  {subset.headline()}")
    print(f"  {SUBSET_DISCLAIMER}")
    print()
    for name, circuit in circuits.items():
        print(f"  {name:<26} neurons={circuit.num_neurons:,} edges={circuit.num_edges:,} "
              f"synapses={circuit.total_synapse_count:,}")

    progress.stage("Verify the drive-matched control against the measured graph")
    # This is the legacy DRIVE_MATCHED_CONTROL that build_arms constructed. It
    # is verified and printed for continuity; the PRIMARY control used by G3,
    # G4 and the paired margin is replaced below by the liveness-gated
    # candidate, which reassigns this variable.
    matched_verification = verify_drive_matching(circuits[PRIMARY_ARM], circuits[PRIMARY_CONTROL])
    existing_verification = verify_degree_preservation(
        circuits[PRIMARY_ARM], circuits[SECONDARY_CONTROL]
    )
    for key in (
        "in_degree_per_neuron_equal", "out_degree_per_neuron_equal",
        "incoming_weight_per_neuron_equal", "outgoing_weight_per_neuron_equal",
        "incoming_weight_max_abs_diff", "outgoing_weight_max_abs_diff",
        "edge_weight_multiset_equal", "total_synapses_equal", "soma_location_equal",
        "edge_change_fraction", "self_loops", "topology_changed",
    ):
        print(f"    {key:<38} {matched_verification[key]}")
    print()
    print(f"  {DRIVE_MATCHED_DISCLAIMER}")
    print("    (legacy DRIVE_MATCHED_CONTROL built by build_arms above; the PRIMARY")
    print("     control for G3/G4 and the paired margin is selected below from the")
    print("     frozen liveness-gated candidate search.)")
    print()
    print(f"  SECONDARY control, retained and labelled drive-mismatched: {SHUFFLED_DISCLAIMER}")
    print(f"    topology_changed={existing_verification['topology_changed']}  "
          f"degree multisets equal="
          f"{existing_verification['in_degree_sequence_equal'] and existing_verification['out_degree_sequence_equal']}")
    print("    NOTE: this control preserves the weight MULTISET, not per-neuron incoming")
    print("    weight. It is reported, not relied on, and is never substituted silently.")

    progress.stage("Build the stimulus, bins and readout indices")
    measured = circuits[PRIMARY_ARM]
    generator = MotionStimulusGenerator(
        measured, spot_fraction=0.15, current=2.0, steps_per_crossing=int(args.n_steps),
        amplitude_perturbation=0.05, input_noise=1.0, randomise_start=True, seed=int(args.seed),
    )
    lateral_bin = lateral_coordinates(measured, int(args.lateral_bins))
    index_map = measured.cell_type_index()
    bin_counts = np.bincount(lateral_bin, minlength=int(args.lateral_bins))
    propagation = measured.propagation_matrix(
        synapse_scale=float(PREDECLARED["synapse_scale"]), coupling="linear"
    )
    print(f"  lateral bin occupancy: {bin_counts.tolist()}  (must have no empty bin)")
    for name, members in READOUT_GROUPS:
        sizes = {m: int(np.asarray(index_map[m]).size) for m in members if m in index_map}
        print(f"  {name:<6} members {members}  sizes {sizes}")
    if (bin_counts == 0).any():
        print("error: an empty lateral bin makes the centroid ill-defined", file=sys.stderr)
        return 2
    drive_g1 = _drive_profile_variance_fraction(
        generator, lateral_bin, tuple(generator.on_types) + tuple(generator.off_types),
        int(args.n_steps)
    )
    print(f"  G1 input locality: drive profile variance fraction "
          f"{drive_g1['variance_fraction']:.4f}  profile {np.round(drive_g1['mean_drive_profile'], 1).tolist()}")

    progress.stage("Select the liveness-gated structural control (G3 acceptance gate)")
    all_counts: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    all_live: dict[str, dict[str, Any]] = {}
    conditions = sorted({PRIMARY_CONTRAST[0], PRIMARY_CONTRAST[1], STATIC_CONTRAST[1]})

    # Primary control: the frozen liveness-gated structural null. The probe
    # exposes ONLY the declared G3 liveness records (T4/T5 pooled
    # zero-activity and saturation fractions, with per-condition details).
    # It never returns a centroid, lateral profile, margin, G4/G5 outcome, or
    # any other scientific quantity, and candidate selection never sees one.
    def _liveness_probe(arm_label: str, circuit: MCNSCircuit) -> dict[str, Any]:
        prop = (
            propagation
            if circuit is measured
            else circuit.propagation_matrix(
                synapse_scale=float(PREDECLARED["synapse_scale"]), coupling="linear"
            )
        )
        _, live = _counts_for_trials(
            circuit, circuit.cell_type_index(), prop, generator, conditions,
            int(trials), int(args.n_steps), READOUT_GROUPS,
        )
        return live

    control_selection = select_liveness_gated_control(measured, _liveness_probe)
    if control_selection["status"] != "accepted":
        blocked_payload = {
            "phase": PREDECLARED_PHASE,
            "stage": "liveness-gated control selection",
            "status": "BLOCKED",
            "reason": control_selection["reason"],
            "control_selection": control_selection,
            "note": (
                "No candidate satisfied the frozen contract within 32 deterministic "
                "attempts. The primary comparison is BLOCKED: thresholds are not "
                "relaxed, seeds are not extended, G3 is not loosened, and the old "
                "DRIVE_MATCHED_CONTROL is not substituted."
            ),
        }
        (output_dir / "phase5a_pilot.json").write_text(
            json.dumps(blocked_payload, indent=2, default=str), encoding="utf-8"
        )
        print()
        print("  BLOCKED: no DRIVE_MATCHED_LIVENESS_GATED candidate qualified within 32 attempts.")
        print(f"  Full diagnostics written to {output_dir / 'phase5a_pilot.json'}")
        return 3
    circuits[PRIMARY_CONTROL] = control_selection["control_circuit"]
    matched_verification = verify_drive_matching(
        circuits[PRIMARY_ARM], circuits[PRIMARY_CONTROL]
    )
    accepted = control_selection["accepted"]
    print(f"  selected {CONTROL_NAME}: attempt_index={accepted['attempt_index']} "
          f"candidate_seed={accepted['candidate_seed']} "
          f"attempts_executed={len(control_selection['attempts'])}")
    progress.stage("Simulate every arm and compute the lateral readout")
    for name, circuit in circuits.items():
        prop = (
            propagation if name == PRIMARY_ARM
            else circuit.propagation_matrix(
                synapse_scale=float(PREDECLARED["synapse_scale"]), coupling="linear"
            )
        )
        arm_counts, arm_live = _counts_for_trials(
            circuit, circuit.cell_type_index(), prop, generator, conditions,
            int(trials), int(args.n_steps), READOUT_GROUPS,
        )
        all_counts[name] = arm_counts
        all_live[name] = arm_live
        for group, _ in READOUT_GROUPS:
            entry = arm_live[group]
            print(f"    {name:<26}{group:<5} mean/trial={entry['mean_spikes_per_trial']:7.2f} "
                  f"zero={entry['fraction_zero_activity_trials']:.3f} "
                  f"sat={entry['spike_saturation_fraction']:.2e} "
                  f"(pooled over {len(conditions)} conditions)")

    centroids: dict[str, dict[str, dict[str, list[float]]]] = {}
    for name, counts in all_counts.items():
        per_group: dict[str, dict[str, list[float]]] = {}
        for group, members in READOUT_GROUPS:
            per_condition: dict[str, list[float]] = {}
            for condition in conditions:
                values = [
                    profile_centroid(
                        lateral_profile(counts[(condition, t)], lateral_bin,
                                        index_map, members)
                    )
                    for t in range(int(trials))
                    if (condition, t) in counts
                ]
                per_condition[condition] = [float(v) for v in values]
            per_group[group] = per_condition
        centroids[name] = per_group

    if estimating:
        # The primary statistic is the *measured minus drive-matched* centroid
        # difference, per trial, on the primary contrast. Estimating the sd of
        # the measured arm's own LEFT-RIGHT spread instead would be a different
        # and easier quantity: it excludes the control's contribution, which is
        # exactly the term the test is about, so the trial count would be set
        # from a variance that is too small to support the planned comparison.
        left, right = PRIMARY_CONTRAST
        diffs = [
            centroids[PRIMARY_ARM][SUBTYPE_PRIMARY][c][t]
            - centroids[PRIMARY_CONTROL][SUBTYPE_PRIMARY][c][t]
            for c in (left, right)
            for t in range(int(trials))
        ]
        finite = [v for v in diffs if np.isfinite(v)]
        nan_values = len(diffs) - len(finite)
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        payload = {
            "phase": PREDECLARED_PHASE,
            "stage": "variance estimation",
            "purpose": (
                "reports the sd of the primary statistic and NOTHING else. No effect is "
                "inspected and no condition is compared, so the trial count can be fixed before "
                "the main run."
            ),
            "predeclaration": predeclared_block(),
            "trials": int(trials),
            "group": SUBTYPE_PRIMARY,
            "contrast": list(PRIMARY_CONTRAST),
            "statistic": "per-trial centroid difference, MEASURED minus DRIVE_MATCHED_CONTROL",
            "sd": sd,
            "n_centroid_values": len(finite),
            "n_nan_centroid_values": int(nan_values),
            "nan_reason": (
                "NaN entries are undefined centroids on zero-activity/empty readout trials. They "
                "are not zero and are not converted to observations; the sd above is conditional on "
                "finite trials only and must be read with n_nan_centroid_values."
            ),
            "next_step": (
                f"python scripts/run_phase5a_pilot.py --sd {sd:.6f} --delta {delta}"
            ),
        }
        (output_dir / "phase5a_variance.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print()
        print("  sd of the per-trial centroid difference = "
              f"{sd:.6f}  over {len(finite)} finite trials ({nan_values} NaN rejected as undefined)")
        print(f"  wrote {output_dir / 'phase5a_variance.json'}")
        return 0

    # ---------------------------------------------------------- pilot proper
    power = power_requirement(float(args.sd), delta, n_trials=int(trials))
    print()
    print(f"  power: sd={power['sd_of_paired_differences']:.6f}  "
          f"delta={power['delta_minimum_detectable']}  "
          f"n required={power['n_trials_required']}  (requested {trials})")
    if power["n_trials_required"] > int(trials):
        print("  WARNING: the requested trial count is below the power requirement. The result is")
        print("  reported, but it is underpowered and must be labelled as such.")

    calibration = calibration_shift_null(
        all_counts[PRIMARY_ARM], lateral_bin, index_map, READOUT_GROUPS[0][1],
        PRIMARY_CONTRAST[0], PRIMARY_CONTRAST[1], int(trials),
        n_scrambles=int(args.scrambles), seed=int(args.seed),
    )
    identity = _input_identity(
        generator, circuits, conditions, int(trials), int(args.n_steps),
        int(args.lateral_bins),
    )
    print(f"  G2 input identity: max spread across arms "
          f"{identity['max_spread_across_arms']:.3e} over {identity['arms_compared']} "
          f"condition-trial pairs  order_identical={identity['order_identical']}  "
          f"lateral_identical={identity['lateral_identical']}  "
          f"direction_energy_equal={identity['direction_energy_equal']}")
    liveness_payload = {
        name: {
            group: {
                "fraction_zero_activity_trials": float(live[group]["fraction_zero_activity_trials"]),
                "spike_saturation_fraction": float(live[group]["spike_saturation_fraction"]),
                "mean_spikes_per_trial": float(live[group]["mean_spikes_per_trial"]),
                "_per_condition": live[group].get("_per_condition", {}),
                "_neurons": live[group].get("_neurons", 0),
            }
            for group, _ in READOUT_GROUPS
        }
        for name, live in all_live.items()
    }
    gate_inputs = {
        "drive_profile_variance_fraction": drive_g1["variance_fraction"],
        "input_identity": identity,
        "liveness": liveness_payload,
        "control_verification": matched_verification,
        "calibration": calibration,
    }
    gates = evaluate_gates(**gate_inputs)

    left, right = PRIMARY_CONTRAST

    def _centroid(arm: str, condition: str, trial: int) -> float:
        return float(centroids[arm][SUBTYPE_PRIMARY][condition][trial])

    # Per-trial paired centroid difference, per arm: the statistic the pilot is
    # powered on. computed with the existing Phase 4C machinery rather than
    # reinvented, so the pairing, the paired t and the sign-flip null are shared.
    per_arm_shift = {
        arm: [_centroid(arm, left, t) - _centroid(arm, right, t) for t in range(int(trials))]
        for arm in ARMS
    }
    paired = {
        control: _phase5a_paired_margin(
            per_arm_shift[PRIMARY_ARM], per_arm_shift[control],
            n_permutations=int(args.permutations), seed=int(args.seed),
        )
        for control in (PRIMARY_CONTROL, SECONDARY_CONTROL)
    }

    total = progress.total_seconds()
    summary = {
        "phase": PREDECLARED_PHASE,
        "kind": "pilot infrastructure; a different and smaller question from Phase 4C",
        "primary_question": PREDECLARED["primary_question"],
        "predeclaration": predeclared_block(),
        "arms": arms(),
        "arm_names": list(ARMS),
        "existing_shuffle_retained": True,
        "existing_shuffle_is_drive_matched": False,
        "population": subset.describe(),
        "circuits": {
            name: {
                "neurons": c.num_neurons, "edges": c.num_edges,
                "biological_synapse_count": c.total_synapse_count,
                "is_measured": name == PRIMARY_ARM,
            }
            for name, c in circuits.items()
        },
        "matched_control_verification": matched_verification,
        "matched_control_disclaimer": DRIVE_MATCHED_DISCLAIMER,
        "existing_shuffle_verification": existing_verification,
        "existing_shuffle_disclaimer": SHUFFLED_DISCLAIMER,
        "readout": {
            "kind": "lateral activity profile and centroid, per cell group",
            "disclaimer": LATERAL_READOUT_DISCLAIMER,
            "lateral_bins": int(args.lateral_bins),
            "bin_occupancy": bin_counts.tolist(),
            "groups": {name: list(members) for name, members in READOUT_GROUPS},
            "subtype_secondary": [name for name, _ in SUBTYPE_GROUPS],
            "decoder_accuracy_used": False,
            "why_not": PREDECLARED["statistic_is_not"],
        },
        "input_locality": drive_g1,
        "centroids": {name: {g: per for g, per in per_group.items()}
                      for name, per_group in centroids.items()},
        "primary_margins": {control: stats for control, stats in paired.items()},
        "primary_control": PRIMARY_CONTROL,
        "primary_control_implementation": CONTROL_NAME,
        "control_selection": {
            "control_name": CONTROL_NAME,
            "structural_constraints": STRUCTURAL_CONSTRAINTS,
            "max_attempts": MAX_ATTEMPTS,
            "attempt_seed_policy": ATTEMPT_SEED_POLICY,
            "accepted_attempt_index": accepted["attempt_index"],
            "accepted_candidate_seed": accepted["candidate_seed"],
            "n_attempts_executed": len(control_selection["attempts"]),
        },
        "secondary_control": SECONDARY_CONTROL,
        "secondary_control_note": (
            "reported and labelled drive-mismatched; the Phase 4C audit measured its per-cell-type "
            "incoming drive moving by 0.16x to 2.75x, so it is not a strength-matched null"
        ),
        "calibration": calibration,
        "liveness": liveness_payload,
        "gates": gates,
        "gate_order": list(GATE_ORDER),
        "power": power,
        "n_trials": int(trials),
        "n_permutations": int(args.permutations),
        "n_scrambles": int(args.scrambles),
        "synapse_scale": float(PREDECLARED["synapse_scale"]),
        "seed": int(args.seed),
        "timings": {"total_wall_clock_seconds": round(total, 3),
                    "graph_seconds": round(graph_seconds, 3)},
        "claims": claim_block(),
        "direction_selectivity_out_of_scope": DIRECTION_SELECTIVITY_OUT_OF_SCOPE,
        "pilot_has_not_been_run_note": (
            "this payload documents infrastructure and the pre-declaration. It contains no "
            "experimental result until the pilot is run with an explicit --sd."
        ),
    }

    written = {}
    for name, payload in (("phase5a_pilot.json", summary),):
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written[name] = str(path)

    print()
    print("=" * _W)
    print("GATES  (evaluated before any interpretation)")
    print("=" * _W)
    for key in GATE_ORDER:
        entry = gates["per_gate"][key]
        print(f"  {key} {entry['status']:<14}{entry.get('name', '')}")
        for line in _wrap(entry["permits"], _W - 8):
            print(f"       {line}")
    print()
    for line in _wrap(gates["statement"], _W - 2):
        print(f"  {line}")
    print()
    print("=" * _W)
    print("PRIMARY MARGIN  (mean centroid difference, measured minus control)")
    print("=" * _W)
    for control, stats in paired.items():
        if stats.get("status") == "ok":
            print(f"  {control:<26} margin {stats['mean_margin']:+.6f}  sd {stats['sd_margin']:.6f}  "
                  f"t {stats['paired_t_statistic']:+.3f}  p {stats['paired_t_p_two_sided']:.4f}")
        else:
            print(f"  {control:<26} not computable: {stats.get('status')}")
    print()
    print("=" * _W)
    print("DIRECTION SELECTIVITY")
    print("=" * _W)
    for line in _wrap(DIRECTION_SELECTIVITY_OUT_OF_SCOPE, _W - 2):
        print(f"  {line}")
    print()
    print("Artifacts")
    print("-" * _W)
    for key, value in written.items():
        print(f"  {key:<26} {value}")
    print()
    print(f"wall clock: {total:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
