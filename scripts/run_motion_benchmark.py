#!/usr/bin/env python
"""Benchmark: does the measured MCNS wiring *transform* spatiotemporal visual input?

The question
------------
**"Does measured MCNS recurrent wiring transform spatiotemporal visual input
differently from a feedforward version of the same wiring, and from a
degree-preserving randomisation of it?"**

It is deliberately *not* "does this model reproduce biological direction
selectivity?". The MCNS connectivity table carries exactly three columns --
``body_pre``, ``body_post``, ``weight`` -- and **no synaptic sign**. Every
coupling in this model is therefore non-negative, the network can only sum and
never subtract, and a Reichardt-type motion detector, which requires ON minus
OFF, is not constructible from this data. No result at any level of this
benchmark licenses a direction-selectivity claim. See
``docs/MCNS_MOTION_BENCHMARK.md`` for the four-level claim ladder.

Circuit conditions
------------------
    MCNS_FEEDFORWARD    documented chains only: L1->Mi1->T4a-d, L2->Tm1/Tm2->T5a-d
    MCNS_RECURRENT      every measured candidate-to-candidate edge
    SHUFFLED_CONTROL    degree-preserving randomisation: same node count, edge
                        count, both degree sequences and synapse-count multiset,
                        different wiring. A computational control, not biology.

All three are built from one extraction, share one bounded neuron subset, one
set of LIF parameters, one timestep, one duration, one ``synapse_scale`` per
sweep point, one input energy, one seed and one trial split. Nothing is tuned
per condition.

The controls that decide whether anything means anything
--------------------------------------------------------
``INPUT_INSTANT``    **the acceptance gate.** Two numbers: the whole-run mean
                     drive of the ON and the OFF channel. If those already
                     separate the conditions, no downstream number can be
                     attributed to MCNS and the run is flagged invalid.
``INPUT_SPATIAL``    Relay bound. The drive pattern at the same lateral
                     resolution as the population readout. A circuit that
                     passed its input through unchanged would score this.
``INPUT_TEMPORAL``   Relay bound from ordering alone: the normalised time at
                     which each lateral bin is driven. The smallest motion
                     statistic that exists.
``MAGNITUDE_ONLY``   Total activity as a single feature.

Population readouts keep T4a/T4b/T4c/T4d and T5a/T5b/T5c/T5d distinct at every
level, and the primary readout resolves space as well as identity: 8 exact
subtypes x 4 lateral bins = 32 columns for ``T4+T5``.

Usage
-----
    python scripts/run_motion_benchmark.py --quick      # bounded subset, ~1 min
    python scripts/run_motion_benchmark.py              # whole population

    python scripts/run_motion_benchmark.py --phase-4c --estimate   # timing estimate
    python scripts/run_motion_benchmark.py --phase-4c             # declared replication

The full-population run is a multi-hour job by design. ``--quick`` is the one to
run first, and it prints progress as it goes so nothing ever runs silently.

``--phase-4c`` is a **pre-declared replication** of the Phase 4B
recurrent-versus-shuffled observation: ``synapse_scale`` fixed at 0.25, 20 trials,
500 permutations, ``subtype_lateral`` primary readout, all three circuit
conditions on one population, and every other Phase 4B setting held. It is not a
new model and not a new question. The declared values were fixed before the run
and the mode **refuses** any flag that would change them, so no gain can be
selected after seeing a result. If 0.25 is silent, saturated or uninformative,
that is reported as the finding and the run is not repeated at another gain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from flybrain.benchmark.controls import (  # noqa: E402
    CONTROL_GROUPS,
    INPUT_INSTANT,
    INPUT_SPATIAL,
    INPUT_TEMPORAL,
    acceptance_gate,
    input_control_features,
    input_instant_identity,
)
from flybrain.benchmark.decoder import (  # noqa: E402
    DECODER_DISCLAIMER,
    cross_validate,
    cv_permutation_null,
)
from flybrain.benchmark.phase4c import (  # noqa: E402
    PREDECLARED_AMPLITUDE_PERTURBATION,
    PREDECLARED_CONTRAST,
    PREDECLARED_CONTRASTS,
    PREDECLARED_COUPLING,
    PREDECLARED_DECODER_ITERATIONS,
    PREDECLARED_DECODER_L2,
    PREDECLARED_DECODER_LEARNING_RATE,
    PREDECLARED_GROUP,
    PREDECLARED_INPUT_NOISE,
    PREDECLARED_MAX_BODIES_PER_TYPE,
    PREDECLARED_N_STEPS,
    PREDECLARED_BIN_STEPS,
    PREDECLARED_SKIP_BINS,
    PREDECLARED_DT,
    PREDECLARED_NULL_CONTRASTS,
    PREDECLARED_NULL_GROUPS,
    PREDECLARED_PERMUTATIONS,
    PREDECLARED_PHASE,
    PREDECLARED_SEED,
    PREDECLARED_SPOT_FRACTION,
    PREDECLARED_STIMULUS_CURRENT,
    PREDECLARED_SYNAPSE_SCALE,
    PREDECLARED_TRIALS,
    PREDECLARED_VARIANTS,
    CONTROL_CIRCUIT,
    PRIMARY_CIRCUIT,
    RELAY_CONTROLS,
    REPLICATION_CIRCUITS,
    claim_block,
    classify_informative,
    enforce_predeclared,
    feature_report,
    fingerprint,
    liveness,
    margin_statistics,
    predeclared_block,
    timing_projection,
)
from flybrain.benchmark.representation import (  # noqa: E402
    AGGREGATE_GROUPS,
    INPUT_GROUP,
    LATREADOUT_BINS,
    MAGNITUDE_GROUP,
    PRIMARY_READOUT,
    READOUTS,
    SUBTYPE_GROUPS,
    PopulationRecord,
    activity_descriptor,
    fast_separation_ratio,
    magnitude_features,
    population_features,
    separation_ratio_null,
    separability,
)
from flybrain.benchmark.shuffled import (  # noqa: E402
    SHUFFLED_DISCLAIMER,
    shuffled_control,
    verify_degree_preservation,
)
from flybrain.benchmark.stimulus import (  # noqa: E402
    STIMULUS_CONDITIONS,
    STIMULUS_DISCLAIMER,
    MotionStimulusGenerator,
    StimulusCondition,
    StimulusMotion,
    StimulusPolarity,
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
CIRCUITS = ("MCNS_FEEDFORWARD", "MCNS_RECURRENT", "SHUFFLED_CONTROL")
DEFAULT_SCALES = (0.25, 0.5, 1.0, 2.0)
QUICK_SCALES = (0.25, 0.5, 1.0, 2.0)
POPULATION_GROUPS: tuple[str, ...] = ("T4", "T5", "T4+T5")
ALL_BENCH_GROUPS: tuple[str, ...] = CONTROL_GROUPS + (MAGNITUDE_GROUP,) + POPULATION_GROUPS
PRIMARY_CONTRAST = "LEFTWARD_vs_RIGHTWARD"

#: Named readout variants. Each is a (readout, unit-normalise) pair.
#:
#: The primary variant resolves space as well as identity and removes overall
#: activity magnitude, which is a confound. ``subtype_lateral_raw`` is the same
#: representation with magnitude kept, and it is reported alongside
#: ``MAGNITUDE_ONLY`` precisely because keeping magnitude re-introduces that
#: confound. ``cell_type`` is the population mean, which cannot represent a moving
#: spot's direction at all; it is reported so the difference is visible.
#:
#: Every variant is computed from the **same** simulation, so adding one costs
#: feature extraction only, never another simulation.
VARIANTS: dict[str, tuple[str, bool]] = {
    "subtype_lateral": (PRIMARY_READOUT, True),
    "subtype_lateral_raw": (PRIMARY_READOUT, False),
    "cell_type": ("cell_type", True),
}
PRIMARY_VARIANT = "subtype_lateral"

#: Stimulus contrasts. Every one is a two-class problem solved by the same
#: leave-one-trial-out protocol, so the numbers are comparable across the table.
CONTRASTS: dict[str, tuple[str, str]] = {
    PRIMARY_CONTRAST: ("LEFTWARD", "RIGHTWARD"),
    "LEFTWARD_vs_STATIC": ("LEFTWARD", "STATIC"),
    "LEFTWARD_vs_POLARITY_REVERSED": ("LEFTWARD", "LEFTWARD_POLARITY_REVERSED"),
}
_WIDTH = 78


class Progress:
    """Stage counter plus flushed output, so a long run is never opaque."""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = int(total)
        self.enabled = bool(enabled)
        self.index = 0
        self.timings: dict[str, float] = {}
        self._started = time.perf_counter()

    def stage(self, label: str) -> None:
        self.index += 1
        if self.enabled:
            print(f"\n[{self.index}/{self.total}] {label}", flush=True)

    def note(self, text: str) -> None:
        if self.enabled:
            print(f"    {text}", flush=True)

    def record(self, key: str, seconds: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + float(seconds)

    def total_seconds(self) -> float:
        return time.perf_counter() - self._started


# ----------------------------------------------------------------- extraction


def _load_extraction(args: argparse.Namespace, populations) -> EdgeExtraction:
    """Reuse the Phase 3/4A extraction cache when it matches the population.

    The cache is keyed on the *full* candidate body list, never on a subset, so
    subset mode never invalidates it and never re-reads the 1 GB connectivity
    table. The bound is applied to the measured edge list, so the edges inside the
    subset are still the measured ones.
    """
    cache = Path(args.cache) if args.cache else Path(args.output_dir) / "extraction_cache.npz"
    if not args.no_cache and cache.is_file():
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
    extraction = extract_candidate_edges(args.connectivity, populations, direction="both")
    if not args.no_cache:
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


# ------------------------------------------------------------------- circuits


class RunContext:
    """Everything a trial needs that does not change between trials.

    Holds the cached propagation matrices, the cached stimulus and drive
    tensors, the lateral bins and the cell-type index maps, so a trial loop never
    rebuilds a graph, a position map or a per-step stimulus.
    """

    def __init__(self, args: argparse.Namespace, generator: MotionStimulusGenerator) -> None:
        self.args = args
        self.generator = generator
        self.params = NeuronParams()
        self._propagation: dict[tuple[str, float], Any] = {}
        self._index_maps: dict[str, dict[str, np.ndarray]] = {}
        self._lateral: dict[str, np.ndarray] = {}

    def propagation(self, circuit_name: str, circuit: MCNSCircuit, scale: float) -> Any:
        key = (circuit_name, float(scale))
        if key not in self._propagation:
            self._propagation[key] = circuit.propagation_matrix(
                synapse_scale=float(scale), coupling=self.args.coupling
            )
        return self._propagation[key]

    def index_map(self, circuit_name: str, circuit: MCNSCircuit) -> dict[str, np.ndarray]:
        if circuit_name not in self._index_maps:
            self._index_maps[circuit_name] = circuit.cell_type_index()
        return self._index_maps[circuit_name]

    def lateral(self, circuit_name: str, circuit: MCNSCircuit) -> np.ndarray:
        if circuit_name not in self._lateral:
            self._lateral[circuit_name] = lateral_coordinates(circuit, int(self.args.lateral_bins))
        return self._lateral[circuit_name]

    def warm(
        self,
        circuits: dict[str, MCNSCircuit],
        scales: list[float],
        trials: int,
        conditions: list[str],
    ) -> dict[str, float]:
        """Build every expensive object once, before any trial runs."""
        started = time.perf_counter()
        for name, circuit in circuits.items():
            for scale in scales:
                self.propagation(name, circuit, scale)
            self.index_map(name, circuit)
            self.lateral(name, circuit)
        for condition in conditions:
            for trial in range(int(trials)):
                self.generator.tensor(
                    StimulusCondition.from_name(condition), trial=trial, n_steps=int(self.args.n_steps)
                )
                self.generator.drive_tensor(
                    StimulusCondition.from_name(condition), trial=trial, n_steps=int(self.args.n_steps)
                )
        elapsed = time.perf_counter() - started
        return {"seconds": elapsed, **self.generator.cache_stats()}


# ---------------------------------------------------------------- stimulus side


def build_control_records(
    context: RunContext,
    circuit: MCNSCircuit,
    circuit_name: str,
    conditions: list[str],
    trials: int,
) -> dict[str, dict[str, PopulationRecord]]:
    """Stimulus-side control records, computed once and shared by every circuit.

    These come from the **noise-free drive**, not from the circuit, so they are
    identical across conditions by construction. The run asserts that rather
    than assuming it.
    """
    args = context.args
    lateral = context.lateral(circuit_name, circuit)
    out: dict[str, dict[str, PopulationRecord]] = {}
    for condition in conditions:
        per_group: dict[str, PopulationRecord] = {}
        for trial in range(int(trials)):
            drive = context.generator.drive_tensor(
                StimulusCondition.from_name(condition), trial=trial, n_steps=int(args.n_steps)
            )
            features = input_control_features(
                drive, circuit, lateral,
                on_types=tuple(context.generator.on_types),
                off_types=tuple(context.generator.off_types),
                bin_steps=int(args.bin_steps), skip_bins=int(args.skip_bins),
            )
            for group, value in features.items():
                centres = np.arange(value.shape[0], dtype=np.float64)
                per_group.setdefault(group, []).append(
                    PopulationRecord(
                        stimulus=condition, trial=trial, group=group,
                        features=value, bin_centres=centres, readout="drive",
                    )
                )
        out[condition] = {group: records for group, records in per_group.items()}
    return {c: {g: list(v) for g, v in d.items()} for c, d in out.items()}


def control_identity_check(
    records: dict[str, dict[str, list[PopulationRecord]]], group: str
) -> dict[str, Any]:
    """Are the drive-derived control features identical across every condition and trial?

    A direct check of the claim that the drive carries no condition-specific
    magnitude. Reported, not assumed.
    """
    values = [
        np.asarray(rec.features, dtype=np.float64).ravel()
        for per_group in records.values()
        for rec in per_group.get(group, [])
    ]
    if len(values) < 2:
        return {"group": group, "compared": 0, "max_spread": float("nan"), "identical": False}
    stack = np.vstack(values)
    spread = float(np.abs(stack - stack[0]).max())
    return {
        "group": group,
        "compared": int(stack.shape[0]),
        "max_spread": spread,
        "identical": bool(spread == 0.0),
    }


# --------------------------------------------------------------------- trials


def run_trials(
    circuit: MCNSCircuit,
    context: RunContext,
    circuit_name: str,
    condition: str,
    population_groups: tuple[str, ...],
    variants: tuple[str, ...],
    scale: float,
    progress: Progress | None = None,
) -> tuple[dict[str, dict[str, list[PopulationRecord]]], dict[str, Any]]:
    """Run every trial of one stimulus condition, once, and bin it every way needed.

    Returns ``{variant: {group: records}}`` plus a liveness descriptor per
    readout group. The liveness descriptor is what lets a report say "T4/T5 were
    silent at this gain" instead of leaving the reader to infer it from a feature
    width, and it carries the **per-trial** spike counts rather than only their
    sum, so a median, a maximum and a zero-activity fraction are all available
    and a mean carried by two lucky trials cannot masquerade as liveness.
    """
    args = context.args
    index_map = context.index_map(circuit_name, circuit)
    lateral = context.lateral(circuit_name, circuit)
    propagation = context.propagation(circuit_name, circuit, scale)
    spec = StimulusCondition.from_name(condition)
    by_variant: dict[str, dict[str, list[PopulationRecord]]] = {
        variant: {g: [] for g in population_groups} for variant in variants
    }
    activity: dict[str, Any] = {}
    simulation_seconds = 0.0
    readout_seconds = 0.0

    for trial in range(int(args.trials)):
        encoder = context.generator.as_encoder(trial=trial)
        sim_started = time.perf_counter()
        result = run_simulation(
            circuit, encoder, spec,
            n_steps=int(args.n_steps), dt=float(args.dt), params=context.params,
            synapse_scale=float(scale), coupling=args.coupling, trace_steps=0,
            propagation=propagation, record_input=False,
        )
        simulation_seconds += time.perf_counter() - sim_started

        read_started = time.perf_counter()
        for variant in variants:
            readout, normalise = VARIANTS[variant]
            magnitude_source: np.ndarray | None = None
            for group in population_groups:
                if group == MAGNITUDE_GROUP:
                    continue
                features, centres = population_features(
                    result.spikes, index_map, group,
                    bin_steps=int(args.bin_steps), skip_bins=int(args.skip_bins),
                    normalise=normalise, readout=readout, lateral_bin=lateral,
                )
                if group == "T4+T5" and MAGNITUDE_GROUP in population_groups:
                    # Unnormalised, so the magnitude control measures total
                    # activity rather than a rescaled direction.
                    magnitude_source = population_features(
                        result.spikes, index_map, group,
                        bin_steps=int(args.bin_steps), skip_bins=int(args.skip_bins),
                        normalise=False, readout=readout, lateral_bin=lateral,
                    )[0]
                by_variant[variant][group].append(
                    PopulationRecord(
                        stimulus=condition, trial=trial, group=group,
                        features=features, bin_centres=centres,
                        readout=readout, lateral_bins=int(args.lateral_bins)
                        if readout == PRIMARY_READOUT else 0,
                    )
                )
            if MAGNITUDE_GROUP in population_groups:
                if magnitude_source is None:
                    magnitude_source = population_features(
                        result.spikes, index_map, "T4+T5",
                        bin_steps=int(args.bin_steps), skip_bins=int(args.skip_bins),
                        normalise=False, readout=readout, lateral_bin=lateral,
                    )[0]
                by_variant[variant][MAGNITUDE_GROUP].append(
                    PopulationRecord(
                        stimulus=condition, trial=trial, group=MAGNITUDE_GROUP,
                        features=magnitude_features(magnitude_source), bin_centres=np.zeros(1),
                        readout="magnitude",
                    )
                )
        readout_seconds += time.perf_counter() - read_started

        for group in population_groups:
            if group == MAGNITUDE_GROUP:
                continue
            descriptor = activity_descriptor(result.spikes, index_map, group)
            entry = activity.setdefault(
                group,
                {"neurons": 0, "trials": 0, "n_steps": int(args.n_steps), "spikes_per_trial": []},
            )
            entry["spikes_per_trial"].append(int(descriptor["spikes"]))
            entry["neurons"] = int(descriptor["neurons"])
            entry["trials"] += 1
        if progress is not None:
            progress.note(
                f"  {circuit_name:<17} {condition:<27} trial {trial}  "
                f"sim {1000 * (read_started - sim_started):6.1f} ms  "
                f"readout {1000 * (time.perf_counter() - read_started):5.1f} ms"
            )

    for group, entry in activity.items():
        entry.update(
            liveness(
                entry["spikes_per_trial"],
                n_neurons=int(entry["neurons"]),
                n_steps=int(entry["n_steps"]),
            )
        )
    if progress is not None:
        progress.record(f"simulation_seconds::{circuit_name}", simulation_seconds)
        progress.record(f"readout_seconds::{circuit_name}", readout_seconds)
    return by_variant, activity


# ------------------------------------------------------------------- analysis


def decode_contrast(
    records_by_group: dict[str, list[PopulationRecord]],
    groups: tuple[str, ...],
    left: str,
    right: str,
    args: argparse.Namespace,
    with_null: bool,
) -> dict[str, Any]:
    """Cross-validate one two-class contrast, group by group.

    ``records_by_group`` must already contain **both** conditions. The decoder
    keys records on ``(stimulus, trial)``, so a stimulus that is missing from the
    input is rejected rather than silently scored against a single class.
    """
    out: dict[str, Any] = {}
    for group in groups:
        payload = records_by_group.get(group, [])
        present = {str(r.stimulus) for r in payload}
        if not {left, right} <= present:
            out[group] = {
                "status": "missing_condition",
                "reason": f"records cover {sorted(present)}, need {sorted({left, right})}",
            }
            continue
        result = cross_validate(
            payload, [left, right],
            learning_rate=float(args.decoder_learning_rate),
            n_iterations=int(args.decoder_iterations),
            l2=float(args.decoder_l2),
        )
        if with_null and int(args.permutations) > 0 and result.get("status") == "ok":
            result["permutation_null"] = cv_permutation_null(
                payload, [left, right],
                n_permutations=int(args.permutations), seed=int(args.seed),
                learning_rate=float(args.decoder_learning_rate),
                n_iterations=int(args.decoder_iterations),
                l2=float(args.decoder_l2),
            )
        result.setdefault("null_ran", bool(with_null and int(args.permutations) > 0))
        out[group] = result
    return out


# --------------------------------------------------------------------- report


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


def _cell(entry: dict[str, Any]) -> str:
    if not entry or entry.get("status") != "ok":
        return f"  n/a"
    accuracy = float(entry["mean_test_accuracy"])
    spread = float(entry.get("sd_test_accuracy", float("nan")))
    null = entry.get("permutation_null", {})
    marker = ""
    if null.get("status") == "ok":
        marker = "*" if null.get("exceeds_null_p95") else " "
    return f"{accuracy:.3f}+-{spread:.3f}{marker}"


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="bounded subset, 8 trials, one scale sweep, 2 stimuli")
    parser.add_argument("--phase-4c", action="store_true",
                        help="pre-declared Phase 4C replication: gain fixed at "
                             f"{PREDECLARED_SYNAPSE_SCALE}, {PREDECLARED_TRIALS} trials, "
                             f"{PREDECLARED_PERMUTATIONS} permutations, no overrides accepted")
    parser.add_argument("--estimate", action="store_true",
                        help="with --phase-4c: time one circuit condition on a few trials and "
                             "project the declared run instead of launching it")
    parser.add_argument("--estimate-trials", type=int, default=4)
    parser.add_argument("--estimate-permutations", type=int, default=25)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    parser.add_argument("--annotations", default=None)
    parser.add_argument("--connectivity", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--bin-steps", type=int, default=None)
    parser.add_argument("--skip-bins", type=int, default=2)
    parser.add_argument("--scales", default=None)
    parser.add_argument("--coupling", choices=("linear", "log1p"), default="linear")
    parser.add_argument("--readout", choices=READOUTS, default=PRIMARY_READOUT,
                        help="primary readout resolution")
    parser.add_argument("--variants", default=None,
                        help=f"comma separated readout variants from {list(VARIANTS)}")
    parser.add_argument("--lateral-bins", type=int, default=LATREADOUT_BINS)
    parser.add_argument("--spot-fraction", type=float, default=0.15)
    parser.add_argument("--stimulus-current", type=float, default=2.0)
    parser.add_argument("--input-noise", type=float, default=1.0)
    parser.add_argument("--amplitude-perturbation", type=float, default=0.05)
    parser.add_argument("--no-randomise-start", action="store_true",
                        help="disable the per-trial randomised sweep start (a confound, off by default)")
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--decoder-iterations", type=int, default=400)
    parser.add_argument("--decoder-learning-rate", type=float, default=0.5)
    parser.add_argument("--decoder-l2", type=float, default=1e-2)
    parser.add_argument("--permutations", type=int, default=None,
                        help="label-permutation null size; 0 disables")
    parser.add_argument("--null-groups", default=None,
                        help="comma separated groups that get a permutation null")
    parser.add_argument("--null-contrasts", default=None,
                        help="comma separated contrasts that get a permutation null")
    parser.add_argument("--contrasts", default=None, help="comma separated subset of contrasts to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-bodies-per-type", type=int, default=None)
    parser.add_argument("--all-groups", action="store_true",
                        help="also report the spike-based INPUT group and every exact subtype")
    parser.add_argument("--quiet", action="store_true")
    return parser


#: Quick-mode defaults, in one place so the preset and the report cannot drift.
QUICK_DEFAULTS: dict[str, Any] = {
    "trials": 8,
    "n_steps": 150,
    "bin_steps": 25,
    "scales": QUICK_SCALES,
    "max_bodies_per_type": 100,
    "permutations": 100,
    "null_groups": (INPUT_INSTANT, "T4", "T5", "T4+T5"),
    "null_contrasts": (PRIMARY_CONTRAST,),
    "output_dir": "outputs/mcns_motion_benchmark_quick",
}

FULL_DEFAULTS: dict[str, Any] = {
    "trials": 12,
    "n_steps": 300,
    "bin_steps": 25,
    "scales": DEFAULT_SCALES,
    "max_bodies_per_type": 0,
    "permutations": 200,
    "null_groups": (INPUT_INSTANT, "T4", "T5", "T4+T5"),
    "null_contrasts": tuple(CONTRASTS),
    "output_dir": "outputs/mcns_motion_benchmark",
}

#: Where a declared Phase 4C run writes its artifacts. This directory is the
#: frozen record of the run and must not receive anything else.
PHASE4C_OUTPUT_DIR = "outputs/mcns_motion_benchmark_phase4c"

#: Where a Phase 4C timing estimate writes. **Separate by construction, not by
#: convention.** An estimate is a reduced-configuration measurement, so letting it
#: land in the same directory as the real run would put a second, smaller
#: `phase4c_timing_estimate.json` next to the frozen artifacts of a completed
#: experiment, where the two are indistinguishable without reading the JSON.
#:
#: The name must not merely *differ* from :data:`PHASE4C_OUTPUT_DIR`; it must not
#: be a prefix-extension of it either, or a glob or a directory listing still
#: sweeps both together. An earlier attempt at this used
#: ``..._phase4c_estimate``, which is a prefix-extension of
#: ``..._phase4c`` and failed exactly that check. Hence the distinct word.
PHASE4C_ESTIMATE_DIR = "outputs/phase4c_timing_estimates"

#: Phase 4C parameters that must not be changed on the command line, and the
#: declared value each is held at. A Phase 4C run is a replication of one declared
#: configuration; a flag that moved any of these would make it a different
#: experiment that happened to be launched from the same script.
#:
#: This list is the *whole* enforcement surface, and it is deliberately complete
#: rather than convenient. An earlier version pinned only the nine numeric
#: parameters above and left the coupling law, the four stimulus parameters and
#: the three decoder settings to be silently accepted, which meant
#: ``--phase-4c --coupling log1p`` would have run a different experiment under the
#: Phase 4C name while the pre-declaration claimed the coupling law was held
#: fixed. Silently accepting a flag is nearly as bad as silently changing it: the
#: operator gets no signal that the declaration was not honoured. Everything
#: scientific is therefore either in this dict (refused if it differs) or forced
#: to the declared value after the check.
PHASE4C_FIXED: dict[str, Any] = {
    "scales": [PREDECLARED_SYNAPSE_SCALE],
    "trials": PREDECLARED_TRIALS,
    "permutations": PREDECLARED_PERMUTATIONS,
    "max_bodies_per_type": PREDECLARED_MAX_BODIES_PER_TYPE,
    "n_steps": PREDECLARED_N_STEPS,
    "bin_steps": PREDECLARED_BIN_STEPS,
    "skip_bins": PREDECLARED_SKIP_BINS,
    "dt": PREDECLARED_DT,
    "seed": PREDECLARED_SEED,
    # --- the coupling law and the stimulus protocol
    "coupling": PREDECLARED_COUPLING,
    "spot_fraction": PREDECLARED_SPOT_FRACTION,
    "stimulus_current": PREDECLARED_STIMULUS_CURRENT,
    "input_noise": PREDECLARED_INPUT_NOISE,
    "amplitude_perturbation": PREDECLARED_AMPLITUDE_PERTURBATION,
    # --- the decoder
    "decoder_iterations": PREDECLARED_DECODER_ITERATIONS,
    "decoder_learning_rate": PREDECLARED_DECODER_LEARNING_RATE,
    "decoder_l2": PREDECLARED_DECODER_L2,
    # --- the readout
    "lateral_bins": LATREADOUT_BINS,
    "readout": PRIMARY_READOUT,
    "all_groups": False,
    "no_normalise": False,
}

#: Phase 4C options a caller may vary without changing the experiment: where the
#: data and the artifacts live, and the two estimation knobs. Everything else is
#: in :data:`PHASE4C_FIXED` and is refused.
PHASE4C_MUTABLE: frozenset[str] = frozenset(
    {"data_dir", "annotations", "connectivity", "output_dir", "cache", "no_cache",
     "quiet", "estimate", "estimate_trials", "estimate_permutations"}
)


def apply_phase4c(args: argparse.Namespace) -> argparse.Namespace:
    """Fill a Phase 4C invocation from the pre-declaration, refusing every override.

    The refusal is the point. Phase 4C exists to answer whether one declared
    observation replicates, and a flag that could move the gain, the trial count
    or the permutation count would let the answer be chosen after the fact. So
    the declared values are applied unconditionally, any explicitly-passed
    conflicting value is reported by name, and the run refuses to start.
    """
    estimating = bool(args.estimate)
    if args.quick:
        raise SystemExit("error: --quick is not a Phase 4C mode; use --phase-4c")
    if args.no_randomise_start:
        raise SystemExit(
            "error: Phase 4C keeps the randomised-per-trial sweep start; it cannot be disabled"
        )

    conflicts = []
    for name, declared in PHASE4C_FIXED.items():
        if estimating and name in {"trials", "permutations"}:
            continue
        given = getattr(args, name, None)
        if given is None:
            continue
        if isinstance(declared, list):
            normalised = [float(s) for s in given.split(",")] if isinstance(given, str) else [
                float(s) for s in given
            ]
            matches = normalised == [float(v) for v in declared]
        else:
            matches = type(given) is type(declared) and given == declared
        if not matches:
            conflicts.append(f"--{name.replace('_', '-')} {given!r} (pre-declared {declared!r})")
    if conflicts:
        raise SystemExit(
            "error: Phase 4C is a pre-declared replication and its declared parameters cannot be "
            "changed: " + "; ".join(conflicts) + ". Remove the overriding flag."
        )

    for name, value in PHASE4C_FIXED.items():
        setattr(args, name, list(value) if isinstance(value, list) else value)
    if estimating:
        args.trials = int(args.estimate_trials)
        args.permutations = int(args.estimate_permutations)

    args.null_groups = tuple(PREDECLARED_NULL_GROUPS)
    args.null_contrasts = tuple(PREDECLARED_NULL_CONTRASTS)
    args.contrasts = tuple(PREDECLARED_CONTRASTS)
    args.variants = tuple(PREDECLARED_VARIANTS)
    args.readout = PRIMARY_READOUT
    args.lateral_bins = LATREADOUT_BINS
    args.output_dir = args.output_dir or PHASE4C_OUTPUT_DIR
    args.no_randomise_start = False
    args.mode = f"phase{PREDECLARED_PHASE.lower()}-estimate" if estimating else f"phase{PREDECLARED_PHASE.lower()}"
    args.is_phase4c = True
    args.is_phase4c_estimate = estimating

    if not estimating:
        # The declared configuration, asserted against the pre-declaration. In
        # estimate mode the trial count and permutation count are deliberately
        # reduced to make the projection cheap, and the estimate says so.
        enforce_predeclared(
            synapse_scale=float(args.scales[0]),
            trials=int(args.trials),
            n_permutations=int(args.permutations),
            max_bodies_per_type=int(args.max_bodies_per_type),
            n_steps=int(args.n_steps),
            bin_steps=int(args.bin_steps),
            skip_bins=int(args.skip_bins),
            seed=int(args.seed),
        )
    return args


def apply_presets(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset options from the mode preset and record what was chosen."""
    if getattr(args, "phase_4c", False):
        return apply_phase4c(args)
    preset = dict(QUICK_DEFAULTS if args.quick else FULL_DEFAULTS)
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if isinstance(args.scales, str):
        args.scales = [float(s) for s in args.scales.split(",") if s.strip()]
    else:
        args.scales = [float(s) for s in args.scales]

    def _names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(v.strip() for v in value.split(",") if v.strip())
        return tuple(value or ())

    args.null_groups = _names(args.null_groups)
    args.null_contrasts = _names(args.null_contrasts)
    requested = _names(args.contrasts) or tuple(CONTRASTS)
    unknown = [c for c in requested if c not in CONTRASTS]
    if unknown:
        raise SystemExit(f"error: unknown contrasts {unknown}; choose from {list(CONTRASTS)}")
    args.contrasts = requested
    variants = _names(args.variants) or (PRIMARY_VARIANT, "subtype_lateral_raw")
    bad_variants = [v for v in variants if v not in VARIANTS]
    if bad_variants:
        raise SystemExit(f"error: unknown readout variants {bad_variants}; choose from {list(VARIANTS)}")
    if PRIMARY_VARIANT not in variants:
        variants = (PRIMARY_VARIANT,) + variants
    args.variants = variants
    args.no_randomise_start = bool(args.no_randomise_start)
    args.mode = "quick" if args.quick else "full"
    args.is_phase4c = False
    args.is_phase4c_estimate = False
    return args


def _population_identity(circuits: dict[str, MCNSCircuit]) -> dict[str, Any]:
    """Do the circuit conditions share one neuron set, in one order, with one labelling?

    Reported per pair and asserted by the caller. The wiring control is only a
    control if it differs from the measured condition in the *wiring* alone, so
    this is checked rather than described.
    """
    names = list(circuits)
    out: dict[str, Any] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = circuits[first], circuits[second]
            out[f"{first}|{second}"] = {
                "neurons": int(a.num_neurons),
                "body_ids_equal": bool(np.array_equal(a.body_ids, b.body_ids)),
                "cell_type_labels_equal": bool(np.array_equal(a.cell_type, b.cell_type)),
                "soma_side_equal": bool(np.array_equal(a.soma_side, b.soma_side)),
                "identical": bool(
                    np.array_equal(a.body_ids, b.body_ids)
                    and np.array_equal(a.cell_type, b.cell_type)
                ),
                "edges": {"first": int(a.num_edges), "second": int(b.num_edges)},
                "shares_neuron_set": True,
            }
    return out


def _phase4c_condition_extras(
    per_condition: dict[str, dict[str, dict[str, list[PopulationRecord]]]],
    results: dict[str, Any],
    scale_key: str,
    variant: str,
    contrast: str,
    circuit_name: str,
    args: argparse.Namespace,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Per-condition Phase 4C extras: separation ratio, features, informativeness.

    Everything here is about the *primary* readout group on the *primary*
    contrast, which is the only place a Phase 4C claim is made. The separation
    ratio and its own permutation null are reported alongside the decoder
    accuracy because the decoder permutation null is uninformative when the two
    classes form tight clusters: it can equal the observed accuracy while the
    geometry is genuinely clean. The separation-ratio null is the statistic that
    does not saturate that way.
    """
    left, right = CONTRASTS[contrast]
    group = PREDECLARED_GROUP
    records = list(per_condition[left][variant][group]) + list(per_condition[right][variant][group])
    decoded = results[scale_key][variant][contrast][circuit_name].get(group, {})

    features = feature_report(records)
    if progress is not None:
        progress.note(
            f"  separation ratio for {circuit_name}: {len(records)} records, "
            f"{features['n_nonzero_features']}/{features['n_features']} non-zero features"
        )
    ratio = fast_separation_ratio(
        [r.features for r in records], [r.stimulus for r in records], [left, right]
    )
    ratio_null = (
        separation_ratio_null(
            records, [left, right], n_permutations=int(args.permutations), seed=int(args.seed)
        )
        if int(args.permutations) > 0
        else {"status": "disabled", "n_permutations": 0}
    )
    null = decoded.get("permutation_null", {}) if isinstance(decoded, dict) else {}
    return {
        "group": group,
        "variant": variant,
        "contrast": contrast,
        "features": features,
        "separation_ratio": float(ratio),
        "separation_ratio_null": ratio_null,
        "mean_test_accuracy": decoded.get("mean_test_accuracy"),
        "fold_accuracies": decoded.get("fold_accuracies"),
        "null_p95": null.get("null_p95"),
        "z_score": null.get("z_score"),
        "fingerprint": fingerprint(records),
        "n_records": len(records),
    }


def _measure_estimate_point(
    args: argparse.Namespace,
    circuits: dict[str, MCNSCircuit],
    context: RunContext,
    control_records: dict[str, dict[str, list[PopulationRecord]]],
    conditions_needed: list[str],
    population_groups: tuple[str, ...],
    variants: tuple[str, ...],
    report_groups: tuple[str, ...],
    contrasts: list[str],
    circuit_name: str,
    scale: float,
    trials: int,
    progress: Progress,
) -> dict[str, Any]:
    """Time one circuit condition at one trial count, and one point on its nulls.

    Separates the three cost centres and, for the nulls, splits fixed from
    marginal by timing the same nulls a second time at twice the permutation
    count. Returns the raw measurements; the projection is built by
    :func:`~flybrain.benchmark.phase4c.timing_projection`.
    """
    saved_trials = int(args.trials)
    args.trials = int(trials)
    progress.stage(
        f"Timing point: {circuit_name}, {trials} trials, {args.permutations} permutations"
    )
    before = dict(progress.timings)
    per_condition: dict[str, dict[str, dict[str, list[PopulationRecord]]]] = {}
    for condition in conditions_needed:
        records, _ = run_trials(
            circuits[circuit_name], context, circuit_name, condition,
            population_groups, variants, scale, progress,
        )
        per_condition[condition] = records

    def _elapsed(prefix: str) -> float:
        return sum(v for k, v in progress.timings.items() if k.startswith(prefix)) - sum(
            v for k, v in before.items() if k.startswith(prefix)
        )

    simulation_seconds = _elapsed(f"simulation_seconds::{circuit_name}")
    readout_seconds = _elapsed(f"readout_seconds::{circuit_name}")

    decode_seconds = 0.0
    null_seconds = 0.0
    null_problems: list[tuple[list[PopulationRecord], str, str, str]] = []
    for contrast in contrasts:
        left, right = CONTRASTS[contrast]
        for group in report_groups:
            if group in CONTROL_GROUPS:
                pool = list(control_records[left][group]) + list(control_records[right][group])
            else:
                pool = (
                    list(per_condition[left][PRIMARY_VARIANT][group])
                    + list(per_condition[right][PRIMARY_VARIANT][group])
                )
            with_null = (
                int(args.permutations) > 0
                and group in args.null_groups
                and contrast in args.null_contrasts
            )
            started = time.perf_counter()
            decode_contrast({group: pool}, (group,), left, right, args, with_null)[group]
            elapsed = time.perf_counter() - started
            if with_null:
                null_seconds += elapsed
                null_problems.append((pool, group, left, right))
            else:
                decode_seconds += elapsed
            if not with_null and group == PREDECLARED_GROUP and contrast == PREDECLARED_CONTRAST:
                progress.note(f"  {contrast} {group} plain decode: {1000 * elapsed:7.1f} ms")

    # Second point on the same nulls, at twice the permutation count, so the
    # per-permutation marginal cost can be separated from the fixed cost of
    # building the leave-one-out design and fitting the observed run. A single
    # small measurement mixes the two, and scaling the mixture linearly in the
    # permutation count is wrong by a large factor in one direction or the other.
    marginal_per_permutation = float("nan")
    null_seconds_double = 0.0
    if null_problems:
        progress.stage(f"  second point: the same nulls at {2 * int(args.permutations)} permutations")
        saved_permutations = int(args.permutations)
        args.permutations = 2 * saved_permutations
        started = time.perf_counter()
        for pool, group, left, right in null_problems:
            decode_contrast({group: pool}, (group,), left, right, args, True)[group]
        null_seconds_double = time.perf_counter() - started
        args.permutations = saved_permutations
        if saved_permutations > 0:
            marginal_per_permutation = (null_seconds_double - null_seconds) / float(
                saved_permutations
            )

    args.trials = saved_trials
    return {
        "trials": int(trials),
        "permutations": int(args.permutations),
        "simulation_seconds": simulation_seconds,
        "readout_seconds": readout_seconds,
        "decode_seconds": decode_seconds,
        "null_seconds": null_seconds,
        "null_seconds_double": null_seconds_double,
        "null_marginal_per_permutation": marginal_per_permutation,
        "fingerprint": _estimate_fingerprint(per_condition),
        "per_condition": per_condition,
    }


def _estimate_fingerprint(
    per_condition: dict[str, dict[str, dict[str, list[PopulationRecord]]]]
) -> str:
    left, right = CONTRASTS[PREDECLARED_CONTRAST]
    return fingerprint(
        list(per_condition[left][PRIMARY_VARIANT][PREDECLARED_GROUP])
        + list(per_condition[right][PRIMARY_VARIANT][PREDECLARED_GROUP])
    )


def _phase4c_estimate(
    args: argparse.Namespace,
    circuits: dict[str, MCNSCircuit],
    context: RunContext,
    generator: MotionStimulusGenerator,
    control_records: dict[str, dict[str, list[PopulationRecord]]],
    conditions_needed: list[str],
    population_groups: tuple[str, ...],
    variants: tuple[str, ...],
    report_groups: tuple[str, ...],
    contrasts: list[str],
    scales: list[float],
    progress: Progress,
    output_dir: Path,
    warm: dict[str, Any],
) -> int:
    """Time one circuit condition, and project the declared run onto the measurement.

    Requirement, not a courtesy: the declared run is 20 trials x 3 circuits with 500
    permutations, and the permutation cost is not guessable, so it is measured.

    **Two points are taken, and the reason is not fussiness.** The first is the
    required cheap one: one circuit condition and a small number of trials. The
    second is the same thing at the *declared* trial count, with the same reduced
    permutation count. Point 2 costs a fraction of a second of simulation and is
    what the launch decision uses, because the batched leave-one-trial-out design
    changes cache behaviour between 5 trials and 20: at 5 trials the whole design
    fits in L2, at 20 it does not, so a per-row-fit cost measured at 5 trials
    under-predicts the declared run by roughly a factor of two. Extrapolating in
    the trial count is what produces that error, and taking a measurement at the
    declared geometry removes the extrapolation instead of modelling it.

    Nothing declared is changed to make the estimate cheap. The graph, the node
    budget, ``n_steps``, the binning, the seed, the gain and the readout are all
    the declared values. Only the permutation count is reduced, and both points
    say so.
    """
    circuit_name = PRIMARY_CIRCUIT
    scale = float(scales[0])
    graph_seconds = progress.timings.get("graph_seconds", 0.0)
    cache_seconds = progress.timings.get("cache_warm_seconds", 0.0)

    common = (
        args, circuits, context, control_records, conditions_needed, population_groups,
        variants, report_groups, contrasts, circuit_name, scale,
    )
    point_small = _measure_estimate_point(*common, int(args.trials), progress)

    # The declared-geometry point. Stimulus tensors for the extra trials are built
    # on demand; the declared run needs them anyway, so this work is not wasted.
    point_declared = (
        point_small
        if int(args.trials) == PREDECLARED_TRIALS
        else _measure_estimate_point(*common, PREDECLARED_TRIALS, progress)
    )

    def _project(point: dict[str, Any]) -> dict[str, Any]:
        return timing_projection(
            measured_simulation_seconds=point["simulation_seconds"],
            measured_decode_seconds=point["decode_seconds"],
            measured_null_seconds=point["null_seconds"],
            measured_null_marginal_per_permutation=point["null_marginal_per_permutation"],
            measured_trials=point["trials"],
            measured_permutations=point["permutations"],
        )

    projections = {
        "small_trial_point": _project(point_small),
        "declared_geometry_point": _project(point_declared),
    }
    projection = dict(projections["declared_geometry_point"])
    overhead = float(graph_seconds) + float(cache_seconds)
    total_projection = float(projection["projected_total_seconds"]) + overhead
    projection["fixed_overhead_seconds"] = overhead
    projection["projected_total_with_overhead_seconds"] = total_projection
    projection["projected_total_with_overhead_minutes"] = total_projection / 60.0
    projection["measured_estimate_wall_clock_seconds"] = progress.total_seconds()
    projection["is_reasonably_bounded"] = bool(total_projection <= 4 * 3600.0)

    small_total = (
        float(projections["small_trial_point"]["projected_total_seconds"]) + overhead
    )
    extrapolation_error_factor = float(total_projection / small_total) if small_total else float("nan")

    print()
    print("=" * _WIDTH)
    print("PHASE 4C TIMING ESTIMATE  (one circuit condition, reduced permutations)")
    print("=" * _WIDTH)
    print(f"  measured on                  : {circuit_name} only")
    print(f"  unchanged, at declared values : synapse_scale {scale:g}, n_steps {args.n_steps}, "
          f"bin_steps {args.bin_steps}, budget {args.max_bodies_per_type}/type, seed {args.seed}")
    print(f"  reduced for the estimate      : permutations {args.permutations} "
          f"(declared {PREDECLARED_PERMUTATIONS}); trials in point 1 only")
    print()
    print(f"  {'timing point':<28}{'trials':>7}{'sim s':>9}{'readout s':>11}"
          f"{'decode s':>10}{'null s':>9}{'null ms/perm':>14}")
    for name, point in (("point 1 (reduced trials)", point_small),
                        ("point 2 (declared geometry)", point_declared)):
        print(f"  {name:<28}{point['trials']:>7}{point['simulation_seconds']:>9.2f}"
              f"{point['readout_seconds']:>11.2f}{point['decode_seconds']:>10.2f}"
              f"{point['null_seconds']:>9.2f}{1000 * point['null_marginal_per_permutation']:>14.2f}")
    print()
    print(f"  graph construction            : {graph_seconds:8.2f} s")
    print(f"  tensor/propagation warm       : {cache_seconds:8.2f} s")
    print()
    print(f"  PROJECTED simulation          : {projection['projected_simulation_seconds'] / 60:8.2f} min")
    print(f"  PROJECTED plain decoding      : {projection['projected_decode_seconds'] / 60:8.2f} min")
    print(f"  PROJECTED permutation nulls   : {projection['projected_null_seconds'] / 60:8.2f} min")
    print(f"  PROJECTED fixed overhead      : {overhead / 60:8.2f} min")
    print(f"  PROJECTED TOTAL               : {total_projection / 60:8.2f} min "
          f"({total_projection / 3600:.2f} h)   <- point 2, used for the decision")
    print()
    print(f"  point 1 alone would have projected {small_total / 60:.2f} min, i.e. it was off by a")
    print(f"  factor of {extrapolation_error_factor:.2f}. The batched leave-one-trial-out design is")
    print("  cache-resident at a small trial count and not at 20, so extrapolating in the trial")
    print("  count is not reliable; the declared-geometry measurement is what is used.")
    print()
    print(f"  simulations in the declared run: {projection['declared']['simulations']:,}")
    print(f"  label-permutation nulls        : {projection['declared']['label_permutation_nulls']:,} "
          f"x {projection['declared']['permutations']} permutations")
    print(f"  reasonably bounded (<= 4 h)    : {projection['is_reasonably_bounded']}")
    print(f"  rule                           : {projection['boundedness_rule']}")
    print()
    print("  The estimate changes nothing declared. The gain stays "
          f"{PREDECLARED_SYNAPSE_SCALE:g}, the trial count stays {PREDECLARED_TRIALS} and the "
          f"permutation count stays {PREDECLARED_PERMUTATIONS} in the real run.")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase4c_timing_estimate.json"
    path.write_text(
        json.dumps(
            {
                "phase": PREDECLARED_PHASE,
                "purpose": "pre-run timing estimate for the declared Phase 4C run",
                "is_not_a_result": (
                    "This file is a reduced-configuration timing measurement, not an experiment "
                    "result. It lives in a directory separate from the frozen Phase 4C artifacts "
                    "so the two can never be confused. No number here describes a circuit."
                ),
                "written_to": PHASE4C_ESTIMATE_DIR,
                "frozen_artifact_dir": PHASE4C_OUTPUT_DIR,
                "predeclaration": predeclared_block(),
                "estimate_settings": {
                    "circuit_condition": circuit_name,
                    "trials_per_condition_point_1": int(args.trials),
                    "trials_per_condition_point_2": PREDECLARED_TRIALS,
                    "permutations": int(args.permutations),
                    "stimulus_conditions": list(conditions_needed),
                    "reduced_relative_to_declaration": ["permutations", "point_1_trials"],
                },
                "measurements": {
                    "point_1_reduced_trials": {
                        k: v for k, v in point_small.items() if k != "per_condition"
                    },
                    "point_2_declared_geometry": {
                        k: v for k, v in point_declared.items() if k != "per_condition"
                    },
                },
                "projections": projections,
                "projection_used_for_the_decision": "declared_geometry_point",
                "projection_used_seconds_with_overhead": total_projection,
                "trial_count_extrapolation_error_factor": extrapolation_error_factor,
                "population": subset_headline(circuits),
                "determinism_fingerprint": point_declared["fingerprint"],
                "determinism_note": (
                    "Two estimates of the same declared configuration must produce this identical "
                    "digest; it hashes the primary-readout feature arrays at the declared 20-trial "
                    "geometry, not the derived accuracies."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print()
    print(f"  wrote {path}")
    return 0


def subset_headline(circuits: dict[str, MCNSCircuit]) -> dict[str, Any]:
    return {
        name: {"neurons": c.num_neurons, "edges": c.num_edges,
               "biological_synapse_count": c.total_synapse_count}
        for name, c in circuits.items()
    }


def _phase4c_liveness(
    condition_activity: dict[str, Any],
    scale_key: str,
    stimulus_conditions: tuple[str, ...],
    groups: tuple[str, ...] = POPULATION_GROUPS,
) -> dict[str, Any]:
    """Per-trial spike liveness per circuit condition, over one contrast's stimuli.

    Per-trial rather than only per-run, because a mean carried by two lucky
    trials is not liveness and a decoder on an empty feature set still returns an
    accuracy. The median, the extremes and the zero-activity fraction are what
    make "is this condition actually firing" answerable.
    """
    out: dict[str, Any] = {}
    for circuit_name, per_condition in condition_activity[scale_key].items():
        per_group: dict[str, Any] = {}
        for group in groups:
            per_trial: list[int] = []
            neurons = 0
            steps = 0
            for condition in stimulus_conditions:
                entry = per_condition.get(condition, {}).get(group, {})
                per_trial.extend(int(v) for v in entry.get("spikes_per_trial", []))
                neurons = max(neurons, int(entry.get("neurons", 0)))
                steps = max(steps, int(entry.get("n_steps", 0)))
            per_group[group] = liveness(per_trial, n_neurons=neurons, n_steps=steps)
        out[circuit_name] = per_group
    return out


def _phase4c_report(
    args: argparse.Namespace,
    scales: list[float],
    results: dict[str, Any],
    activity: dict[str, Any],
    condition_activity: dict[str, Any],
    extras: dict[str, Any],
    population_identity: dict[str, Any],
    gate_by_scale: dict[str, Any],
    identity: dict[str, Any],
    generator: MotionStimulusGenerator,
    verification: dict[str, Any],
    subset: Any,
    circuits: dict[str, MCNSCircuit],
    progress: Progress,
) -> dict[str, Any]:
    """Build and print the Phase 4C report, in the order the reader must read it.

    Liveness first, then the primary comparison, then the relay controls, then
    the statistics, then the claims. The order is deliberate: a decoder accuracy
    printed before the liveness of the population it was fitted on invites the
    reader to treat noise as a result, and a downstream number printed before the
    relay controls invites them to attribute relay information to the circuit.
    """
    scale_key = str(scales[0])
    left, right = CONTRASTS[PREDECLARED_CONTRAST]
    sweep_conditions = (left, right)
    liveness_by_circuit = _phase4c_liveness(
        condition_activity, scale_key, sweep_conditions
    )
    primary = results[scale_key][PRIMARY_VARIANT][PREDECLARED_CONTRAST]

    # ---------------------------------------------------------- 1. liveness
    informativeness: dict[str, Any] = {}
    for circuit_name in CIRCUITS:
        extra = extras.get(circuit_name, {})
        features = extra.get("features", {})
        verdict = classify_informative(
            liveness_by_circuit[circuit_name][PREDECLARED_GROUP],
            features=features,
            accuracy=extra.get("mean_test_accuracy"),
            null_p95=extra.get("null_p95"),
        )
        informativeness[circuit_name] = {
            "T4+T5": verdict,
            "T4": classify_informative(liveness_by_circuit[circuit_name]["T4"]),
            "T5": classify_informative(liveness_by_circuit[circuit_name]["T5"]),
        }

    print()
    print("=" * _WIDTH)
    print("PHASE 4C  1. LIVENESS  (read this before any accuracy below)")
    print("=" * _WIDTH)
    print(f"  synapse_scale {scales[0]:g} (PRE-DECLARED), {args.trials} trials per condition, "
          f"over {left} and {right}")
    print()
    for circuit_name in CIRCUITS:
        print(f"  {circuit_name}")
        print(f"    {'group':<8}{'mean':>9}{'median':>9}{'min':>7}{'max':>8}{'zero %':>9}"
              f"{'nonzero feats':>15}  status")
        for group in POPULATION_GROUPS:
            live = liveness_by_circuit[circuit_name][group]
            features = extras.get(circuit_name, {}).get("features", {}) if group == PREDECLARED_GROUP else {}
            verdict = informativeness[circuit_name][group]
            nonzero = (
                f"{features.get('n_nonzero_features', '?')}/{features.get('n_features', '?')}"
                if features else "-"
            )
            print(
                f"    {group:<8}{live['mean_spikes_per_trial']:>9.1f}"
                f"{live['median_spikes_per_trial']:>9.1f}"
                f"{live['min_spikes_per_trial']:>7}{live['max_spikes_per_trial']:>8}"
                f"{100 * live['fraction_zero_activity_trials']:>9.1f}{nonzero:>15}  "
                f"{verdict['status']}"
            )
        for group in POPULATION_GROUPS:
            verdict = informativeness[circuit_name][group]
            for reason in verdict["reasons"]:
                for line in _wrap(f"{group}: {reason}", _WIDTH - 6):
                    print(f"      {line}")
    print()
    print("  A condition marked uninformative_ MUST NOT have its decoder accuracy")
    print("  interpreted. It is reported for completeness and is scientifically empty.")
    print()

    # --------------------------------------------- 2. the primary comparison
    recurrent = primary.get(PRIMARY_CIRCUIT, {}).get(PREDECLARED_GROUP, {})
    shuffled = primary.get(CONTROL_CIRCUIT, {}).get(PREDECLARED_GROUP, {})
    both_informative = bool(
        informativeness[PRIMARY_CIRCUIT][PREDECLARED_GROUP]["may_interpret_accuracy"]
        and informativeness[CONTROL_CIRCUIT][PREDECLARED_GROUP]["may_interpret_accuracy"]
    )
    margin = margin_statistics(
        recurrent.get("fold_accuracies", []), shuffled.get("fold_accuracies", []),
        n_permutations=int(args.permutations), seed=int(args.seed),
    )
    margin["both_conditions_informative"] = both_informative
    margin["interpretable"] = bool(
        both_informative and margin.get("status") == "ok" and margin.get("effect_supported")
    )
    if not both_informative:
        margin["statement_uninformative"] = (
            "At least one of the two conditions is silent or saturated at the pre-declared gain, "
            "so the recurrent-minus-shuffled margin is reported but is SCIENTIFICALLY "
            "UNINFORMATIVE and no accuracy difference may be interpreted from it."
        )

    print("=" * _WIDTH)
    print("PHASE 4C  2. PRIMARY COMPARISON")
    print("=" * _WIDTH)
    print(f"  primary quantity: {PRIMARY_CIRCUIT} accuracy - {CONTROL_CIRCUIT} accuracy")
    print(f"  readout: {PRIMARY_VARIANT} on {PREDECLARED_GROUP}   contrast: {PREDECLARED_CONTRAST}")
    print()
    print("  TWO DISTINCT NULL FAMILIES ARE USED IN THIS REPORT. They are not")
    print("  interchangeable and neither is privileged:")
    print("    NULL FAMILY 1 - decoder permutation null. Rescores the leave-one-trial-out")
    print("      ACCURACY with the trial-to-condition mapping permuted. Answers: can a")
    print("      linear decoder beat relabelled trials on accuracy? The 'null p95' and 'z'")
    print("      columns below are from this family.")
    print("    NULL FAMILY 2 - separation-ratio permutation null. Rescores a CONTINUOUS")
    print("      between/within centroid-distance RATIO with the same permutation scheme.")
    print("      Answers: is the observed distance ratio larger than relabelled trials give?")
    print("      The 'sep ratio' column below is from this family and is reported with its")
    print("      own null in section 2b.")
    print("  The two can disagree, and when they do neither is 'the answer': they are")
    print("  measuring different quantities with different power.")
    print()
    print(f"  {'':<22}{'accuracy':>10}{'sd':>8}{'chance':>8}{'null p95':>10}{'z':>8}"
          f"{'sep ratio':>11}  status")
    print(f"  {'':<22}{'':>10}{'':>8}{'':>8}{'(family 1)':>10}{'(fam 1)':>8}"
          f"{'(family 2)':>11}")
    for circuit_name in (PRIMARY_CIRCUIT, CONTROL_CIRCUIT, "MCNS_FEEDFORWARD"):
        entry = primary.get(circuit_name, {}).get(PREDECLARED_GROUP, {})
        extra = extras.get(circuit_name, {})
        if entry.get("status") != "ok":
            print(f"  {circuit_name:<22}{'n/a':>10}  ({entry.get('status')})")
            continue
        print(
            f"  {circuit_name:<22}{entry['mean_test_accuracy']:>10.4f}"
            f"{entry['sd_test_accuracy']:>8.4f}{entry['chance_accuracy']:>8.3f}"
            f"{_fmt(extra.get('null_p95')):>10}{_fmt(extra.get('z_score')):>8}"
            f"{extra.get('separation_ratio', float('nan')):>11.3f}  "
            f"{informativeness[circuit_name][PREDECLARED_GROUP]['status']}"
        )
    print()
    if margin.get("status") == "ok":
        print(f"  recurrent-minus-shuffled margin : {margin['mean_margin']:+.4f} "
              f"(sd {margin['sd_margin']:.4f} over {margin['n_folds']} shared folds)")
        print(f"  paired t {margin['paired_t_statistic']:+.3f} on {margin['paired_t_df']} df, "
              f"95% CI [{margin['confidence_interval'][0]:+.4f}, "
              f"{margin['confidence_interval'][1]:+.4f}], "
              f"p={margin['paired_t_p_two_sided']:.4f}")
        null_block = margin["margin_permutation_null"]
        print(f"  paired sign-flip null: {null_block['n_permutations']} permutations, "
              f"p95 |mean|={_fmt(null_block['null_p95'])}, p={_fmt(null_block['p_two_sided'])}")
        print(f"  uncertainty {'SUPPORTS' if margin['effect_supported'] else 'DOES NOT SUPPORT'} "
              f"calling this a difference rather than an observation")
        print(f"  margin interpretable: {margin['interpretable']}")
    else:
        print(f"  margin not computable: {margin.get('status')} ({margin.get('reason', '')})")
    if margin.get("statement_uninformative"):
        print()
        for line in _wrap(margin["statement_uninformative"], _WIDTH - 2):
            print(f"  {line}")
    print()
    print("  Conditions are NOT ranked and no condition is called a winner. A positive")
    print("  margin and a negative margin are reported with identical statistics.")
    print()
    if margin.get("status") == "ok":
        # The power statement is part of the result, not a footnote. A null from a
        # design that cannot see small effects bounds the difference; it does not
        # establish that the difference is zero, and the report has to say so.
        half_width = 0.5 * (margin["confidence_interval"][1] - margin["confidence_interval"][0])
        try:
            from scipy.stats import t as _student_t

            _n = int(margin["n_folds"])
            _sd = float(margin["sd_margin"])
            _critical = float(_student_t.ppf(0.975, _n - 1))
            _power_crit = float(_student_t.ppf(0.80, _n - 1))
            mde = float(_sd * np.sqrt((_critical + _power_crit) ** 2 / _n))
        except Exception:  # pragma: no cover - scipy is a hard dependency
            mde = float("nan")
        print(f"  POWER: this design is UNDERPOWERED for small effects. The 95% interval bounds the")
        print(f"  true margin at +-{half_width:.4f}, and the minimum detectable |margin| at 80% power is")
        print(f"  about {mde:.4f}. A null here is therefore a BOUND on the difference, not evidence")
        print("  that the difference is zero, and it is not evidence that the circuit does nothing.")
        print()

    # ------------------------------------- 2b. separation ratio against its own null
    separation_verdicts: dict[str, Any] = {}
    print("-" * _WIDTH)
    print("  NULL FAMILY 2 of 2 -- SEPARATION-RATIO trial-permutation null "
          f"({args.permutations} permutations)")
    print("  This null permutes the trial-to-condition mapping and rescores a CONTINUOUS")
    print("  geometry statistic (between/within centroid distance). It is NOT the same test as")
    print("  the decoder accuracy null in NULL FAMILY 1 above, and the two answer different")
    print("  questions. Neither is 'the' null and neither supersedes the other:")
    print("    family 1 asks: can a linear decoder beat relabelled trials on ACCURACY?")
    print("    family 2 asks: is the observed between/within DISTANCE RATIO bigger than")
    print("                    relabelled trials produce?")
    for circuit_name in CIRCUITS:
        extra = extras.get(circuit_name, {})
        ratio = extra.get("separation_ratio")
        ratio_null = extra.get("separation_ratio_null", {})
        if ratio is None or ratio_null.get("status") != "ok":
            separation_verdicts[circuit_name] = {
                "status": "unavailable", "separation_ratio": ratio
            }
            continue
        exceeds = bool(float(ratio) > float(ratio_null["null_p95"]))
        separation_verdicts[circuit_name] = {
            "status": "ok",
            "separation_ratio": float(ratio),
            "null_p95": ratio_null["null_p95"],
            "null_mean": ratio_null["null_mean"],
            "z_score": ratio_null["z_score"],
            "n_permutations": ratio_null["n_permutations"],
            "exceeds_null_p95": exceeds,
            "interpretation": (
                "the two sweep directions are further apart than repeated presentations of the "
                "same stimulus, above this null"
                if exceeds else
                "the two sweep directions are NOT further apart than repeated presentations of the "
                "same stimulus, so this readout carries no direction information above "
                "trial-to-trial noise at the pre-declared gain"
            ),
        }
        print(f"    {circuit_name:<18} ratio={_fmt(ratio):>8}  null p95={_fmt(ratio_null['null_p95']):>8}"
              f"  z={_fmt(ratio_null['z_score']):>8}  above null: {exceeds}")
    print()
    print("  A separation ratio below 1 means repeated presentations of the SAME stimulus are")
    print("  further apart than the two directions are from each other. A ratio at or below its")
    print("  own 95th-percentile null means the readout is not separating the conditions at all,")
    print("  whatever the decoder accuracy reports. This is reported for every condition because")
    print("  the decoder permutation null is uninformative when the classes are tight.")
    print()

    # -------------------------------------------------- 3. relay / input control
    print("=" * _WIDTH)
    print("PHASE 4C  3. RELAY / INPUT CONTROLS  (how much direction information")
    print("          exists BEFORE the MCNS circuit?)")
    print("=" * _WIDTH)
    print(f"  {'':<22}" + "".join(f"{c:>22}" for c in CIRCUITS))
    for group in (INPUT_INSTANT,) + RELAY_CONTROLS + (MAGNITUDE_GROUP,):
        line = f"  {group:<22}"
        for circuit_name in CIRCUITS:
            entry = primary.get(circuit_name, {}).get(group, {})
            line += f"{_cell(entry):>22}"
        print(line)
    print()
    print("  INPUT_SPATIAL and INPUT_TEMPORAL are the relay bounds. They are computed")
    print("  from the injected drive, are identical across the three circuit conditions by")
    print("  construction, and establish how much direction information is present before")
    print("  any MCNS circuit acts on it. Downstream decoding above the input relay is NOT")
    print("  evidence of MCNS computation unless the circuit-vs-shuffled comparison supports")
    print("  it. A T4/T5 score at or below these bounds is consistent with pure relay.")
    print()

    # ------------------------------- 3b. MAGNITUDE_ONLY against the structural readout
    magnitude_verdicts: dict[str, Any] = {}
    print("-" * _WIDTH)
    print("  IS THE STRUCTURAL READOUT MEASURING STRUCTURE?  "
          f"{MAGNITUDE_GROUP} vs {PREDECLARED_GROUP}")
    print("  This is the comparison that decides whether the 32-column readout is doing")
    print("  any work at all. MAGNITUDE_ONLY is a ONE-feature control: total T4+T5 activity.")
    print("  If a single number matches the 32-column subtype x lateral representation, the")
    print("  representation is not resolving structure at this operating point, and any")
    print("  accuracy it produces is a restatement of drive level.")
    print()
    print(f"  {'':<22}{MAGNITUDE_GROUP:>16}{PREDECLARED_GROUP:>16}{'gap':>10}  reading")
    for circuit_name in CIRCUITS:
        mag = primary.get(circuit_name, {}).get(MAGNITUDE_GROUP, {})
        struct = primary.get(circuit_name, {}).get(PREDECLARED_GROUP, {})
        if mag.get("status") != "ok" or struct.get("status") != "ok":
            magnitude_verdicts[circuit_name] = {"status": "unavailable"}
            continue
        mag_acc = float(mag["mean_test_accuracy"])
        struct_acc = float(struct["mean_test_accuracy"])
        gap = struct_acc - mag_acc
        if gap <= 0.0:
            reading = ("1-feature control MATCHES OR EXCEEDS the 32-column readout: the "
                       "readout is not resolving structure")
        else:
            reading = "structural readout above the 1-feature control, by a small margin"
        magnitude_verdicts[circuit_name] = {
            "status": "ok",
            "magnitude_only_accuracy": mag_acc,
            "structural_accuracy": struct_acc,
            "structural_minus_magnitude": float(gap),
            "structural_exceeds_magnitude_control": bool(gap > 0.0),
            "reading": reading,
        }
        print(f"  {circuit_name:<22}{mag_acc:>16.4f}{struct_acc:>16.4f}{gap:>+10.4f}  {reading}")
    print()
    print("  READ THIS BEFORE ANY T4/T5 NUMBER. Neither control has a permutation null of")
    print("  its own -- null_groups covers INPUT_INSTANT, T4, T5 and T4+T5 only -- so this")
    print("  is a comparison of point estimates, not a significance statement. It is")
    print("  reported because the audit found it to be the most consequential omitted")
    print("  comparison in the Phase 4C report.")
    print()

    # ---------------------------------------------- 4. statistical reporting
    statistics: dict[str, Any] = {}
    print("=" * _WIDTH)
    print("PHASE 4C  4. STATISTICAL REPORTING")
    print("=" * _WIDTH)
    for circuit_name in CIRCUITS:
        entry = primary.get(circuit_name, {}).get(PREDECLARED_GROUP, {})
        if entry.get("status") != "ok":
            continue
        extra = extras.get(circuit_name, {})
        statistics[circuit_name] = {
            "per_fold_accuracy": entry["fold_accuracies"],
            "mean_accuracy": entry["mean_test_accuracy"],
            "sd_accuracy": entry["sd_test_accuracy"],
            "chance_level": entry["chance_accuracy"],
            "null_p95": extra.get("null_p95"),
            "z_score": extra.get("z_score"),
            "separation_ratio": extra.get("separation_ratio"),
            "separation_ratio_null": extra.get("separation_ratio_null"),
            "n_trials": entry["n_trials"],
            "n_folds": entry["n_folds"],
            "n_permutations": int(args.permutations),
            "n_test_samples": entry["n_test_samples"],
            "n_test_samples_expected": entry["n_test_samples_expected"],
            "sample_count_matches": entry["sample_count_matches"],
            "confusion_matrix": entry["confusion_matrix"],
        }
        print()
        print(f"  {circuit_name}")
        print(f"    per-fold accuracy : "
              + " ".join(f"{v:.4f}" for v in entry["fold_accuracies"]))
        print(f"    mean accuracy     : {entry['mean_test_accuracy']:.4f}")
        print(f"    sd of folds       : {entry['sd_test_accuracy']:.4f}")
        print(f"    chance level      : {entry['chance_accuracy']:.3f}")
        print(f"    null p95          : {_fmt(extra.get('null_p95'))}")
        print(f"    z-score           : {_fmt(extra.get('z_score'))}")
        print(f"    separation ratio  : {_fmt(extra.get('separation_ratio'))}")
        ratio_null = extra.get("separation_ratio_null", {})
        if ratio_null.get("status") == "ok":
            print(f"    sep-ratio null p95: {_fmt(ratio_null.get('null_p95'))} "
                  f"(z={_fmt(ratio_null.get('z_score'))}, "
                  f"{ratio_null.get('n_permutations')} permutations)")
        print(f"    trials / folds / permutations : {entry['n_trials']} / "
              f"{entry['n_folds']} / {int(args.permutations)}")
        print(f"    test samples      : {entry['n_test_samples']} "
              f"(expected {entry['n_test_samples_expected']}, "
              f"match={entry['sample_count_matches']})")
    print()
    print(f"  recurrent-minus-shuffled margin: "
          f"{_fmt(margin.get('mean_margin'))} "
          f"[{_fmt((margin.get('confidence_interval') or [float('nan'), float('nan')])[0])}, "
          f"{_fmt((margin.get('confidence_interval') or [float('nan'), float('nan')])[1])}]")
    print(f"  n_trials={PREDECLARED_TRIALS}  n_folds={PREDECLARED_TRIALS}  "
          f"n_permutations={int(args.permutations)} (pre-declared, not reduced)")
    print()
    print("  Train/test structure: leave-one-trial-out. Each fold holds out one trial")
    print("  index across BOTH sweep directions, so no fold can be advantaged by which")
    print("  direction was excluded, and no held-out trial ever appears in its own")
    print("  training set. The sample count above is asserted, not trusted.")
    print()

    # ------------------------------------------------------------ 5. claims
    claims = claim_block()
    replication = _replication_verdict(
        margin=margin,
        both_informative=both_informative,
        separation_verdicts=separation_verdicts,
        n_trials=PREDECLARED_TRIALS,
    )
    print()
    print("=" * _WIDTH)
    print("PHASE 4C  REPLICATION VERDICT")
    print("=" * _WIDTH)
    for line in _wrap(replication["verdict"], _WIDTH - 2):
        print(f"  {line}")
    print()
    print(f"  rule: {replication['rule']}")
    print()
    print("  This verdict is about ONE COMPARISON, IN ONE SIMULATION, AT ONE GAIN. It is not a")
    print("  statement about the MCNS circuit and not a statement about a fly.")
    print()
    print("=" * _WIDTH)
    print("PHASE 4C  5. WHAT THIS IS: MEASURED / SIMULATED / ASSUMED / UNRESOLVED")
    print("=" * _WIDTH)
    for heading in ("MEASURED", "SIMULATED", "ASSUMED", "UNRESOLVED"):
        print()
        print(f"  {heading}")
        for item in claims[heading]:
            for line in _wrap(f"- {item}", _WIDTH - 4):
                print(f"    {line}")
    print()
    print("  CANNOT BE ESTABLISHED")
    for line in _wrap(claims["CANNOT_BE_ESTABLISHED"], _WIDTH - 4):
        print(f"    {line}")
    print()

    return {
        "phase": PREDECLARED_PHASE,
        "kind": "pre-declared replication of the Phase 4B observation, not a new model",
        "replication_target": (
            "the Phase 4B observation that MCNS_RECURRENT scored above SHUFFLED_CONTROL on the "
            "T4+T5 subtype_lateral readout at synapse_scale 0.25 with 8 trials"
        ),
        "predeclaration": predeclared_block(),
        "predeclared_synapse_scale": PREDECLARED_SYNAPSE_SCALE,
        "predeclared_trials": PREDECLARED_TRIALS,
        "predeclared_permutations": PREDECLARED_PERMUTATIONS,
        "permutations_run": int(args.permutations),
        "population_identity_across_conditions": population_identity,
        "same_population_and_edge_subset": bool(
            all(entry["identical"] for entry in population_identity.values())
        ),
        "liveness": liveness_by_circuit,
        "informativeness": informativeness,
        "feature_census": {
            name: extras.get(name, {}).get("features", {}) for name in CIRCUITS
        },
        "determinism": {
            "method": (
                "sha256 over the primary-readout feature arrays, ordered by (stimulus, trial) and "
                "rounded to 12 decimals. Two runs of the declared configuration must produce the "
                "same digest."
            ),
            "fingerprint_by_circuit": {
                name: extras.get(name, {}).get("fingerprint") for name in CIRCUITS
            },
        },
        "primary_comparison": {
            "quantity": f"{PRIMARY_CIRCUIT} accuracy - {CONTROL_CIRCUIT} accuracy",
            "readout": PRIMARY_VARIANT,
            "group": PREDECLARED_GROUP,
            "contrast": PREDECLARED_CONTRAST,
            "recurrent_accuracy": recurrent.get("mean_test_accuracy"),
            "shuffled_accuracy": shuffled.get("mean_test_accuracy"),
            "feedforward_accuracy": primary.get("MCNS_FEEDFORWARD", {})
            .get(PREDECLARED_GROUP, {}).get("mean_test_accuracy"),
            "recurrent_separation_ratio": extras.get(PRIMARY_CIRCUIT, {}).get("separation_ratio"),
            "shuffled_separation_ratio": extras.get(CONTROL_CIRCUIT, {}).get("separation_ratio"),
            "recurrent_minus_shuffled_margin": margin.get("mean_margin"),
            "margin_statistics": margin,
            "effect_supported_by_uncertainty": bool(margin.get("effect_supported", False)),
            "no_winner_declared": True,
        },
        "replication_verdict": replication,
        "power": _power_statement(margin),
        "separation_ratio_verdicts": separation_verdicts,
        "relay_controls": {
            group: {
                circuit_name: primary.get(circuit_name, {}).get(group, {})
                for circuit_name in CIRCUITS
            }
            for group in (INPUT_INSTANT,) + RELAY_CONTROLS + (MAGNITUDE_GROUP,)
        },
        "null_families": {
            "decoder_permutation_null": {
                "family": 1,
                "statistic": "leave-one-trial-out cross-validated accuracy",
                "null_of": "the trial-to-condition mapping",
                "answers": "can a linear decoder beat relabelled trials on accuracy?",
                "n_permutations": int(args.permutations),
                "null_groups": list(args.null_groups),
                "note": (
                    "Saturates when the two classes form tight, well-separated clusters, because "
                    "permuting labels destroys the mapping but not the geometry, so a linear "
                    "model still recovers the true grouping."
                ),
            },
            "separation_ratio_permutation_null": {
                "family": 2,
                "statistic": "between/within centroid-distance ratio (euclidean)",
                "null_of": "the trial-to-condition mapping",
                "answers": "is the observed distance ratio larger than relabelled trials give?",
                "n_permutations": int(args.permutations),
                "groups": list(PREDECLARED_GROUP for _ in range(1)),
                "note": (
                    "Continuous, so it does not saturate the way an accuracy can. It answers a "
                    "different question from family 1, and the two can disagree; when they do, "
                    "neither is privileged and neither is 'the' null."
                ),
            },
            "families_are_distinct": True,
            "neither_is_privileged": True,
        },
        "magnitude_only_versus_structural_readout": magnitude_verdicts,
        "magnitude_only_note": (
            "A one-feature total-activity control compared against the 32-column "
            "subtype x lateral readout. If the control matches the structural readout, the "
            "representation is not resolving structure at this operating point, and any "
            "accuracy it produces restates drive level. Neither side carries a permutation "
            "null, so this is a point-estimate comparison and not a significance statement."
        ),
        "relay_control_purpose": (
            "INPUT_SPATIAL and INPUT_TEMPORAL establish how much direction information exists "
            "before the MCNS circuit acts on the input. They are computed from the drive and are "
            "identical across the three circuit conditions by construction. Downstream decoding "
            "above the input relay is not evidence of MCNS computation unless the "
            "circuit-versus-shuffled comparison supports it."
        ),
        "statistics": statistics,
        "n_trials": PREDECLARED_TRIALS,
        "n_folds": PREDECLARED_TRIALS,
        "n_permutations": int(args.permutations),
        "chance_level": 0.5,
        "acceptance_gate": {
            "status": gate_by_scale[scale_key]["status"],
            "per_condition": gate_by_scale[scale_key]["per_condition"],
            "gate_feature_identical_across_conditions": identity["identical"],
            "gate_feature_max_spread": identity["max_spread"],
        },
        "claims": claims,
        "no_post_hoc_tuning": {
            "statement": (
                "If the pre-declared 0.25 condition is silent, saturated or otherwise "
                "uninformative, that is the finding. The gain is NOT changed and the run is NOT "
                "repeated. The population budget, the feature bins, the stimulus and the trial "
                "protocol are NOT changed either."
            ),
            "decl": list(predeclared_block()["not_done"]),
        },
    }


def _replication_verdict(
    margin: dict[str, Any],
    both_informative: bool,
    separation_verdicts: dict[str, Any],
    n_trials: int,
) -> dict[str, Any]:
    """The one thing a replication run has to say, decided by a stated rule.

    Three outcomes, not two. A replication that can only say "yes" or "no"
    invites the reader to assume a negative result means the circuit does
    nothing, so the third outcome, *uninformative*, exists and is stated as
    plainly as the other two: if the pre-declared gain left a condition silent
    or saturated, the experiment did not test the question and says so instead of
    reporting the number it happened to produce.

    The secondary separation-ratio verdicts are reported alongside but do not
    decide the verdict. They can say the readout is not separating the two
    directions at all, which is a statement about the readout and the gain rather
    than about the wiring, and conflating the two would be the exact error this
    experiment exists to avoid.
    """
    rule = (
        "replicated if the paired-t and paired sign-flip tests on the recurrent-minus-shuffled "
        "margin both reject at alpha=0.05; uninformative if either condition is silent or "
        "saturated at the pre-declared gain; otherwise not reproduced. The separation-ratio "
        "verdicts are reported but do not decide it, because a ratio at its own null is a "
        "statement about the readout at this gain, not about the wiring."
    )
    secondary = {
        circuit_name: entry.get("interpretation")
        for circuit_name, entry in separation_verdicts.items()
        if entry.get("status") == "ok"
    }
    if not both_informative:
        outcome = "uninformative"
        verdict = (
            "UNDECIDABLE. At least one of the two compared conditions is silent or saturated at "
            f"the pre-declared synapse_scale, with {n_trials} trials, so this run did not test the "
            "comparison. The gain is NOT changed and the run is NOT repeated; that is the finding."
        )
    elif margin.get("status") != "ok":
        outcome = "undecidable"
        verdict = (
            "UNDECIDABLE. The recurrent-minus-shuffled margin could not be computed "
            f"({margin.get('status')}: {margin.get('reason', '')})."
        )
    elif bool(margin.get("effect_supported")):
        outcome = "replicated"
        verdict = (
            f"REPRODUCED. The recurrent-minus-shuffled margin is {margin['mean_margin']:+.4f} "
            f"with a paired 95% interval of [{margin['confidence_interval'][0]:+.4f}, "
            f"{margin['confidence_interval'][1]:+.4f}] that excludes zero, and both the paired t "
            f"test (p={margin['paired_t_p_two_sided']:.4f}) and the paired sign-flip permutation "
            f"null (p={margin['margin_permutation_null']['p_two_sided']:.4f}, "
            f"{margin['margin_permutation_null']['n_permutations']} permutations) reject at "
            f"alpha=0.05, on {n_trials} trials. This is a reproducible property OF THIS SIMULATION "
            "AT THIS GAIN and nothing more."
        )
    else:
        outcome = "not_reproduced"
        low, high = margin["confidence_interval"]
        verdict = (
            f"NOT REPRODUCED, AND THE PHASE 4B PRELIMINARY RESULT IS RETRACTED. The Phase 4B "
            f"recurrent-minus-shuffled difference does not survive the pre-declared replication. "
            f"At {n_trials} trials the margin is {margin['mean_margin']:+.4f} with a paired 95% "
            f"interval of [{low:+.4f}, {high:+.4f}], which contains zero; the paired t test gives "
            f"p={margin['paired_t_p_two_sided']:.4f} and the paired sign-flip permutation null "
            f"({margin['margin_permutation_null']['n_permutations']} permutations) gives "
            f"p={margin['margin_permutation_null']['p_two_sided']:.4f}. The 8-trial Phase 4B margin "
            "of +0.079 is therefore WITHDRAWN: it was a small-sample fluctuation at this gain, not "
            "a property of the measured wiring, and it must not be cited, quoted or reused. This is "
            "a statement about one contrast, in one simulation, at one gain. It is NOT a statement "
            "that the MCNS circuit computes nothing, and it is not a direction-selectivity result. "
            "No gain, budget, bin, stimulus or protocol change is made in response, and none will "
            "be made to obtain a positive result."
        )
    return {
        "outcome": outcome,
        "verdict": verdict,
        "rule": rule,
        "n_trials": n_trials,
        "both_conditions_informative": bool(both_informative),
        "separation_ratio_verdicts": secondary,
        "scope": (
            "one comparison, in one simulation, at synapse_scale 0.25, on a bounded subset of the "
            "MCNS candidate circuit. Not a statement about the MCNS circuit, not a statement about "
            "a fly, and not a direction-selectivity result."
        ),
    }


def _power_statement(margin: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """What this design could and could not have seen. Part of the result.

    A null from a design that cannot resolve small effects bounds the difference;
    it does not establish that the difference is zero. Reporting a p-value without
    this invites exactly the over-reading the Phase 4C audit found, so the
    detectable effect size is computed and stored next to the verdict.
    """
    if margin.get("status") != "ok":
        return {"status": "unavailable", "reason": margin.get("status")}
    from scipy.stats import t as student_t

    n = int(margin["n_folds"])
    sd = float(margin["sd_margin"])
    critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
    power_crit = float(student_t.ppf(0.80, n - 1))
    low, high = margin["confidence_interval"]
    return {
        "status": "ok",
        "n_folds": n,
        "sd_of_paired_differences": sd,
        "ci_half_width": float(0.5 * (high - low)),
        "bounds_true_margin_at": float(alpha),
        "minimum_detectable_margin_at_80pct_power": float(
            sd * np.sqrt((critical + power_crit) ** 2 / n)
        ),
        "trials_for_80pct_power_at_0p10": int(np.ceil(((critical + power_crit) * sd / 0.10) ** 2)),
        "underpowered_for_small_effects": True,
        "statement": (
            "A null from this design is a BOUND on the recurrent-minus-shuffled difference, not "
            "evidence that the difference is zero, and it is not evidence that the MCNS circuit "
            "computes nothing. Any true wiring effect smaller than the minimum detectable margin "
            "would be invisible at this trial count."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = apply_presets(parser.parse_args(argv))

    data_dir = Path(args.data_dir)
    args.annotations = args.annotations or str(data_dir / ANNOTATION_NAME)
    args.connectivity = args.connectivity or str(data_dir / CONNECTIVITY_NAME)
    for path in (Path(args.annotations), Path(args.connectivity)):
        if not Path(path).is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
    if args.trials < 3:
        print("error: --trials must be at least 3 for leave-one-trial-out", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not args.is_phase4c_estimate:
        output_dir.mkdir(parents=True, exist_ok=True)
    scales = list(args.scales)
    contrasts = list(args.contrasts)
    variants = tuple(args.variants)
    conditions_needed = sorted({c for name in contrasts for c in CONTRASTS[name]})
    population_groups = POPULATION_GROUPS + ((MAGNITUDE_GROUP,) if True else ())
    report_groups = ALL_BENCH_GROUPS + ((INPUT_GROUP,) + SUBTYPE_GROUPS if args.all_groups else ())

    total_stages = 3 + len(scales) * len(CIRCUITS) + 1
    if args.is_phase4c:
        # One extras stage per circuit condition, plus the Phase 4C report.
        total_stages += len(scales) * len(CIRCUITS) + 1
    if args.is_phase4c_estimate:
        # Two timing points, each with its own second-permutation-count point.
        total_stages += 4
    progress = Progress(total=total_stages, enabled=not args.quiet)

    print("=" * _WIDTH)
    print(f"MCNS motion representation benchmark  [{args.mode} mode]")
    print("=" * _WIDTH)
    if args.is_phase4c:
        print("  PHASE 4C: PRE-DECLARED REPLICATION of the Phase 4B observation.")
        print(f"    question           : is MCNS_RECURRENT - SHUFFLED_CONTROL reproducible at")
        print(f"                        {PREDECLARED_TRIALS} trials instead of 8, same conditions?")
        print(f"    synapse_scale      : {PREDECLARED_SYNAPSE_SCALE}  PRE-DECLARED, no sweep, "
              f"not tunable after the fact")
        print(f"    trials per condition: {PREDECLARED_TRIALS} (from Phase 4B's 8)")
        print(f"    permutations       : {PREDECLARED_PERMUTATIONS} (from Phase 4B quick's 100)")
        print(f"    readout            : subtype_lateral, {LATREADOUT_BINS} lateral bins "
              f"({len(SUBTYPE_GROUPS) * LATREADOUT_BINS} T4+T5 columns)")
        print(f"    everything else    : unchanged from Phase 4B")
        print("  A silent or saturated result at this gain is REPORTED, not rerun at another")
        print("  gain. No gain, budget, bin, stimulus or protocol change follows this run.")
    else:
        print("  QUESTION: does measured MCNS recurrent wiring transform spatiotemporal")
        print("            visual input differently from a feedforward version of the same")
        print("            wiring and from a degree-preserving randomisation of it?")
    print("  NOT the question: does this model reproduce direction selectivity? It cannot.")
    print("            The connectivity table has no synaptic sign, so every coupling is")
    print("            non-negative and the network can only sum, never subtract.")
    print()

    # ---------------------------------------------------------------- stage 1
    progress.stage("Load MCNS annotations, candidate population, extraction cache")
    load_started = time.perf_counter()
    annotations = load_annotations(args.annotations)
    populations = build_populations(annotations, MOTION_CELL_TYPES, require_superclass=True)
    extraction = _load_extraction(args, populations)
    full_ff = MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_FEEDFORWARD)
    full_rc = MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT)
    subset = bounded_population(
        populations, extraction.pre_index, extraction.post_index,
        synapse_count=extraction.synapse_count, bodies_per_type=int(args.max_bodies_per_type),
    )
    circuits: dict[str, MCNSCircuit] = {
        "MCNS_FEEDFORWARD": restrict_circuit(full_ff, subset.positions),
        "MCNS_RECURRENT": restrict_circuit(full_rc, subset.positions),
    }
    if not np.array_equal(circuits["MCNS_FEEDFORWARD"].body_ids, circuits["MCNS_RECURRENT"].body_ids):
        raise RuntimeError("subset selection is not identical across circuit modes")
    meta = candidate_metadata(args.annotations, circuits["MCNS_RECURRENT"].body_ids)
    for circuit in circuits.values():
        circuit.with_soma_locations(meta)
    shuffle_started = time.perf_counter()
    shuffled = shuffled_control(circuits["MCNS_RECURRENT"], seed=int(args.seed))
    progress.record("shuffle_seconds", time.perf_counter() - shuffle_started)
    shuffled.soma_location = circuits["MCNS_RECURRENT"].soma_location.copy()
    circuits["SHUFFLED_CONTROL"] = shuffled
    progress.record("graph_seconds", time.perf_counter() - load_started)
    print(f"  {subset.headline()}")
    if subset.is_subset:
        print(f"  {SUBSET_DISCLAIMER}")
        print(f"  {'type':<6}{'budget':>7}{'seeds':>7}{'closure':>9}{'top-up':>8}{'driven':>9}{'L':>6}{'R':>6}")
        for name in sorted(subset.selected_per_type()):
            balance = subset.hemisphere_balance.get(name, {})
            print(
                f"  {name:<6}{args.max_bodies_per_type:>7}{subset.seeds_per_type.get(name, 0):>7}"
                f"{subset.closure_per_type.get(name, 0):>9}{subset.topped_up_per_type.get(name, 0):>8}"
                f"{subset.driven_per_type.get(name, 0):>9}{balance.get('L', 0):>6}{balance.get('R', 0):>6}"
            )
        for channel, entry in subset.chain_connectivity.items():
            if not channel.startswith("_"):
                print(f"  chain {channel:<4} {entry['pairs_present']}/{entry['pairs_expected']} "
                      f"documented pairs carry measured edges inside the subset")
    else:
        print("  FULL POPULATION: every annotated candidate body is simulated.")

    print()
    print("Circuit conditions (one bounded neuron set, identical in all three)")
    print("-" * _WIDTH)
    for name, circuit in circuits.items():
        print(f"  {name:<18} neurons={circuit.num_neurons:,} edges={circuit.num_edges:,} "
              f"synapses={circuit.total_synapse_count:,}")
    # Asserted rather than assumed: the whole comparison is only fair if the
    # three conditions differ in wiring and in nothing else, so the identity of
    # the neuron set is checked here and reported per pair.
    population_identity = _population_identity(circuits)
    print()
    print("  Population identity across conditions (neuron set, order, cell types)")
    for pair, entry in population_identity.items():
        print(f"    {pair:<34} identical={entry['identical']}  "
              f"neurons={entry['neurons']:,}")
    if not all(entry["identical"] for entry in population_identity.values()):
        raise RuntimeError(
            "the three circuit conditions do not share one neuron set, so the "
            "circuit-versus-shuffled comparison is not a fair contrast"
        )
    verification = verify_degree_preservation(circuits["MCNS_RECURRENT"], shuffled)
    print()
    print("  Shuffled-control invariants (one graph, built once, reused for every trial)")
    for key in (
        "neurons_equal", "edges_equal", "in_degree_sequence_equal",
        "out_degree_sequence_equal", "synapse_count_total_equal",
        "synapse_count_distribution_equal", "topology_changed", "self_loops",
    ):
        print(f"    {key:<34} {verification[key]}")

    # ---------------------------------------------------------------- stage 2
    progress.stage("Cache graphs, propagation matrices, stimulus and drive tensors")
    generator = MotionStimulusGenerator(
        circuits["MCNS_RECURRENT"],
        spot_fraction=float(args.spot_fraction), current=float(args.stimulus_current),
        steps_per_crossing=int(args.n_steps), amplitude_perturbation=float(args.amplitude_perturbation),
        input_noise=float(args.input_noise), randomise_start=not args.no_randomise_start,
        seed=int(args.seed),
    )
    context = RunContext(args, generator)
    warm = context.warm(circuits, scales, int(args.trials), conditions_needed)
    progress.record("cache_warm_seconds", warm["seconds"])
    print(f"  {len(circuits) * len(scales)} propagation matrices, "
          f"{warm['cached_tensors']} drive+noise tensors, {warm['cached_drive_tensors']} drive "
          f"tensors ({(warm['cached_bytes'] + warm['cached_drive_bytes']) / 1024 ** 2:.1f} MB) "
          f"in {warm['seconds']:.2f}s")
    print()
    print("Stimulus protocol")
    print("-" * _WIDTH)
    print(f"  conditions presented      : {', '.join(conditions_needed)}")
    print(f"  contrasts decoded         : {', '.join(contrasts)}")
    print(f"  units per channel/step    : {generator.n_units} of {generator.channel_sizes}")
    print(f"  per-neuron current        : {args.stimulus_current}")
    print(f"  amplitude perturbation    : +/-{100 * args.amplitude_perturbation:.0f}% "
          f"(keyed on seed+trial, shared by every condition)")
    print(f"  input noise               : SD {args.input_noise}, common random numbers across conditions")
    print(f"  randomised sweep start    : {not args.no_randomise_start}, "
          f"offsets {generator.start_offset(0)}, {generator.start_offset(1)}, ...")
    print(f"  drive magnitude / run     : {generator.total_energy(int(args.n_steps)):,.0f} "
          f"(identical for every condition by construction)")
    print(f"  trials per condition      : {args.trials}")
    print(f"  readout                   : {args.readout}"
          f"{f' ({args.lateral_bins} lateral bins)' if args.readout == 'subtype_lateral' else ''}")
    print(f"  T4+T5 columns             : {_group_width(args.readout, args.lateral_bins)}")
    print(f"  decoder                   : leave-one-trial-out, {args.decoder_iterations} iterations, "
          f"lr {args.decoder_learning_rate}, l2 {args.decoder_l2}")
    print(f"  permutation null          : {args.permutations} for groups "
          f"{list(args.null_groups) or 'none'} on contrasts {list(args.null_contrasts) or 'none'}")

    control_records = build_control_records(
        context, circuits["MCNS_RECURRENT"], "MCNS_RECURRENT", conditions_needed, int(args.trials)
    )
    identity = input_instant_identity(control_records, conditions_needed)
    print()
    print(f"  {INPUT_INSTANT} within-trial spread across conditions: "
          f"{identity['max_spread']:.3e} over {identity['trials_compared']} trials "
          f"(exactly equal = {identity['identical']})")

    # The estimate writes to its own directory, so it must not create the real
    # run's output directory as a side effect. The real directory is created
    # below, after the estimate has returned.
    if args.is_phase4c_estimate:
        args.output_dir = PHASE4C_ESTIMATE_DIR
        output_dir = Path(args.output_dir)
        return _phase4c_estimate(
            args, circuits, context, generator, control_records, conditions_needed,
            population_groups, variants, report_groups, contrasts, scales,
            progress, output_dir, warm,
        )

    # ---------------------------------------------------------------- stages 3+
    results: dict[str, Any] = {}
    activity: dict[str, Any] = {}
    condition_activity: dict[str, Any] = {}
    phase4c_extras: dict[str, Any] = {}
    for scale in scales:
        results[str(scale)] = {variant: {} for variant in variants}
        for circuit_name in CIRCUITS:
            progress.stage(f"{circuit_name}  (synapse_scale={scale:g})")
            circuit = circuits[circuit_name]
            per_condition: dict[str, dict[str, dict[str, list[PopulationRecord]]]] = {}
            for condition in conditions_needed:
                records, descriptor = run_trials(
                    circuit, context, circuit_name, condition, population_groups, variants,
                    scale, progress,
                )
                per_condition[condition] = records
                for group, entry in descriptor.items():
                    bucket = activity.setdefault(str(scale), {}).setdefault(circuit_name, {})
                    target = bucket.setdefault(
                        group,
                        {"neurons": 0, "trials": 0, "n_steps": int(args.n_steps),
                         "spikes_per_trial": []},
                    )
                    target["neurons"] = int(entry["neurons"])
                    target["trials"] += int(entry["trials"])
                    target["spikes_per_trial"].extend(entry["spikes_per_trial"])
                    per_cond = condition_activity.setdefault(str(scale), {}).setdefault(
                        circuit_name, {}
                    ).setdefault(condition, {})
                    per_cond[group] = dict(entry)
            for circuit_bucket in activity[str(scale)].values():
                for group, entry in circuit_bucket.items():
                    entry.update(
                        liveness(
                            entry["spikes_per_trial"],
                            n_neurons=int(entry["neurons"]),
                            n_steps=int(entry["n_steps"]),
                        )
                    )

            for variant in variants:
                for contrast in contrasts:
                    left, right = CONTRASTS[contrast]
                    for group in report_groups:
                        if group in CONTROL_GROUPS:
                            pool = (
                                list(control_records[left][group])
                                + list(control_records[right][group])
                            )
                        else:
                            pool = (
                                list(per_condition[left][variant][group])
                                + list(per_condition[right][variant][group])
                            )
                        with_null = (
                            int(args.permutations) > 0
                            and group in args.null_groups
                            and contrast in args.null_contrasts
                        )
                        decoded = decode_contrast(
                            {group: pool}, (group,), left, right, args, with_null
                        )[group]
                        results[str(scale)][variant].setdefault(contrast, {}).setdefault(
                            circuit_name, {}
                        )[group] = decoded

            if args.is_phase4c:
                progress.stage(
                    f"Phase 4C extras for {circuit_name}: separation ratio and feature census"
                )
                phase4c_extras[circuit_name] = _phase4c_condition_extras(
                    per_condition, results, str(scale), PRIMARY_VARIANT,
                    PREDECLARED_CONTRAST, circuit_name, args, progress,
                )

            gate_cell = (
                results[str(scale)][PRIMARY_VARIANT].get(PRIMARY_CONTRAST, {})
                .get(circuit_name, {}).get(INPUT_INSTANT, {})
            )
            probe = activity[str(scale)][circuit_name].get("T4+T5", {})
            print(f"    {circuit_name:<18} gate={_cell(gate_cell)}   "
                  f"T4+T5 mean spikes/trial={float(probe.get('mean_spikes_per_trial', float('nan'))):.1f}")

    # ---------------------------------------------------------------- last stage
    progress.stage("Acceptance gate and report")
    gate_by_scale: dict[str, Any] = {}
    for scale in scales:
        payload = {
            circuit_name: results[str(scale)][PRIMARY_VARIANT]
            .get(PRIMARY_CONTRAST, {})
            .get(circuit_name, {})
            .get(INPUT_INSTANT, {})
            for circuit_name in CIRCUITS
        }
        gate_by_scale[str(scale)] = acceptance_gate(payload)
    gate_statuses = {k: g["status"] for k, g in gate_by_scale.items()}
    gate_passed = all(g["passed"] and g["status"] != "undecidable" for g in gate_by_scale.values())

    print()
    print("=" * _WIDTH)
    print("ACCEPTANCE GATE")
    print("=" * _WIDTH)
    for scale in scales:
        gate = gate_by_scale[str(scale)]
        status = {"pass": "PASS", "pass_with_undecidable": "PASS*", "fail": "FAIL",
                  "undecidable": "UNDECIDABLE"}[gate["status"]]
        print(f"  synapse_scale={scale:<5g} {status}"
              f"{'  (undecidable conditions: ' + str(len(gate['undecidable_conditions'])) + ')' if gate['undecidable_conditions'] else ''}")
        for circuit_name in CIRCUITS:
            entry = gate["per_condition"][circuit_name]
            print(f"    {circuit_name:<18} observed={_fmt(entry['observed_mean_test_accuracy'])} "
                  f"null p95={_fmt(entry['null_p95'])} z={_fmt(entry['z_score'])} "
                  f"-> {entry['verdict']}")
    print()
    print(f"  gate features identical across conditions within a trial: "
          f"{identity['identical']} (max spread {identity['max_spread']:.3e} over "
          f"{identity['trials_compared']} trials)")
    print()
    for line in _wrap(gate_by_scale[str(scales[0])]["instruction"], _WIDTH - 2):
        print(f"  {line}")
    print()

    def table(title: str, groups: tuple[str, ...], notes: tuple[str, ...]) -> None:
        print(title)
        print("-" * _WIDTH)
        for scale in scales:
            first = results[str(scale)][PRIMARY_VARIANT][contrasts[0]][CIRCUITS[0]].get(groups[0], {})
            print(f"  synapse_scale = {scale:g}   (chance 0.500, mean +- sd over "
                  f"{first.get('n_folds', 0)} leave-one-trial-out folds)")
            print(f"    {'group':<18}" + "".join(f"{c:>26}" for c in CIRCUITS))
            for group in groups:
                line = f"    {group:<18}"
                for circuit_name in CIRCUITS:
                    cell = results[str(scale)][PRIMARY_VARIANT][contrasts[0]][circuit_name].get(group, {})
                    line += f"{_cell(cell):>26}"
                print(line)
            print()
        print("  Cells are mean +- sd of the per-fold accuracy. A trailing * marks a score")
        print("  above the 95th percentile of its own label-permutation null.")
        for note in notes:
            for line in _wrap(note, _WIDTH - 2):
                print(f"  {line}")
        print()

    table(
        "CONTROLS",
        CONTROL_GROUPS + (MAGNITUDE_GROUP,),
        (
            "INPUT_INSTANT is the acceptance gate: it must not separate. INPUT_SPATIAL and",
            "INPUT_TEMPORAL are the relay bounds, so a circuit that passed its input through",
            "unchanged would score no better than these. MAGNITUDE_ONLY separates on drive",
            "level alone, so a group matching it is reporting activity, not pattern.",
        ),
    )
    for variant in variants:
        readout, normalise = VARIANTS[variant]
        marker = "  [PRIMARY]" if variant == PRIMARY_VARIANT else "  [secondary]"
        table(
            f"POPULATION READOUT  variant={variant}  "
            f"({readout}, unit_normalise={normalise}){marker}",
            POPULATION_GROUPS,
            (
                "Read against the relay bounds above and against SHUFFLED_CONTROL, never alone.",
                "A T4 or T5 score that SHUFFLED_CONTROL also achieves is not a property of the",
                "measured wiring.",
            ),
        )

    print("T4/T5 LIVENESS  (is the readout population firing at all?)")
    print("-" * _WIDTH)
    print(f"  {'scale':>7}{'condition':<20}" + "".join(f"{g:>12}" for g in POPULATION_GROUPS))
    for scale in scales:
        for circuit_name in CIRCUITS:
            line = f"  {scale:>7g}{circuit_name:<20}"
            for group in POPULATION_GROUPS:
                entry = activity[str(scale)][circuit_name].get(group, {})
                line += f"{float(entry.get('mean_spikes_per_trial', float('nan'))):>12.1f}"
            print(line)
    print()
    print("  Values are mean T4 (or T5) spikes per trial across the whole subpopulation.")
    print("  A decoder score on a silent population is noise; read the liveness first.")
    print()

    phase4c_block: dict[str, Any] | None = None
    if args.is_phase4c:
        progress.stage("Phase 4C report: liveness, primary comparison, relay controls, claims")
        phase4c_block = _phase4c_report(
            args=args,
            scales=scales,
            results=results,
            activity=activity,
            condition_activity=condition_activity,
            extras=phase4c_extras,
            population_identity=population_identity,
            gate_by_scale=gate_by_scale,
            identity=identity,
            generator=generator,
            verification=verification,
            subset=subset,
            circuits=circuits,
            progress=progress,
        )

    simulation_seconds = sum(v for k, v in progress.timings.items() if k.startswith("simulation_seconds::"))
    graph_seconds = progress.timings.get("graph_seconds", 0.0)
    total = progress.total_seconds()

    summary: dict[str, Any] = {
        "experiment": "MCNS motion representation benchmark",
        "mode": args.mode,
        "primary_question": (
            "Does measured MCNS recurrent wiring transform spatiotemporal visual input "
            "differently from a feedforward version of the same wiring, and from a "
            "degree-preserving randomisation of it?"
        ),
        "direction_selectivity_claim": (
            "NONE, AND NOT TESTABLE HERE. The MCNS connectivity table carries body_pre, body_post "
            "and weight and no synaptic sign, so every coupling in this model is non-negative and "
            "the network can only sum. A Reichardt-type detector requires ON minus OFF "
            "subtraction and cannot be built from this data. No number in this report may be "
            "presented as biological direction selectivity, as reproduction of T4/T5 directional "
            "tuning, or as Reichardt computation."
        ),
        "claim_ladder": [
            "1. ANATOMICAL CONNECTIVITY (measured): the extract contains edges "
            "L1->Mi1->T4a-d and L2->Tm1/Tm2->T5a-d with integer synapse counts.",
            "2. SIMULATED SPIKE PROPAGATION (simulated): a sparse LIF model over that graph "
            "propagates drive from L1/L2 to Mi1/Tm1/Tm2 and, at sufficient coupling gain, to T4/T5.",
            "3. REPRESENTATION OF A SYNTHETIC MOTION TRAJECTORY (simulated, with an assumed "
            "stimulus and readout): T4/T5 activity may differ between two energy-matched sweeps "
            "more than its own trial-to-trial spread, and more than a degree-preserving "
            "randomisation of the same graph does.",
            "4. BIOLOGICAL DIRECTION SELECTIVITY (not claimed, not testable with this data): "
            "reaching level 4 would require signed connectivity and recorded responses to a "
            "controlled stimulus sweep. Neither exists here.",
        ],
        "measured": [
            "MCNS body ids, exact cell-type labels, superclass, somaSide, soma coordinates",
            "directed connectivity between candidate bodies",
            "integer biological synapse counts, unmodified",
        ],
        "simulated": [
            "all membrane potentials, spikes, firing rates",
            "all population representations, distances and decoder accuracies",
        ],
        "assumed": [
            "synapse_scale and the coupling law mapping a synapse count to a driving current",
            "generic LIF parameters, not measured for these neurons",
            "the synthetic stimulus protocol, including the position-map fallback",
            "the linear decoder, its optimiser settings and its cross-validation folds",
            "the SHUFFLED_CONTROL null construction",
            "the bounded neuron subset, when the run is in subset mode",
            "the readout resolution (8 subtypes x 4 lateral bins)",
        ],
        "stimulus": {**generator.describe(), "disclaimer": STIMULUS_DISCLAIMER,
                     "conditions_presented": conditions_needed},
        "population": subset.describe(),
        "circuits": {
            name: {
                "neurons": circuit.num_neurons, "edges": circuit.num_edges,
                "biological_synapse_count": circuit.total_synapse_count,
                "is_measured": name != "SHUFFLED_CONTROL",
                "storage": "scipy CSR (sparse); no dense adjacency is ever constructed",
            }
            for name, circuit in circuits.items()
        },
        "shuffled_control": {"disclaimer": SHUFFLED_DISCLAIMER, "verification": verification,
                             "built_once_per_run": True},
        "acceptance_gate": {
            "passed_all_scales": bool(gate_passed),
            "status_per_scale": gate_statuses,
            "gate_feature_identical_across_conditions": identity["identical"],
            "gate_feature_max_spread": identity["max_spread"],
            "per_scale": gate_by_scale,
            "disclaimer": (
                "If the gate fails, the injected current already distinguishes the conditions and "
                "no downstream number may be attributed to MCNS processing."
            ),
        },
        "config": {
            "mode": args.mode, "trials_per_condition": int(args.trials),
            "conditions_presented": conditions_needed, "contrasts": contrasts,
            "readout": str(args.readout), "lateral_bins": int(args.lateral_bins),
            "variants": {v: {"readout": VARIANTS[v][0], "unit_normalise": VARIANTS[v][1],
                             "is_primary": v == PRIMARY_VARIANT} for v in variants},
            "n_steps": int(args.n_steps), "dt": float(args.dt),
            "bin_steps": int(args.bin_steps), "skip_bins": int(args.skip_bins),
            "synapse_scales": scales, "coupling": args.coupling, "seed": int(args.seed),
            "max_bodies_per_type": int(args.max_bodies_per_type),
            "decoder_iterations": int(args.decoder_iterations),
            "decoder_learning_rate": float(args.decoder_learning_rate),
            "decoder_l2": float(args.decoder_l2),
            "permutations": int(args.permutations),
            "null_groups": list(args.null_groups), "null_contrasts": list(args.null_contrasts),
            "neuron_params": vars(NeuronParams()),
            "groups": list(report_groups),
            "identical_across_conditions": [
                "selected neuron set", "LIF parameters", "timestep", "duration", "synapse_scale",
                "input energy", "noise realisation (common random numbers)",
                "sweep start offset", "refractory period", "threshold", "reset", "seed",
                "decoder settings and cross-validation folds",
            ],
        },
        "timings": {
            "total_wall_clock_seconds": round(total, 3),
            "graph_seconds": round(graph_seconds, 3),
            "cache_warm_seconds": round(progress.timings.get("cache_warm_seconds", 0.0), 3),
            "simulation_seconds": round(simulation_seconds, 3),
            "n_simulations": len(scales) * len(CIRCUITS) * len(conditions_needed) * int(args.trials),
        },
        "memory": {
            "circulation_csr_bytes": int(sum(
                c.csr.data.nbytes + c.csr.indices.nbytes + c.csr.indptr.nbytes
                for c in circuits.values())),
            "stimulus_cache_bytes": int(warm["cached_bytes"] + warm["cached_drive_bytes"]),
            "peak_spike_log_bytes": int(int(args.n_steps) * int(circuits["MCNS_RECURRENT"].num_neurons)),
        },
        "activity": activity,
        "population_identity_across_conditions": population_identity,
        "decoder_disclaimer": DECODER_DISCLAIMER,
    }
    if phase4c_block is not None:
        # The Phase 4C report is a top-level key, not a variant of the Phase 4B
        # one, so a reader holding only benchmark_summary.json still gets the
        # pre-declaration, the liveness verdicts, the margin with its
        # uncertainty, and the MEASURED / SIMULATED / ASSUMED / UNRESOLVED split.
        summary["phase4c"] = phase4c_block
        summary["primary_question"] = phase4c_block["predeclaration"]["configuration"][
            "primary_question"
        ]
        summary["unresolved"] = phase4c_block["claims"]["UNRESOLVED"]
        summary["assumed"] = summary["assumed"] + phase4c_block["claims"]["ASSUMED"]
        summary["direction_selectivity_claim"] = (
            phase4c_block["claims"]["CANNOT_BE_ESTABLISHED"]
        )

    written: dict[str, str] = {}
    for name, payload in (
        ("benchmark_summary.json", summary),
        ("decoder_results.json", results),
        ("activity.json", activity),
    ):
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written[name] = str(path)
    if phase4c_block is not None:
        path = output_dir / "phase4c_report.json"
        path.write_text(
            json.dumps(
                {
                    "phase4c": phase4c_block,
                    "activity": condition_activity,
                    "acceptance_gate": gate_by_scale,
                    "timings": summary["timings"],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        written["phase4c_report.json"] = str(path)

    print("Artifacts")
    print("-" * _WIDTH)
    for key, value in written.items():
        print(f"  {key:<28} {value}")
    print()
    print("Cost")
    print("-" * _WIDTH)
    print(f"  total wall clock       : {total:8.2f} s")
    print(f"  graph construction     : {graph_seconds:8.2f} s")
    print(f"  simulation             : {simulation_seconds:8.2f} s")
    print(f"  simulations run        : {summary['timings']['n_simulations']}")
    print(f"  neurons simulated      : {circuits['MCNS_RECURRENT'].num_neurons:,}")
    print(f"  CSR bytes (3 circuits) : {summary['memory']['circulation_csr_bytes'] / 1024 ** 2:.2f} MB")
    print(f"  stimulus + drive cache : {summary['memory']['stimulus_cache_bytes'] / 1024 ** 2:.2f} MB")
    print()
    print("Language")
    print("-" * _WIDTH)
    for line in _wrap(summary["direction_selectivity_claim"], _WIDTH - 2):
        print(f"  {line}")
    return 0


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "  n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.3f}"


def _group_width(readout: str, lateral_bins: int) -> str:
    subtypes = len(SUBTYPE_GROUPS)
    if readout == "subtype_lateral":
        return f"{subtypes * lateral_bins} (8 subtypes x {lateral_bins} lateral bins)"
    if readout == "cell_type":
        return f"{subtypes} (one per exact subtype)"
    return "one per neuron (overparameterised at this trial count)"


if __name__ == "__main__":
    raise SystemExit(main())
