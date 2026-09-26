#!/usr/bin/env python
"""Simulate the extracted MCNS motion candidate circuit and record its activity.

Runs the same synthetic stimuli through two connectivity configurations and
reports whether the circuit produces distinguishable activity patterns.

    NO_MOTION
    LEFTWARD_MOTION
    RIGHTWARD_MOTION

What this measures
------------------
Whether a sparse LIF model built on real MCNS connectivity responds differently
to different injected drive patterns.

What this does NOT measure
--------------------------
**Not direction selectivity.** T4a-d and T5a-d are direction-selective neurons in
the fly. This model does not reproduce that tuning, and a difference between the
LEFTWARD and RIGHTWARD conditions is a consequence of the asymmetry that was
*injected*, not evidence of a computed directional code. The stimulus names
label the drive pattern and nothing more.

Usage
-----
    python scripts/run_mcns_activity.py --data-dir data/raw
    python scripts/run_mcns_activity.py --data-dir data/raw --n-steps 500
    python scripts/run_mcns_activity.py --annotations A.feather --connectivity B.feather
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

from flybrain.brain.mcns_circuit import (  # noqa: E402
    FEEDFORWARD_CHAINS,
    MODE_FEEDFORWARD,
    MODE_RECURRENT,
    MCNSCircuit,
    candidate_metadata,
)
from flybrain.brain.mcns_simulation import (  # noqa: E402
    MotionInputEncoder,
    Stimulus,
    activity_by_group,
    active_body_ids,
    run_simulation,
)
from flybrain.brain.neuron import NeuronParams  # noqa: E402
from flybrain.data.motion_candidate import (  # noqa: E402
    MOTION_CELL_TYPES,
    MOTION_PATHWAYS,
    EdgeExtraction,
    build_populations,
    extract_candidate_edges,
    load_annotations,
)
from flybrain.visualization.mcns_activity import (  # noqa: E402
    FIGURE_BANNER,
    NOT_BIOLOGICAL_ACTIVITY,
    active_body_ids_series,
    plot_cell_type_activity,
    plot_population_activity,
    plot_raster,
)

DEFAULT_DATA_DIR = "data/raw"
ANNOTATION_NAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CONNECTIVITY_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
STIMULI = (Stimulus.NO_MOTION, Stimulus.LEFTWARD_MOTION, Stimulus.RIGHTWARD_MOTION)
_WIDTH = 78


def _peak_bytes() -> int | None:
    """Peak Python allocation for this process.

    ``tracemalloc`` rather than RSS: it is stdlib and cross-platform, and it
    measures the arrays and matrices this script actually allocates. Returns
    ``None`` rather than a guess if it cannot be determined.
    """
    try:
        import tracemalloc

        current, peak = tracemalloc.get_traced_memory()
        return int(peak or current)
    except Exception:  # noqa: BLE001
        return None


def _load_cached_extraction(path: Path, populations, annotations) -> EdgeExtraction | None:
    """Reuse a previously extracted edge table so iteration does not rescan 151M rows."""
    if not path.is_file():
        return None
    try:
        blob = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001
        return None
    if int(blob["n_bodies"]) != int(populations.num_bodies):
        return None
    if not np.array_equal(blob["body_ids"], np.asarray(populations.body_ids, dtype=np.int64)):
        print("  cache body ids do not match the current population; re-extracting")
        return None
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
    )


def _save_cached_extraction(path: Path, extraction: EdgeExtraction, populations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
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


def build_circuits(args: argparse.Namespace) -> tuple[MCNSCircuit, MCNSCircuit, dict, EdgeExtraction]:
    """Extract once (or reuse the cache), then build both configurations."""
    load_start = time.perf_counter()
    annotations = load_annotations(args.annotations)
    populations = build_populations(
        annotations, MOTION_CELL_TYPES, require_superclass=True, body_budget_per_type=args.body_budget
    )

    cache_path = Path(args.cache) if args.cache else Path(args.output_dir) / "extraction_cache.npz"
    extraction = None
    if not args.no_cache:
        extraction = _load_cached_extraction(cache_path, populations, annotations)
        if extraction is not None:
            print(f"reusing cached extraction: {cache_path}")
    if extraction is None:
        extraction = extract_candidate_edges(args.connectivity, populations, direction="both")
        if not args.no_cache:
            _save_cached_extraction(cache_path, extraction, populations)
            print(f"wrote extraction cache: {cache_path}")
    load_seconds = time.perf_counter() - load_start

    feedforward = MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_FEEDFORWARD)
    recurrent = MCNSCircuit.from_extraction(populations, extraction, annotations, mode=MODE_RECURRENT)

    meta = candidate_metadata(args.annotations, feedforward.body_ids)
    feedforward.with_soma_locations(meta)
    recurrent.with_soma_locations(meta)

    info = {
        "load_seconds": round(load_seconds, 2),
        "extraction_edges": int(extraction.num_edges),
        "candidate_neurons": int(populations.num_bodies),
        "rows_scanned": int(extraction.rows_scanned),
        "batches_processed": int(extraction.batches_processed),
        "batches_total": int(extraction.batches_total),
        "from_cache": extraction is not None and not args.no_cache,
    }
    return feedforward, recurrent, info, extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--connectivity", default=None)
    parser.add_argument("--output-dir", default="outputs/mcns_activity")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--trace-steps", type=int, default=120, help="membrane traces to retain")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--synapse-scale", type=float, default=0.25)
    parser.add_argument("--coupling", choices=("linear", "log1p"), default="linear")
    parser.add_argument("--body-budget", type=int, default=0)
    parser.add_argument("--cache", default=None, help="path for the extracted edge table cache")
    parser.add_argument("--no-cache", action="store_true", help="always re-extract from the Feather file")
    parser.add_argument(
        "--sweep-scales",
        default="0.05,0.25,1.0",
        help="comma-separated coupling gains to report a sensitivity table for ('' to skip)",
    )
    parser.add_argument("--benchmark-steps", type=int, default=0, help="extra timing run of this many steps")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    args.annotations = args.annotations or str(data_dir / ANNOTATION_NAME)
    args.connectivity = args.connectivity or str(data_dir / CONNECTIVITY_NAME)

    for path in (Path(args.annotations), Path(args.connectivity)):
        if not Path(path).is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import tracemalloc

    tracemalloc.start()

    print("=" * _WIDTH)
    print("MCNS motion circuit — sparse LIF activity")
    print("=" * _WIDTH)
    print(f"banner : {FIGURE_BANNER}")
    print(f"caveat : {NOT_BIOLOGICAL_ACTIVITY}")
    print()

    feedforward, recurrent, load_info, extraction = build_circuits(args)

    print("Extraction")
    print("-" * _WIDTH)
    print(f"candidate neurons      : {load_info['candidate_neurons']:,}")
    print(f"extraction edges       : {load_info['extraction_edges']:,}")
    print(f"rows scanned           : {load_info['rows_scanned']:,}")
    print(f"record batches         : {load_info['batches_processed']:,} of {load_info['batches_total']:,}")
    print(f"load seconds           : {load_info['load_seconds']}")
    print()
    print("Circuits")
    print("-" * _WIDTH)
    for circuit in (feedforward, recurrent):
        print(
            f"{circuit.mode:<12} neurons={circuit.num_neurons:,}  edges={circuit.num_edges:,}  "
            f"synapses={circuit.total_synapse_count:,}  self-loops={circuit.self_loop_count}"
        )
    removed = recurrent.num_edges - feedforward.num_edges
    print(
        f"\nMODE FEEDFORWARD removed {removed:,} of {recurrent.num_edges:,} edges "
        f"({100.0 * removed / max(recurrent.num_edges, 1):.1f}%)."
    )
    print("  kept chains:")
    for name, chain in FEEDFORWARD_CHAINS.items():
        print(f"    {name:<4} {' -> '.join(chain[0])}  then  {' , '.join(f'{a} -> {b}' for a, b in chain[1:])}")
    print("  MODE_RECURRENT is the measured connectivity; MODE_FEEDFORWARD is a documented")
    print("  subset of it chosen by this project. Neither is 'more correct'.")
    print()

    encoder = MotionInputEncoder()
    params = NeuronParams()

    benchmark: dict[str, float] = {}
    if args.benchmark_steps > 0:
        start = time.perf_counter()
        run_simulation(feedforward, encoder, Stimulus.NO_MOTION, n_steps=args.benchmark_steps, dt=args.dt,
                       params=params, synapse_scale=args.synapse_scale, coupling=args.coupling)
        benchmark[f"{args.benchmark_steps}_steps_seconds"] = round(time.perf_counter() - start, 3)

    summary: dict[str, object] = {
        "experiment": "MCNS motion circuit activity",
        "banner": FIGURE_BANNER,
        "not_biological_activity": NOT_BIOLOGICAL_ACTIVITY,
        "direction_selectivity_claim": (
            "NONE. This model does not reproduce T4/T5 direction selectivity. Any difference "
            "between stimulus conditions reflects the drive pattern that was injected."
        ),
        "measured_vs_assumed": {
            "measured": [
                "MCNS body ids, cell-type labels, superclass, somaSide, soma coordinates",
                "directed connectivity between candidate bodies",
                "integer biological synapse count per connection (unmodified)",
            ],
            "simulated": [
                "membrane potential, spikes, firing rates, and all activity figures",
            ],
            "assumed": [
                "synapse_scale and coupling law mapping a synapse count to a driving current",
                "leaky integrate-and-fire parameters (generic, not measured for these neurons)",
                "synthetic stimulus currents injected into the L1 and L2 populations",
            ],
        },
        "extraction": load_info,
        "circuits": {"feedforward": feedforward.summary(), "recurrent": recurrent.summary()},
        "feedforward_edges_removed": int(removed),
        "config": {
            "n_steps": args.n_steps,
            "dt": args.dt,
            "trace_steps": args.trace_steps,
            "synapse_scale": args.synapse_scale,
            "coupling": args.coupling,
            "body_budget_per_type": args.body_budget,
            "stimulus_encoder": encoder.describe(),
            "neuron_params": vars(params),
        },
        "benchmark": benchmark,
    }

    results: dict[str, object] = {}
    spikes_payload: dict[str, np.ndarray] = {}
    feedforward_results: dict[str, Any] = {}
    for stimulus in STIMULI:
        result = run_simulation(
            feedforward, encoder, stimulus, n_steps=args.n_steps, dt=args.dt, params=params,
            synapse_scale=args.synapse_scale, coupling=args.coupling, trace_steps=args.trace_steps,
        )
        feedforward_results[stimulus.value] = result
        spikes_payload[f"{stimulus.value}__spikes"] = result.spikes
        results[stimulus.value] = {
            "circuit_mode": "feedforward",
            "total_spikes": result.total_spikes(),
            "spikes_per_step_mean": float(result.spikes.sum(axis=1).mean()),
            "ms_per_step": result.timings["ms_per_step"],
            **activity_by_group(feedforward, result),
        }
        print(
            f"{stimulus.value:<17} feedforward  spikes={result.total_spikes():>9,}  "
            f"mean/step={result.spikes.sum(axis=1).mean():8.2f}  {result.timings['ms_per_step']:.2f} ms/step"
        )

    # Recurrent configuration on the first stimulus only: it is a connectivity
    # variant, not a third stimulus, and it is not claimed to be more correct.
    recurrent_result = run_simulation(
        recurrent, encoder, Stimulus.LEFTWARD_MOTION, n_steps=args.n_steps, dt=args.dt, params=params,
        synapse_scale=args.synapse_scale, coupling=args.coupling, trace_steps=args.trace_steps,
    )
    results["LEFTWARD_MOTION_recurrent"] = {
        "circuit_mode": "recurrent",
        "total_spikes": recurrent_result.total_spikes(),
        "spikes_per_step_mean": float(recurrent_result.spikes.sum(axis=1).mean()),
        "ms_per_step": recurrent_result.timings["ms_per_step"],
        "note": "same stimulus as LEFTWARD_MOTION, different connectivity configuration",
        **activity_by_group(recurrent, recurrent_result),
    }
    print(
        f"{'LEFTWARD (recurrent)':<17} recurrent   spikes={recurrent_result.total_spikes():>9,}  "
        f"mean/step={recurrent_result.spikes.sum(axis=1).mean():8.2f}  {recurrent_result.timings['ms_per_step']:.2f} ms/step"
    )

    # Distinguishability, reported without interpretation.
    totals = {s: int(feedforward_results[s].total_spikes()) for s in feedforward_results}
    summary["stimulus_distinguishability"] = {
        "spike_totals_by_stimulus": totals,
        "leftward_minus_no_motion": totals[Stimulus.LEFTWARD_MOTION.value] - totals[Stimulus.NO_MOTION.value],
        "rightward_minus_no_motion": totals[Stimulus.RIGHTWARD_MOTION.value] - totals[Stimulus.NO_MOTION.value],
        "leftward_vs_rightward": totals[Stimulus.LEFTWARD_MOTION.value] - totals[Stimulus.RIGHTWARD_MOTION.value],
        "caveat": (
            "A non-zero difference shows only that the circuit responds differently to the drive "
            "patterns that were injected. It is not evidence of a computed directional code."
        ),
    }

    print()
    print("Cell-type spike counts (feedforward, exact types, never merged)")
    print("-" * _WIDTH)
    types = feedforward.cell_type_names
    header = f"{'type':<8}" + "".join(f"{s.value.split('_')[0][:6]:>12}" for s in STIMULI)
    print(header)
    for name in types:
        row = f"{name:<8}"
        for stimulus in STIMULI:
            counts = feedforward_results[stimulus].spikes.astype(np.int64).sum(axis=0)
            row += f"{int(counts[feedforward.indices_of_type(name)].sum()):>12,}"
        print(row)
    print()
    print("  note: T4a/T4b/T4c/T4d and T5a/T5b/T5c/T5d are reported separately and are")
    print("        never summed into a bare 'T4' or 'T5'.")
    print()

    # Sensitivity to the coupling scale. This parameter is an assumption, so its
    # effect is reported rather than hidden behind one chosen value.
    sweep: dict[str, object] = {}
    if args.sweep_scales:
        print()
        print("Coupling-scale sensitivity (LEFTWARD_MOTION, feedforward)")
        print("-" * _WIDTH)
        print("  the whole result depends on this arbitrary gain, so here is how much")
        sweep_rows = []
        for scale in [float(s) for s in args.sweep_scales.split(",") if s.strip()]:
            probe = run_simulation(
                feedforward, encoder, Stimulus.LEFTWARD_MOTION, n_steps=args.n_steps, dt=args.dt,
                params=params, synapse_scale=scale, coupling=args.coupling, trace_steps=0,
            )
            counts = probe.spikes.astype(np.int64).sum(axis=0)
            per_type = {
                name: int(counts[feedforward.indices_of_type(name)].sum())
                for name in feedforward.cell_type_names
            }
            sweep_rows.append(
                {
                    "synapse_scale": scale,
                    "total_spikes": probe.total_spikes(),
                    "spikes_per_neuron": probe.total_spikes() / max(feedforward.num_neurons, 1),
                    "by_cell_type": per_type,
                }
            )
            shown = " ".join(f"{t}={per_type[t]:,}" for t in ("L1", "Mi1", "T4a", "L2", "Tm1", "T5a"))
            print(f"  scale={scale:<7} total={probe.total_spikes():>9,}  {shown}")
        sweep = {
            "note": (
                "Spike counts at several coupling gains. The gain is a modelling assumption, not a "
                "measurement, so the spread across this range is the honest uncertainty on any "
                "absolute spike number reported here."
            ),
            "stimulus": Stimulus.LEFTWARD_MOTION.value,
            "rows": sweep_rows,
        }

    # ------------------------------------------------------------- artifacts
    summary["results"] = results
    summary["coupling_scale_sensitivity"] = sweep
    summary["memory_peak_bytes"] = _peak_bytes()
    summary["memory_peak_note"] = "tracemalloc peak Python allocation; RSS is higher"
    written: dict[str, str] = {}
    summary_path = output_dir / "activity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written["activity_summary"] = str(summary_path)

    spikes_path = output_dir / "spikes.npz"
    payload = dict(spikes_payload)
    payload["body_ids"] = feedforward.body_ids
    payload["cell_type"] = feedforward.cell_type.astype(str)
    np.savez_compressed(spikes_path, **payload)
    written["spikes"] = str(spikes_path)

    meta = {
        "banner": FIGURE_BANNER,
        "not_biological_activity": NOT_BIOLOGICAL_ACTIVITY,
        "index_to_body_id": {str(i): feedforward.to_body_id(i) for i in range(feedforward.num_neurons)},
        "body_id_to_index": {str(b): feedforward.to_index(b) for b in feedforward.body_ids},
        "cell_type_names": feedforward.cell_type_names,
        "soma_side": feedforward.soma_side.astype(str),
        "superclass": feedforward.superclass.astype(str),
        "has_soma_location": (feedforward.soma_location[:, 0] >= 0).tolist(),
        "soma_location": feedforward.soma_location.tolist(),
        "feedforward_chains": {k: [list(e) for e in v] for k, v in FEEDFORWARD_CHAINS.items()},
        "modes": {"feedforward": feedforward.summary(), "recurrent": recurrent.summary()},
        "pathway_framing": {
            name: {"input": spec["input"], "relay": list(spec["relay"]), "outputs": list(spec["outputs"])}
            for name, spec in MOTION_PATHWAYS.items()
        },
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    written["metadata"] = str(meta_path)

    written["raster"] = str(
        plot_raster(feedforward, feedforward_results[Stimulus.LEFTWARD_MOTION], output_dir / "raster.png")
    )
    written["population_activity"] = str(
        plot_population_activity(feedforward, feedforward_results[Stimulus.LEFTWARD_MOTION],
                                 output_dir / "population_activity.png")
    )
    written["cell_type_activity"] = str(
        plot_cell_type_activity(feedforward, feedforward_results, output_dir / "cell_type_activity.png")
    )

    # The future glow view's data source, for a handful of steps.
    series = active_body_ids_series(feedforward, feedforward_results[Stimulus.LEFTWARD_MOTION], max_steps=25)
    (output_dir / "active_body_ids_sample.json").write_text(
        json.dumps(
            {
                "note": "MCNS body ids firing per timestep; a future 3D glow view consumes this",
                "steps": {str(i): ids for i, ids in enumerate(series)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["active_body_ids_sample"] = str(output_dir / "active_body_ids_sample.json")

    print("Artifacts")
    print("-" * _WIDTH)
    for key, value in written.items():
        print(f"  {key:<24} {value}")
    print()
    print("Reminders")
    print("-" * _WIDTH)
    print("  * simulated activity, not recorded from a fly")
    print("  * no direction selectivity is demonstrated")
    print("  * synapse counts are unmodified; coupling is a separate assumption")
    print("  * MODE_RECURRENT is measured connectivity, not 'more correct'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
