"""Visualisation primitives for MCNS circuit activity.

Backend only: these functions return Matplotlib figures and data. The eventual
dashboard will consume them, but no UI is built here.

Every figure is titled so it cannot be mistaken for a recording. The banner
:data:`FIGURE_BANNER` is applied to all of them.

Two honesty rules are enforced in code, not just convention:

* the banner states *MCNS connectivity + simulated LIF dynamics*;
* no figure or returned label implies recorded biological neural activity, and
  nothing here claims direction selectivity.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flybrain.brain.mcns_circuit import MCNSCircuit  # noqa: E402
from flybrain.brain.mcns_simulation import SimulationResult  # noqa: E402

__all__ = [
    "FIGURE_BANNER",
    "NOT_BIOLOGICAL_ACTIVITY",
    "active_body_ids_series",
    "plot_cell_type_activity",
    "plot_population_activity",
    "plot_raster",
]

#: Required wording on every figure produced here.
FIGURE_BANNER = "MCNS connectivity + simulated LIF dynamics"

#: Required caveat on every figure produced here.
NOT_BIOLOGICAL_ACTIVITY = (
    "SIMULATED activity from a computational model. Not neural activity recorded from a fly. "
    "Direction selectivity is not demonstrated by this model."
)

_BACKGROUND = "#0b1020"
_TEXT = "#c8d2f0"
_MUTED = "#8892b0"
_GRID = "#2a3350"
_RAMP = ("#101426", "#1f4e79", "#2ec4b6", "#ffd166", "#fff6d8")


def _style(axes: Any) -> None:
    axes.set_facecolor(_BACKGROUND)
    axes.tick_params(colors=_MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(_GRID)


def _footer(figure: Any) -> None:
    figure.text(
        0.5,
        0.012,
        f"{FIGURE_BANNER} — {NOT_BIOLOGICAL_ACTIVITY}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=_MUTED,
    )


def _finish(figure: Any, path: str | Path | None) -> Any:
    figure.patch.set_facecolor(_BACKGROUND)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=130, facecolor=figure.get_facecolor())
        plt.close(figure)
        return target
    return figure


def plot_raster(
    circuit: MCNSCircuit,
    result: SimulationResult,
    path: str | Path | None = None,
    max_neurons: int = 400,
    window: int | None = 250,
) -> Any:
    """Spike raster over time, one row per neuron, ordered by cell type.

    Neurons are sorted by cell type so the raster shows type blocks rather than
    an arbitrary index order. ``T4a``-``T4d`` remain separate rows: they are
    never merged into a single ``T4`` band.
    """
    order: list[int] = []
    for name in circuit.cell_type_names:
        order.extend(int(i) for i in circuit.indices_of_type(name))
    if max_neurons and len(order) > max_neurons:
        step = max(1, len(order) // max_neurons)
        order = order[::step][:max_neurons]
    order_arr = np.asarray(order, dtype=np.int64)

    start = 0 if not window else max(0, result.n_steps - window)
    block = result.spikes[start:, :][:, order_arr] if order_arr.size else np.zeros((0, 0), bool)

    figure, (axes_raster, axes_count) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[3, 1]
    )
    if block.size:
        axes_raster.imshow(
            block,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            origin="lower",
            extent=(float(result.times[start]), float(result.times[-1]) or 1.0, 0, block.shape[1]),
        )
    axes_raster.set_ylabel(f"neuron (of {len(order):,} shown, grouped by cell type)")
    axes_raster.set_title(
        f"{FIGURE_BANNER} — spike raster | stimulus={result.stimulus} | mode={result.circuit_mode}",
        fontsize=9,
        color=_TEXT,
    )

    if result.n_steps:
        axes_count.plot(result.times, result.spikes.sum(axis=1), color="#2ec4b6", linewidth=0.9)
    axes_count.set_xlabel("time (s, simulated)")
    axes_count.set_ylabel("spikes")
    axes_count.set_ylim(bottom=0)

    _style(axes_raster)
    _style(axes_count)
    return _finish(figure, path)


def plot_population_activity(
    circuit: MCNSCircuit,
    result: SimulationResult,
    path: str | Path | None = None,
) -> Any:
    """Population-level spike rate over time, split by hemisphere."""
    from flybrain.brain.mcns_simulation import as_labels

    sides = as_labels(circuit.soma_side)
    figure, axes = plt.subplots(figsize=(11, 4))
    rates = result.spikes.sum(axis=1)
    if result.n_steps:
        axes.plot(result.times, rates, color="#ffd166", linewidth=1.0, label="all candidate neurons")
        for label, colour in (("L", "#2ec4b6"), ("R", "#ef6f6c")):
            idx = np.flatnonzero(sides == label)
            if idx.size:
                axes.plot(
                    result.times,
                    result.spikes[:, idx].sum(axis=1),
                    color=colour,
                    linewidth=0.9,
                    label=f"somaSide={label} (n={idx.size:,})",
                )
    axes.set_xlabel("time (s, simulated)")
    axes.set_ylabel("spikes per timestep")
    axes.set_ylim(bottom=0)
    axes.set_title(
        f"{FIGURE_BANNER} — population activity | stimulus={result.stimulus} | mode={result.circuit_mode}",
        fontsize=9,
        color=_TEXT,
    )
    legend = axes.legend(loc="upper right", fontsize=7, frameon=False)
    for text in legend.get_texts() if legend else []:
        text.set_color(_TEXT)
    _style(axes)
    return _finish(figure, path)


def _has_opposed_stimuli(results: Mapping[str, Any]) -> bool:
    """True when the two opposing motion conditions are both being compared."""
    names = {str(k).upper() for k in results}
    return any("LEFTWARD" in n for n in names) and any("RIGHTWARD" in n for n in names)


def plot_cell_type_activity(
    circuit: MCNSCircuit,
    results: Mapping[str, SimulationResult],
    path: str | Path | None = None,
) -> Any:
    """Spike counts per exact cell type, one panel per synthetic stimulus.

    T4a-d and T5a-d appear as separate bars. They are not summed, and no bar is
    labelled with a bare "T4" or "T5".
    """
    types = list(circuit.cell_type_names)
    figure, axes = plt.subplots(1, 1, figsize=(12, 4.5))
    width = 0.8 / max(len(results), 1)
    positions = np.arange(len(types), dtype=np.float64)

    for offset, (stimulus, result) in enumerate(sorted(results.items())):
        counts = result.spikes.astype(np.int64).sum(axis=0)
        per_type = [int(counts[circuit.indices_of_type(name)].sum()) for name in types]
        axes.bar(positions + offset * width, per_type, width=width, label=stimulus)

    axes.set_xticks(positions + 0.4 - width / 2)
    axes.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    axes.set_ylabel("spikes (simulated)")
    title = f"{FIGURE_BANNER} — spike count by exact cell type\n{NOT_BIOLOGICAL_ACTIVITY}"
    if _has_opposed_stimuli(results):
        title += (
            "\nA T4-favours-one-condition / T5-favours-the-other pattern here mirrors the drive "
            "that was injected.\nIt is a property of the synthetic input, not a computed "
            "direction-selective response."
        )
    axes.set_title(title, fontsize=9, color=_TEXT)
    legend = axes.legend(fontsize=7, frameon=False)
    for text in legend.get_texts() if legend else []:
        text.set_color(_TEXT)
    _style(axes)
    return _finish(figure, path)


def active_body_ids_series(
    circuit: MCNSCircuit,
    result: SimulationResult,
    max_steps: int | None = None,
) -> list[list[int]]:
    """Body ids firing at each timestep, as plain lists.

    This is the data source a future glowing-neuron view would consume: one list
    of MCNS body ids per timestep, ready to be mapped to 3D positions and lit.
    """
    steps = result.n_steps if max_steps is None else min(result.n_steps, max(0, max_steps))
    body_ids = circuit.body_ids
    return [
        [int(b) for b in body_ids[result.spikes[step]]] if result.spikes[step].any() else []
        for step in range(steps)
    ]
