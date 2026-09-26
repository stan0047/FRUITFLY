"""Recording and plotting neural activity.

The dashboard will need four things: how many neurons are firing, when each
neuron fired, how active each brain region is, and details for one selected
neuron. :class:`ActivityRecorder` produces exactly those four from a list of
:class:`~flybrain.brain.simulation.StepResult` objects, and the ``save_*``
functions render them as static Matplotlib figures.

Everything plotted here is **simulated activity**. A bright node means a model
neuron crossed threshold in a numerical integration, not that a real fly neuron
fired.

Matplotlib is forced onto the ``Agg`` backend so figures can be written on
headless machines and in Colab without a display.
"""

from __future__ import annotations

import colorsys
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.brain.graph import BrainGraph
from flybrain.brain.simulation import StepResult

__all__ = [
    "ActivityRecorder",
    "activity_color",
    "region_activity_matrix",
    "save_activity_dashboard",
    "save_activity_timeline",
    "save_brain_graph",
    "save_region_activity",
    "summarize_activity",
]

_BACKGROUND = "#0b1020"
_TEXT = "#c8d2f0"
_MUTED = "#8892b0"
_GRID = "#2a3350"
_ACCENT = "#2ec4b6"

# Perceptually ordered ramp for activity: dark -> blue -> teal -> amber -> hot.
_ACTIVITY_COLORS = ("#101426", "#1f4e79", "#2ec4b6", "#ffd166", "#fff6d8")


def _use_agg() -> tuple[Any, Any]:
    """Return ``(matplotlib, pyplot)`` with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def activity_color(activity: float) -> tuple[float, float, float]:
    """Map a value in ``[0, 1]`` to an RGB colour on the activity ramp."""
    clamped = float(np.clip(activity, 0.0, 1.0))
    position = clamped * (len(_ACTIVITY_COLORS) - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(_ACTIVITY_COLORS) - 1)
    blend = position - lower

    def _rgb(hex_color: str) -> np.ndarray:
        value = hex_color.lstrip("#")
        return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0

    blended = _rgb(_ACTIVITY_COLORS[lower]) * (1.0 - blend) + _rgb(_ACTIVITY_COLORS[upper]) * blend
    return float(blended[0]), float(blended[1]), float(blended[2])


@dataclass
class ActivityRecorder:
    """Rolling history of :class:`StepResult` snapshots.

    Records are appended with :meth:`record`; at most ``max_history`` steps are
    kept so a long run cannot exhaust memory. Read methods return copies, so
    downstream plotting cannot corrupt the history.
    """

    node_ids: list[str]
    max_history: int = 500
    results: list[StepResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        self.node_ids = [str(node) for node in self.node_ids]

    # ----------------------------------------------------------------- write

    def record(self, result: StepResult) -> None:
        """Append one timestep, dropping the oldest if the buffer is full."""
        self.results.append(result)
        if len(self.results) > self.max_history:
            del self.results[: len(self.results) - self.max_history]

    def record_all(self, results: Iterable[StepResult]) -> None:
        for result in results:
            self.record(result)

    def clear(self) -> None:
        self.results.clear()

    # ------------------------------------------------------------------ read

    def __len__(self) -> int:
        return len(self.results)

    @property
    def num_neurons(self) -> int:
        return len(self.node_ids)

    @property
    def latest(self) -> StepResult | None:
        return self.results[-1] if self.results else None

    def times(self) -> np.ndarray:
        return np.array([result.time for result in self.results], dtype=np.float64)

    def steps(self) -> np.ndarray:
        return np.array([result.step for result in self.results], dtype=np.int64)

    def spike_matrix(self) -> np.ndarray:
        """``(time, neuron)`` boolean spike matrix."""
        if not self.results:
            return np.zeros((0, self.num_neurons), dtype=bool)
        return np.stack([result.spikes for result in self.results])

    def activity_matrix(self) -> np.ndarray:
        """``(time, neuron)`` activity matrix in ``[0, 1]``."""
        if not self.results:
            return np.zeros((0, self.num_neurons), dtype=np.float32)
        return np.stack([result.activity for result in self.results])

    def active_neuron_count(self) -> np.ndarray:
        """Number of firing neurons per timestep."""
        if not self.results:
            return np.zeros(0, dtype=np.int64)
        return np.array([result.n_active for result in self.results], dtype=np.int64)

    def firing_counts(self) -> np.ndarray:
        """Total spikes per neuron across the recorded history."""
        spikes = self.spike_matrix()
        if spikes.size == 0:
            return np.zeros(self.num_neurons, dtype=np.int64)
        return spikes.sum(axis=0).astype(np.int64)

    def firing_timeline(self, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(times, counts)`` for the last ``window`` steps.

        ``counts`` has shape ``(num_neurons, n_steps)`` of 0/1 values: one row per
        neuron, which is what a firing-timeline panel plots as a raster.
        """
        results = self.results[-window:] if window else self.results
        if not results:
            return np.zeros(0, dtype=np.float64), np.zeros((self.num_neurons, 0), dtype=np.int64)
        times = np.array([result.time for result in results], dtype=np.float64)
        counts = np.stack([result.spikes.astype(np.int64) for result in results], axis=1)
        return times, counts

    def spike_times(self, node_id: str) -> np.ndarray:
        """Times at which ``node_id`` fired, within the recorded history."""
        index = self._index_of(node_id)
        spikes = self.spike_matrix()
        if spikes.size == 0:
            return np.zeros(0, dtype=np.float64)
        return self.times()[spikes[:, index]]

    def most_active_neurons(self, k: int = 5) -> list[tuple[str, int]]:
        """Top-``k`` neurons by spike count in the recorded history."""
        counts = self.firing_counts()
        order = np.argsort(counts)[::-1][: max(k, 0)]
        return [(self.node_ids[i], int(counts[i])) for i in order]

    def neuron_detail(self, node_id: str, graph: BrainGraph | None = None) -> dict[str, Any]:
        """Details for one selected neuron, as a dashboard panel would show."""
        index = self._index_of(node_id)
        counts = self.firing_counts()
        activity = self.activity_matrix()
        latest = self.latest

        detail: dict[str, Any] = {
            "node_id": node_id,
            "index": index,
            "total_spikes": int(counts[index]) if counts.size else 0,
            "mean_activity": float(activity[:, index].mean()) if activity.size else 0.0,
            "peak_activity": float(activity[:, index].max()) if activity.size else 0.0,
            "firing_now": bool(latest.spikes[index]) if latest is not None else False,
            "last_spike_time": None,
        }
        spike_times = self.spike_times(node_id)
        if spike_times.size:
            detail["first_spike_time"] = float(spike_times[0])
            detail["last_spike_time"] = float(spike_times[-1])
        if graph is not None:
            detail["region"] = graph.region_of(node_id)
            detail["in_degree"] = graph.in_degree(node_id)
            detail["out_degree"] = graph.out_degree(node_id)
        return detail

    def summary(self) -> dict[str, Any]:
        return summarize_activity(self)

    def _index_of(self, node_id: str) -> int:
        if node_id not in self.node_ids:
            raise KeyError(f"unknown neuron: {node_id!r}")
        return self.node_ids.index(node_id)


def region_activity_matrix(recorder: ActivityRecorder) -> tuple[list[str], np.ndarray]:
    """Aggregate recorded activity per region.

    Returns ``(region_names, matrix)`` with ``matrix`` of shape
    ``(time, region)``, holding mean activity per region per timestep.
    """
    if len(recorder) == 0:
        return [], np.zeros((0, 0), dtype=np.float32)
    regions = sorted({region for result in recorder.results for region in result.region_activity})
    matrix = np.array(
        [[result.region_activity.get(region, 0.0) for region in regions] for result in recorder.results],
        dtype=np.float32,
    )
    return regions, matrix


def summarize_activity(recorder: ActivityRecorder) -> dict[str, Any]:
    """Headline statistics: firing rate, active count, busiest region."""
    if len(recorder) == 0:
        return {
            "steps_recorded": 0,
            "num_neurons": recorder.num_neurons,
            "mean_active_neurons": 0.0,
            "max_active_neurons": 0,
            "mean_firing_rate": 0.0,
            "neurons_that_fired": 0,
            "busiest_region": None,
            "duration": 0.0,
        }

    regions, region_matrix = region_activity_matrix(recorder)
    active = recorder.active_neuron_count()
    counts = recorder.firing_counts()
    busiest: str | None = None
    if regions and region_matrix.size:
        busiest = regions[int(np.argmax(region_matrix.mean(axis=0)))]

    times = recorder.times()
    return {
        "steps_recorded": len(recorder),
        "num_neurons": recorder.num_neurons,
        "mean_active_neurons": float(active.mean()),
        "max_active_neurons": int(active.max()),
        "mean_firing_rate": float(active.sum() / max(len(recorder) * recorder.num_neurons, 1)),
        "neurons_that_fired": int(np.count_nonzero(counts)),
        "busiest_region": busiest,
        "duration": float(times[-1] - times[0]) if times.size > 1 else 0.0,
    }


def _style_axes(axes: Any) -> None:
    axes.set_facecolor(_BACKGROUND)
    axes.tick_params(colors=_MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(_GRID)


def save_brain_graph(
    graph: BrainGraph,
    recorder: ActivityRecorder | None = None,
    path: str | None = None,
    node_size: float = 60.0,
    figsize: tuple[float, float] = (7.0, 6.0),
    dpi: int = 120,
) -> Any:
    """Draw the circuit, colouring and sizing nodes by simulated activity.

    Nodes that fired on the most recent step are ringed, so "currently firing"
    is readable at a glance. This is a 2D spring layout; the glowing 3D brain is
    a later milestone.
    """
    _, plt = _use_agg()
    import networkx as nx

    networkx_graph = graph.to_networkx()
    figure, axes = plt.subplots(figsize=figsize)
    positions = nx.spring_layout(networkx_graph, seed=0)

    latest = recorder.latest if recorder is not None else None
    activity = latest.activity if latest is not None else np.zeros(graph.num_nodes)

    nx.draw_networkx_edges(networkx_graph, positions, ax=axes, alpha=0.15, edge_color="#7f9cf5", width=0.5)
    nx.draw_networkx_nodes(
        networkx_graph,
        positions,
        ax=axes,
        node_color=[activity_color(value) for value in activity],
        node_size=node_size * (0.4 + 1.6 * np.asarray(activity, dtype=np.float64)),
        edgecolors=_BACKGROUND,
        linewidths=0.6,
    )
    if latest is not None and latest.n_active:
        fired = [graph.node_id(i) for i in latest.spike_indices]
        nx.draw_networkx_nodes(
            networkx_graph,
            positions,
            nodelist=fired,
            ax=axes,
            node_color="none",
            node_size=node_size * 2.4,
            edgecolors="#fff6d8",
            linewidths=1.2,
        )
    if graph.num_nodes <= 40:
        nx.draw_networkx_labels(networkx_graph, positions, ax=axes, font_size=6, font_color=_TEXT)

    axes.set_title(
        f"Brain graph '{graph.name}' — topology source: {graph.source} · activity is simulated",
        fontsize=9,
        color=_TEXT,
    )
    axes.set_axis_off()
    figure.patch.set_facecolor(_BACKGROUND)
    axes.set_facecolor(_BACKGROUND)
    figure.tight_layout()

    if path:
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
        return Path(path)
    return figure


def save_activity_timeline(
    recorder: ActivityRecorder,
    path: str | None = None,
    window: int | None = None,
    figsize: tuple[float, float] = (10.0, 4.5),
    dpi: int = 120,
) -> Any:
    """Raster plot of spikes over time, one row per neuron, plus active count."""
    _, plt = _use_agg()

    times, counts = recorder.firing_timeline(window=window)
    figure, (axes_raster, axes_count) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, height_ratios=[3, 1]
    )

    if counts.size:
        span = float(times[-1] - times[0]) or 1e-9
        axes_raster.imshow(
            counts,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            origin="lower",
            extent=(float(times[0]), float(times[0]) + span, 0, recorder.num_neurons),
        )
    axes_raster.set_ylabel("neuron")
    axes_raster.set_title("Simulated spike raster (neuron firing timeline)", fontsize=10, color=_TEXT)

    if len(recorder):
        axes_count.plot(recorder.times(), recorder.active_neuron_count(), color=_ACCENT, linewidth=1.0)
    axes_count.set_xlabel("time (s, simulated)")
    axes_count.set_ylabel("active")
    axes_count.set_ylim(bottom=0)

    for axes in (axes_raster, axes_count):
        _style_axes(axes)
    figure.patch.set_facecolor(_BACKGROUND)
    figure.tight_layout()

    if path:
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
        return Path(path)
    return figure


def save_region_activity(
    recorder: ActivityRecorder,
    path: str | None = None,
    figsize: tuple[float, float] = (10.0, 3.5),
    dpi: int = 120,
) -> Any:
    """Mean simulated activity per brain region over time."""
    _, plt = _use_agg()

    regions, matrix = region_activity_matrix(recorder)
    figure, axes = plt.subplots(figsize=figsize)
    if regions and matrix.size:
        times = recorder.times()
        for index, region in enumerate(regions):
            hue = index / max(len(regions), 1)
            axes.plot(
                times,
                matrix[:, index],
                label=region,
                color=colorsys.hsv_to_rgb(hue, 0.55, 0.95),
                linewidth=1.4,
            )
        axes.legend(loc="upper right", fontsize=7, frameon=False)
    axes.set_xlabel("time (s, simulated)")
    axes.set_ylabel("mean activity")
    axes.set_title("Simulated activity per brain region", fontsize=10, color=_TEXT)
    axes.set_ylim(bottom=0.0)
    _style_axes(axes)
    figure.patch.set_facecolor(_BACKGROUND)
    figure.tight_layout()

    if path:
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
        return Path(path)
    return figure


def save_activity_dashboard(
    graph: BrainGraph,
    recorder: ActivityRecorder,
    directory: str | Path,
    prefix: str = "activity",
    window: int | None = None,
    dpi: int = 120,
) -> dict[str, str]:
    """Write the static panels plus a JSON summary to ``directory``.

    This is the CPU-only stand-in for the live dashboard: the same panels,
    rendered to disk after a run instead of streamed to a browser.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    written = {
        "graph": str(save_brain_graph(graph, recorder, target / f"{prefix}_brain.png", dpi=dpi)),
        "timeline": str(save_activity_timeline(recorder, target / f"{prefix}_timeline.png", window=window, dpi=dpi)),
        "regions": str(save_region_activity(recorder, target / f"{prefix}_regions.png", dpi=dpi)),
    }

    summary = summarize_activity(recorder)
    summary["provenance"] = {
        "graph_source": graph.source,
        "is_biological_topology": graph.is_biological,
        "activity": "simulated (not a recording from a real fly)",
    }
    summary_path = target / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    written["summary"] = str(summary_path)
    return written
