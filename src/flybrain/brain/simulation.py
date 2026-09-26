"""The simulation engine: external input + synaptic weights -> activity.

This module is the boundary between the *circuit* and everything else. It has
no knowledge of actions, rewards, gymnasium, or plotting, which keeps the four
separation rules intact:

* data enters through a :class:`~flybrain.brain.graph.BrainGraph`;
* results leave through :class:`StepResult` plain arrays;
* visualisation only ever reads those arrays.

Everything produced here is a **simulation**. A :class:`NeuralSimulation` built
on a synthetic graph is a random network firing; one built on real connectome
topology is still a simulation, because the connectome supplies structure only,
never dynamics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.brain.graph import BrainGraph
from flybrain.brain.neuron import LIFPopulation, NeuronParams

__all__ = ["NeuralSimulation", "SimulationConfig", "StepResult"]


@dataclass(frozen=True)
class SimulationConfig:
    """Timing and scaling parameters for the simulation."""

    dt: float = 0.001
    seed: int = 0
    weight_scale: float = 0.35
    external_input_scale: float = 1.2
    synaptic_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.synaptic_decay <= 0.0 or self.synaptic_decay > 1.0:
            raise ValueError("synaptic_decay must be in (0, 1]")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SimulationConfig":
        if not mapping:
            return cls()
        fields = set(cls.__dataclass_fields__)
        return cls(**{str(k): v for k, v in mapping.items() if str(k) in fields})


@dataclass(frozen=True)
class StepResult:
    """Immutable record of one simulation timestep.

    Every array has shape ``(num_neurons,)`` unless noted, and is indexed by
    ``graph.node_ids``.
    """

    step: int
    time: float
    spikes: np.ndarray
    membrane: np.ndarray
    activity: np.ndarray
    region_activity: dict[str, float]
    region_spikes: dict[str, int]
    total_input: float
    recurrent_input: float
    external_input: float

    @property
    def spike_indices(self) -> np.ndarray:
        """Indices of neurons that fired on this step."""
        return np.flatnonzero(self.spikes)

    @property
    def n_active(self) -> int:
        """Number of neurons that fired on this step."""
        return int(np.count_nonzero(self.spikes))

    def summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "time": self.time,
            "n_active": self.n_active,
            "spike_rate": self.spike_rate,
            "mean_activity": float(self.activity.mean()) if self.activity.size else 0.0,
            "total_input": self.total_input,
        }

    @property
    def spike_rate(self) -> float:
        return self.n_active / float(self.spikes.size) if self.spikes.size else 0.0


class NeuralSimulation:
    """Drive a population of LIF neurons through a :class:`BrainGraph`.

    The interface is deliberately the minimum a later, more realistic model
    needs: it accepts neuron states (owned by :class:`LIFPopulation`), directed
    connections and synaptic weights (from the graph), and external sensory
    input; it produces activity and firing events.

    Example
    -------
    >>> from flybrain.brain.graph import synthetic_brain_graph
    >>> graph = synthetic_brain_graph(num_neurons=16, seed=0)
    >>> sim = NeuralSimulation(graph)
    >>> result = sim.step(np.ones(graph.num_nodes))
    >>> 0 <= result.n_active <= graph.num_nodes
    True

    Notes
    -----
    Weights are held in a dense ``(n, n)`` array. That is fine for the toy
    graphs used now; a real connectome has millions of synapses and will need a
    sparse backend (see the TODO in :meth:`_sparse_fallback`).
    """

    def __init__(
        self,
        graph: BrainGraph,
        config: SimulationConfig | None = None,
        neuron_params: NeuronParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or SimulationConfig()
        self.params = neuron_params or NeuronParams()
        self._seed = self.config.seed if seed is None else seed
        self._rng = np.random.default_rng(self._seed)
        self._population = LIFPopulation(graph.num_nodes, self.params, seed=self._seed)
        self._region_masks = graph.region_matrix()
        self.step_index = 0

    # ------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> None:
        """Clear all state and restart the clock."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._population.reset(seed=seed)
        self.step_index = 0

    @property
    def population(self) -> LIFPopulation:
        return self._population

    @property
    def node_ids(self) -> list[str]:
        return self.graph.node_ids

    @property
    def current_time(self) -> float:
        return self.step_index * self.config.dt

    @property
    def membrane(self) -> np.ndarray:
        return self._population.membrane

    @property
    def weights(self) -> np.ndarray:
        """Effective synaptic weights, i.e. the graph weights times ``weight_scale``."""
        return self.graph.weights * self.config.weight_scale

    # ------------------------------------------------------------------ step

    def step(self, external_input: Sequence[float] | np.ndarray | None = None) -> StepResult:
        """Advance the simulation by one timestep.

        Parameters
        ----------
        external_input:
            Sensory drive per neuron, shape ``(num_neurons,)``. ``None`` means
            zero input, which with no drive produces silence (the caller can then
            inspect that state rather than being surprised by spikes).
        """
        n = self.graph.num_nodes
        external = self._as_vector(external_input, n, name="external_input")

        recurrent = self.weights @ self._population.activity()
        total = external + self.config.synaptic_decay * recurrent

        spikes = self._population.step(total, dt=self.config.dt)
        activity = self._population.activity()

        region_activity, region_spikes = self._aggregate_regions(activity, spikes)
        result = StepResult(
            step=self.step_index,
            time=self.current_time,
            spikes=spikes,
            membrane=self._population.membrane.copy(),
            activity=activity,
            region_activity=region_activity,
            region_spikes=region_spikes,
            total_input=float(total.sum()),
            recurrent_input=float(recurrent.sum()),
            external_input=float(external.sum()),
        )
        self.step_index += 1
        return result

    def run(
        self,
        external_inputs: Iterable[Sequence[float] | np.ndarray] | np.ndarray | None = None,
        n_steps: int = 10,
    ) -> list[StepResult]:
        """Run for ``n_steps`` steps.

        ``external_inputs`` may be a sequence of per-step vectors, a single
        constant vector, or ``None`` (random drive scaled by
        ``external_input_scale``, which is what the smoke test uses).
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        n = self.graph.num_nodes
        results: list[StepResult] = []

        if external_inputs is None:
            for _ in range(n_steps):
                results.append(self.step(self._rng.uniform(0.0, self.config.external_input_scale, size=n)))
        elif isinstance(external_inputs, np.ndarray) and external_inputs.ndim == 1:
            for _ in range(n_steps):
                results.append(self.step(external_inputs))
        else:
            for drive in external_inputs:
                results.append(self.step(drive))
        return results

    def random_input(self) -> np.ndarray:
        """One random sensory drive vector, useful for tests and demos."""
        return self._rng.uniform(0.0, self.config.external_input_scale, size=self.graph.num_nodes)

    # ------------------------------------------------------------ aggregation

    def _aggregate_regions(self, activity: np.ndarray, spikes: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
        region_activity: dict[str, float] = {}
        region_spikes: dict[str, int] = {}
        for region, mask in self._region_masks.items():
            if not mask.any():
                region_activity[region] = 0.0
                region_spikes[region] = 0
                continue
            region_activity[region] = float(activity[mask].mean())
            region_spikes[region] = int(np.count_nonzero(spikes[mask]))
        return region_activity, region_spikes

    def _as_vector(self, value: Sequence[float] | np.ndarray | None, size: int, name: str) -> np.ndarray:
        if value is None:
            return np.zeros(size, dtype=np.float64)
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size != size:
            raise ValueError(f"{name} must have {size} elements, got {vector.size}")
        return vector

    # ------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Provenance, size and region breakdown of the simulated circuit."""
        return {
            "graph": self.graph.summary(),
            "config": {
                "dt": self.config.dt,
                "seed": self._seed,
                "weight_scale": self.config.weight_scale,
                "steps_run": self.step_index,
            },
            "population": self._population.summary(),
            "is_biological_topology": self.graph.is_biological,
            "dynamics": "simulated (LIF approximation, not a biological neuron model)",
        }

    def neuron_detail(self, node_id: str, result: StepResult | None = None) -> dict[str, Any]:
        """Per-neuron readout, the data a dashboard neuron panel would display."""
        index = self.graph.node_index(node_id)
        current = result
        return {
            "node_id": node_id,
            "index": index,
            "region": self.graph.region_of(node_id),
            "membrane": float(self._population.membrane[index]),
            "last_spike_step": int(self._population.last_spike_step[index]),
            "in_degree": self.graph.in_degree(node_id),
            "out_degree": self.graph.out_degree(node_id),
            "fired_recently": bool(current is not None and current.spikes[index]),
        }

    def _sparse_fallback(self) -> None:  # pragma: no cover - future work
        """TODO(connectome-scale): switch to a sparse weight matrix.

        A real connectome has millions of synapses. Dense ``(n, n)`` float32
        storage becomes infeasible well before that, and the recurrent update
        will need a sparse matmul. No behaviour change is intended.
        """
        raise NotImplementedError
