"""Sparse leaky integrate-and-fire dynamics over an extracted MCNS circuit.

The model is deliberately plain. For each neuron, per timestep::

    V <- V_rest + (V - V_rest) * exp(-dt / tau_m) + (I_ext + I_rec) * dt / tau_m
    if V >= threshold and refractory == 0:  spike;  V <- V_reset;  refractory = n

``I_ext`` comes from a :class:`MotionInputEncoder`. ``I_rec`` is the recurrent
synaptic current, ``coupling @ spikes_recent``. Nothing else is modelled: no
conductances, no synaptic kinetics, no STDP, no calcium, no compartments. A first
model should be readable end to end.

What this is and is not
-----------------------
This is a **computational model built from real MCNS connectivity**. It is not a
reconstruction of the fly brain, and its output is not neural activity recorded
from a fly. Two specific limits are load-bearing:

1. **Direction selectivity is not demonstrated.** The recurrent circuit contains
   the kinds of connections that might support it, but a sparse LIF model with a
   uniform coupling law does not reproduce T4/T5 direction tuning, and nothing
   here should be read as evidence that it does. Whether any distinguishable
   output under different synthetic inputs reflects biology or just reflects the
   injected asymmetry is an open question this phase does not answer.
2. **Coupling is assumed.** :meth:`~flybrain.brain.mcns_circuit.MCNSCircuit.coupling_matrix`
   maps a synapse *count* to a driving current. That mapping is a modelling
   choice, not a measurement, and it is the single largest assumption in the
   model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from flybrain.brain.mcns_circuit import (
    MAX_SIMULABLE_NODES,
    MCNSCircuit,
    MCNSCircuitError,
)
from flybrain.brain.neuron import NeuronParams, membrane_update, spike_mask

__all__ = [
    "MotionInputEncoder",
    "SimulationResult",
    "Stimulus",
    "SyntheticMotionEncoder",
    "run_simulation",
]


class Stimulus(str, Enum):
    """Synthetic motion stimulus conditions.

    These are **labels for injected drive patterns**, not measurements of visual
    input and not claims about the preferred directions of T4a-d or T5a-d.
    """

    NO_MOTION = "NO_MOTION"
    LEFTWARD_MOTION = "LEFTWARD_MOTION"
    RIGHTWARD_MOTION = "RIGHTWARD_MOTION"


def as_labels(values: np.ndarray) -> np.ndarray:
    """Normalise a metadata column to plain strings, mapping missing values to ""."""
    return np.asarray(
        ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values]
    )


# --------------------------------------------------------------- encoder


@dataclass
class MotionInputEncoder:
    """Injects a spatially distributed drive into the L1 and L2 populations.

    The L1 population is treated as the ON channel and L2 as the OFF channel,
    following the standard fly literature. The spatial pattern is a smooth
    gradient across the two hemispheres: one stimulus condition biases drive
    toward the right-hemisphere population, the other toward the left, and
    ``NO_MOTION`` drives both equally and weakly.

    **These currents are synthetic.** They are not photoreceptor responses, not
    measured luminance, and not derived from any retinal model. The mapping from
    stimulus name to drive pattern is a convention chosen here so that two
    conditions can be compared; it carries no claim about the sign or magnitude
    of any real direction selectivity.

    The default ``amplitude`` is set so the input neurons sit comfortably above
    ``v_threshold`` and fire regularly. That is a choice about putting the model
    in a working regime, not a measurement; ``scripts/run_mcns_activity.py
    --sweep-scales`` reports how the output depends on it.
    """

    on_types: tuple[str, ...] = ("L1",)
    off_types: tuple[str, ...] = ("L2",)
    amplitude: float = 2.0
    baseline: float = 0.2
    gradient: float = 0.5
    seed: int = 0

    def _channel(self, circuit: MCNSCircuit, types: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Per-neuron membership mask and signed lateral position for one channel.

        ``mask`` is 1 for neurons of the requested types. ``lateral`` is +0.5 in
        the right hemisphere, -0.5 on the left, and 0 where the side is unknown,
        so a stimulus can shift drive toward one side without ever silencing the
        other channel entirely.
        """
        mask = np.zeros(circuit.num_neurons, dtype=np.float64)
        lateral = np.zeros(circuit.num_neurons, dtype=np.float64)
        available = [t for t in types if circuit.has(t)]
        if not available:
            return mask, lateral
        indices = np.concatenate([circuit.indices_of_type(t) for t in available])
        mask[indices] = 1.0
        sides = as_labels(circuit.soma_side[indices])
        lateral[indices] = np.where(sides == "R", 0.5, np.where(sides == "L", -0.5, 0.0))
        return mask, lateral

    def encode(
        self, circuit: MCNSCircuit, stimulus: Stimulus | str, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Return the injected current vector for one timestep.

        Deterministic: the same circuit and stimulus always give the same vector.
        ``LEFTWARD_MOTION`` biases drive toward the left hemisphere and
        ``RIGHTWARD_MOTION`` toward the right. This sign choice is a convention
        for comparing two conditions; it is not a claim about which hemisphere
        responds to which direction in a real fly.
        """
        condition = stimulus if isinstance(stimulus, Stimulus) else Stimulus(str(stimulus))
        on_mask, on_lateral = self._channel(circuit, self.on_types)
        off_mask, off_lateral = self._channel(circuit, self.off_types)

        bias = {
            Stimulus.NO_MOTION: 0.0,
            Stimulus.LEFTWARD_MOTION: -1.0,
            Stimulus.RIGHTWARD_MOTION: 1.0,
        }[condition]

        # A mild fixed per-neuron offset breaks exact ties within a hemisphere so
        # the drive is spatially distributed rather than a single flat value.
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        jitter = generator.uniform(0.85, 1.15, size=circuit.num_neurons)

        def drive(mask: np.ndarray, lateral: np.ndarray, shift: float) -> np.ndarray:
            # 0.5 is the neutral level; the stimulus shifts it laterally. With
            # gradient=0.5 a full stimulus saturates one side and floors the other.
            level = np.clip(0.5 + shift * self.gradient * lateral * 2.0, 0.0, 1.0)
            return (self.baseline + self.amplitude * level * jitter) * mask

        return drive(on_mask, on_lateral, bias) + drive(off_mask, off_lateral, -bias)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "synthetic stimulus",
            "is_measured": False,
            "note": (
                "deterministic drive injected into the L1 (ON) and L2 (OFF) populations. Not a "
                "photoreceptor or luminance model. The stimulus name labels the drive pattern and "
                "carries no claim about T4/T5 direction selectivity."
            ),
            "on_types": list(self.on_types),
            "off_types": list(self.off_types),
            "amplitude": self.amplitude,
            "baseline": self.baseline,
            "gradient": self.gradient,
            "seed": self.seed,
        }


# ------------------------------------------------------------ simulation


@dataclass
class SimulationResult:
    """Recorded activity. Arrays are indexed by simulation index, not body id."""

    times: np.ndarray
    spikes: np.ndarray            # (n_steps, n_neurons) bool
    membrane: np.ndarray          # (n_steps, n_neurons) float32
    input_current: np.ndarray     # (n_steps, n_neurons) float32
    stimulus: str
    circuit_mode: str
    config: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.times.size)

    @property
    def n_neurons(self) -> int:
        return int(self.spikes.shape[1]) if self.spikes.ndim == 2 else 0

    def total_spikes(self) -> int:
        return int(np.count_nonzero(self.spikes))

    def active_indices(self, step: int) -> np.ndarray:
        """Simulation indices firing at one timestep."""
        if not 0 <= step < self.n_steps:
            raise IndexError(f"step {step} out of range (0..{self.n_steps - 1})")
        return np.flatnonzero(self.spikes[step])


def run_simulation(
    circuit: MCNSCircuit,
    encoder: MotionInputEncoder,
    stimulus: Stimulus | str,
    n_steps: int = 500,
    dt: float = 0.001,
    params: NeuronParams | None = None,
    synapse_scale: float = 0.25,
    coupling: str = "linear",
    trace_steps: int = 0,
) -> SimulationResult:
    """Run ``n_steps`` of sparse LIF dynamics over ``circuit``.

    The recurrent term is a single sparse mat-vec per timestep,
    ``recurrent = propagation_csr @ spikes_recent``. The matrix is never
    densified, and the code path contains no dense allocation proportional to
    ``num_neurons ** 2``.

    ``synapse_scale`` default of 0.25 is the value this project settled on
    because it sits in the sparse, propagating regime. It is a choice, not a
    calibration, and the model's output is very sensitive to it: at 0.05 nothing
    propagates past the input layer at all. See ``docs/MCNS_ACTIVITY_MODEL.md``.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if circuit.num_neurons > MAX_SIMULABLE_NODES:
        raise MCNSCircuitError(
            f"{circuit.num_neurons} neurons exceeds the simulable limit ({MAX_SIMULABLE_NODES})"
        )

    neuron_params = params or NeuronParams()
    # Target-by-source orientation, so `propagation @ spikes` is the drive each
    # neuron receives. See MCNSCircuit.propagation_matrix for why this matters.
    propagation = circuit.propagation_matrix(synapse_scale=synapse_scale, coupling=coupling)
    n = circuit.num_neurons
    trace_steps = max(0, min(int(trace_steps), n_steps))

    membrane = np.zeros(n, dtype=np.float64)
    refractory = np.zeros(n, dtype=np.int64)
    spikes_recent = np.zeros(n, dtype=np.float64)

    times = np.arange(n_steps, dtype=np.float64) * dt
    spike_log = np.zeros((n_steps, n), dtype=bool)
    membrane_log = np.zeros((trace_steps, n), dtype=np.float32) if trace_steps else np.zeros((0, n), np.float32)
    input_log = np.zeros((n_steps, n), dtype=np.float32)

    condition = stimulus if isinstance(stimulus, Stimulus) else Stimulus(str(stimulus))
    generator = np.random.default_rng(getattr(encoder, "seed", 0))

    started = time.perf_counter()
    for step in range(n_steps):
        injected = encoder.encode(circuit, condition, rng=generator)
        # One sparse mat-vec for the recurrent term. Nothing here is dense.
        recurrent = propagation @ spikes_recent
        membrane = membrane_update(
            membrane, injected + recurrent, neuron_params.tau_membrane, dt
        )

        fired = spike_mask(membrane, neuron_params.v_threshold, refractory)
        membrane = np.where(fired, neuron_params.v_reset, membrane)
        if neuron_params.refractory_periods > 0:
            refractory = np.where(fired, neuron_params.refractory_periods, np.maximum(refractory - 1, 0))

        spikes_recent = fired.astype(np.float64)
        spike_log[step] = fired
        input_log[step] = injected
        if trace_steps and step < trace_steps:
            membrane_log[step] = membrane
    elapsed = time.perf_counter() - started

    return SimulationResult(
        times=times,
        spikes=spike_log,
        membrane=membrane_log,
        input_current=input_log,
        stimulus=condition.value,
        circuit_mode=circuit.mode,
        config={
            "dt": dt,
            "n_steps": n_steps,
            "trace_steps": trace_steps,
            "synapse_scale": synapse_scale,
            "coupling": coupling,
            "neuron_params": {
                "tau_membrane": neuron_params.tau_membrane,
                "v_threshold": neuron_params.v_threshold,
                "v_reset": neuron_params.v_reset,
                "v_rest": neuron_params.v_rest,
                "refractory_periods": neuron_params.refractory_periods,
            },
            "stimulus": encoder.describe(),
            "recurrent_update": "single sparse mat-vec: propagation_csr @ spikes_recent (rows are targets)",
        },
        timings={
            "seconds_total": elapsed,
            "ms_per_step": 1000.0 * elapsed / max(n_steps, 1),
        },
    )


def activity_by_group(
    circuit: MCNSCircuit,
    result: SimulationResult,
) -> dict[str, dict[str, Any]]:
    """Spike counts grouped by cell type and by hemisphere.

    Cell types stay exact: T4a, T4b, T4c and T4d are reported separately and are
    never summed into a "T4" row here. Aggregation for display happens in the
    visualisation layer, which is a presentation choice rather than a change to
    the underlying identity.
    """
    counts = result.spikes.astype(np.int64).sum(axis=0)
    duration = float(result.times[-1] - result.times[0]) if result.n_steps > 1 else 0.0

    by_type = {
        name: {
            "neurons": int(circuit.indices_of_type(name).size),
            "spikes": int(counts[circuit.indices_of_type(name)].sum()),
            "rate_hz": float(counts[circuit.indices_of_type(name)].sum() / duration) if duration > 0 else 0.0,
        }
        for name in circuit.cell_type_names
    }

    sides = as_labels(circuit.soma_side)
    by_hemisphere: dict[str, Any] = {}
    for label in ("L", "R", "unlabelled"):
        idx = np.flatnonzero(sides == label) if label != "unlabelled" else np.flatnonzero(sides == "")
        by_hemisphere[label] = {
            "neurons": int(idx.size),
            "spikes": int(counts[idx].sum()) if idx.size else 0,
            "rate_hz": float(counts[idx].sum() / duration) if duration > 0 and idx.size else 0.0,
        }

    return {"by_cell_type": by_type, "by_hemisphere": by_hemisphere, "total_spikes": int(counts.sum())}


def active_body_ids(circuit: MCNSCircuit, result: SimulationResult, step: int) -> list[int]:
    """MCNS body ids firing at ``step``. The future glow view's data source."""
    return [circuit.to_body_id(int(i)) for i in result.active_indices(step)]
