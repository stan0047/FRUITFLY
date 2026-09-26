"""Leaky integrate-and-fire neuron model (a simulation).

Model summary
-------------
Each neuron relaxes toward ``v_rest + I(t)`` with a membrane time constant::

    dV/dt = ( I(t) - (V - V_rest) ) / tau_m

which is integrated exactly over a timestep ``dt`` as
``V <- decay * V + (1 - decay) * (v_rest + I)``. A spike is emitted when ``V``
reaches ``v_threshold``; the neuron is then reset to ``v_reset`` and held there
for ``refractory_periods`` steps.

Input ``I`` is therefore expressed in the same units as the voltage, i.e. a
synaptic drive rather than an absolute current. With ``I`` held constant, a
neuron reaches threshold after roughly ``-tau_m * ln(1 - (v_threshold - V_rest) / I) / dt``
steps.

This is a **generic, non-biological** model. It is used because it is cheap and
well understood, not because it reproduces a real fly neuron. Replacing it with
a Hodgkin-Huxley-style multi-compartment model is a future step and would change
no interface here.

Example
-------
>>> import numpy as np
>>> from flybrain.brain.neuron import LIFPopulation, NeuronParams
>>> pop = LIFPopulation(2, NeuronParams(tau_membrane=0.02, v_threshold=1.0))
>>> pop.reset()
>>> fired = np.zeros(2, dtype=bool)
>>> for _ in range(20):
...     fired = pop.step(np.array([5.0, 0.0]), dt=0.001)
>>> bool(fired[0])
True
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["LIFPopulation", "NeuronParams", "NeuronState", "membrane_update", "spike_mask"]


@dataclass(frozen=True)
class NeuronParams:
    """Parameters of the LIF model. Time is measured in simulation steps."""

    tau_membrane: float = 0.02
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    tau_synapse: float = 0.005
    refractory_periods: int = 2

    def __post_init__(self) -> None:
        if self.tau_membrane <= 0.0:
            raise ValueError("tau_membrane must be positive")
        if self.tau_synapse <= 0.0:
            raise ValueError("tau_synapse must be positive")
        if self.v_reset > self.v_threshold:
            raise ValueError("v_reset must not exceed v_threshold")
        if self.refractory_periods < 0:
            raise ValueError("refractory_periods must be non-negative")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "NeuronParams":
        """Build parameters from a config section, ignoring unknown keys."""
        if not mapping:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{str(k): v for k, v in mapping.items() if str(k) in fields})


@dataclass
class NeuronState:
    """Mutable state of a homogeneous population of LIF neurons."""

    membrane: np.ndarray
    refractory: np.ndarray
    spikes: np.ndarray
    last_spike_step: np.ndarray

    @classmethod
    def zeros(cls, size: int, dtype: Any = np.float64) -> "NeuronState":
        return cls(
            membrane=np.zeros(size, dtype=dtype),
            refractory=np.zeros(size, dtype=np.int64),
            spikes=np.zeros(size, dtype=bool),
            last_spike_step=np.full(size, -1, dtype=np.int64),
        )

    @property
    def size(self) -> int:
        return int(self.membrane.shape[0])

    def reset(self) -> None:
        self.membrane.fill(0.0)
        self.refractory.fill(0)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1)

    def copy(self) -> "NeuronState":
        return NeuronState(
            membrane=self.membrane.copy(),
            refractory=self.refractory.copy(),
            spikes=self.spikes.copy(),
            last_spike_step=self.last_spike_step.copy(),
        )

    def spike_indices(self) -> np.ndarray:
        """Indices of neurons that fired on the most recent step."""
        return np.flatnonzero(self.spikes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "membrane": self.membrane.copy(),
            "spikes": self.spikes.copy(),
            "last_spike_step": self.last_spike_step.copy(),
        }


def membrane_update(membrane: np.ndarray, current: np.ndarray, tau_membrane: float, dt: float) -> np.ndarray:
    """One exact step of the leak integration toward ``v_rest + current``.

    Kept as a free function so the update rule is unit-testable on its own.
    """
    decay = float(np.exp(-dt / max(tau_membrane, 1e-12)))
    return decay * membrane + (1.0 - decay) * current


def spike_mask(membrane: np.ndarray, v_threshold: float, refractory: np.ndarray) -> np.ndarray:
    """Boolean mask of neurons eligible to fire this step."""
    return (membrane >= v_threshold) & (refractory <= 0)


class LIFPopulation:
    """A vectorised population of :class:`NeuronParams` neurons.

    Example
    -------
    >>> import numpy as np
    >>> pop = LIFPopulation(3, NeuronParams(tau_membrane=0.02, v_threshold=1.0))
    >>> pop.reset()
    >>> spikes = pop.step(np.array([5.0, 0.0, 0.0]), dt=0.001)
    >>> bool(spikes[0])
    True
    """

    def __init__(self, size: int, params: NeuronParams | None = None, seed: int = 0) -> None:
        if size <= 0:
            raise ValueError("population size must be positive")
        self.size = int(size)
        self.params = params or NeuronParams()
        self._rng = np.random.default_rng(seed)
        self._state = NeuronState.zeros(self.size)
        self._state.v_threshold = self.params.v_threshold
        self.step_count = 0

    # ------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> None:
        """Reset the population to its resting state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state.reset()
        self.step_count = 0

    @property
    def state(self) -> NeuronState:
        return self._state

    @property
    def membrane(self) -> np.ndarray:
        return self._state.membrane

    @property
    def spikes(self) -> np.ndarray:
        return self._state.spikes

    @property
    def last_spike_step(self) -> np.ndarray:
        return self._state.last_spike_step

    def activity(self) -> np.ndarray:
        """Non-negative activity signal in ``[0, 1]``, for visualisation only."""
        if self.size == 0:
            return self._state.membrane.copy()
        return np.clip(self._state.membrane / max(self.params.v_threshold, 1e-9), 0.0, 1.0)

    # ----------------------------------------------------------------- step

    def step(self, current: np.ndarray, dt: float = 0.001) -> np.ndarray:
        """Advance the population by one step.

        Parameters
        ----------
        current:
            Input current per neuron, shape ``(size,)``.
        dt:
            Timestep, same units as ``tau_membrane``.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(size,)`` flagging neurons that fired.
        """
        drive = np.asarray(current, dtype=np.float64).reshape(self.size)
        if self.step_count == 0:
            self._state.membrane.fill(self.params.v_rest)

        refractory = self._state.refractory
        active = refractory <= 0
        self._state.membrane = np.where(
            active,
            membrane_update(self._state.membrane, drive, self.params.tau_membrane, dt),
            self._state.membrane,
        )

        fired = spike_mask(self._state.membrane, self.params.v_threshold, refractory)
        self._state.membrane = np.where(fired, self.params.v_reset, self._state.membrane)
        if self.params.refractory_periods > 0:
            self._state.refractory = np.where(fired, self.params.refractory_periods, np.maximum(refractory - 1, 0))

        self._state.spikes = fired
        self._state.last_spike_step = np.where(fired, self.step_count, self._state.last_spike_step)
        self.step_count += 1
        return fired

    def step_batch(self, currents: np.ndarray, dt: float = 0.001) -> np.ndarray:
        """Advance over a ``(n_steps, size)`` array of currents.

        Returns a ``(n_steps, size)`` boolean array of spikes.
        """
        batch = np.atleast_2d(np.asarray(currents, dtype=np.float64))
        if batch.shape[1] != self.size:
            raise ValueError(f"expected {self.size} neurons, got {batch.shape[1]}")
        return np.stack([self.step(batch[step], dt=dt) for step in range(batch.shape[0])])

    def random_current(self, scale: float = 1.0) -> np.ndarray:
        """Uniform random input current, useful for smoke tests."""
        return self._rng.uniform(0.0, max(scale, 0.0), size=self.size)

    def summary(self) -> dict[str, Any]:
        """Small statistics block, handy for logs and the dashboard."""
        total_spikes = int(np.count_nonzero(self._state.last_spike_step >= 0))
        return {
            "size": self.size,
            "step_count": self.step_count,
            "neurons_that_fired": total_spikes,
            "mean_membrane": float(np.mean(self._state.membrane)) if self.size else 0.0,
            "max_membrane": float(np.max(self._state.membrane)) if self.size else 0.0,
        }
