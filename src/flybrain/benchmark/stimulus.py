"""Energy-matched synthetic motion stimuli, and the controls that keep them honest.

Design, and why
---------------
The driving question is **not** "does this model reproduce direction selectivity?".
It is: *does the measured MCNS recurrent wiring treat two spatiotemporal visual
input patterns differently from a feedforward version of the same wiring, and from
a degree-preserving randomisation of it?*

For that question the stimulus must not hand the answer over. Concretely:

* **Direction is a trajectory, never a label.** ``LEFTWARD`` and ``RIGHTWARD``
  differ only in the sign of the sweep velocity, ``x(t) = -1 + 2t`` against
  ``x(t) = 1 - 2t``. The ON/OFF balance is identical to within ~1e-4 of the
  total drive, so no instantaneous channel-balance statistic can read the answer
  off the input. There is a regression test pinning that number.
* **Energy is matched exactly.** At every timestep each channel drives exactly
  ``n_units`` neurons at exactly ``current``, so total drive magnitude
  ``2 * n_units * current * n_steps`` is identical for every condition, in every
  polarity. Only the pattern changes.
* **The start position is randomised per trial.** A sweep that always begins at
  the same edge lets a decoder key on that edge rather than on the direction. The
  offset is drawn from ``(seed, trial)`` only — **not** from the condition — so
  the two directions within a trial still traverse the same part of the field and
  the comparison stays paired.
* **The polarity twin** inverts the contrast sign on the OFF channel while
  leaving the ON channel, the trajectory and the energy untouched. It is a
  *channel-assignment* control, not a discrimination task: the prediction is that
  the OFF-driven group (L2 -> Tm1/Tm2 -> T5) is suppressed and the ON-driven group
  (L1 -> Mi1 -> T4) is relatively spared. Expect it to separate trivially, which
  is why it is reported next to the magnitude control.

Nothing here is photoreceptor activity. It is not a luminance model, not a
contrast model, and not derived from any retina.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from flybrain.brain.mcns_circuit import MCNSCircuit
from flybrain.brain.mcns_simulation import MotionInputEncoder

__all__ = [
    "MOTION_STIMULI",
    "STIMULUS_CONDITIONS",
    "STIMULUS_DISCLAIMER",
    "MotionStimulusGenerator",
    "StimulusCondition",
    "StimulusMotion",
    "StimulusPolarity",
    "build_position_map",
    "condition",
    "lateral_coordinates",
]


class StimulusMotion(str, Enum):
    """Spatiotemporal conditions. These label injected drive patterns only.

    ``STATIC`` is a matched-energy control: the same spot, the same drive, the
    same duration, held at the field centre. It differs from the moving
    conditions in trajectory and in nothing else.
    """

    STATIC = "STATIC"
    LEFTWARD = "LEFTWARD"
    RIGHTWARD = "RIGHTWARD"


class StimulusPolarity(str, Enum):
    """Contrast polarity of the OFF channel relative to the ON channel."""

    NORMAL = "NORMAL"
    REVERSED = "REVERSED"


#: Every motion the benchmark can present.
MOTION_STIMULI: tuple[StimulusMotion, ...] = tuple(StimulusMotion)

STIMULUS_DISCLAIMER = (
    "Synthetic computational stimulus, not biological photoreceptor activity. Every condition is "
    "energy-matched: identical n_units, identical per-neuron current magnitude, identical duration, "
    "therefore identical total drive magnitude. Direction is a trajectory, never an explicit label."
)

_POLARITY_SUFFIX = {StimulusPolarity.NORMAL: "", StimulusPolarity.REVERSED: "_POLARITY_REVERSED"}


@dataclass(frozen=True, order=True)
class StimulusCondition:
    """One complete stimulus specification: what moves, and with what contrast sign."""

    motion: StimulusMotion
    polarity: StimulusPolarity = StimulusPolarity.NORMAL

    @property
    def name(self) -> str:
        """Stable string label, round-trippable through :meth:`from_name`."""
        return f"{self.motion.value}{_POLARITY_SUFFIX[self.polarity]}"

    @property
    def value(self) -> str:
        """Alias for :attr:`name`.

        :func:`~flybrain.brain.mcns_simulation.run_simulation` passes the
        stimulus through the simulator as ``str(getattr(stimulus, "value",
        stimulus))``, so a condition needs a ``value`` to survive that hop.
        """
        return self.name

    @classmethod
    def from_name(cls, value: Any) -> "StimulusCondition":
        """Parse a condition label. Accepts an existing condition or its name."""
        if isinstance(value, cls):
            return value
        text = str(getattr(value, "value", value)).upper()
        for condition in STIMULUS_CONDITIONS:
            if condition.name == text:
                return condition
        raise KeyError(
            f"unknown stimulus condition {text!r}; known: {[c.name for c in STIMULUS_CONDITIONS]}"
        )

    @property
    def off_sign(self) -> float:
        """Sign applied to the OFF channel. Reversed contrast drives it negative."""
        return 1.0 if self.polarity is StimulusPolarity.NORMAL else -1.0

    def is_moving(self) -> bool:
        return self.motion is not StimulusMotion.STATIC

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "motion": self.motion.value,
            "polarity": self.polarity.value,
            "off_channel_sign": self.off_sign,
            "moving": self.is_moving(),
        }


#: Every condition the benchmark can present, in a fixed order.
STIMULUS_CONDITIONS: tuple[StimulusCondition, ...] = tuple(
    StimulusCondition(m, p) for m in MOTION_STIMULI for p in StimulusPolarity
)


def condition(name: Any) -> StimulusCondition:
    """Module-level alias for :meth:`StimulusCondition.from_name`."""
    return StimulusCondition.from_name(name)


# ------------------------------------------------------------------- geometry


def build_position_map(
    circuit: MCNSCircuit,
    cell_type: str,
    lateral_gap: float = 0.35,
    side_spread: float = 0.15,
) -> np.ndarray:
    """Deterministic 2D layout in ``[-1, 1] ** 2`` for one cell type.

    Uses real ``somaLocation`` where the annotation table has it, and fills the
    rest deterministically. That fallback matters: only about 17 percent of L1/L2
    bodies in MaleCNS carry soma coordinates, because photoreceptor somata sit
    outside the annotated neuropil.

    ``x`` carries the lateral axis because MCNS ``somaSide`` maps onto it cleanly
    (left hemisphere x in the 77k-93k nm band, right in 3k-21k nm). ``y`` carries
    the vertical axis. Neurons without coordinates are placed on the correct side
    and spread by rank, so the layout stays deterministic and side-respecting.

    The layout is **not** an exact mirror between hemispheres, because the
    coordinate coverage and the rank fallback differ per side. That asymmetry is
    reported, not hidden, and it is the reason the direction cue has to live in
    the trajectory rather than in a per-hemisphere drive level.
    """
    indices = circuit.indices_of_type(cell_type)
    count = int(indices.size)
    if count == 0:
        return np.zeros((0, 2), dtype=np.float64)

    sides = np.asarray(["" if s is None else str(s) for s in circuit.soma_side[indices]])
    loc = circuit.soma_location[indices]
    has_xy = (loc[:, 0] >= 0) & (loc[:, 1] >= 0)

    x = np.zeros(count, dtype=np.float64)
    y = np.zeros(count, dtype=np.float64)

    def _rank_spread(mask: np.ndarray, low: float, high: float) -> np.ndarray:
        """Deterministic even spread over [low, high] for the masked rows."""
        out = np.zeros(int(mask.sum()), dtype=np.float64)
        if out.size == 0:
            return out
        # Rank by body id so the layout does not depend on array ordering luck.
        order = np.argsort(indices[mask], kind="stable")
        for position, slot in enumerate(order):
            out[slot] = low + (high - low) * (position / max(out.size - 1, 1))
        return out

    for side, sign in (("R", -1.0), ("L", 1.0)):
        side_mask = sides == side
        if not side_mask.any():
            continue
        inner, outer = lateral_gap, 1.0
        if sign > 0:
            lo, hi = inner, outer
        else:
            lo, hi = -outer, -inner

        known = side_mask & has_xy
        if known.any():
            raw = loc[known, 0].astype(np.float64)
            span_lo, span_hi = np.percentile(raw, 1.0), np.percentile(raw, 99.0)
            if span_hi > span_lo:
                scaled = lo + (raw - span_lo) / (span_hi - span_lo) * (hi - lo)
            else:  # pragma: no cover - degenerate coordinate column
                scaled = np.full(raw.shape, 0.5 * (lo + hi))
            x[known] = scaled
        unknown = side_mask & ~has_xy
        if unknown.any():
            # Same side, nudged off the exact side line so the fallback is not a
            # single point, and spread vertically.
            jitter = _rank_spread(unknown, -side_spread, side_spread)
            x[unknown] = (0.5 * (lo + hi)) + jitter
            y[unknown] = _rank_spread(unknown, -0.9, 0.9)

    unplaced = ~(sides == "L") & ~(sides == "R")
    if unplaced.any():
        x[unplaced] = _rank_spread(unplaced, -0.2, 0.2)
        y[unplaced] = _rank_spread(unplaced, -0.9, 0.9)

    if has_xy.any():
        raw_y = loc[has_xy, 1].astype(np.float64)
        lo, hi = np.percentile(raw_y, 1.0), np.percentile(raw_y, 99.0)
        if hi > lo:
            y[has_xy] = -0.9 + (raw_y - lo) / (hi - lo) * 1.8
        else:  # pragma: no cover
            y[has_xy] = 0.0

    return np.clip(np.column_stack([x, y]), -1.0, 1.0)


def lateral_coordinates(circuit: MCNSCircuit, n_bins: int = 4) -> np.ndarray:
    """Lateral bin index per neuron, from the same geometry the stimulus uses.

    The population readout is binned by this, so a readout column answers "which
    part of the field responded" rather than only "how much responded in total".
    A direction cue for a moving stimulus lives in *where*, not in *how much*.

    Edges are **quantiles** of the lateral coordinate, not equal-width cuts. The
    position map concentrates neurons near the two hemisphere lines, so equal-width
    bins over that field come out as roughly 562 / 88 / 102 / 548 neurons on the
    real subset: two fat outer bins and two starved inner ones. Quantile edges give
    four comparable columns, which is what a decoder needs.
    """
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive")
    x = np.zeros(circuit.num_neurons, dtype=np.float64)
    for name in circuit.cell_type_names:
        indices = circuit.indices_of_type(name)
        if indices.size:
            x[indices] = build_position_map(circuit, name)[:, 0]
    if int(n_bins) == 1 or circuit.num_neurons == 0:
        return np.zeros(circuit.num_neurons, dtype=np.int64)
    edges = np.quantile(x, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    edges = np.unique(edges)
    if edges.size == 0:
        return np.zeros(circuit.num_neurons, dtype=np.int64)
    return np.clip(np.digitize(x, edges), 0, int(n_bins) - 1).astype(np.int64)


# ------------------------------------------------------------------ generator


@dataclass
class MotionStimulusGenerator:
    """Energy-matched moving-spot drive for the L1 and L2 input populations.

    Parameters
    ----------
    circuit:
        Provides cell types, ``somaLocation`` and ``somaSide``.
    spot_fraction:
        Fraction of the smaller input population driven at any one timestep. The
        resulting **absolute** count is used for both channels, so L1 and L2
        receive identical energy.
    current:
        Per-neuron current amplitude. Identical everywhere and in both polarities.
    steps_per_crossing:
        How many timesteps a moving stimulus takes to traverse the field. All
        moving conditions share this, so their temporal profiles match.
    amplitude_perturbation:
        Trial-to-trial current jitter, as a fraction of ``current``.
    input_noise:
        Standard deviation of additive Gaussian current, applied to every neuron
        on every timestep. Reported, not hidden.
    randomise_start:
        Draw each trial's sweep start offset from ``(seed, trial)``. The offset
        does **not** depend on the condition, so the two directions within a trial
        stay paired.
    seed:
        Seed for the start offsets, the trial perturbations and the noise.
    on_types, off_types:
        Which cell types carry the ON and OFF drive. L1 and L2 respectively, per
        the standard fly literature.
    """

    circuit: MCNSCircuit
    spot_fraction: float = 0.15
    current: float = 2.0
    steps_per_crossing: int = 100
    amplitude_perturbation: float = 0.0
    input_noise: float = 1.0
    randomise_start: bool = True
    seed: int = 0
    on_types: tuple[str, ...] = ("L1",)
    off_types: tuple[str, ...] = ("L2",)

    def __post_init__(self) -> None:
        self._channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._tensor_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._drive_cache: dict[tuple[str, int, int], np.ndarray] = {}
        sizes: list[int] = []
        for label, types in (("ON", self.on_types), ("OFF", self.off_types)):
            indices = np.concatenate(
                [self.circuit.indices_of_type(t) for t in types if self.circuit.has(t)]
            ).astype(np.int64)
            if indices.size == 0:
                raise ValueError(f"stimulus channel {label} matched no neurons")
            self._channels[label] = (indices, build_position_map(self.circuit, types[0]))
            sizes.append(int(indices.size))
        # One absolute count for both channels: this is what makes the energy
        # guarantee hold across ON and OFF as well as across conditions.
        self.n_units = int(max(1, round(self.spot_fraction * min(sizes))))
        self.channel_sizes = {label: int(indices.size) for label, (indices, _) in self._channels.items()}

    # ------------------------------------------------------------------ drive

    def start_offset(self, trial: int) -> int:
        """Sweep start offset for a trial, in steps.

        A function of ``(seed, trial)`` and deliberately **not** of the condition,
        so LEFTWARD and RIGHTWARD in the same trial traverse the same stretch of
        field and remain a paired comparison.
        """
        period = max(int(self.steps_per_crossing), 1)
        if not self.randomise_start:
            return 0
        return int(np.random.default_rng([int(self.seed), int(trial), 0xA17]).integers(0, period))

    def _centre(
        self, condition: StimulusCondition, step: int, trial: int = 0
    ) -> tuple[float, float]:
        """Spot centre in normalised field coordinates at ``step``.

        The phase runs 0..1 across one crossing, identical for both sweep
        directions, so the two conditions differ only in the sign of the
        velocity and nothing in their temporal envelope.
        """
        if not condition.is_moving():
            return 0.0, 0.0
        period = max(int(self.steps_per_crossing), 1)
        phase = (((int(step) + self.start_offset(trial)) % period) + 0.5) / period
        if condition.motion is StimulusMotion.RIGHTWARD:
            return -1.0 + 2.0 * phase, 0.0
        return 1.0 - 2.0 * phase, 0.0

    def _select(
        self, indices: np.ndarray, positions: np.ndarray, centre: tuple[float, float], n_units: int
    ) -> np.ndarray:
        """The ``n_units`` neurons nearest the centre, chosen deterministically.

        ``lexsort`` on (distance, index) rather than ``argpartition`` so ties
        resolve by index and the selection is reproducible run to run.
        """
        dx = positions[:, 0] - centre[0]
        dy = positions[:, 1] - centre[1]
        distance = dx * dx + dy * dy
        order = np.lexsort((indices, distance))
        return indices[order[:n_units]]

    def encode(
        self,
        condition: StimulusCondition | str,
        step: int,
        trial: int = 0,
        n_steps: int | None = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Full-length current vector for one timestep.

        The returned vector has exactly ``n_units`` non-zero entries in each of
        the two channels, every one of magnitude ``current``. Total drive
        **magnitude** is therefore independent of the condition, and only the
        pattern and the sign of the OFF channel differ.

        ``add_noise=False`` returns the drive alone, which is what makes the
        energy guarantee checkable rather than merely asserted.
        """
        spec = StimulusCondition.from_name(condition)
        steps = int(n_steps or self.steps_per_crossing)
        centre = self._centre(spec, int(step), trial=int(trial))

        amplitude = float(self.current)
        if self.amplitude_perturbation:
            # Keyed on (seed, trial) ONLY, never on the condition. A
            # condition-dependent jitter would make the total drive magnitude
            # differ between conditions and quietly break the energy match, and
            # it would unpair the comparison. Every condition in a trial gets
            # the same amplitude.
            jitter_rng = np.random.default_rng([self.seed, int(trial), 0x5EED])
            amplitude *= 1.0 + float(
                jitter_rng.uniform(-self.amplitude_perturbation, self.amplitude_perturbation)
            )

        vector = np.zeros(self.circuit.num_neurons, dtype=np.float64)
        for label, (indices, positions) in self._channels.items():
            chosen = self._select(indices, positions, centre, self.n_units)
            sign = 1.0 if label == "ON" else spec.off_sign
            vector[chosen] = amplitude * sign
        if add_noise and self.input_noise:
            # Common random numbers: the same noise field in every condition of a
            # given (trial, step). Conditions then differ *only* in their drive
            # pattern, so any difference a decoder finds is stimulus structure
            # rather than an independent noise draw. Without this, two conditions
            # get uncorrelated noise and the difference of their ON/OFF balance is
            # dominated by noise rather than by the stimulus.
            noise_rng = np.random.default_rng([self.seed, int(trial), int(step), 0x0FF5])
            vector += noise_rng.normal(0.0, float(self.input_noise), size=vector.shape)
        return vector

    def tensor(
        self,
        condition: StimulusCondition | str,
        trial: int = 0,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """The whole run as one ``(n_steps, n_neurons)`` array, cached.

        Values are bit-identical to calling :meth:`encode` once per step, because
        the per-step RNG seed is a pure function of ``(seed, trial, step,
        condition)``.
        """
        spec = StimulusCondition.from_name(condition)
        steps = int(n_steps or self.steps_per_crossing)
        key = (spec.name, int(trial), steps)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = np.vstack(
                [self.encode(spec, step, trial=trial, n_steps=steps) for step in range(steps)]
            ).astype(np.float32)
            self._tensor_cache[key] = cached
        return cached

    def drive_tensor(
        self,
        condition: StimulusCondition | str,
        trial: int = 0,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """The noise-free drive as one ``(n_steps, n_neurons)`` array, cached.

        This, not the injected current, is what the stimulus-side controls are
        computed from. The relay bound should be the most favourable assumption
        available to a pure relay, and the additive noise is common across
        conditions by construction, so including it can only blur the bound.
        """
        spec = StimulusCondition.from_name(condition)
        steps = int(n_steps or self.steps_per_crossing)
        key = (spec.name, int(trial), steps)
        cached = self._drive_cache.get(key)
        if cached is None:
            cached = np.vstack(
                [
                    self.encode(spec, step, trial=trial, n_steps=steps, add_noise=False)
                    for step in range(steps)
                ]
            ).astype(np.float32)
            self._drive_cache[key] = cached
        return cached

    def cache_stats(self) -> dict[str, int]:
        return {
            "cached_tensors": len(self._tensor_cache),
            "cached_bytes": int(sum(v.nbytes for v in self._tensor_cache.values())),
            "cached_drive_tensors": len(self._drive_cache),
            "cached_drive_bytes": int(sum(v.nbytes for v in self._drive_cache.values())),
        }

    def clear_cache(self) -> None:
        self._tensor_cache.clear()
        self._drive_cache.clear()

    def drive_energy(
        self,
        condition: StimulusCondition | str,
        n_steps: int,
        trial: int = 0,
    ) -> float:
        """Total drive **magnitude** over a run, noise excluded.

        Identical for every condition by construction, in both polarities. This
        is the number the energy-equality test compares across conditions.
        """
        spec = StimulusCondition.from_name(condition)
        total = 0.0
        for step in range(int(n_steps)):
            vector = self.encode(spec, step, trial=trial, n_steps=int(n_steps), add_noise=False)
            total += float(np.abs(vector).sum())
        return total

    def on_off_share(
        self, condition: StimulusCondition | str, n_steps: int, trial: int = 0
    ) -> float:
        """Fraction of drive magnitude carried by the ON channel, noise excluded.

        This is the statistic that must **not** separate the two sweep
        directions. If it ever does, the input is carrying the answer.
        """
        spec = StimulusCondition.from_name(condition)
        on_total = off_total = 0.0
        for step in range(int(n_steps)):
            vector = self.encode(spec, step, trial=trial, n_steps=int(n_steps), add_noise=False)
            for label, (indices, _) in self._channels.items():
                part = float(np.abs(vector[indices]).sum())
                if label == "ON":
                    on_total += part
                else:
                    off_total += part
        return float(on_total / (on_total + off_total)) if (on_total + off_total) else float("nan")

    def total_energy(self, n_steps: int) -> float:
        """Total drive magnitude over a run. Identical for every condition."""
        return float(2 * self.n_units * self.current * n_steps)

    def as_encoder(self, trial: int = 0) -> MotionInputEncoder:
        """Adapt this generator to the Phase 4A encoder interface.

        The timestep index is taken from a mutable counter, because
        :class:`MotionInputEncoder` is called once per step with no step argument.
        """
        return _StimulusEncoderAdapter(self, trial)

    def describe(self) -> dict[str, Any]:
        period = max(int(self.steps_per_crossing), 1)
        return {
            "kind": "synthetic computational stimulus, not biological photoreceptor activity",
            "is_measured": False,
            "disclaimer": STIMULUS_DISCLAIMER,
            "conditions": [c.describe() for c in STIMULUS_CONDITIONS],
            "n_units_per_channel_per_step": self.n_units,
            "channels": self.channel_sizes,
            "on_types": list(self.on_types),
            "off_types": list(self.off_types),
            "per_neuron_current": self.current,
            "steps_per_crossing": self.steps_per_crossing,
            "amplitude_perturbation": self.amplitude_perturbation,
            "input_noise": self.input_noise,
            "snr_current_over_noise": (
                float(self.current / self.input_noise) if self.input_noise else float("inf")
            ),
            "spot_fraction": self.spot_fraction,
            "randomise_start": self.randomise_start,
            "start_offset_rule": (
                "drawn from (seed, trial) only, never from the condition, so the two sweep "
                "directions within a trial traverse the same stretch of field"
            ),
            "example_start_offsets": [self.start_offset(t) for t in range(4)],
            "seed": self.seed,
            "energy_guarantee": (
                "exactly n_units neurons at magnitude `current` in each of the two channels at "
                "every timestep, so total drive magnitude is identical for every condition"
            ),
            "position_map": "real somaLocation where present, deterministic rank fallback otherwise",
            "position_map_caveat": (
                "the per-side layout is not an exact mirror, so a per-hemisphere drive level "
                "carries a small residual asymmetry; the direction cue is the trajectory"
            ),
        }


class StepCounter:
    """Mutable timestep cursor, so a stateless encoder call can be time-aware."""

    def __init__(self, start: int = 0) -> None:
        self.value = int(start)

    def reset(self, start: int = 0) -> None:
        self.value = int(start)


class _StimulusEncoderAdapter(MotionInputEncoder):
    """Presents :class:`MotionStimulusGenerator` through the Phase 4A interface.

    Subclassing keeps ``run_simulation`` unchanged, which matters: the simulation
    loop should not need to know whether it is being driven by the Phase 4A
    gradient encoder or by this energy-matched generator.
    """

    def __init__(self, generator: MotionStimulusGenerator, trial: int = 0) -> None:
        self._generator = generator
        self._trial = int(trial)
        self.counter = StepCounter()
        super().__init__(seed=generator.seed)

    def _cached(self, spec: StimulusCondition) -> np.ndarray | None:
        key = (spec.name, self._trial, int(self._generator.steps_per_crossing))
        return self._generator._tensor_cache.get(key)

    def encode(self, circuit, stimulus, rng=None):  # type: ignore[override]
        spec = StimulusCondition.from_name(stimulus)
        step = self.counter.value
        cached = self._cached(spec)
        if cached is not None and 0 <= step < cached.shape[0]:
            self.counter.value += 1
            return cached[step].astype(np.float64)
        vector = self._generator.encode(spec, step, trial=self._trial)
        self.counter.value += 1
        return vector

    def describe(self) -> dict[str, Any]:  # type: ignore[override]
        described = self._generator.describe()
        described["trial"] = self._trial
        return described
