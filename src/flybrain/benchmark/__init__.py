"""Phase 4B benchmark: does measured MCNS wiring *transform* visual input?

Scope, stated once so it does not have to be restated everywhere:

* The primary question is whether measured MCNS **recurrent** wiring transforms
  spatiotemporal visual input differently from a **feedforward** version of the
  same wiring and from a **degree-preserving randomisation** of it.
* This package does **not** test direction selectivity and cannot. The MCNS
  connectivity table carries ``body_pre``, ``body_post`` and ``weight`` and no
  synaptic sign, so every coupling is non-negative, the network can only sum, and
  a Reichardt-type ON-minus-OFF detector is not constructible from it. See
  ``docs/MCNS_MOTION_BENCHMARK.md`` for the four-level claim ladder.
* Every stimulus is synthetic and exactly energy-matched. Nothing here is a
  measurement of a fly.
* The MCNS-derived graphs are the only measured inputs. The stimulus, the neuron
  model, the readout resolution and the decoder are all modelling choices made by
  this project, and all of them are listed under ``assumed`` in the run summary.
* No T4 or T5 number means anything on its own. It must be read against
  ``INPUT_SPATIAL`` and ``INPUT_TEMPORAL`` (the relay bounds) and against
  ``SHUFFLED_CONTROL`` (the wiring control), and only when
  :func:`~flybrain.benchmark.controls.acceptance_gate` has passed.
"""

from __future__ import annotations

__all__: list[str] = []
