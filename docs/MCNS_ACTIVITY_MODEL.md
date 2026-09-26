# MCNS activity model

What this phase built, and — more importantly — what it does not show.

Run it with:

```bash
python scripts/run_mcns_activity.py --data-dir data/raw
```

## The one-paragraph version

A sparse leaky integrate-and-fire model was run over the real MCNS motion
candidate circuit: 22,451 bodies and 340,310 measured connections, in two
connectivity configurations, under three synthetic stimulus conditions. The
circuit produces sparse, structured, layer-by-layer activity. **That activity is
a property of this model, not a recording of a fly, and it does not demonstrate
direction selectivity.**

## What is measured, what is simulated, what is assumed

This is the most important section in the document. Anything not in the first
column is a modelling decision.

| | |
|---|---|
| **Measured** (from the MaleCNS dataset) | body ids, cell-type labels, superclass, `somaSide`, soma coordinates, directed connectivity, and the integer `biological_synapse_count` per connection |
| **Simulated** (computed here) | membrane potentials, spikes, firing rates, every figure and every number in `spikes.npz` |
| **Assumed** (chosen by this project) | the coupling law and gain, the LIF parameters, the synthetic stimulus currents, and the choice to run a feedforward subset |

### The synapse count is not a strength

`biological_synapse_count` is the integer the dataset provides: how many chemical
synapses connect two bodies. It is stored as `int64` in the circuit and is never
normalised, rescaled, or reinterpreted.

It is **not** synaptic efficacy, not a conductance, and not a strength. The
dataset does not measure any of those.

Turning a count into a driving current is a separate, explicit step:

```
simulation_coupling = f(biological_synapse_count) * synapse_scale
```

with `f` either `identity` (default) or `log1p`. `MCNSCircuit.csr` holds the
counts; `MCNSCircuit.coupling_matrix()` and `propagation_matrix()` hold the
derived coupling. A test asserts that asking for coupling never rewrites the
stored counts.

## The model

Per neuron, per timestep of `dt = 0.001 s`:

```
V <- V_rest + (V - V_rest) * exp(-dt / tau_m) + I * (1 - exp(-dt / tau_m))
if V >= v_threshold and not refractory:  spike;  V <- V_reset;  refractory = 2
I   = I_stimulus + W_propagation @ spikes_recent
```

Defaults: `tau_membrane = 0.02 s`, `v_threshold = 1.0`, `v_reset = 0`,
`refractory_periods = 2`. These are **generic LIF defaults, not measured for
these neurons.** They are inherited from `flybrain.brain.neuron` so that there is
one LIF definition in the project rather than two.

Deliberately **not** modelled: synaptic kinetics, conductances, short-term
plasticity, STDP, calcium, dendritic compartments, photoreceptor dynamics,
electrode noise. A first model should be readable end to end.

## Two connectivity configurations

Both keep the **same 22,451 nodes and the same index mapping**, so results are
directly comparable.

| mode | edges | synapses | meaning |
|---|---|---|---|
| `MODE_RECURRENT` | 340,310 | 1,852,867 | every measured candidate-to-candidate edge |
| `MODE_FEEDFORWARD` | 123,019 | 1,396,645 | only the documented chains |

`MODE_FEEDFORWARD` keeps only `L1 -> Mi1 -> T4a-d` and `L2 -> Tm1/Tm2 -> T5a-d`,
and **removes 217,291 edges (63.9%)**, almost all of it measured feedback.

Neither mode is "more correct". `MODE_RECURRENT` is the measured connectivity.
`MODE_FEEDFORWARD` is a documented subset chosen by this project to make the
signal flow legible. The script prints the removed-edge count on every run.

## The two stimuli are synthetic

`NO_MOTION`, `LEFTWARD_MOTION`, and `RIGHTWARD_MOTION` label **injected drive
patterns**, nothing more.

- L1 is treated as the ON channel and L2 as the OFF channel, following the
  standard fly literature.
- The spatial drive is a left/right gradient across the two hemispheres, chosen
  so two conditions can be compared.
- These are **not** photoreceptor responses, **not** measured luminance, and
  **not** derived from any retinal model.

The stimulus names carry no claim about the preferred direction of any neuron.

## Direction selectivity is not demonstrated

T4a-d and T5a-d are direction-selective in the fly. This model does not
reproduce that tuning.

It does produce this:

| | NO_MOTION | LEFTWARD | RIGHTWARD |
|---|---|---|---|
| T4a-d total | 2,321 | 24,441 | 46,055 |
| T5a-d total | 2,910 | 58,321 | 37,565 |

It is tempting to read the left/right mirror between the T4 and T5 groups as a
directional code. **It is not one.** The encoder biases the ON channel toward
one hemisphere and the OFF channel toward the other, so a T4-favours-one-side /
T5-favours-the-other split is a restatement of the asymmetry that was *put in*.
Demonstrating direction selectivity would need recorded responses to a
controlled stimulus sweep, or a model whose tuning emerges from the connectivity
without the input already containing it. Neither exists here.

## Sensitivity: the result depends on an arbitrary gain

`activity_summary.json` contains a `coupling_scale_sensitivity` block. For
`LEFTWARD_MOTION`, feedforward, 500 steps:

| `synapse_scale` | total spikes | L1 | Mi1 | T4a |
|---|---|---|---|---|
| 0.05 | 69,396 | 34,516 | **0** | **0** |
| 0.25 | 240,492 | 34,516 | 23,009 | 7,954 |
| 1.0 | 607,249 | 34,516 | 34,498 | 55,786 |

Two things worth stating plainly:

1. The input layer is insensitive to the gain, because its drive is set directly
   by the stimulus. Everything downstream is extremely sensitive to it.
2. At `synapse_scale = 0.05` **nothing propagates past the input layer at all.**
   The model has a threshold-like transition between "silent" and "spreading".
   That fragility is a property of the model, not of the fly, and it is the main
   reason no absolute spike number here should be quoted as a result.

The default `0.25` was chosen because it sits in the sparse, propagating regime.
It is a choice, not a calibration.

## One implementation note worth knowing

`MCNSCircuit.csr` is stored `csr[source, target]`, which is what in-degree,
out-degree, and `edge_type_matrix()` want. But `matrix @ spikes` computes
row-by-column, so feeding that matrix a spike vector delivers each spike back to
the source's row instead of to the target.

This fails silently: nothing crashes, the run completes, and the activity is
simply wrong — in this case, nothing ever propagated past L1/L2. Use
`MCNSCircuit.propagation_matrix()`, which is transposed so that rows are targets.
There is a test that checks the direction of a single known edge for exactly this
reason.

## Performance

Measured on CPU, 22,451 neurons, no GPU.

| | |
|---|---|
| extraction (first run only, 151.9M rows scanned) | ~12 s, 2,318 batches |
| simulation | ~3.0 ms/step (~0.6 s for 200 steps) |
| 500 steps, 4 runs | ~6 s |
| peak Python allocation | ~415 MB (RSS is higher) |
| extraction cache | 1.2 MB, reused on later runs |

The dense equivalent of this circuit would be 22,451² × 8 B = **4.03 GB** per
matrix. `to_dense()` refuses any circuit whose dense form would exceed 64 MB, so
it refuses this one. The recurrent update is one sparse mat-vec per timestep and
the whole thing is not close to memory-bound.

## Outputs

Written to `outputs/mcns_activity/` (gitignored):

| file | contents |
|---|---|
| `activity_summary.json` | structure, per-type counts, config, sensitivity, memory |
| `spikes.npz` | `(n_steps, n_neurons)` boolean spike matrix per stimulus, plus body ids and types |
| `metadata.json` | index↔bodyId maps, cell types, superclass, sides, soma coordinates, both mode summaries |
| `extraction_cache.npz` | the measured edge table, so later runs skip the 151.9M-row scan |
| `raster.png` | spike raster, rows grouped by cell type |
| `population_activity.png` | population rate over time, split by hemisphere |
| `cell_type_activity.png` | spikes per exact cell type per stimulus |
| `active_body_ids_sample.json` | firing body ids per timestep, the future glow view's input |

## What a reader should take away

The MCNS candidate circuit is real, the connectivity is real, and a simple sparse
LIF model over it settles into sparse layer-by-layer activity with a network
transient. Everything about *how much* and *which way* is downstream of three
choices this project made: a generic LIF, an arbitrary coupling gain, and a
synthetic input that already contains the asymmetry you might later "discover".
