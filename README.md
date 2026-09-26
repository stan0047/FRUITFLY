# FlyBrain — Bio-Inspired Vision and Navigation

A zero-cost, fully open-source research and portfolio project investigating whether
**fruit-fly connectome-inspired neural architectures** can be used for visual navigation.

> **Status: development scaffold.** The repository currently contains architecture,
> interfaces, configuration and tests. It does **not** contain a working fly brain
> simulation, a trained policy, or the real *Drosophila* connectome. Nothing here
> should be read as a claim of biological fidelity.

---

## 1. Project goal

Build an end-to-end system in which a camera-like visual input is processed by a
neural circuit whose *topology is inspired by* the fruit-fly connectome, whose
*activity is produced by a numerical simulation*, and whose output drives a
navigation policy in a simulated environment that is trained with reinforcement
learning.

```
CAMERA / VISUAL INPUT
        ↓
FLY-BRAIN-INSPIRED VISUAL CIRCUIT
        ↓
NEURAL ACTIVITY / REPRESENTATION
        ↓
NAVIGATION POLICY
        ↓
LEFT / FORWARD / RIGHT
        ↓
SIMULATED ENVIRONMENT
        ↓
REWARD
        ↓
TRAINING
```

The final artefact is a live dashboard:

| Left half | Right half |
| --- | --- |
| camera / agent view | glowing 3D fruit-fly brain |
| | active neurons visibly firing / glowing |
| | neuron firing timeline |
| | active neuron count |
| | brain-region activity |
| | selected neuron details |
| navigation controls | reward and simulation statistics |

---

## 2. Scientific motivation

The fly solves navigation with a small, well-characterised circuit. A few
properties make it attractive as an engineering template:

* **Small.** A few hundred thousand neurons, a known set of visual pathways
  (compound eye → optic lobe → central brain → descending neurons).
* **Explicit headings.** The fly maintains an internal direction estimate
  (the central complex is the canonical candidate for a heading "ring attractor"),
  which is exactly the kind of latent state a navigation policy needs.
* **Fast, low-power.** A cheap, local, spiking computation — attractive for
  embodied and edge applications.
* **Known circuits.** Multiple independent connectome reconstructions now exist,
  so a *data-driven* prior over circuit structure is possible instead of a purely
  hand-designed one.

The research question is deliberately narrow and testable:

> **Can a network built from real connectome topology (or a topology abstracted
> from it) produce representations that make visual navigation easier to learn,
> compared to unstructured baselines of similar size?**

Note the phrasing: *topology inspired by / abstracted from* the connectome. The
connectome is **biological data**, not a neural network you can run. It is a
measured graph of neurons and synapses; it contains no weights, no activation
functions, no dynamics, and no learning rules. Supplying those is our modelling
choice, and must always be reported as such.

---

## 3. What is real biological data vs. what is simulation

This distinction is enforced in the code (`Connectome.is_biological`,
`BrainGraph.source`, `src/flybrain/data/connectome.py`) and is the single most
important honesty constraint of the project.

| Layer | Nature | Where |
| --- | --- | --- |
| Connectome topology (neuron identities, which neuron connects to which) | **Real published biological data** — once a real export is loaded | `flybrain.data.connectome` |
| Neuron / synapse attributes as published | **Real biological data** (synapse counts, types, weights *if the source provides them*) | `flybrain.data.connectome` |
| Region labels and neuropil grouping | **Real biological data** when taken from the source; otherwise author-assigned and flagged | `flybrain.data.connectome` |
| The synthetic graph used in tests and the smoke test | **NOT biological.** Random graph, clearly labelled `source="synthetic"` | `flybrain.brain.graph.synthetic_brain_graph` |
| LIF membrane integration, spike times, firing rates | **Simulation.** A standard, generic, non-biological model chosen for convenience | `flybrain.brain.neuron` |
| Region↔neuron mapping for the synthetic graph | **Simulation scaffolding.** Arbitrary round-robin assignment | `flybrain.brain.graph` |
| Observation encoding, action set, reward function | **Simulation.** A toy gridworld invented for this project | `flybrain.navigation.environment` |
| Learned policy, weights, training curves | **Simulation / machine learning.** Nothing to do with the fly | `flybrain.models`, `flybrain.training` |
| 3D brain rendering and glow | **Visualisation of simulated values** | `flybrain.visualization` |

Hard rules for this project:

1. Never present simulated activity as a model of a real neuron.
2. Never hardcode invented numbers under a citation. Unknown → `TODO` + not loaded.
3. Every artefact that mixes real topology with simulated dynamics must state
   both parts.
4. When the real connectome lands, `connectome.source` becomes `"flywire"` (or
   whichever source), `is_biological` becomes `True` for topology, and the README
   status section is updated in the same commit.

---

## 4. Current status

Implemented now:

* Repository layout, packaging (`pyproject.toml`, `requirements.txt`).
* YAML configuration system with relative-path resolution and dotted overrides.
* Connectome *interface* with explicit provenance flags and an unintegrated
  `load_flywire_connectome` stub that raises (no fabricated data).
* `BrainGraph` wrapper over `networkx`, plus a small synthetic graph generator.
* LIF neuron population (vectorised NumPy) and a `NeuralSimulation` that
  consumes external input + synaptic weights and returns activity and spikes.
* Vision preprocessing helpers (crop / resize / grayscale / normalise / stack).
* `FlyBrainNavEnv`: a tiny 2D gridworld with `LEFT` / `FORWARD` / `RIGHT`,
  obstacles, reward, termination.
* `TinyBaselineNet`: a placeholder MLP policy + value head used only to prove the
  data flow and training loop work. **It is not a FlyBrain model.**
* REINFORCE-style `Trainer` with reward baseline and checkpointing.
* `ActivityRecorder` and static Matplotlib plots (timeline, region activity,
  graph activity). The live dashboard is deliberately *not* built yet.
* Tests for imports, graph construction, simulation, environment stepping, and
  the baseline model; plus a runnable `scripts/smoke_test.py`.

Deliberately not implemented yet: real connectome download, biologically detailed
neuron models, optic-flow processing, a learned FlyBrain circuit, PPO, the live
dashboard, and any frontend.

---

## 5. Architecture

```
configs/default.yaml            all tunables; no hardcoded values in code

src/flybrain/
  config.py                     YAML loading, dotted get/set, path resolution
  data/connectome.py            I/O + provenance for biological data ONLY
  brain/graph.py                graph container, synthetic generator
  brain/neuron.py               LIF parameters, state, population
  brain/simulation.py           step function: input + weights -> spikes
  vision/preprocessing.py       frames -> model-ready arrays
  navigation/environment.py     gymnasium gridworld
  models/baseline.py            placeholder MLP policy/value net (torch)
  training/trainer.py           REINFORCE loop, checkpointing
  visualization/activity.py     recording, timelines, static plots
```

The four separation rules:

1. **Data ≠ simulation.** `data/` never imports `brain/`. Biological facts enter
   the system only as an explicit, attributed graph.
2. **Simulation ≠ RL.** `brain/` has no knowledge of rewards, actions or gym.
   `training/` consumes simulation output and nothing else.
3. **Simulation ≠ visualisation.** `visualization/` only reads recorded arrays.
   It cannot step or mutate the simulation.
4. **Config everywhere.** `brain/`, `navigation/`, `models/`, `training/` all take
   their parameters from dataclasses that can be built from the YAML config.

Data flow (current, scaffold level):

```
FlyBrainNavEnv.reset/step
        -> observation (float32 vector)
        -> vision.preprocessing.apply (identity by default)
        -> models.baseline.TinyBaselineNet.forward
        -> action in {LEFT, FORWARD, RIGHT}
NeuralSimulation.step
        -> StepResult(membrane, spikes, firing_rates, region_activity)
        -> visualization.ActivityRecorder.record
```

Note that the simulation and the baseline policy are **not yet wired together**.
That wiring is a roadmap item, not a scaffold feature — doing it now would imply a
biological claim the scaffold cannot support.

---

## 6. Free compute strategy

No paid API, no paid cloud, no credit card, ever.

| Stage | Where | Hardware |
| --- | --- | --- |
| Unit tests, smoke test | local laptop | CPU only, seconds |
| Graph / activity analysis | local laptop | CPU, `networkx` + `numpy` |
| RL training (small) | local laptop | CPU, `torch` (few minutes) |
| RL training (large) | Google Colab free tier | free T4 GPU |
| Connectome parsing | local + Colab | CPU + streaming `pandas` |

* `requirements.txt` pins the CPU wheel of `torch` so nothing tries to download
  CUDA runtimes. On Colab, `pip install torch` picks up the free GPU build.
* Device selection is explicit (`model.baseline.device`, `training.device`) and
  always falls back to CPU with a warning if CUDA is requested but missing.
* The synthetic graph is tiny on purpose: the scaffold must run on a CPU-only
  machine in seconds, and a full connectome is millions of synapses.

Colab notebook workflow lives in `notebooks/README.md`.

---

## 7. Roadmap

1. **Scaffold** — layout, config, interfaces, tests, smoke test. *(this commit)*
2. **Real connectome ingestion** — stream a public export into `data/processed/`
   as a tidy CSV; record source, licence, version, and citation.
3. **Circuit extraction** — pick documented visual and heading pathways; build
   region-level and neuron-level subgraphs; define which weights are measured
   vs. initialised.
4. **Neuromodelled simulation** — graded potentials, synaptic kinetics, spike
   propagation, sparse weights at connectome scale; validate on CPU at toy scale.
5. **Learned FlyBrain circuit** — initialise from connectome topology, learn
   dynamics, compare against `TinyBaselineNet` on equal parameter budgets.
6. **PPO and curriculum** — robust long-horizon training; multiple seeds;
   report mean ± std.
7. **Live dashboard** — camera view left, glowing 3D brain right, neuron firing
   timeline, region activity, selected-neuron panel, controls, statistics.
8. **Publication-grade write-up** — ablations, honest limitations, reproducible
   seeds and configs.

---

## 8. Install and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q            # tests
python scripts/smoke_test.py   # end-to-end scaffold check, writes outputs/figures
```

Or install the package itself in editable mode:

```bash
pip install -e ".[dev]"
```

## 9. Licence

Code: [MIT](LICENSE). Any real connectome data is **not** redistributed here;
see `data/README.md` for the licensing and attribution requirements that apply
once a real dataset is added.
