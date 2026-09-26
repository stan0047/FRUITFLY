# Data directory

**No dataset is stored in this repository.** Connectome data is large,
attribution-heavy and usually licence-conditional, so it is obtained locally and
ignored by git (see `.gitignore`).

```
data/
  raw/        # untouched, as downloaded from the original source
  processed/  # tidy, documented, derived tables
```

Rules:

1. `raw/` is write-once. Never edit a raw file; re-download instead.
2. Every file in `processed/` must be reproducible by a script in `scripts/`
   from a file in `raw/`.
3. Any real dataset must ship a sibling `*.source.json` recording
   `source_name`, `dataset_name`, `dataset_version`, `source_url`, `license`
   and `citation`. The importer writes this for you.

---

## 1. What a connectome is

A **connectome** is a measurement of the wiring of a nervous system: which
neurons are connected to which, how many synapses, and often where those synapses
sit. For *Drosophila* the reference is the adult female central brain
(`FAFB`/`FCNP` volume in FlyWire terms), reconstructed from electron microscopy
and published with per-synapse resolution.

Three things a connectome is **not**, which this project keeps strictly separate:

| Not this | Why |
| --- | --- |
| a neural network | it has no activation function, no dynamics, no learning rule |
| a set of weights | synaptic strength is a physical measurement, not a trained parameter |
| a recorded brain state | it says nothing about what the neurons were *doing* |

A connectome is **data**. It is the single piece of real biology in FlyBrain, and
it constrains what we build; it does not become a model by being loaded.

## 2. What FlyWire provides

FlyWire (flywire.ai) is the current public resource for the *Drosophila*
connectome. Its downloadable products include per-synapse tables, neuron and
cell-type annotations, and neuropil/region assignments.

**FlyBrain does not download any of it, and this repository contains no
hardcoded download URL.** There are several exports and releases, they change,
and a URL baked into a library goes stale without anyone noticing. The division
of responsibility is deliberate:

* **You** download the export from the official FlyWire / Codex resources, and
  read the licence and citation terms that apply.
* **FlyBrain** inspects the local file you point it at, tells you what is in it,
  and refuses to guess when the schema is unclear.

If a URL breaks, that is a documentation fix, not a code fix.

## 3. Why start with a downloadable public dataset

* **Zero cost.** No credentials, no licence negotiation, no paid API. It fits the
  project's free-only constraint.
* **Reproducible.** A specific, citable release can be named in a report. A
  scraped live endpoint cannot.
* **Small enough to be honest about.** A subset is a legitimate experimental
  unit, provided every statistic says it is a subset.
* **Citable.** A result derived from a public dataset can name that dataset.

## 4. How to supply the dataset

```bash
# 1. Download the export yourself into data/raw/ (git-ignored).
#    data/raw/neuroprops.csv

# 2. Look at it. This reads only the first 10k rows, whatever the file size.
python scripts/inspect_connectome.py --input data/raw/neuroprops.csv

# 3. Import it, declaring the provenance the source actually states.
python scripts/import_connectome.py \
    --input data/raw/neuroprops.csv \
    --output data/processed/ \
    --source-name flywire \
    --dataset-name "<the exact export name>" \
    --dataset-version "<the release or date the file states>" \
    --source-url "<the page you downloaded from>" \
    --license "<the licence the source states>" \
    --citation "<the citation the source states>" \
    --max-nodes 5000 --max-edges 50000 --on-limit subsample
```

Anything you leave out stays `None` and is listed by the report under
`Unverified`. FlyBrain does not fill these in for you.

## 5. Memory considerations

Measured on a synthetic 2,000,000-edge / 120,000-neuron file (development check,
not a real dataset):

| Approach | Time | Peak Python allocation | Graph storage |
| --- | --- | --- | --- |
| Full import, all columns kept | 17 s | ~690 MB frame | 16 MB (CSR) |
| Budgeted (`max_edges=50000`), chunked, extras dropped | 0.5 s | 16 MB | 16 MB (CSR) |

Practical guidance:

* **Always set a node/edge budget while experimenting.** It is the difference
  between 0.5 s and 17 s, and the reader stops early instead of parsing
  everything.
* **A dense `N x N` matrix is never the answer.** For 120k neurons that is
  58 GB. `Connectome.sparse_matrix()` always returns CSR, and
  `Connectome.weight_matrix()` refuses above a node limit.
* **NetworkX is for extracted subgraphs only.** A NetworkX edge costs far more
  than a CSR entry, so `to_networkx()` refuses above 2,000 nodes by default.
* **`keep_extra_columns=False`** drops unmapped source columns. This is the main
  per-row memory cost after the string columns.
* **`chunksize=N`** bounds parse memory and enables the early exit above.
* **Prefer CSV or Parquet over JSON** for large files; JSON is loaded whole.
* **Parquet needs an engine**: `pip install -e ".[parquet]"` (free, `pyarrow`).
* **The simulation is not the importer.** `NeuralSimulation` keeps dense
  per-step activity arrays and refuses graphs above
  `brain.simulation.max_nodes` (default 2000). Extract a subgraph first.

## 6. Free Google Colab workflow

`notebooks/connectome_ingestion_colab.ipynb` walks through the whole pipeline on
the Colab free tier: install, inspect, validate, budgeted import, sparse graph,
report, and a plot of a small extracted subgraph.

Colab's free T4 is not needed for ingestion — this work is memory bound, not
compute bound — but the free tier is what makes a 12 GB RAM machine available
for the full import. Upload a small export directly, or mount Drive for a
larger one. The runtime is temporary: anything you want to keep must be written
to `data/processed/` before the session ends.

## 7. Biological data vs. simulation

| Layer | Nature | Reported as |
| --- | --- | --- |
| Connectome topology from a real export | **biological data** | `Biological connectome data` |
| `BrainGraph` built from it | **derived representation** | inherits the source's provenance |
| LIF integration, spikes, firing rates | **simulation** | `SIMULATED, not recorded from a fly` |
| Learned policy weights | **machine learning** | no biological claim |
| The hand-made files in `tests/fixtures/` | **test data** | `TEST FIXTURE - NOT REAL BIOLOGICAL DATA` |

Two guarantees enforced in code:

* `is_biological` is **declared**, never inferred. A file called `flywire.csv`
  is not biological data on the strength of its name; only an explicit
  `Provenance(biological_data=True)` with a source that is allowed to be
  biological sets it.
* `DataSource.TEST_FIXTURE` is structurally incapable of reporting
  `is_biological=True`, so a fixture cannot be mistaken for real data even if a
  caller asks for it.

## 8. The processed-table contract

One row per synapse, `data/processed/<dataset>.csv`:

| column | required | biological? | notes |
| --- | --- | --- | --- |
| `source_id` | yes | yes | pre-synaptic neuron id, as published |
| `target_id` | yes | yes | post-synaptic neuron id, as published |
| `weight` | yes | **maybe** | only if the source reports strength or counts; otherwise a flagged placeholder |
| `weight_is_biological` | no | — | `False` whenever `weight` is a placeholder |
| `source_region` / `target_region` | no | yes | only if the source reports regions |
| `source_type` / `target_type` | no | yes | only if the source reports neuron types |
| `original_*` | no | yes | unmapped source columns, preserved rather than dropped |

Unknown values stay null. Inventing a plausible-looking number and citing a paper
for it is the failure mode this project exists to avoid.

## 9. Test fixtures are not data

`tests/fixtures/` contains hand-written files used to test the importer without
a dataset. They are labelled `TEST FIXTURE - NOT REAL BIOLOGICAL DATA`, are not
derived from FlyWire or any publication, and are imported as
`DataSource.TEST_FIXTURE`. See `tests/fixtures/README.md`.

## 10. Getting a real dataset later (free)

Candidate public sources, all free to access with attribution:

* **FlyWire** — `flywire.ai` open access; per-synapse tables for the adult
  female *Drosophila* central brain.
* **Janelia / neuPrint hosted datasets** — brain-wide synapse counts per
  compartment.
* **NEAT and other derived tables** — hemi-brain derivatives.

Licences differ (typically CC-BY or CC-BY-SA for data, separate for code). Read
them before redistributing anything. This project will not commit third-party
data; it commits the loader, the schema, and the attribution.
