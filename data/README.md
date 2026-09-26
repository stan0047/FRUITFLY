# Data directory

**No dataset is stored in this repository.** Connectome data is large,
attribution-heavy and usually licence-conditional, so it is fetched or
generated locally and ignored by git (see `.gitignore`).

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
   `name`, `url`, `version`/`date`, `licence`, and `citation`.

## Connectome status

**Not yet integrated.** The real fruit-fly connectome is *not* downloaded,
parsed, or bundled here. Today the only graph in the codebase is
`flybrain.brain.graph.synthetic_brain_graph()`, a random directed graph that
is explicitly labelled `source="synthetic"`.

`flybrain.data.connectome.load_flywire_connectome()` is a stub that raises
`NotImplementedError`. It exists to fix the interface contract — required
columns, attribution fields, provenance flags — so that integration later is a
data task rather than an architecture change.

## What a processed connectome will look like

One CSV row per synapse, `data/processed/connectome.csv`:

| column | type | biological? | notes |
| --- | --- | --- | --- |
| `pre` | str | yes | pre-synaptic neuron / cell type id, as published |
| `post` | str | yes | post-synaptic neuron / cell type id, as published |
| `weight` | float | **maybe** | only if the source reports synaptic strength; otherwise initialised by us and flagged |
| `synapse_type` | str | yes | if reported by the source |
| `neuropil` | str | yes | if reported by the source |
| `neurotransmitter` | str | no | typically *not* reported; do not guess |

Anything not reported by the source stays `null` and is marked as an
author assumption. Inventing plausible-looking values and citing a paper for
them is the failure mode this project must avoid.

## Getting a real dataset later (free)

Candidate public sources, all free to access with attribution:

* **FlyWire** — `flywire.ai` open access; per-synapse connectome for
  *Drosophila melanogaster* adult female (FAFB).
* **Janelia/Connectome Annotations** — hemi-brain / FAFB derivative tables.
* **NEAT / neuPrint** hosted datasets — brain-wide synapse counts per
  compartment.

Licences differ (typically CC-BY or CC-BY-SA for the data, code licences
separate). Read them before redistributing anything. This project will not
commit third-party data; it will commit the loader, the schema and the
attribution.
