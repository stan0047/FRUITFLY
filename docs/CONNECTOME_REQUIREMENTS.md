# Connectome data requirements for the FlyBrain visual-navigation experiment

**Status:** specification only. No dataset has been supplied or inspected yet.
**Scope:** what the data must contain for the pipeline below to be runnable, and
what it must not be assumed to contain.

---

## 1. Purpose

FlyBrain's intended pipeline is:

```
REAL CONNECTOME
      ↓
visual-system subgraph
      ↓
neural simulation
      ↓
visual representation
      ↓
navigation policy
```

Each stage consumes something specific from the stage above it:

| Stage | Consumes | Fails without |
| --- | --- | --- |
| visual-system subgraph | neuron identity, directed edges, and something to select neurons *by* (region or type) | a way to define "visual system" |
| neural simulation | directed edges, one weight per edge, node identity | a connectivity structure and a weight |
| visual representation | a simulation output plus a definition of which neurons carry which visual signal | a labelled readout of the circuit |
| navigation policy | a fixed-size observation vector and `LEFT` / `FORWARD` / `RIGHT` | nothing from the data layer |

This document specifies the minimum input to make that pipeline runnable, and
nothing more. It is a data-requirements document, not a circuit design.

## 2. Provenance taxonomy

Every field below is labelled with how it enters FlyBrain. The distinctions
matter because they determine what may be claimed in a write-up.

| Label | Meaning | Example |
| --- | --- | --- |
| **MEASURED** | read out of the experimental reconstruction, subject to reconstruction and segmentation error | which neuron is pre-synaptic to which |
| **DERIVED** | computed or predicted from the measurement, by an annotator or an algorithm | a predicted cell-type label; a synapse count obtained by counting rows |
| **ASSUMED** | supplied by FlyBrain because the dataset does not provide it | a uniform placeholder edge weight |

A DERIVED field is a real result, but it is a result *about* the data, not a
direct observation. A model-predicted cell type and a manually annotated one do
not deserve the same confidence, which is why confidence and annotation source
are called out separately below.

---

## 3. REQUIRED

Without these, the pipeline cannot start. Six items.

### R1 — Neuron identifier (node table)

* **Biologically:** the identity of a single reconstructed neuron. In a
  connectome this is a *root* identifier: a segmentation "root" that all the
  fragments (neurites) of one cell are merged into. A neuron is not a voxel blob
  and not a named cell type; it is one reconstructed cell.
* **Why FlyBrain needs it:** the node key for the graph. Every edge endpoint must
  resolve to one, and the simulation indexes its activity arrays by it.
* **Expected type:** string, unique, stable across releases if the provider
  claims stability. FlyBrain stores it as `str` and never parses it.
* **Table:** node table (or, in a single-table export, the same column used on
  both edges).
* **Status:** **MEASURED** (with the caveat that segmentation is a reconstruction,
  so "one neuron" is itself an inference from image data).

### R2 — Source neuron identifier (edge table)

* **Biologically:** the pre-synaptic neuron. Chemical synapses are directional,
  so this end of the connection is not interchangeable with the other.
* **Why FlyBrain needs it:** the row index of the recurrent update
  `W[i, j]`; also the anchor for propagating activity in the intended direction.
* **Expected type:** string, values drawn from the same identifier space as R1.
* **Table:** edge table, canonical name `source_id`.
* **Status:** **MEASURED**.

### R3 — Target neuron identifier (edge table)

* **Biologically:** the post-synaptic neuron.
* **Why FlyBrain needs it:** the column index of the recurrent update.
* **Expected type:** string, same identifier space as R1.
* **Table:** edge table, canonical name `target_id`.
* **Status:** **MEASURED**.

> R1–R3 together are the *only* strictly structural requirements. Everything else
> exists to make the graph interpretable rather than merely well-formed.

### R4 — Stated edge granularity

* **Biologically:** not a field. It is the answer to "what does one row mean?"
  — one chemical synapse, or one connected pair summarised in some way.
* **Why FlyBrain needs it:** it determines what a weight means and whether a
  synapse count is already present. If a row is one synapse, the count for a
  `(source, target)` pair is just the number of rows, and FlyBrain derives it by
  aggregation. If a row is one connected pair, the count is not available and
  any weight is whatever the file says it is. Reading `count` as a synapse
  multiplicity when rows are already aggregated is a silent, order-of-magnitude
  error, so this must be established before any weight is trusted.
* **Expected type:** a statement, recorded in the provenance `notes`, plus
  whatever the source documents.
* **Table:** neither; provenance metadata.
* **Status:** **MEASURED** (a property of the file, recorded by the user).

### R5 — A weight field, biological or explicitly flagged

* **Biologically:** the strength or magnitude attributed to a connection. This
  is the field most often over-claimed.
* **Why FlyBrain needs it:** the canonical schema requires a `weight` column, and
  the simulation needs a number per edge. What it must *not* do is present an
  assumed number as a measured one.
* **Expected type:** float, non-negative. FlyBrain stores
  `weight` alongside `weight_is_biological`, and a file with no weight column
  yields a uniform placeholder of `1.0` with that flag `False`.
* **Table:** edge table, canonical name `weight`.
* **Status:** **MEASURED** if the source reports it; otherwise **ASSUMED**, and
  the flag stays `False` all the way into the report and the figures.

If the dataset supplies a per-synapse size or volume measurement, that is
**MEASURED**. Treating it as a *functional* synaptic weight is an **ASSUMED**
step, because structural size and functional efficacy are related but not
identical. If FlyBrain adopts that step it must be a documented, flagged
modelling choice, not an implicit one.

### R6 — Provenance the provider actually states

* **Biologically:** not applicable; this is attribution.
* **Why FlyBrain needs it:** a result that cannot name its dataset, version,
  licence and citation cannot be checked by anyone, including us in a year.
* **Expected type:** `source_name`, `dataset_name`, `dataset_version`,
  `source_url`, `license`, `citation`, and whatever else the source states.
* **Table:** provenance sidecar, `<dataset>.source.json`.
* **Status:** **MEASURED** from the source's own documentation. Fields the source
  does not state stay `None` and are listed as `Unverified`; FlyBrain does not
  fill them in.

---

## 4. STRONGLY PREFERRED

Not needed to build a graph. Each of these is what makes the graph *interpretable
as a visual system*, and without them the "visual-system subgraph" stage becomes
guesswork.

### P1 — Neuron type / cell type (node table)

* **Biologically:** a categorical label for what kind of cell a neuron is —
  for a visual pathway, the cell types of the lamina, medulla and lobula
  retinotopic columns, the motion-detecting interneurons, and so on.
* **Why FlyBrain needs it:** the primary way to select a visual pathway. With
  types, "take the medulla" or "take the known motion detectors" is a defined
  operation. Without them, the only route is purely connectivity-based selection
  from a seed neuron, which is weaker and harder to justify.
* **Expected type:** string label, ideally accompanied by a confidence value and
  a statement of how the label was produced.
* **Table:** node table, canonical name `neuron_type`.
* **Status:** typically **DERIVED**, and often *model-predicted* rather than
  hand-annotated. That is a legitimate and useful result, but it must be
  reported as a prediction with its confidence, not as a measured identity. A
  dataset that does not distinguish these should be treated as doing so.

### P2 — Brain region / neuropil (node table)

* **Biologically:** the brain compartment a neuron's arborisation occupies —
  the neuropils that tile the insect central brain.
* **Why FlyBrain needs it:** the second, independent route to pathway selection,
  and the one that survives when type annotations are absent or unreliable. It
  also drives the region-activity panel the dashboard will need.
* **Expected type:** string label per neuron, or per edge endpoint.
* **Table:** node table preferred (`region`); per-edge endpoint columns
  (`source_region`, `target_region`) are accepted and mapped into the same place.
* **Status:** usually **DERIVED** (annotation, sometimes assisted by automated
  assignment). Neuropil assignment is a judgement about where a process
  terminates, not a direct observation of it.

> P1 and P2 are alternatives, not requirements on each other. A dataset with
> strong regions and no types is workable; a dataset with types and no regions is
> workable. A dataset with neither leaves only connectivity-based selection, which
> is legitimate for a first look but should be recorded as such.

### P3 — Synapse count, or a per-synapse granularity that makes it derivable

* **Biologically:** the number of synaptic contacts between a pair of neurons. It
  is the most common structural correlate of connection strength used in
  connectome work.
* **Why FlyBrain needs it:** a weight that is at least monotonically related to
  connection strength is far more useful than a uniform `1.0`, and it lets
  aggregate statistics such as degree-weighted connectivity mean something.
* **Expected type:** non-negative integer, either as a column or as the row
  multiplicity under per-synapse granularity (see R4).
* **Table:** edge table, folded into `weight`.
* **Status:** **DERIVED** when it is a row count aggregated by FlyBrain;
  **MEASURED** only in the sense that the underlying synapses were reconstructed,
  which is not the same as the count being a reported measurement.

### P4 — Directionality semantics, and synapse class

* **Biologically:** chemical synapses are directed; **gap junctions are
  electrical and not directed.** A table mixing both cannot be represented as a
  plain directed graph without deciding what to do with the undirected half.
* **Why FlyBrain needs it:** to avoid silently corrupting the recurrent update.
  Representing an electrical connection as if it were directional is a
  physiological error, not a formatting one.
* **Expected type:** a synapse class or connection type column, if the source
  distinguishes them.
* **Table:** edge table; currently carried as an unmapped source column.
* **Status:** **MEASURED** where the source distinguishes the classes.

---

## 5. OPTIONAL

Useful later, not needed for the first prototype.

### O1 — Anatomical coordinates

* **Biologically:** the position of a neuron in the volume — typically a soma
  centroid, sometimes a cell-body or arborisation bounding box. Real spatial
  information from the reconstruction.
* **Why it is optional *for the experiment*:** position contributes nothing to
  subgraph selection, simulation, representation learning, or the policy. Those
  are graph and dynamics problems.
* **Why it is wanted *for the visualisation*:** the dashboard's glowing 3D brain
  needs somewhere to put each neuron. Without coordinates, a 3D view is a
  spring layout with a brain-shaped colour scheme, which would be decoration
  masquerading as anatomy. **This is the honest reason to want coordinates, and
  it is a presentation requirement, not a scientific one.**
* **Expected type:** three floats per neuron, in the source's coordinate frame
  and unit, which must be recorded.
* **Table:** node table.
* **Status:** **MEASURED** for a soma centroid, with segmentation error; the
  *unit and frame* are metadata that must be stated, not guessed.

### O2 — Neurotransmitter or synaptic receptor annotation

Would let activity be interpreted by signalling class. Not needed to simulate
electrical activity. **DERIVED** where assigned from a reference atlas.

### O3 — Neuron morphology, arborisation size, compartment

Useful for weighting or for a size-resolved view. Not needed. **MEASURED** where
reconstructed.

### O4 — Cross-references to published neuron sets

Lineages or named-cell-type identifiers that tie a neuron to a published
circuit description. This is the bridge from "a graph" to "the known visual
system", and it materially raises the quality of pathway selection — but it is a
convenience, and a dataset can be scientifically useful without it. **DERIVED.**

### O5 — Multi-connectome alignment or cross-species data

Out of scope. Recorded here only so it is not mistaken for a missing requirement.

### O6 — Experimental activity data

**Not part of a connectome at all.** Nothing in this document asks for it,
because FlyBrain has none. Every spike, membrane potential and firing rate this
project produces is a simulation. No future milestone should imply otherwise
without an actual dataset of recorded activity.

---

## 6. NOT REQUIRED for the first prototype

Stated explicitly, because these are the requests most likely to arrive and most
likely to be satisfiable only by fabrication:

* **A 3D anatomical brain mesh or neuropil volumes.** Not required. The 3D
  visualisation milestone can ship without one, and must then say that node
  positions are schematic.
* **Anatomical coordinates.** See O1. Optional.
* **Electrophysiological properties** — membrane time constant, threshold,
  reversal potentials, channel densities. FlyBrain uses a generic leaky
  integrate-and-fire model. Substituting real values for real neurons is a
  later modelling milestone, not a data requirement.
* **Synaptic conductance, kinetics or receptor types.**
* **A visual input stimulus or photoreceptor mosaic.** The simulation is driven
  by an abstract external input vector. Wiring a real optic flow to a real
  photoreceptor layout is a separate milestone.
* **Behavioural labels, conditioned stimulus preferences, or per-neuron tuning.**
* **Developmental or sex / age metadata**, unless a specific claim depends on it.
* **Every neuron typed.** A first prototype can restrict itself to a subgraph
  where annotations are dense and say so.
* **Whole-brain coverage.** A budgeted subset is a legitimate experimental unit
  provided every statistic is labelled `SUBSET`.

---

## 7. DO NOT ASSUME

The implementation must not assume any of the following. Each is a way this
project could produce a confident, wrong, unfalsifiable result.

**1. A particular FlyWire column naming convention.**
Column names differ between exports, releases and providers, and a substring
heuristic over `pre` / `post` / `root` is not a specification. FlyBrain infers
roles structurally and, when a role cannot be resolved with confidence, stops and
asks. It must not carry a hardcoded list of expected FlyWire column names, and it
must not silently fall back to "the first column that looks like an id".

**2. A particular dataset release.**
No release, version, date, or download URL is recorded anywhere in this
repository, and none should be until a real file is in hand. There are several
exports; they change; a URL baked into a library goes stale silently. The
ingested dataset's version comes from the file's own documentation, entered by
the user.

**3. That every connection has a biological weight.**
Most connectome exports measure *structure*, not *function*. Synapse count and
synapse size are structural. A placeholder weight must stay flagged
`weight_is_biological=False` through the graph, the report and every figure. A
graph built on uniform weights is a structural skeleton, and any claim about
signal strength from it is unsupported.

**4. That neuron type annotations exist for every neuron.**
Annotations are typically model-predicted, carry confidence, and have gaps. Code
must tolerate missing types, must not impute a label for an unlabelled neuron, and
must not assume a closed set of type names. A neuron with no type is unlabelled,
not "unlabelled type".

**5. That anatomical coordinates are present.**
Optional. A neuron without coordinates must be placeable by a clearly-schematic
fallback, and any 3D rendering without coordinates must be labelled as schematic
rather than anatomical.

**6. That all visual pathways are represented in one export.**
Exports differ in volume, in whether they are per-synapse or aggregated, and in
which compartments they resolve. A single file may contain the central brain and
nothing of the optic lobe, or may be a subset of one. Subgraph selection must
report what was found and what was absent, and must not assume that "no neurons
matched" means the pathway is absent rather than the export being incomplete.

Two more that follow from the same principle:

**7. That a filename indicates provenance.** A file named `flywire.csv` is not
biological data by name. `is_biological` is declared, never inferred.

**8. That a citation implies endorsement of a derived field.** Citing a dataset
cites its reconstruction. A model-predicted cell type inherits the citation of
the annotation it came from, not the authority of the paper being cited for it.

---

## 8. PHASE 3 DECISION GATE

**Circuit extraction cannot begin until the actual user-provided dataset has been
inspected, and its schema and provenance verified.**

Extraction rules — which regions constitute the visual system, which cell types
to select, how to seed the subgraph, which weights are measured versus assumed —
must be written against the columns a real file actually contains. Writing them
against a guessed schema produces code that is confidently wrong, and the failure
is invisible because the pipeline runs.

The gate is passed when all of the following are true:

1. **A real file exists locally**, obtained by the user from the official
   provider, with its licence and citation terms read.
2. **Its schema has been inspected** with
   `python scripts/inspect_connectome.py --input <file>`, and the output is
   recorded: file type, row count, columns, dtypes, detected roles, ambiguities.
3. **Its granularity is established** (R4): whether a row is one synapse or one
   connected pair.
4. **Its provenance is recorded** (R6): source name, dataset name, version as
   stated, URL, licence, citation.
5. **A test import has succeeded** with
   `python scripts/import_connectome.py`, and the resulting report has been read
   — particularly the `Biological`, `Statistics`, `Edge weight is biological`,
   and `Unverified` lines.
6. **The R1–R6 requirements have been checked against the file**, item by item,
   and any that are unmet are recorded as unmet rather than worked around.

If any item cannot be satisfied, that is a finding to report, not a gap to paper
over. "This export has no region labels" is a legitimate and useful result that
changes the Phase 3 plan; quietly assigning regions from a different dataset is
not.

Until the gate is passed, the correct state of the project is: pipeline built,
dataset not yet seen, no circuit claims made.

---

## 9. Related documents

* `data/README.md` — ingestion commands, memory characteristics, the processed
  table contract.
* `README.md` §3 — biological data versus derived graph versus simulation versus
  learned behaviour.
* `notebooks/connectome_ingestion_colab.ipynb` — the free-tier Colab walkthrough.
