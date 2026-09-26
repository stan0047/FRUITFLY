# MaleCNS v1.0 — verified data semantics and provenance

**Status:** semantics verified against official documentation on 2026-09-26.
Two questions remain **unresolved** and are recorded as such below. No value was
invented to fill either gap.

This document is the human-readable companion to
`src/flybrain/data/mcns_provenance.py`, which holds the same facts in
machine-readable form and is covered by `tests/test_mcns_provenance.py`.

> **This is still data validation.** Nothing here identifies a visual neuron or
> selects a pathway. The bounded experiment reported at the end is a
> *top-weight MCNS connectivity subset* — a data-engineering feasibility check.

---

## 1. Official sources

| Purpose | URL |
| --- | --- |
| Project home | https://male-cns.janelia.org/ |
| **Downloads and file descriptions** | https://male-cns.janelia.org/download |
| Release notes | https://male-cns.janelia.org/release |
| neuPrint user guide | https://neuprint.janelia.org/public/neuprintuserguide.pdf |
| Published paper | https://doi.org/10.1016/j.cell.2026.08.015 |
| Preprint | https://doi.org/10.1101/2025.10.09.680999 |
| neuPrint Python client | https://connectome-neuprint.github.io/neuprint-python/ |
| neuPrint endpoint | `https://neuprint.janelia.org`, dataset `male-cns:v1.0` |

Bulk files live under `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`.

## 2. Dataset identity

| Field | Value | Status |
| --- | --- | --- |
| Source | Janelia MaleCNS | DOCUMENTED |
| Dataset | Male CNS v1.0 | DOCUMENTED |
| Release date | 2026-06-08 | DOCUMENTED |
| Published | 2026-09-03, *Cell* 189(18) | DOCUMENTED |
| Licence | **CC-BY 4.0** | DOCUMENTED |
| Canonical citation | Berg, S. et al. (2026). *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* Cell 189(18). https://doi.org/10.1016/j.cell.2026.08.015 | DOCUMENTED |
| Preprint | https://doi.org/10.1101/2025.10.09.680999 | DOCUMENTED |
| **Companion paper for Phase 3** | Hoeller, J., Zhao, A., Nern, A., Rogers, E. M., Romani, S., & Reiser, M. B. (2026). *The organization of visual pathways in the Drosophila brain.* Cell 189(18). | DOCUMENTED (journal DOI not independently verified) |

Collaboration: FlyEM (HHMI Janelia), University of Cambridge (Dept. of Zoology),
MRC Laboratory of Molecular Biology, Google Research.

## 3. Verified field semantics

Quotations are short fragments for re-checking, not copied passages.

| Field | Meaning | Status |
| --- | --- | --- |
| `body_pre` | Body (segment) id of the **pre-synaptic** neuron. One row is one directed connection from this body to the one in `body_post`. | DOCUMENTED |
| `body_post` | Body (segment) id of the **post-synaptic** neuron. | DOCUMENTED |
| `weight` | **Number of synapses** from `body_pre` to `body_post`. Positive integer. A structural count, **not** a functional efficacy. | DOCUMENTED |
| `bodyId` | Body id in the annotation table. Joins to `body_pre`/`body_post`. One row per *body*, including glia, orphans and fragments — not one row per proofread neuron. | DOCUMENTED |
| `class` | Broad sensory/functional class (`visual`, `olfactory`, `gustatory`, …). | DOCUMENTED |
| `superclass` | Finer grouping: `cb_intrinsic`, `ol_intrinsic`, `visual_projection`, `visual_centrifugal`, `ascending_neuron`, `descending_neuron`, VNC compartments. | INFERRED |
| `type` | Cell type, 11,691 types reported. | INFERRED |
| `status` | Reconstruction status: `Traced`, `Roughly traced`, `Assign`, `Anchor`, `Orphan`, `Glia`, `Unimportant`. | INFERRED |
| `somaSide` | Soma side: `L`, `R`, `M` (midline). | INFERRED |
| `somaNeuromere` | Neuromere containing the soma (`LB`, `T1`, `CG`, …). Heavily null. | INFERRED |
| `somaLocation` | Soma centroid as three integers in MaleCNS EM voxel coordinates, 8 nm isotropic. Partial. | INFERRED |

Source wording, paraphrased: the download page describes the connectivity table
as *segment-to-segment connection strengths for all segments in the dataset,
excluding those with no synapses*, and adds *this is the full connection graph*.
The neuPrint user guide states that the connection weight is *the total number of
synapses from the presynaptic neuron to the postsynaptic neuron*.

### Granularity: AGGREGATED

One row per **ordered body pair**, with `weight` equal to the synapse count. This
satisfies requirement **R4** of `docs/CONNECTOME_REQUIREMENTS.md`. The local file
confirms it: 1,000,000 rows → 1,000,000 unique ordered pairs, **0 duplicates**.

### Measured locally (OBSERVED — facts about the file, not biology)

| Property | Value |
| --- | --- |
| Rows (from Arrow footer, no scan) | 151,856,684 |
| Record batches | 2,318, of 65,536 rows (last 9,772) |
| Ordering | sorted by `weight` **descending**, in contiguous equal-weight blocks |
| First batch weight range | 92 – 2,591 |
| Batches ≥ 1,159 | `weight` == 1 throughout (~76 M rows) |
| Annotation rows / columns | 211,577 / 36 |
| `bodyId` nulls, duplicates | 0, 0 |

The descending sort is the fact that makes bounded extraction possible: the
strongest N edges are the first N rows, so no full scan is ever required.

## 4. The edge-count discrepancy — **RESOLVED**

The 25.58M-vs-151,856,684 gap is explained by an **officially documented node
population difference**, not by a threshold FlyBrain failed to apply.

The published Cell STAR Methods states:

> "After proofreading, the neuron segmentation and synaptic connections jointly
> define a connectome graph containing 25.58M edges between 166,483 neurons
> **with a superclass annotation**."

and in the main text:

> "This **proofread connectome** containing 124.2M synaptic connections defines a
> graph of 25.6M edges between 166,483 neurons (217 neurons without synapses are
> disconnected)."

The discriminator is stated explicitly:

> "Bodies in the dataset are defined as **neurons if they have a superclass**.
> Bodies without one are **fragments of neurons**."

The two numbers are therefore:

| | Node population | Edges | Source |
| --- | --- | --- | --- |
| Paper | **neurons** — bodies carrying a `superclass` | 25.58M | published Cell |
| Local file | **all segments**, including fragments | 151,856,684 | bulk download |

Both are ordered **body-pair** rows. The difference is the node set, not the
granularity and not a synapse threshold.

### Ruled out, by official statement

* **Not a minimum synapse-count threshold.** The ≥5-synapse filter is documented
  as a *further* reduction, to a different graph: *"Considering only connections
  with a strength of at least 5 synapses … the remaining graph contains 6.24M
  edges between 165,536 neurons."* The 25.58M figure is the unthresholded one.
* **Not a different release.** The file is `…-v1.0-minconf-0.5.feather` under
  `gs://flyem-male-cns/v1.0/`, matching the paper. Release notes list only v0.9
  and v1.0.
* **Not synapse-level vs body-pair-level.** Both figures are body-pair.

### Verified locally

| Quantity | Local file | Paper | Match |
| --- | --- | --- | --- |
| Bodies with a `superclass` | **166,700** | "166,700 neurons" | **exact** |
| Bodies without a `superclass` | 44,877 | "fragments of neurons" | consistent |
| Total bodies | 211,577 | not stated | — |

The 166,700 correspondence is exact and is now the practical definition of a
neuron in FlyBrain: **`superclass` is non-null**. Selecting neurons by that
predicate is documented, not guessed.

### `minconf-0.5` — now resolved

Documented in the paper's figure caption: *"Our published synapses are filtered
with a confidence threshold of 0.5 … resulting in overall precision of 0.82 and
recall of 0.81."* It is a synapse-detection confidence filter, it applies to both
sides of the comparison, and therefore explains none of the edge-count gap.
Detection totals: 46M presynapses, 312M postsynapses; 84.6M orphan fragments in
the dataset; only 40.1% of connections have both sites on a proofread neuron.

### Version caveat

The **preprint (bioRxiv v2) and the published Cell paper differ** on this point.
The preprint says *"25.6M edges between 166,391 neurons"* with no superclass
qualifier; the published version says 25.58M / 166,483 and adds the superclass
condition. **The published version is canonical here.** Citing the preprint alone
would support a weaker and incorrect conclusion.

### Still unresolved

No official source states the row count of the bulk file, nor the total number of
bodies/segments in the dataset. So the residual — 124.2M proofread synaptic
connections versus 151.86M segment-pair rows — is quantified by neither side.
This no longer blocks subgraph work, because the node predicate is now known.

## 5. A material limitation for a visual-navigation experiment

The paper documents that the visual periphery is the **least complete** part of
this dataset:

* some photoreceptors are missing or incompletely reconstructed, and the
  corresponding columns were assigned from anatomical markers alone;
* initial validation cubes in the lamina showed poor synapse-detector recall,
  and the T-bar detector was subsequently fine-tuned on them;
* proofreading was executed primarily in the central brain and **right** optic
  lobe, while the **left** optic lobe and the VNC relied on cell typing to flag
  errors;
* technical limitations prevented exhaustive annotation of the VNC, and some
  efferent and sensory VNC neurons could not be typed.

Also relevant: only **40.1%** of synaptic connections have *both* the pre- and
post-synaptic site on a proofread neuron, with 94% pre- and 42% post-synaptic
completion. Most of the raw graph is therefore not between proofread neurons.

This does not make MaleCNS unsuitable — it is the only complete whole-animal
connectome — but it fixes what a visual-pathway experiment on this dataset can
and cannot claim.

## 6. Bounded experiment: top-1M connectivity subset

`python scripts/analyze_mcns_top_edges.py --max-edges 1000000`

| Metric | Value |
| --- | --- |
| Edges read | 1,000,000 (0.66% of the file) |
| Record batches read | **16 of 2,318** (2,302 untouched) |
| Edge arrays in memory | **22.9 MiB** |
| Dense *N*×*N* float32 equivalent | **78.8 GiB** — never built |
| Unique ordered pairs / duplicates | 1,000,000 / **0** |
| Self-loops | **2** (on 2 neurons) — measured and **retained** |
| Unique source / target / overall neurons | 124,612 / 120,193 / **145,476** |
| Weights min / median / mean / max | 20 / 32 / 43.6 / 2,591 |
| Annotation coverage | 99.62% (557 unannotated) |

For comparison, `--max-edges 100000` reads **2 of 2,318** batches, holds 2.3 MiB,
and touches 38,442 neurons.

### The finding that matters most for Phase 3

**The top 1,000,000 connections already touch 145,476 neurons — about 87% of the
entire connectome.** A weight threshold does *not* isolate a circuit. It produces
a sparser version of the whole brain, not a functional subgraph.

Consequence: circuit extraction cannot be a weight-cut operation. It must be
annotation-driven or connectivity-driven (seed a known cell type, follow its
projections), and it must state a budget. Annotation breakdowns over this subset
show the strong-connection mass is optic-lobe-local (`superclass`:
`ol_intrinsic` 72,766, `cb_intrinsic` 30,717, `visual_projection` 8,452), and the
most frequent `type` values are visual ones (`R1-R6` 3,043, `Tm3` 2,031, `L5`
1,783, `L2` 1,778, `L1` 1,776) — recorded here as a dataset property, **not** as a
visual-circuit selection.

`class` is annotated for only 22,205 of the subset's neurons; `somaNeuromere` for
20,993. An unannotated neuron has an **unknown label**: that is not evidence of
non-membership in any category, and it is reported as such rather than folded
into a class.

## 7. Standing data-ownership rule

**LOCAL BULK FEATHER = authoritative.** Primary connectome, reproducible graph
construction, parameterised subgraph extraction, all local experiments.

**neuPrint API = targeted queries only.** Discovery, specific-neuron
investigation, annotations we deliberately do not bulk-download, skeleton
retrieval. A graph built from API results is an **API-derived subset** and must
be labelled as such; it is never presented as the local MCNS graph.

The token is required only for neuPrint. Every local workflow — inspection,
import, validation, sparse graph construction, reports, seeded exploration, and
all unit tests — runs with no token at all. No test in this repository makes a
live API request.

**NEUPRINT_AUTH_STATUS = `INVALID_OR_UNVERIFIED`.** The token currently in
`.env` returns HTTP 401. Diagnosis without exposing the value: it contains
internal whitespace and only 18 distinct characters across 71, which is
inconsistent with a neuPrint token, so it is treated as a placeholder. It has
**not** been modified and will not be retried automatically. Replace it with a
real token from your neuPrint account and run
`python scripts/test_neuprint.py` to re-check. Until then, plan Phase 3+ around
local bulk data only.

## 8. Seeded connectivity neighborhood (Phase 2.5 feasibility)

A **seeded connectivity neighborhood** is a bounded, weight-truncated expansion
from a seed set. It is a data-structure artifact. It is **not** the visual
circuit, and nothing below is a claim about a biological pathway.

Seeds: cell types `L1`, `L2`, `L5`, `Tm1`, `Tm3` — **9,173 bodies**, all
`superclass = ol_intrinsic`, all `status = Traced`, L/R symmetric (892/884,
893/886, 898/889, 890/887, 1037/1017). Prefix: top 2,000,000 edges (31 of 2,318
batches, 172,498 nodes, dense equivalent 119 GB — never built).

| Stage | New nodes | Cumulative | Discovery edges | Budget that bound it |
| --- | --- | --- | --- | --- |
| seeds | 9,173 | 9,173 | 0 | — |
| 1-hop downstream | 32,742 | **41,915** | 32,742 | none |
| 1-hop upstream | 14,737 | **23,910** | 14,737 | none |
| 1-hop both | 39,697 | 48,870 | 39,697 | none |
| 2-hop downstream | 29,252 | **71,167** | 61,994 | none |
| 2-hop both | 37,814 | **86,684** | 77,511 | none |

"Discovery edges" are the edges that first brought a new node in — not the
induced edge count among visited nodes, which is larger.

Annotation coverage of the 2-hop downstream neighborhood: 98.39% present in the
annotation table, 97.93% with a cell-type label.

### What the neighborhood contains

2-hop downstream is **92.6% optic-lobe intrinsic** (`ol_intrinsic` 64,801 of
69,714 annotated), with `visual_projection` 4,110, `cb_intrinsic` 592,
`descending_neuron` 95, `visual_centrifugal` 77. Top cell types: `Tm3` 2,054,
`T3` 1,922, `L5` 1,787, `L2` 1,779, `Tm1` 1,777, `L1` 1,776, `C3` 1,774,
`T1` 1,772, `Mi1` 1,771, `Tm2` 1,765, `T4c` 1,764, `T2a` 1,719.

Two things follow, and both matter more than the counts:

1. **Hop-limited expansion does not escape the optic lobe.** After two hops from
   lamina/medulla neurons the neighborhood is still overwhelmingly
   `ol_intrinsic`, with only 592 central-brain neurons. The optic lobe is a
   densely local module. Reaching central-brain circuits needs either more hops,
   or bridge neurons chosen for their projection pattern.
2. **Growth is bounded by the read prefix, not by biology.** 86,684 of the
   172,498 nodes in a 2M-edge prefix is half the prefix. A third hop would
   saturate it, and any "explosion" measured this way is an artifact of the
   prefix size. Growth rates must not be quoted from a truncated prefix.

Also recorded: `class`, `subclass`, `supertype`, `somaNeuromere` and
`hemibrainType` are **null for all 9,173 seeds**. The visual lamina and medulla
neurons are *not* reachable through `class == "visual"` (which covers only 6,091
bodies dataset-wide). `superclass`, `type`, `flywireType` and `somaSide` are the
usable annotation fields for this seed set.

## 9. Provenance of the seed labels

`type` is a **curated assignment, not an experimental measurement.** The paper
describes cell types as defined *"based on morphology and connectivity of
individual neurons across the CNS"*; the download page calls the annotations
*curated* and the nuclei-derived cell-body annotations *manually reviewed*. The
method behind each individual field is not stated per field, so per-field
provenance remains unknown. No annotation field in FlyBrain is reported as a
direct measurement.


