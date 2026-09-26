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

## 4. Unresolved — recorded as gaps, not filled with guesses

**`minconf-0.5` is not defined anywhere we checked.** The tag appears in three
filenames and on no documentation page. The paper's only 0.5 confidence cut-offs
apply to *neurotransmitter* assignment, not synapse detection. What the 0.5 is a
threshold on is therefore **unknown to FlyBrain**. Observation only: the file
retains `weight == 1` connections, so the filter is permissive.

**The edge count does not reconcile.** The paper states a graph of *25.6M edges
between 166,391 neurons* with no filter named in that sentence. The local file
has **151,856,684 rows** — roughly six times more. Candidate explanations include
a minimum-synapse threshold applied in the paper's analysis, and the inclusion
here of bodies that are not proofread neurons. **FlyBrain does not choose between
them.** Until this is resolved, no statement about "the MaleCNS connectome has N
edges" should be made from either source without saying which is meant.

**Neuron counts differ three ways.** 166,691 (paper abstract), 166,391 (paper
results), 211,577 (annotation rows). The first two differ by 300 with no
explanation; the third counts all bodies including glia and fragments. The
annotation table's `status == "Traced"` count of 165,122 sits closest to the
published figure.

**Per-field annotation provenance is unknown.** The download page calls the
annotations *curated* and the nuclei-derived cell body annotations *manually
reviewed*, but does not state the method per field. **No annotation field is
treated by FlyBrain as a direct measurement.** Cell types in particular are
curated assignments, and must be reported as annotations, not observations.

**Gap junctions are neither confirmed nor excluded.** The table has no synapse
class column, and neuPrint exposes a single directed `ConnectsTo` relationship
with a weight. Whether electrical connections are present is unknown, so the
graph is a chemical-synapse graph *by assumption*.

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
import, validation, sparse graph construction, reports, and all unit tests — runs
with no token at all. No test in this repository makes a live API request.
