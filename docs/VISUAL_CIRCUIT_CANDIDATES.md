# Visual circuit candidates, grounded in Hoeller et al. 2026 and MCNS v1.0

**Status:** literature and feasibility analysis. **No pathway is selected. No
simulation code is written.** This document exists so that the choice of circuit
is made deliberately, against measured numbers, rather than by accident.

**Primary source:** Hoeller, J., Zhao, A., Nern, A., Rogers, E. M., Romani, S., &
Reiser, M. B. (2026). *The organization of visual pathways in the Drosophila
brain.* **Cell 189(18): 5552–5570.e10.** DOI
[10.1016/j.cell.2026.08.014](https://doi.org/10.1016/j.cell.2026.08.014). Open
access. Preprint: [10.64898/2025.12.22.696097](https://doi.org/10.64898/2025.12.22.696097).
Read 2026-09-26, published version canonical.

**Target pipeline:** `camera → visual processing → motion/spatial representation
→ navigation controller → LEFT / FORWARD / RIGHT`

---

## 1. The single most important thing to understand about this paper

**The paper does not classify visual pathways by function.** It classifies them by
**propagated input composition** — an 8-dimensional vector of how much each visual
input channel contributes — and then attaches a functional label *as a hypothesis*
based on whichever channel dominates.

The paper is explicit about both halves of this:

> "Pathway classes constrain which visual features may reach each VPN, **not what
> it computes**. VPNs within a class draw on similar input mixtures but, as
> distinct cell types, likely perform different computations."

> "the VCBN pathway classes, feature selectivity, and spatial maps **should be
> interpreted as transparent, testable hypotheses that await targeted
> physiological and behavioral experiments**."

So `oc9` is not "the motion pathway". It is *the class whose propagated input is
dominated by L1/L2*, and the paper notes that many known motion-sensitive neurons
happen to fall in it — which it presents as a **recovered correspondence**, i.e.
a validation, not a definition.

**Consequence for this project:** the paper supplies the *inputs*, the *depth
ordering*, the *projection topology* and the *spatial maps*. It does **not**
supply a motion pathway, a direction-selectivity pathway, or a navigation
pathway. Those must come from elsewhere, and that is stated explicitly in §3.

## 2. What the paper provides, precisely

| Contribution | Definition, in the paper's terms |
| --- | --- |
| **8 visual input channels** | L1 (ON), L2 (OFF), L3 (luminance), R7, R8 (short/long wavelength), R7d, R8d (dorsal-rim polarisation), HB eyelet (non-spatial, circadian) |
| **Columnar sampling grid** | 892 medulla columns, typically five inputs each (L1, L2, L3, R7, R8); dorsal-rim columns substitute R7d/R8d |
| **Layer** | Average number of synaptic steps from a visual input, under a deterministic cumulative-activation flow model. One free parameter (saturation threshold 0.1), calibrated on T4 |
| **Direction per edge** | feedforward > 0.5 layers deeper; lateral ±0.5; feedback shallower |
| **VIC** | Visual Input Contribution: sum of effective weights from all visual inputs. Explicitly *"a relative, layer-dependent score rather than an absolute fraction of visual drive"* |
| **Propagated ARF** | Receptive-field estimate: column-by-column summed effective weight from columnar inputs, fitted with an anisotropic 2D Gaussian |
| **VPN** | Visual projection neuron: arbor in optic lobe, axon to central brain. ~4,350 right-side neurons / ~350 cell types |
| **VCBN** | Visual central brain neuron: VIC > 5×10⁻⁴. 5,527 cell types, ~11k neurons, 53% of central-brain types |
| **VCN** | Visual centrifugal neuron: feedback relay, right optic lobe, 264 neurons / 104 types |
| **9 optic-lobe classes** | oc1–oc9, clustered from 352 right-side VPN types |
| **10 central-brain classes** | cc1–cc10, clustered from VCBN types |
| **Depth** | Optic lobes form an emergent 3-layer architecture; central brain adds layers 3–5; deepest cell at layer 6; whole brain reached within ~5 synaptic steps |

### Validated against physiology (the paper's own checks)

| Check | Result |
| --- | --- |
| ON/OFF contrast selectivity from L1/L2 relative weights | 45 of 54 cell types correct, *p* = 3.9×10⁻⁶ |
| Propagated ARF size vs measured RF size | Pearson *r* = 0.96 (direct ARF: 0.84); aspect ratio only 0.41 |
| Central-brain ON/OFF predictions | 8/8 curated VCBN entries correct, *p* = 0.0039 |
| VCBN identification | all 66 cell types with documented visual responses exceeded threshold; 4 non-responsive KCab fell below |
| Left/right consistency | propagation from the left lobe "produced nearly mirror-symmetric patterns" |
| Portability | FlyWire female reproduces ON/OFF polarity and VCBN estimates (45/53) |
| Robustness | layer assignments vs threshold 0.05/0.1/0.2: Spearman 0.99 / 1.00 / 0.98; vs a >2-synapse cutoff: 0.98; vs synthetic R1–R6 seeding: 0.94 |

## 3. What the paper does NOT provide — and where that must come from

Stated plainly, because these are exactly the gaps that matter for a
motion-to-navigation project.

| Needed | In Hoeller 2026? | Where it actually comes from |
| --- | --- | --- |
| **A motion-detection pathway class** | **No.** Motion appears only as *prior known circuitry* used to calibrate the threshold and as a sanity check | Takemura et al. 2013 *Nature*; Behnia et al. 2014; Pinto-Teixeira et al. 2015; Apolonio et al. 2017; Gruntman et al. 2019 |
| **Direction-selectivity predictions** | **No.** The paper says: *"As more VIN computations such as the directional selectivity of T4/T5 become characterized, derived features can be propagated in the same way"* — i.e. explicitly future work | Prior physiology only |
| **A navigation / compass / head-direction class** | **No.** No such class exists in the paper | The paper only *names* the ellipsoid body as "central to navigation" and notes convergence zones (EB, PB, LAL) |
| **Absolute central-brain neuron counts** | **No.** Only the derived VCBN count | — |
| **Per-class node/edge counts** | **No** in text; only in figure panels and Tables S1/S3 | Supplementary data files, not verified cell-by-cell here |

**This is the key literature gap for Phase 3.** Hoeller et al. give a rigorous,
validated *spatial and depth* framework but no motion pathway definition. If the
project wants motion, the motion circuit has to be taken from the older
connectomic papers and must be cited to them — **not** to Hoeller et al. The one
chain the paper does reference is T4 being *"widely described as being two
synapses downstream of L1 and L3"*, and its own layer analysis is consistent with
that (T4 at layer 2, VS at layer 3), but the paper presents this as prior
knowledge, not as its finding.

---

## 4. Paper cell types → MCNS v1.0 label availability

Checked verbatim against the `type` field of
`body-annotations-male-cns-v1.0-minconf-0.5.feather` by
`scripts/check_mcns_cell_types.py`. **No similar name is ever substituted.**

Totals: 211,577 bodies · 11,751 distinct type labels · 164,506 bodies with a
type · 166,700 bodies with a superclass (the official neuron predicate).

### 4.1 Exact matches — 55 labels, ~52,300 bodies

| Literature group | Labels (bodies) | superclass | somaSide | flywireType |
| --- | --- | --- | --- | --- |
| **Visual inputs** | L1 1,776 · L2 1,779 · L3 1,772 · L4 1,770 · L5 1,787 · R7d 82 · R8d 76 · R1-R6 3,377 | `ol_intrinsic` (lamina) · `ol_sensory` (R7d/R8d/R1-R6) | present, L/R balanced | 100% |
| **Lamina, other** | C2 1,745 · C3 1,779 · T1 1,777 | `ol_intrinsic` | present | 100% |
| **Motion / direction** | T4a 1,684 · T4b 1,690 · T4c 1,778 · T4d 1,709 · T5a 1,664 · T5b 1,715 · T5c 1,720 · T5d 1,620 · VS 18 | `ol_intrinsic`; **VS is `visual_projection`** | present | 100% |
| **Medulla, columnar** | Mi1 1,773 · Mi4 1,772 · Mi9 1,775 · Tm1 1,777 · Tm2 1,766 · Tm3 2,054 · Tm4 1,670 · Tm5a 624 · Tm5b 522 · Tm5c 750 · Tm9 1,771 · Tm20 1,762 | `ol_intrinsic` | present | 100% |
| **Lobula Dm/Pm** | Dm4 99 · Dm8a 572 · Dm8b 532 · Dm9 273 · Dm11 158 · **Pm12 4** | `ol_intrinsic` | present | 100% |
| **Lobula → central** | LC4 126 · LC6 124 · LPLC1 134 · LPLC2 185 · LPLC4 97 · **LPT23 6** · LC16 182 · LC22 73 | `visual_projection` | present | 100% |
| **Central / projection** | MeTu2a 70 · MeTu3b 80 · MeTu3c 173 · MeVP11 55 · **MeVPMe12 4** · MeVP57 1 · MeVP64 1 · aMe12 6 · pC1_2a 4 · pC1_5b 4 · Nod5 2 · ER2_c 20 · ER4d 26 · DNp03 2 · PVLP133 24 · DNg13 2 · DNge104 2 | mixed `visual_projection` / `cb_intrinsic` / `descending_neuron` | present | 100% except MeVP64, pC1_2a, pC1_5b (0%) |

### 4.2 Present as sets — the literature label is a group MCNS subdivides

| Literature label | MCNS subtypes | Bodies |
| --- | --- | --- |
| **T4** | T4a, T4b, T4c, T4d | 6,861 |
| **T5** | T5a, T5b, T5c, T5d | 6,719 |
| **Tm1-Tm2** | Tm1, Tm2 | 3,543 |

Treating `T4` as a single MCNS type would be a silent misrepresentation. Note
also that Hoeller et al. discuss only **T4a** by name; T4b/c/d are not mentioned
individually anywhere in the paper.

### 4.3 NOT present — flagged, not substituted

| Paper label | Bodies | Near-miss strings found (**NOT applied**) | Consequence |
| --- | --- | --- | --- |
| **R7** | 0 | `ExR7`, `R7R8_unclear`, `R7_unclear` | The short-wavelength colour input channel is **unusable**. A human must adjudicate which, if any, is the right label. |
| **R8** | 0 | `ExR8`, `R7R8_unclear`, `R8_unclear` | The long-wavelength colour input channel is **unusable** |
| **HB** (eyelet) | 0 | `HBeyelet` | The circadian / non-spatial channel is **unusable** as labelled |
| **KCab** | 0 | `KC`, `KCab-c`, `KCab-m` | Only used by the paper as a negative control; not needed |
| **PVLP008_a** | 0 | `PVLP008_a1`, `PVLP008_a2`, `PVLP008_a3` | Appears only in a supplementary ARF decomposition; not needed |

**Net effect:** the three input channels that survive cleanly in MCNS are
**L1 (ON), L2 (OFF) and L3 (luminance)**. Every colour, polarisation and circadian
channel the paper defines is either absent or label-ambiguous. For a
motion/spatial project this is fortunate rather than lucky — L1/L2/L3 are exactly
the contrast channels the paper validates its ON/OFF predictions on.

### 4.4 A quantified data gap worth stating

MCNS contains **3,377 `R1-R6` bodies**. The fly has roughly 800 ommatidia per eye
× 6 R1-R6 photoreceptors × 2 eyes ≈ 9,600. So **~35% of R1-R6 are present**.

This independently corroborates the paper's own reason for excluding R1–R6:
*"large parts of both laminae, and as a result many R1-6 photoreceptors, are
missing in the EM dataset (we exclude all lamina connectivity data from our
analysis as it is less reliable)."* Hoeller et al. also state that some
photoreceptors are *"missing or incompletely reconstructed"* and that their
columns were *"assigned based on the anatomical markers alone"*, and that
photoreceptors were excluded from their Fig. 4c analysis *"as many are missing in
the male CNS"*.

**Any candidate circuit must start at L1/L2/L3, not at the photoreceptors.**

---

## 5. Candidate circuits, classified by role

**These are not ranked.** Each is a different functional commitment. The choice
belongs to the project, and different choices answer different research
questions.

Every candidate draws on the same measured substrate: 892 medulla columns, ~5
columnar inputs each, MCNS labels as audited in §4.

---

### ROLE: MOTION — ON contrast channel

| Element | Detail |
| --- | --- |
| **Chain** | L1 → Mi1 → T4a/b/c/d → (VS, layer 3) |
| **Biology source** | **Prior literature**, not Hoeller et al. Hoeller et al. *reference* T4 as *"two synapses downstream of L1 and L3"* and independently place T4 at layer 2 and VS at layer 3, which is consistent. The L1→Mi1→T4 chain itself is Takemura 2013 / Behnia 2014 / Pinto-Teixeira 2015. |
| **Documented role** | ON-motion pathway. T4 is the first direction-selective neuron; T4a is the only subtype Hoeller et al. name, and it *"participate[s] mainly in oc8 and oc9"* |
| **MCNS availability** | L1 1,776 · Mi1 1,773 · T4a-d 6,861 · VS 18 — all EXACT |
| **Columnar** | Yes. L1, Mi1 and T4 are all columnar, so the circuit tiles cleanly |
| **Node estimate** | ~7.7 nodes/column measured (see §6) → 100 columns ≈ 770 nodes; full 892 columns ≈ 6,900 |
| **Simulation concern** | Direction selectivity is **not** in the connectome. It emerges from the L1→Mi1→T4 *microcircuit* including its inhibitory and lateral components. A pure feedforward LIF chain will not reproduce direction tuning. This is the single biggest scientific risk in the project. |

### ROLE: MOTION — OFF contrast channel

| Element | Detail |
| --- | --- |
| **Chain** | L2 → Tm1 / Tm2 → T5a/b/c/d → (VS) |
| **Biology source** | Prior literature. Hoeller et al. reference *"VPNs downstream of both T4 and T5"* for LPLC2 |
| **Documented role** | OFF-motion pathway |
| **MCNS availability** | L2 1,779 · Tm1 1,777 · Tm2 1,766 · T5a-d 6,719 — all EXACT |
| **Node estimate** | Same ~7.7 nodes/column scaling |
| **Simulation concern** | Same as ON. Also: Tm1-Tm2 is a literature group that MCNS splits — the choice between them is a modelling decision, not a lookup. |

### ROLE: DIRECTION — lobula plate output

| Element | Detail |
| --- | --- |
| **Chain** | T4/T5 → lobula plate tangential cells (LPTs) → central brain |
| **Biology source** | Hoeller et al. place *"nearly all of the large and small lobula plate motion-sensitive neurons LPTs, LPCs, and LLPCs"* in **oc9**, and note *"T4a neurons participate mainly in oc8 and oc9"* |
| **Important** | oc9 is defined by **L1/L2-dominated input**, not by motion. The paper presents the motion-neuron membership as a **recovered correspondence** |
| **MCNS availability** | LPT23 present but only **6 bodies**. The paper's bare "LPTs"/"LPCs"/"LLPCs" groups are **not** resolvable to specific MCNS labels without a crosswalk we have not built |
| **Assessment** | **Blocked on labels.** Not usable until the LPT/LPC/LLPC group names are mapped to MCNS types by hand. Recorded as a gap, not a candidate. |

### ROLE: SPATIAL ACUITY — ARF-based receptive field

| Element | Detail |
| --- | --- |
| **Chain** | columnar inputs → columnar relays → lobula → central brain; receptive field measured as propagated ARF |
| **Biology source** | Hoeller et al. core contribution. ARF expands *"at least 2-fold"* from VIN to VPN, sometimes *"exceeding 60-fold"*; VPN→central targets have *"a modal value of 16-fold"* |
| **Documented role** | Multi-resolution spatial sampling; *"within each of the optic lobe pathway classes, ARF sizes span more than an order of magnitude"* |
| **MCNS availability** | Everything the paper uses is present |
| **What this gives the project** | A **principled, published spatial-representation scheme** with a validated ground truth (*r* = 0.96 against measured RF size). This is the most directly reusable quantitative result in the paper for a camera-driven model |
| **Node estimate** | Independent of hop count — a projection cell plus its receptive field. A few hundred VPN-type cells suffice |
| **Simulation concern** | The ARF is a *static* anatomical/connectomic estimate. A dynamic model must *reproduce* receptive-field emergence; the paper does not provide a dynamics model. |

### ROLE: VISUAL PROJECTION — contrast channels to central brain

| Element | Detail |
| --- | --- |
| **Chain** | L1/L2 → … → LC4, LC6, LPLC1/2/4 → central brain |
| **Biology source** | Hoeller et al. LC4: *"measured OFF sensitivity is consistent with its strong L2-associated OFF-pathway inputs"*. LPLC2: *"more balanced effective weights from L1, L2, and L3 as expected of VPNs downstream of both T4 and T5"* |
| **MCNS availability** | LC4 126 · LC6 124 · LPLC1 134 · LPLC2 185 · LPLC4 97 — all EXACT, all `visual_projection` |
| **Attraction** | **Small.** 927 bodies total across these five types. This is the most compact route from optic lobe to central brain that the paper documents |
| **Node estimate** | 927 projection neurons + their upstream columnar relays. At ~7.7 nodes/column, 200 columns feeding these VPNs ≈ 1,500–3,000 nodes |
| **Simulation concern** | LC6's defining feature in the paper is **>25% effective weight from R7/R8** — channels that are *absent from MCNS*. LC6 would lose its documented specialisation. LC4 and LPLC2 do not depend on R7/R8 and are cleaner candidates. |

### ROLE: VISUAL PROJECTION — luminance

| Element | Detail |
| --- | --- |
| **Chain** | L3 → … → MeVP11, MeVPMe12, oc6/oc7 → cc1–cc4 |
| **Biology source** | Hoeller et al.: *"oc6-7 (L3) for luminance"*; MeVP11 is *"the early MeVP11 in oc7"*; MeVPMe12 at layer 2 in oc7 with a large ARF |
| **MCNS availability** | L3 1,772 · MeVP11 55 · MeVPMe12 only **4 bodies** |
| **Assessment** | MeVPMe12 is too small to be a useful population. MeVP11 (55) is marginal |

### ROLE: CENTRAL INTEGRATION — ring attractor / compass

| Element | Detail |
| --- | --- |
| **Cell types** | ER2_c 20 · ER4d 26 · Nod5 2 · (EB, NO named as regions) |
| **Biology source** | Hoeller et al. do **not** define a compass class. They report ER2_c and ER4d *"fell close to the equality line"* in a predicted-vs-measured spatial comparison, and that noduli show *"strong, region-wide spatial biases"* |
| **MCNS availability** | Present but very small: 20 + 26 + 2 bodies |
| **Honest assessment** | A ring attractor is a *known* fly circuit from a **different** literature (Giraldo et al. 2018; Green et al. 2019; Hulse et al. 2026 preprint on ring-attractor dynamics). Hoeller et al. supply no attractor analysis. This candidate rests on other papers, and those must be read before it is designed. |
| **Simulation concern** | A ring attractor needs **recurrent** dynamics with tuned connectivity. Hoeller et al. explicitly *remove* recurrence for their own analysis: their trimmed graph *"neglects the contributions of recurrence, particularly inhibitory feedback"*. So the paper's framework is structurally unable to certify a recurrent attractor. |

### ROLE: DESCENDING — behavioural readout

| Element | Detail |
| --- | --- |
| **Cell types** | DNp03 2 · DNg13 2 · DNge104 2 |
| **Biology source** | Janelia's own project page describes a visual-to-motor example pathway running *"from R1-R6 visual neurons to the DNg13 motor neuron"*. Since R1-R6 is only ~35% present, that specific chain is **not** reconstructable from MCNS. Hoeller et al. do not analyse descending control. |
| **MCNS availability** | 2 bodies each — a bilateral pair, i.e. a single cell type per side |
| **Assessment** | Too small to form a population, but **exactly right** as a *readout label*: a target cell type whose activity could be compared against a navigation decision. |

---

## 6. Measured computational budget

These are **engineering simulation budgets for a free Colab T4**. They are **not
biological boundaries**, and nothing in the fly recognises them.

### Measured growth rates (bounded reads, top-2M-edge prefix, 172,498 nodes)

Sampled L1 bodies as a proxy for medulla columns (L1 is columnar, ~1 body per
column per eye), 1-hop downstream:

| Columns (L1 bodies) | Nodes after 1 hop | Nodes/column |
| --- | --- | --- |
| 50 | 391 | 7.8 |
| 100 | 773 | 7.7 |
| 200 | 1,533 | 7.7 |

Depth cost, 50 columns, downstream:

| Hop | Nodes | Ratio |
| --- | --- | --- |
| 1 | 391 | — |
| 2 | 2,962 | 7.6× |
| 3 | 15,685 | 5.3× |

**Each hop multiplies the node count by roughly 5–8×.** Hop count, not cell type,
is what determines whether a circuit fits a budget.

### Budget mapping

| Budget | Nodes | Fits | Notes |
| --- | --- | --- | --- |
| **small** | < 1,000 | ~100 columns at 1 hop (≈770) | Enough for a columnar contrast relay. Not enough for depth. |
| **medium** | 1,000–10,000 | 100–200 columns at 1–2 hops; **or** the full 892 columns at 1 hop (≈6,900); **or** ~900 projection-only cells | The most useful range. Contains the entire contrast→projection route. |
| **large** | 10,000–50,000 | 50 columns at 3 hops (15,685); ~200 columns at 2 hops (est. ≈12,000) | Depth becomes available, but recall this is still a **weight-truncated** prefix. |

**Sparse-storage context:** the 2M-edge prefix is a 172,498-node graph and a dense
float32 matrix would be 119 GB. All measurements above use CSR only.

## 7. MEASURED / DERIVED / ASSUMED

The distinctions this project is built on, restated for the eventual circuit.

**MEASURED** — read out of the EM reconstruction, provided by the dataset:
- Directed body→body connectivity, at synapse resolution.
- `weight` = **number of synapses** from `body_pre` to `body_post` (documented).
- Body identity, `superclass`, `somaSide`, soma coordinates (8 nm voxels).
- Neuron `status` (Traced / Orphan / Glia / …).

**DERIVED** — computed or assigned by the dataset authors or by this project:
- **All cell-type, class and superclass labels.** Curated assignments, not
  measurements. Per-field method is undocumented. The paper's own functional
  labels are explicitly *"testable hypotheses"*.
- The paper's **layer** assignments, **VIC**, **effective weight** and **ARF** — all
  outputs of a specific algorithm with one calibrated free parameter (0.1, fitted
  on T4). The paper reports Spearman 0.97–0.99 sensitivity to that parameter.
- **This project's** columnar sampling, node budgets, hop limits, and every
  coverage percentage.
- Truncation by the weight-sorted prefix: a prefix-limited neighbourhood is
  **biased against weak connections** and is not a random sample.

**ASSUMED** — supplied by this project because no data supports it:
- **Synapse count ≠ synaptic efficacy.** Converting `weight` into a synaptic
  conductance or efficacy is an **explicit modelling choice**, not a measurement.
  Any such mapping must be a flagged, documented function with a stated rationale.
- **All LIF parameters.** Membrane time constant, threshold, reset, refractory
  period, leak — generic. Unless a future source supplies real values for real
  neurons, every one is an assumption.
- **The choice of which cell types form a circuit.** A curated list of types is a
  modelling decision, not a fact about the fly.
- **The stimulus model.** Driving the circuit with synthetic image motion is a
  modelling choice; Hoeller et al. provide receptive-field *predictions*, not a
  stimulus set.
- **Direction selectivity as a simulable property** is *not* provided by the
  connectome. Whether a given simplified circuit reproduces it is an open
  experimental question, and a negative result would be scientifically
  informative.

## 8. Major biological uncertainties

1. **The motion pathway is not in Hoeller et al.** Using `oc9` as "the motion
   pathway" would misrepresent the paper. The ON/OFF motion chains must be cited
   to Takemura 2013 / Behnia 2014 / Apolonio 2017 / Gruntman 2019, and those papers
   have not been read in this phase.
2. **Colour and circadian input channels are unavailable in MCNS** (R7, R8, HB
   absent or label-ambiguous). Any candidate leaning on spectral or polarisation
   computation is blocked until the `R7_unclear` / `ExR7` / `HBeyelet` labels are
   adjudicated by a human against the paper.
3. **The photoreceptor input stage is ~65% missing** (3,377 of ~9,600 R1-R6). The
   circuit must start at L1/L2/L3, and no claim can be made about retinal
   processing.
4. **Lamina connectivity is excluded by the paper** as *"less reliable"*, yet
   L1/L2/L3 are the project's only clean inputs. Their incoming synapses are the
   least trustworthy part of the input stage — the paper's own caveat.
5. **L5 is a documented failure case.** Hoeller et al.: *"Many of the mismatches
   involve L5, whose contributions to visual processing remain underexplored."*
   L5 is in our seed set from the earlier feasibility work; it should probably be
   dropped or treated separately.
6. **The paper's framework is feedforward by construction.** It trims lateral and
   feedback edges, so it cannot certify any recurrent computation. A ring-attractor
   or compass candidate is outside what this paper can support.
7. **Per-class membership is in supplementary tables** (Table S1 right-OL
   inventory, Table S3 VCBN inventory, Data S1–S6) that have **not** been
   verified cell-by-cell here. Any claim of the form "type X is in class oc4"
   is currently unsupported.
8. **The `Pm12` (4 bodies), `LPT23` (6), `aMe12` (6), `Nod5` (2) populations are
   too small** to serve as circuit layers in a simulation.
9. **Right/right-only bias.** The paper traces from the *right* optic lobe. Any
   candidate should state which side it uses and why; MCNS `somaSide` supports
   both.

## 9. What to decide before implementation

No ranking is offered. These are the open decisions, in dependency order.

1. **Which functional role is the project actually about?** MOTION (needs prior
   literature, carries the direction-selectivity risk), SPATIAL ACUITY (best
   supported by the paper, has validated ground truth), VISUAL PROJECTION (most
   compact), or CENTRAL INTEGRATION (outside this paper's scope).
2. **Which side?** Right (paper convention) or both.
3. **How many columns?** 50 / 100 / 200 / 892. This single number sets the node
   budget and is the main cost lever.
4. **How many hops?** Each hop costs 5–8×.
5. **Start at L1/L2/L3** (only viable input stage) or invest in adjudicating the
   R7/R8/HB labels first.
6. **What is the readout?** A descending cell type as a sparse behavioural label
   (DNg13, DNp03), or a population of central-brain projection cells.
7. **Read the motion-pathway prior literature** (Takemura 2013, Behnia 2014,
   Apolonio 2017, Gruntman 2019) before committing to a MOTION candidate. Without
   it, any motion circuit is invented rather than grounded.
8. **Decide the synapse-count → weight mapping explicitly**, and record it as an
   assumption with a stated rationale.

## 10. Related documents

- `docs/MCNS_DATA_SEMANTICS.md` — verified dataset semantics, the 25.6M vs 151.86M
  reconciliation, seeded-neighborhood measurements, provenance.
- `docs/CONNECTOME_REQUIREMENTS.md` — the R1–R6 / P1–P4 / O1–O6 data requirements
  this candidate set is checked against.
- `scripts/check_mcns_cell_types.py` — reproduces the §4 audit.
- `scripts/explore_seeded_neighborhood.py` — reproduces the §6 measurements.
