"""Verified provenance and field semantics for the Janelia MaleCNS v1.0 dataset.

Everything in this module was checked against the dataset's own documentation on
2026-09-26. Each fact carries one of four statuses, because the difference
between "the provider says this", "we inferred it", and "nobody has said this"
is the difference between a measurement and a guess:

``DOCUMENTED``
    Stated by Janelia, the neuPrint user guide, or the published paper. The
    source URL and a short verbatim phrase are recorded so the claim can be
    re-checked.

``INFERRED``
    Consistent with the file's own structure and with the published description,
    but not stated outright. Usable, but must be reported as inference.

``UNRESOLVED``
    Not stated anywhere we checked. Recorded as a gap. **No value is invented
    to fill it**, and downstream code must treat it as unknown.

``OBSERVED``
    Measured by FlyBrain from the local file. Not a biological fact, a fact
    about the file.

Official sources
----------------
* Project and downloads: https://male-cns.janelia.org/download
* Project home:          https://male-cns.janelia.org/
* Release notes:         https://male-cns.janelia.org/release
* neuPrint user guide:   https://neuprint.janelia.org/public/neuprintuserguide.pdf
* Paper (published):     https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
* Preprint:              https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2
* neuPrint client docs:  https://connectome-neuprint.github.io/neuprint-python/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flybrain.data.provenance import DataSource, Provenance

__all__ = [
    "MCNS_CITATION",
    "MCNS_CITATION_PAPER",
    "MCNS_CITATION_PREPRINT",
    "MCNS_CONNECTOME_FILE",
    "MCNS_ANNOTATION_FILE",
    "MCNS_DATASET",
    "MCNS_FACTS",
    "MCNS_FIELD_SEMANTICS",
    "MCNS_LICENSE",
    "MCNS_LICENSE_URL",
    "MCNS_PREPRINT_DOI",
    "MCNS_PUBLISHED_DOI",
    "MCNS_PUBLICATION_DATE",
    "MCNS_RELEASE_DATE",
    "MCNS_SERVER",
    "MCNS_SOURCE",
    "MCNS_SOURCE_URL",
    "MCNS_UNRESOLVED",
    "MCNS_VISUAL_COMPANION_PAPER",
    "FieldFact",
    "mcns_provenance",
]

# ---------------------------------------------------------------- identity

MCNS_SOURCE = "Janelia MaleCNS"
MCNS_DATASET = "Male CNS v1.0"
MCNS_SOURCE_URL = "https://male-cns.janelia.org/download"
MCNS_RELEASE_DATE = "2026-06-08"  # docs/release.md: "v1.0 (June 8, 2026)"
MCNS_PUBLICATION_DATE = "2026-09-03"  # project news: "MaleCNS paper published"
MCNS_LICENSE = "CC-BY-4.0"
MCNS_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

MCNS_PUBLISHED_DOI = "10.1016/j.cell.2026.08.015"
MCNS_PREPRINT_DOI = "10.1101/2025.10.09.680999"
MCNS_CITATION_PAPER = (
    "Berg, S. et al. (2026). Sexual dimorphism in the complete connectome of the "
    "Drosophila male central nervous system. Cell 189(18). "
    f"https://doi.org/{MCNS_PUBLISHED_DOI}"
)
MCNS_CITATION_PREPRINT = (
    "Berg, S. et al. (2025). Sexual dimorphism in the complete connectome of the "
    f"Drosophila male central nervous system. bioRxiv. https://doi.org/{MCNS_PREPRINT_DOI}"
)
#: The canonical citation. The published version supersedes the preprint.
MCNS_CITATION = MCNS_CITATION_PAPER

#: Companion paper in the same Cell issue, on the topic this project needs next.
#: Recorded now so Phase 3 is designed against the real literature, not from memory.
MCNS_VISUAL_COMPANION_PAPER = (
    "Hoeller, J., Zhao, A., Nern, A., Rogers, E. M., Romani, S., & Reiser, M. B. (2026). "
    "The organization of visual pathways in the Drosophila brain. Cell 189(18). "
    "Companion article to Berg et al. 2026; journal DOI not independently verified by FlyBrain."
)

#: neuPrint endpoint and dataset identifier for the same data.
MCNS_SERVER = "https://neuprint.janelia.org"
MCNS_NEUPRINT_DATASET = "male-cns:v1.0"

MCNS_ANNOTATION_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
MCNS_CONNECTOME_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


@dataclass(frozen=True)
class FieldFact:
    """One verified claim about one field of the dataset."""

    field: str
    meaning: str
    status: str
    source: str
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "meaning": self.meaning,
            "status": self.status,
            "source": self.source,
            "quote": self.quote,
        }


_DOWNLOAD = "https://male-cns.janelia.org/download"
_USERGUIDE = "https://neuprint.janelia.org/public/neuprintuserguide.pdf"
_PAPER = f"https://doi.org/{MCNS_PREPRINT_DOI}"

#: Documented meaning of every field FlyBrain depends on.
MCNS_FIELD_SEMANTICS: dict[str, FieldFact] = {
    "body_pre": FieldFact(
        field="body_pre",
        meaning=(
            "Body (segment) identifier of the pre-synaptic neuron. One row is one directed "
            "chemical connection from this body to the body named by body_post."
        ),
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="segment-to-segment connection strengths for all segments in the dataset",
    ),
    "body_post": FieldFact(
        field="body_post",
        meaning="Body (segment) identifier of the post-synaptic neuron, the target of the connection.",
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="segment-to-segment connection strengths for all segments in the dataset",
    ),
    "weight": FieldFact(
        field="weight",
        meaning=(
            "Number of synapses on that connection, i.e. the synapse count from body_pre to "
            "body_post. Integer, positive. It is a structural count, not a functional efficacy."
        ),
        status="DOCUMENTED",
        source=_USERGUIDE,
        quote="The 'weight' is the total number of synapses from the presynaptic neuron to the postsynaptic neuron.",
    ),
    "bodyId": FieldFact(
        field="bodyId",
        meaning=(
            "Body (segment) identifier in the neuron annotation table. Joins to body_pre and "
            "body_post. One row per body in the dataset, not one row per proofread neuron: the "
            "table also contains glia, orphans and fragments."
        ),
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="Curated neuron annotations (classes, types, sides, etc.)",
    ),
    "class": FieldFact(
        field="class",
        meaning="Broad sensory or functional class of the neuron, e.g. visual, olfactory, gustatory.",
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="Curated neuron annotations (classes, types, sides, etc.)",
    ),
    "superclass": FieldFact(
        field="superclass",
        meaning=(
            "Finer functional grouping; values distinguish central-brain intrinsic "
            "(cb_intrinsic), optic-lobe intrinsic (ol_intrinsic), visual_projection, "
            "visual_centrifugal, ascending/descending neurons, and VNC compartments."
        ),
        status="INFERRED",
        source=_DOWNLOAD,
        quote="Curated neuron annotations (classes, types, sides, etc.)",
    ),
    "type": FieldFact(
        field="type",
        meaning=(
            "Cell type of the neuron, at the finest annotated granularity (11,691 types are "
            "reported for the dataset). Strongly populated with visual cell types."
        ),
        status="INFERRED",
        source=_PAPER,
        quote="we defined 11,691 unique cell types based on morphology and connectivity of individual neurons",
    ),
    "status": FieldFact(
        field="status",
        meaning=(
            "Reconstruction status of the body: Traced, Roughly traced, Assign, Anchor, Orphan, "
            "Glia, Unimportant. 'Traced' bodies are proofread neurons."
        ),
        status="INFERRED",
        source=_DOWNLOAD,
        quote="The v1.0 proofread neuron segmentation.",
    ),
    "somaSide": FieldFact(
        field="somaSide",
        meaning="Which side of the brain the soma sits on: L, R, or M for midline.",
        status="INFERRED",
        source=_DOWNLOAD,
        quote="Curated neuron annotations (classes, types, sides, etc.)",
    ),
    "somaNeuromere": FieldFact(
        field="somaNeuromere",
        meaning="Neuromere containing the soma, e.g. LB, T1, CG. Heavily null for most bodies.",
        status="INFERRED",
        source=_DOWNLOAD,
        quote="Curated neuron annotations (classes, types, sides, etc.)",
    ),
    "somaLocation": FieldFact(
        field="somaLocation",
        meaning=(
            "Three integers: the soma centroid in MaleCNS EM voxel coordinates. Voxels are 8 nm "
            "isotropic. Present for a subset of bodies only."
        ),
        status="INFERRED",
        source=_DOWNLOAD,
        quote="Male CNS EM coordinate space, with coordinates specified in 8nm units.",
    ),
}

#: Facts FlyBrain could not verify. These are gaps, not defaults.
MCNS_UNRESOLVED: dict[str, str] = {
    "minconf-0.5": (
        "The 'minconf-0.5' tag in the filenames is not defined on the download page and is not "
        "explained in the paper. The only 0.5 confidence cut-offs described in the paper apply to "
        "neurotransmitter assignment, not to synapse detection. What the 0.5 is a threshold on is "
        "therefore UNKNOWN to FlyBrain. Observation only: the local file retains connections with "
        "weight 1, so the filter is permissive."
    ),
    "edge_count": (
        "The paper states a connectome graph of '25.6M edges between 166,391 neurons' with no "
        "filter named in that sentence, but the local connectome-weights file contains 151,856,684 "
        "rows. The roughly six-fold difference is UNEXPLAINED. Candidate explanations include a "
        "minimum-synapse threshold applied in the paper's analysis, and the inclusion here of "
        "bodies that are not proofread neurons. FlyBrain does not choose between them."
    ),
    "neuron_count": (
        "The paper gives 166,691 neurons in the abstract and 166,391 in the results; the 300-neuron "
        "difference is not explained in the paper. The annotation table has 211,577 bodies, which "
        "includes glia, orphans and fragments, so it is not directly comparable."
    ),
    "annotation_provenance": (
        "The download page calls the annotations 'curated' and the nuclei-derived cell body "
        "annotations 'manually reviewed', but the method behind each individual field (class, "
        "type, superclass) is not stated per-field. Per-field provenance is therefore UNKNOWN, and "
        "no annotation field is treated by FlyBrain as a direct measurement."
    ),
    "gap_junctions": (
        "Whether the weights table contains only chemical synapses is NOT stated. The neuPrint data "
        "model exposes a single directed ConnectsTo relationship with a weight, and no synapse-class "
        "column is present in the local file. Electrical connectivity is therefore neither confirmed "
        "nor excluded."
    ),
}

#: Dataset-level facts, for reports and provenance sidecars.
MCNS_FACTS: dict[str, FieldFact] = {
    "granularity": FieldFact(
        field="granularity",
        meaning=(
            "AGGREGATED. One row per ordered body pair, with weight equal to the synapse count. "
            "This satisfies requirement R4 of docs/CONNECTOME_REQUIREMENTS.md."
        ),
        status="INFERRED",
        source=_DOWNLOAD,
        quote="segment-to-segment connection strengths for all segments in the dataset",
    ),
    "completeness": FieldFact(
        field="completeness",
        meaning=(
            "The provider describes this as the full connection graph for the dataset, restricted "
            "to segments that have at least one synapse. The synapse-level filter behind the "
            "'minconf-0.5' tag is not documented."
        ),
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="This is the full connection graph.",
    ),
    "stated_neurons": FieldFact(
        field="stated_neurons",
        meaning="166,691 neurons across brain and nerve cord, annotated with 11,691 cell types.",
        status="DOCUMENTED",
        source=_PAPER,
        quote="This contains 166,691 neurons spanning the brain and nerve cord",
    ),
    "stated_edges": FieldFact(
        field="stated_edges",
        meaning=(
            "25.6M edges between 166,391 neurons, as stated in the paper's results. See "
            "MCNS_UNRESOLVED['edge_count']: the local file has 151,856,684 rows."
        ),
        status="DOCUMENTED",
        source=_PAPER,
        quote="a connectome graph containing 25.6M edges between 166,391 neurons",
    ),
    "proofread_completion": FieldFact(
        field="proofread_completion",
        meaning=(
            "40.1% of synaptic connections have both the pre- and the post-synaptic site on a "
            "proofread neuron. Most of the raw graph is therefore not between proofread neurons."
        ),
        status="DOCUMENTED",
        source=_PAPER,
        quote="the fraction of synaptic connections for which both pre- and postsynaptic sites belong to a proofread neuron is ... 40.1%",
    ),
    "visual_system_completeness": FieldFact(
        field="visual_system_completeness",
        meaning=(
            "The paper documents that the visual periphery is the least complete part of this "
            "dataset: some photoreceptors are missing or incompletely reconstructed and their "
            "columns were assigned from anatomical markers alone; the lamina had poor synapse-"
            "detector recall; the left optic lobe and the VNC were not proofread top-down. This is "
            "a material limitation for any visual-pathway experiment on MaleCNS."
        ),
        status="DOCUMENTED",
        source=_PAPER,
        quote="we found some photoreceptors to be missing or incompletely reconstructed",
    ),
    "license": FieldFact(
        field="license",
        meaning="The Male CNS dataset is licensed CC-BY 4.0.",
        status="DOCUMENTED",
        source=_DOWNLOAD,
        quote="The Male CNS dataset is licensed under CC-BY.",
    ),
}


def mcns_provenance(
    dataset_name: str = MCNS_DATASET,
    dataset_version: str = "v1.0",
    original_filename: str | None = None,
    notes: str = "",
    biological_data: bool = True,
) -> Provenance:
    """Provenance for a dataset the user obtained from Janelia MaleCNS v1.0.

    The dataset is treated as biological **only** when this factory is used, which
    is the explicit declaration the Phase 2 spec requires. A file merely named
    ``...-minconf-0.5.feather`` does not become biological on its own.
    """
    return Provenance(
        source_name=MCNS_SOURCE,
        dataset_name=dataset_name,
        source=DataSource.EXTERNAL,
        dataset_version=dataset_version,
        source_url=MCNS_SOURCE_URL,
        license=MCNS_LICENSE,
        citation=MCNS_CITATION,
        original_filename=original_filename,
        biological_data=biological_data,
        notes=notes or (
            "Janelia MaleCNS v1.0 bulk connectome, obtained by the user from the official "
            "download page. Field meanings verified against the dataset documentation; see "
            "flybrain.data.mcns_provenance for per-field status and unresolved questions."
        ),
        attributes={
            "license_url": MCNS_LICENSE_URL,
            "preprint_doi": MCNS_PREPRINT_DOI,
            "release_date": MCNS_RELEASE_DATE,
            "publication_date": MCNS_PUBLICATION_DATE,
            "neuprint_server": MCNS_SERVER,
            "neuprint_dataset": MCNS_NEUPRINT_DATASET,
            "visual_companion_paper": MCNS_VISUAL_COMPANION_PAPER,
        },
    )


def semantics_table() -> list[dict[str, Any]]:
    """Flat list of every verified field fact, for reports."""
    return [fact.to_dict() for fact in MCNS_FIELD_SEMANTICS.values()]


def dataset_facts_table() -> list[dict[str, Any]]:
    """Flat list of dataset-level facts, for reports."""
    return [fact.to_dict() for fact in MCNS_FACTS.values()]
