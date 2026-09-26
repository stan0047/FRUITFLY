"""Tests for the MaleCNS provenance record.

These tests do not touch the network, do not need a neuPrint token, and do not
read the 1.1 GB connectome file. They check the thing that matters most: that
biological status is only ever claimed when it was explicitly declared, and that
the unresolved questions stay recorded as unresolved.
"""

from __future__ import annotations

import pytest

from flybrain.data.mcns_provenance import (
    MCNS_ANNOTATION_FILE,
    MCNS_CITATION,
    MCNS_CITATION_PAPER,
    MCNS_CITATION_PREPRINT,
    MCNS_CONNECTOME_FILE,
    MCNS_DATASET,
    MCNS_FACTS,
    MCNS_FIELD_SEMANTICS,
    MCNS_LICENSE,
    MCNS_LICENSE_URL,
    MCNS_PUBLISHED_DOI,
    MCNS_SOURCE,
    MCNS_SOURCE_URL,
    MCNS_UNRESOLVED,
    MCNS_VISUAL_COMPANION_PAPER,
    mcns_provenance,
)
from flybrain.data.provenance import BIOLOGICAL_LABEL, DataSource, Provenance

DOCUMENTED_STATUSES = {"DOCUMENTED", "INFERRED", "UNRESOLVED", "OBSERVED"}


def test_identity_fields_match_the_official_dataset() -> None:
    assert MCNS_SOURCE == "Janelia MaleCNS"
    assert MCNS_DATASET == "Male CNS v1.0"
    assert MCNS_LICENSE == "CC-BY-4.0"
    assert MCNS_LICENSE_URL == "https://creativecommons.org/licenses/by/4.0/"
    assert MCNS_SOURCE_URL.startswith("https://male-cns.janelia.org/")
    assert MCNS_PUBLISHED_DOI == "10.1016/j.cell.2026.08.015"
    assert MCNS_CITATION.startswith("Berg, S. et al. (2026)")
    assert "Cell 189(18)" in MCNS_CITATION
    assert MCNS_CITATION == MCNS_CITATION_PAPER
    assert "bioRxiv" in MCNS_CITATION_PREPRINT
    assert MCNS_CITATION != MCNS_CITATION_PREPRINT, "the published version is the canonical citation"
    assert "Hoeller" in MCNS_VISUAL_COMPANION_PAPER


def test_filenames_match_the_files_the_user_downloaded() -> None:
    assert MCNS_ANNOTATION_FILE == "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    assert MCNS_CONNECTOME_FILE == "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


@pytest.mark.parametrize("field", sorted(MCNS_FIELD_SEMANTICS))
def test_every_field_fact_is_well_formed(field: str) -> None:
    fact = MCNS_FIELD_SEMANTICS[field]
    assert fact.field == field
    assert fact.status in DOCUMENTED_STATUSES
    assert fact.source.startswith("https://")
    assert fact.meaning.strip()
    # Quotes are kept short on purpose: a paraphrase plus a checkable fragment,
    # never a copied passage.
    assert len(fact.quote) < 200, "quotes must be short fragments, not copied passages"


@pytest.mark.parametrize("field", ["body_pre", "body_post", "weight", "bodyId"])
def test_core_fields_are_marked_documented(field: str) -> None:
    assert MCNS_FIELD_SEMANTICS[field].status == "DOCUMENTED", (
        f"{field} semantics were verified against the official documentation and must not be downgraded"
    )


def test_weight_is_documented_as_synapse_count_not_efficacy() -> None:
    meaning = MCNS_FIELD_SEMANTICS["weight"].meaning.lower()
    assert "synapse" in meaning
    assert "not a functional efficacy" in meaning


def test_granularity_is_recorded_as_aggregated() -> None:
    fact = MCNS_FACTS["granularity"]
    assert "AGGREGATED" in fact.meaning
    assert "R4" in fact.meaning, "must reference the spec requirement it satisfies"


def test_unresolved_questions_stay_unresolved() -> None:
    for key in ("minconf-0.5", "edge_count", "neuron_count", "annotation_provenance", "gap_junctions"):
        assert key in MCNS_UNRESOLVED, f"{key} must remain recorded as an open question"
        assert MCNS_UNRESOLVED[key].strip()


def test_visual_system_incompleteness_is_recorded() -> None:
    fact = MCNS_FACTS["visual_system_completeness"]
    assert fact.status == "DOCUMENTED"
    assert "photoreceptor" in fact.meaning.lower()
    assert "lamina" in fact.meaning.lower()


def test_provenance_factory_declares_biological() -> None:
    provenance = mcns_provenance(original_filename=MCNS_CONNECTOME_FILE)
    assert isinstance(provenance, Provenance)
    assert provenance.is_biological is True
    assert provenance.source is DataSource.EXTERNAL
    assert provenance.source_name == MCNS_SOURCE
    assert provenance.label == BIOLOGICAL_LABEL
    assert provenance.citation == MCNS_CITATION
    assert provenance.license == MCNS_LICENSE
    assert provenance.attributes["neuprint_dataset"] == "male-cns:v1.0"
    assert provenance.missing_fields() == [], "all provenance fields are supplied by the factory"


def test_provenance_factory_can_declare_non_biological() -> None:
    """A user can still say 'this is not biological data', e.g. for a mock file."""
    provenance = mcns_provenance(dataset_name="mock", biological_data=False)
    assert provenance.is_biological is False
    assert provenance.label != BIOLOGICAL_LABEL


def test_no_token_material_in_the_module() -> None:
    """The provenance record must never carry a credential."""
    import importlib

    # Resolve via importlib, not ``import ... as``: the package re-exports a
    # function called ``mcns_provenance``, which shadows the submodule name.
    module = importlib.import_module("flybrain.data.mcns_provenance")
    assert module.__file__ is not None
    text = open(module.__file__, encoding="utf-8").read().lower()
    for forbidden in ("neuprint_token", "api_key", "apikey", "password", "authorization:"):
        assert forbidden not in text, f"provenance module must not contain {forbidden!r}"
