"""Tests for bounded seeded connectivity exploration.

These run entirely on a small hand-made Feather fixture. They never touch the
1.1 GB MaleCNS file, never make a network request, and never need a token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybrain.data.seeded_neighborhood import (
    PrefixIndex,
    SeedSet,
    build_prefix_index,
    explore,
    load_annotation_table,
    load_seed_file,
    seeds_from_types,
)

FIXTURES = Path(__file__).parent / "fixtures"
EDGES = FIXTURES / "sample_mcns_edges.feather"
NODES = FIXTURES / "sample_mcns_nodes.feather"


@pytest.fixture(scope="module")
def index() -> PrefixIndex:
    return build_prefix_index(EDGES, max_edges=10_000)


@pytest.fixture(scope="module")
def annotations() -> dict[str, np.ndarray]:
    return load_annotation_table(NODES, ("type", "superclass", "status"))


# ------------------------------------------------------------------- index


def test_prefix_index_reads_bounded_batches(index: PrefixIndex) -> None:
    assert index.record_batches_total == 3
    assert index.edges_read == 158
    assert index.record_batches_read == 3
    assert index.edges_after_threshold == 158
    assert index.truncated is False
    assert index.num_nodes == 40


def test_prefix_index_node_ids_are_sorted(index: PrefixIndex) -> None:
    """Regression: factorize returns uniques in appearance order, not sorted.

    An unsorted node_ids silently breaks every searchsorted lookup, which made
    every seed miss. Ascending order is part of the contract.
    """
    assert np.all(np.diff(index.node_ids) > 0)
    found = index.index_of([int(index.node_ids[0]), int(index.node_ids[-1])])
    assert found.size == 2


def test_prefix_index_is_sparse_and_reports_dense_cost(index: PrefixIndex) -> None:
    import scipy.sparse as sp

    assert sp.issparse(index.csr)
    assert sp.issparse(index.weights)
    assert index.dense_bytes_if_materialised() == 40 * 40 * 4


def test_prefix_index_row_budget_is_respected() -> None:
    partial = build_prefix_index(EDGES, max_edges=60)
    assert partial.edges_read == 60
    assert partial.record_batches_read == 2
    assert partial.record_batches_untouched == 1
    assert partial.truncated is True
    assert partial.edges_after_threshold == 60


def test_minimum_synapse_count_threshold_filters_and_reports() -> None:
    filtered = build_prefix_index(EDGES, max_edges=10_000, min_weight=20)
    assert filtered.min_weight == 20
    assert filtered.edges_after_threshold < filtered.edges_read
    assert filtered.weights.data.min() >= 20


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        build_prefix_index(FIXTURES / "nope.feather", max_edges=10)
    with pytest.raises(ValueError):
        build_prefix_index(EDGES, max_edges=0)


# ------------------------------------------------------------------- seeds


def test_seeds_from_types_selects_exact_matches(annotations: dict[str, np.ndarray]) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2"])
    assert len(seeds) == 18
    assert seeds.body_ids.size == 18
    assert np.all(np.diff(seeds.body_ids) > 0), "seed ids must be sorted and unique"
    assert "type" in seeds.origin


def test_unlabelled_neurons_are_never_seeded() -> None:
    """A neuron with a null type label must not be selected by a type query."""
    seeds = seeds_from_types(NODES, ["L1", "L2", "Tm3", "Mi1"])
    assert len(seeds) == 30, "the 10 null-typed rows must be excluded"
    unlabelled = 10
    assert len(seeds) + unlabelled == 40


def test_unknown_type_yields_no_seeds() -> None:
    assert len(seeds_from_types(NODES, ["NoSuchType"])) == 0


def test_seed_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text("10005\n10001\n10003\n", encoding="utf-8")
    seeds = load_seed_file(path)
    assert list(seeds.body_ids) == [10001, 10003, 10005]
    assert seeds.origin == "seeds.txt"


def test_seed_file_csv_with_id_column(tmp_path: Path) -> None:
    path = tmp_path / "seeds.csv"
    path.write_text("bodyId\n10002\n10004\n", encoding="utf-8")
    assert list(load_seed_file(path).body_ids) == [10002, 10004]


def test_seed_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_seed_file(tmp_path / "absent.txt")


# ----------------------------------------------------------------- explore


def test_zero_hops_returns_seeds_only(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1"])
    result = explore(index, seeds, hops=0, direction="downstream", max_nodes=1000, max_edges=1000)
    assert result.final_nodes == len(seeds)
    assert result.final_edges == 0
    assert result.stages[0]["stage"] == "seed"


def test_one_hop_expands(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1"])
    result = explore(index, seeds, hops=1, direction="downstream", max_nodes=10_000, max_edges=10_000)
    assert result.final_nodes > len(seeds)
    assert result.final_edges == result.stages[-1]["cumulative_edges"]
    # The run completed its hop budget without hitting a node or edge budget,
    # so nothing bound it. Growth stopped because the frontier ran out.
    assert result.budget_binding == "none"


@pytest.mark.parametrize("direction", ["downstream", "upstream", "both"])
def test_directions_differ(index: PrefixIndex, direction: str) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2"])
    result = explore(index, seeds, hops=1, direction=direction, max_nodes=10_000, max_edges=10_000)
    assert result.final_nodes >= len(seeds)
    assert result.direction == direction


def test_both_direction_is_a_superset_of_downstream(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2"])
    down = explore(index, seeds, hops=1, direction="downstream", max_nodes=10_000, max_edges=10_000)
    both = explore(index, seeds, hops=1, direction="both", max_nodes=10_000, max_edges=10_000)
    assert both.final_nodes >= down.final_nodes


def test_node_budget_binds_deterministically(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2", "Tm3", "Mi1"])
    result = explore(index, seeds, hops=1, direction="downstream", max_nodes=33, max_edges=100_000)
    assert result.final_nodes <= 33
    assert result.budget_binding == "max_nodes"


def test_edge_budget_binds(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2", "Tm3", "Mi1"])
    result = explore(index, seeds, hops=1, direction="downstream", max_nodes=100_000, max_edges=5)
    assert result.final_edges <= 5
    assert result.budget_binding == "max_edges"


def test_exploration_is_deterministic(index: PrefixIndex) -> None:
    """Same inputs must give the same node set, whatever the iteration order."""
    seeds = seeds_from_types(NODES, ["L1", "L2", "Tm3"])
    runs = [
        explore(index, seeds, hops=2, direction="both", max_nodes=500, max_edges=5_000)
        for _ in range(3)
    ]
    first = [(s["stage"], s["new_nodes"], s["cumulative_nodes"]) for s in runs[0].stages]
    for run in runs[1:]:
        assert [(s["stage"], s["new_nodes"], s["cumulative_nodes"]) for s in run.stages] == first
    assert runs[0].final_nodes == runs[1].final_nodes == runs[2].final_nodes


def test_two_hops_superset_of_one_hop(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2"])
    one = explore(index, seeds, hops=1, direction="downstream", max_nodes=10_000, max_edges=10_000)
    two = explore(index, seeds, hops=2, direction="downstream", max_nodes=10_000, max_edges=10_000)
    assert two.final_nodes >= one.final_nodes
    assert len(two.stages) == 3


def test_invalid_direction_and_hops(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1"])
    with pytest.raises(ValueError, match="direction must be"):
        explore(index, seeds, hops=1, direction="sideways")
    with pytest.raises(ValueError, match="hops must be"):
        explore(index, seeds, hops=-1)


# ---------------------------------------------------------------- coverage


def test_coverage_reports_nulls_separately(index: PrefixIndex, annotations) -> None:
    seeds = seeds_from_types(NODES, ["L1", "L2", "Tm3", "Mi1"])
    result = explore(
        index, seeds, hops=1, direction="downstream", max_nodes=10_000, max_edges=10_000, annotations=annotations
    )
    coverage = result.coverage
    assert coverage["nodes"] == result.final_nodes
    assert coverage["in_annotation_table"] <= coverage["nodes"]
    assert coverage["with_type_label"] <= coverage["in_annotation_table"]
    assert "not evidence of non-membership" in coverage["note"]
    assert isinstance(result.node_type_top, dict)


def test_result_declares_it_is_not_the_visual_circuit(index: PrefixIndex) -> None:
    seeds = seeds_from_types(NODES, ["L1"])
    report = explore(index, seeds, hops=1, direction="downstream").to_dict()
    assert "NOT the complete visual circuit" in report["what_this_is"]
    assert "DISCOVERY edges" in report["edge_count_semantics"]
    assert report["minimum_synapse_count_threshold"] == index.min_weight
    assert "prefix" in report and "truncated" in report["prefix"]


def test_explore_without_annotations_is_allowed(index: PrefixIndex) -> None:
    result = explore(index, seeds_from_types(NODES, ["L1"]), hops=1, direction="upstream")
    assert result.coverage == {}
    assert result.node_type_top == {}
