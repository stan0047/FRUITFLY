"""End-to-end checks of the scaffold.

One test per component named in the design: synthetic graph creation, neural
simulation, environment reset/step, and the baseline model producing an output.
Fast enough to run on every save; the heavier integration path lives in
``scripts/smoke_test.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybrain.brain.graph import BrainGraph, synthetic_brain_graph
from flybrain.brain.neuron import LIFPopulation, NeuronParams
from flybrain.brain.simulation import NeuralSimulation, SimulationConfig
from flybrain.data.connectome import (
    Connectome,
    ConnectomeError,
    DataSource,
    load_connectome,
    load_flywire_connectome,
    synthetic_connectome,
)
from flybrain.models.baseline import BaselineConfig, TinyBaselineNet, count_parameters, resolve_device
from flybrain.navigation.environment import Action, FlyBrainNavEnv, GridSpec, make_env
from flybrain.vision.preprocessing import VisionPreprocessor
from flybrain.visualization.activity import ActivityRecorder, region_activity_matrix, summarize_activity

SMALL = 16


# --------------------------------------------------------------------- data


def test_synthetic_connectome_is_flagged_non_biological() -> None:
    connectome = synthetic_connectome(num_neurons=SMALL, seed=0)
    assert isinstance(connectome, Connectome)
    assert connectome.source is DataSource.SYNTHETIC
    assert connectome.is_biological is False
    assert connectome.num_neurons == SMALL
    assert connectome.num_edges > 0
    assert "SIMULATION INPUT" in connectome.warn_if_synthetic()


def test_real_connectome_is_not_bundled_or_fetched() -> None:
    """The project must never ship or download a FlyWire dataset."""
    from pathlib import Path

    import flybrain

    package_root = Path(flybrain.__file__).resolve().parent
    repository_root = package_root.parent.parent
    data_root = repository_root / "data"
    if data_root.is_dir():
        for path in data_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".h5", ".hdf5", ".zip", ".gz"}:
                pytest.fail(f"unexpected data file committed: {path}")

    with pytest.raises(ConnectomeError, match="never downloads"):
        load_flywire_connectome("https://example.invalid/flywire.csv")
    with pytest.raises(ConnectomeError, match="does not bundle or download"):
        load_flywire_connectome("no_such_local_file.csv")


def test_connectome_roundtrip(tmp_path) -> None:
    connectome = synthetic_connectome(num_neurons=8, seed=1, name="unit-test")
    path = connectome.save(tmp_path / "connectome.csv")
    assert path.is_file()
    assert (tmp_path / "connectome.source.json").is_file()

    from flybrain.data.connectome import load_connectome

    reloaded = load_connectome(path)
    assert reloaded.name == "unit-test"
    assert reloaded.num_edges == connectome.num_edges
    assert reloaded.is_biological is False


# ------------------------------------------------------------------ graph


def test_synthetic_graph_creation() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, connectivity=0.2, seed=0)
    assert isinstance(graph, BrainGraph)
    assert graph.num_nodes == SMALL
    assert graph.num_edges > 0
    assert graph.is_biological is False
    assert graph.source == DataSource.SYNTHETIC.value
    assert "synthetic" in graph.provenance_warning()


def test_synthetic_graph_is_reproducible_and_connected() -> None:
    first = synthetic_brain_graph(num_neurons=24, seed=7)
    second = synthetic_brain_graph(num_neurons=24, seed=7)
    assert np.array_equal(first.weights, second.weights)

    # Every neuron must have in- and out-degree so the population can fire.
    for node in first.node_ids:
        assert first.in_degree(node) >= 1
        assert first.out_degree(node) >= 1


def test_graph_regions_and_networkx_roundtrip() -> None:
    import networkx as nx

    graph = synthetic_brain_graph(num_neurons=12, regions=["a", "b"], seed=2)
    assert set(graph.region_names()) == {"a", "b"}
    assert sum(len(graph.nodes_in_region(r)) for r in graph.region_names()) == graph.num_nodes

    networkx_graph = graph.to_networkx()
    assert isinstance(networkx_graph, nx.DiGraph)
    assert networkx_graph.number_of_nodes() == graph.num_nodes
    assert networkx_graph.number_of_edges() == graph.num_edges

    restored = BrainGraph.from_networkx(networkx_graph)
    assert restored.num_nodes == graph.num_nodes


def test_graph_from_connectome_preserves_provenance() -> None:
    connectome = synthetic_connectome(num_neurons=10, seed=0)
    graph = BrainGraph.from_connectome(connectome)
    assert graph.num_nodes == connectome.num_neurons
    assert graph.is_biological == connectome.is_biological


# ------------------------------------------------------------- simulation


def test_lif_population_fires_under_drive() -> None:
    population = LIFPopulation(3, NeuronParams(tau_membrane=0.02, v_threshold=1.0))
    population.reset()

    # A constant drive of 5.0 (in voltage units) relaxes past the 1.0 threshold
    # after a handful of steps; the other neurons never receive any input.
    fired_at_least_once = np.zeros(3, dtype=bool)
    for _ in range(20):
        fired_at_least_once |= population.step(np.array([5.0, 0.0, 0.0]), dt=0.001)
    assert fired_at_least_once[0] and not fired_at_least_once[1:].any()

    # Silence after the drive is removed.
    population.reset()
    assert not population.step(np.zeros(3), dt=0.001).any()
    assert population.summary()["neurons_that_fired"] == 0


def test_lif_refractory_period_blocks_immediate_refiring() -> None:
    params = NeuronParams(tau_membrane=0.02, v_threshold=1.0, refractory_periods=3)
    population = LIFPopulation(1, params)
    population.reset()
    drive = np.array([50.0])
    spikes = [bool(population.step(drive, dt=0.001)[0]) for _ in range(12)]
    assert sum(spikes) >= 1
    fired_steps = [index for index, value in enumerate(spikes) if value]
    assert all(b - a >= 4 for a, b in zip(fired_steps, fired_steps[1:])), "refractory period not respected"


def test_neural_simulation_executes() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, seed=0)
    sim = NeuralSimulation(graph, SimulationConfig(dt=0.001, seed=0))

    result = sim.step(np.ones(graph.num_nodes))
    assert result.spikes.shape == (graph.num_nodes,)
    assert result.activity.shape == (graph.num_nodes,)
    assert 0.0 <= result.n_active <= graph.num_nodes
    assert result.step == 0
    assert abs(result.time - 0.0) < 1e-12
    assert set(result.region_activity) == set(graph.region_names())
    assert result.total_input > 0.0

    results = sim.run(n_steps=5)
    assert len(results) == 5
    assert [r.step for r in results] == [1, 2, 3, 4, 5]
    assert results[-1].time == pytest.approx(5 * 0.001)

    sim.reset()
    assert sim.step_index == 0
    assert sim.population.step_count == 0


def test_neural_simulation_produces_firing_events() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, connectivity=0.5, seed=1)
    sim = NeuralSimulation(graph, SimulationConfig(external_input_scale=4.0))
    results = sim.run(n_steps=20)

    assert any(result.n_active > 0 for result in results), "expected at least one spike"
    recorder = ActivityRecorder(graph.node_ids)
    recorder.record_all(results)
    assert recorder.firing_counts().sum() > 0
    assert summarize_activity(recorder)["neurons_that_fired"] > 0
    assert summarize_activity(recorder)["mean_firing_rate"] > 0.0


def test_simulation_rejects_wrong_input_shape() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, seed=0)
    sim = NeuralSimulation(graph)
    with pytest.raises(ValueError):
        sim.step(np.ones(SMALL + 1))


def test_simulation_reports_simulation_provenance() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, seed=0)
    summary = NeuralSimulation(graph).summary()
    assert summary["is_biological_topology"] is False
    assert "simulated" in summary["dynamics"]


# ----------------------------------------------------------- environment


def test_environment_reset_and_step() -> None:
    env = FlyBrainNavEnv(GridSpec(width=5, height=5, num_obstacles=0, randomize_layout=False, max_steps=20))

    observation, info = env.reset(seed=0, options={"agent_start": (0, 0), "target": (0, 2)})
    assert observation.shape == (env.observation_dim,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.agent_pos == (0, 0)
    assert env.target_pos == (0, 2)
    assert info["steps"] == 0

    observation, reward, terminated, truncated, info = env.step(Action.FORWARD)
    assert env.agent_pos == (0, 1)
    assert reward == pytest.approx(GridSpec().step_penalty)
    assert not terminated and not truncated
    assert info["heading"] == "north"
    assert env.observation_space.contains(observation)

    env.step(Action.RIGHT)
    assert env.heading == "east"
    assert env.agent_pos == (0, 1), "turning must not move the agent"
    env.step(Action.LEFT)
    assert env.heading == "north"

    observation, reward, terminated, truncated, info = env.step(Action.FORWARD)
    assert env.agent_pos == env.target_pos
    assert terminated and not truncated
    assert info["success"] is True
    assert reward == pytest.approx(GridSpec().goal_reward + GridSpec().step_penalty)


def test_environment_walls_block_and_penalise() -> None:
    env = FlyBrainNavEnv(GridSpec(width=5, height=5, num_obstacles=0, randomize_layout=False, max_steps=20))
    env.reset(seed=0)
    env.step(Action.LEFT)  # face west
    assert env.heading == "west"

    _, reward, terminated, _, _ = env.step(Action.FORWARD)  # blocked by the wall
    assert not terminated
    assert env.agent_pos == (0, 0)
    assert reward < GridSpec().step_penalty  # step penalty plus collision penalty
    assert reward == pytest.approx(GridSpec().step_penalty + GridSpec().collision_penalty)


def test_environment_truncates_at_max_steps() -> None:
    env = FlyBrainNavEnv(GridSpec(width=4, height=4, num_obstacles=0, randomize_layout=False, max_steps=3))
    env.reset(seed=0)
    for step in range(3):
        _, _, terminated, truncated, _ = env.step(Action.LEFT)
    assert truncated and not terminated
    assert step == 2


def test_environment_randomises_layout_reproducibly() -> None:
    spec = GridSpec(width=8, height=8, num_obstacles=6, randomize_layout=True)
    first = make_env({"navigation": {"width": 8, "height": 8, "num_obstacles": 6}}, seed=3)
    second = make_env({"navigation": {"width": 8, "height": 8, "num_obstacles": 6}}, seed=3)
    assert first.obstacles == second.obstacles
    assert first.num_obstacles == 6
    assert spec.width == 8


def test_environment_renders_ascii() -> None:
    env = FlyBrainNavEnv(GridSpec(width=4, height=4, num_obstacles=0, randomize_layout=False))
    env.reset(seed=0)
    text = env.render()
    assert "agent=" in text and "target=" in text
    assert "A" in text and "T" in text


# --------------------------------------------------------------- baseline


def test_baseline_model_produces_output() -> None:
    env = FlyBrainNavEnv(GridSpec(width=5, height=5, num_obstacles=0))
    observation, _ = env.reset(seed=0)

    net = TinyBaselineNet(BaselineConfig(input_dim=env.observation_dim, num_actions=len(Action), seed=0))
    action, log_prob, value = net.act(observation)
    assert 0 <= action < len(Action)
    assert np.isfinite(log_prob) and np.isfinite(value)
    assert count_parameters(net) > 0

    batch = np.stack([observation, observation])
    logits, values = net(batch)
    assert logits.shape == (2, len(Action))
    assert values.shape == (2,)


def test_baseline_model_backpropagates() -> None:
    net = TinyBaselineNet(BaselineConfig(input_dim=10, num_actions=3, hidden_sizes=(8,), seed=0))
    observations = torch.randn(4, 10)
    actions = torch.tensor([0, 1, 2, 1])
    log_probs, entropies, values = net.evaluate_actions(observations, actions)
    assert log_probs.shape == entropies.shape == values.shape == (4,)

    loss = -(log_probs.mean()) + entropies.mean() + values.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_baseline_model_saves_and_loads(tmp_path) -> None:
    net = TinyBaselineNet(BaselineConfig(input_dim=6, num_actions=3, seed=0))
    path = net.save(tmp_path / "net.pt")
    restored = TinyBaselineNet.load(path)
    assert restored.config.input_dim == 6
    assert restored.config.num_actions == 3
    for a, b in zip(net.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_device_resolution_falls_back_to_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.warns(RuntimeWarning):
            assert resolve_device("cuda").type == "cpu"


def test_tiny_training_loop_runs() -> None:
    from flybrain.training.trainer import Trainer, TrainerConfig

    env = FlyBrainNavEnv(GridSpec(width=5, height=5, num_obstacles=2, max_steps=20))
    net = TinyBaselineNet(BaselineConfig(input_dim=env.observation_dim, num_actions=len(Action), seed=0))
    trainer = Trainer(net, TrainerConfig(episodes=2, verbose=False, seed=0))

    report = trainer.train(env, num_episodes=2)
    assert report.episodes == 2
    assert report.timesteps > 0
    assert len(report.losses) == 2
    assert np.isfinite(report.losses).all()
    assert np.isfinite(report.mean_reward)

    metrics = trainer.evaluate(env, episodes=1)
    assert metrics["episodes"] == 1.0


# ------------------------------------------------------------------ vision


def test_vision_preprocessing_shapes() -> None:
    preprocessor = VisionPreprocessor(width=16, height=12, grayscale=True)
    frame = np.random.default_rng(0).random((40, 40, 3)).astype(np.float32)
    output = preprocessor.process(frame)
    assert output.shape == (12, 16)
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0
    assert preprocessor.observation_dim() == 12 * 16


def test_vision_preprocessing_motion_channel() -> None:
    preprocessor = VisionPreprocessor(width=8, height=8, use_motion_channel=True)
    previous = np.zeros((8, 8), dtype=np.float32)
    current = np.ones((8, 8), dtype=np.float32)
    output = preprocessor.process(current, previous=previous)
    assert output.shape[-1] == 2
    assert preprocessor.observation_dim() == 2 * 8 * 8


# ---------------------------------------------------------- visualization


def test_activity_recorder_statistics() -> None:
    graph = synthetic_brain_graph(num_neurons=SMALL, seed=0)
    sim = NeuralSimulation(graph, SimulationConfig(external_input_scale=4.0))
    recorder = ActivityRecorder(graph.node_ids, max_history=10)
    recorder.record_all(sim.run(n_steps=25))

    assert len(recorder) == 10  # ring buffer respected
    assert recorder.spike_matrix().shape == (10, SMALL)
    assert recorder.activity_matrix().shape == (10, SMALL)
    assert len(recorder.active_neuron_count()) == 10

    times, counts = recorder.firing_timeline(window=5)
    assert counts.shape == (SMALL, 5)
    assert times.shape == (5,)

    regions, matrix = region_activity_matrix(recorder)
    assert matrix.shape == (10, len(regions))

    top_neuron, spike_count = recorder.most_active_neurons(1)[0]
    assert top_neuron in graph.node_ids
    assert spike_count >= 0

    detail = recorder.neuron_detail(top_neuron, graph)
    assert detail["node_id"] == top_neuron
    assert "region" in detail and "in_degree" in detail

    with pytest.raises(KeyError):
        recorder.neuron_detail("not-a-neuron")


def test_activity_figures_are_written(tmp_path) -> None:
    from flybrain.visualization.activity import save_activity_dashboard

    graph = synthetic_brain_graph(num_neurons=SMALL, seed=0)
    sim = NeuralSimulation(graph, SimulationConfig(external_input_scale=4.0))
    recorder = ActivityRecorder(graph.node_ids)
    recorder.record_all(sim.run(n_steps=12))

    written = save_activity_dashboard(graph, recorder, tmp_path / "figures")
    for key in ("graph", "timeline", "regions", "summary"):
        assert key in written

    import json
    from pathlib import Path

    for value in written.values():
        assert Path(value).is_file() and Path(value).stat().st_size > 0

    summary = json.loads((tmp_path / "figures" / "activity_summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["is_biological_topology"] is False
    assert "simulated" in summary["provenance"]["activity"]
