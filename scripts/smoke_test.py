#!/usr/bin/env python
"""End-to-end scaffold check.

Runs the whole pipeline once on CPU and prints a readable report:

1. load the YAML config;
2. build a synthetic connectome and convert it to a brain graph;
3. run the neural simulation and record activity;
4. roll out the gridworld with the placeholder policy;
5. run a short training loop;
6. write the static activity figures and a JSON summary.

Every stage reports provenance, so the output is unambiguous that the graph is
synthetic and the activity is simulated.

Usage
-----
    python scripts/smoke_test.py [--config configs/default.yaml] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a clone without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from flybrain import __version__  # noqa: E402
from flybrain.brain.graph import BrainGraph, synthetic_brain_graph  # noqa: E402
from flybrain.brain.simulation import NeuralSimulation, SimulationConfig  # noqa: E402
from flybrain.brain.neuron import NeuronParams  # noqa: E402
from flybrain.config import load_config  # noqa: E402
from flybrain.data.connectome import synthetic_connectome  # noqa: E402
from flybrain.models.baseline import BaselineConfig, TinyBaselineNet, count_parameters  # noqa: E402
from flybrain.navigation.environment import Action, make_env  # noqa: E402
from flybrain.training.trainer import Trainer, TrainerConfig  # noqa: E402
from flybrain.vision.preprocessing import VisionPreprocessor  # noqa: E402
from flybrain.visualization.activity import ActivityRecorder, save_activity_dashboard, summarize_activity  # noqa: E402

BANNER = "=" * 78


def _step(index: int, title: str) -> None:
    print(f"\n{BANNER}\n[{index}] {title}\n{BANNER}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlyBrain scaffold smoke test")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument("--quick", action="store_true", help="fewer simulation steps and episodes")
    args = parser.parse_args(argv)

    print(f"FlyBrain v{__version__} — scaffold smoke test")
    print("NOTE: synthetic graph + simulated activity + placeholder policy. Not a fly-brain model.")

    # ---------------------------------------------------------------- 1. config
    _step(1, "Configuration")
    config = load_config(args.config)
    print(f"config file : {config.source}")
    print(f"project root: {config.root}")
    print(f"seed        : {config.seed}")
    print(f"connectome  : {config.get('connectome.source')}  (no real connectome is integrated)")

    n_steps = 8 if args.quick else 40
    n_episodes = 2 if args.quick else int(config.get("training.episodes", 20))

    # ------------------------------------------------------------- 2. connectome
    _step(2, "Synthetic connectome (NOT biological data)")
    connectome = synthetic_connectome(
        num_neurons=int(config.get("brain.graph.num_neurons", 64)),
        connectivity=float(config.get("brain.graph.connectivity", 0.15)),
        seed=int(config.get("brain.graph.seed", 0)),
    )
    print(f"name        : {connectome.name}")
    print(f"source      : {connectome.source.value}  is_biological={connectome.is_biological}")
    print(f"neurons     : {connectome.num_neurons}   synapses: {connectome.num_edges}")
    print(f"warning     : {connectome.warn_if_synthetic()}")

    # Check that the data layer converts cleanly, but keep region labels out of it:
    # the data layer must not invent neuropil names.
    print(f"\ndata->graph : {BrainGraph.from_connectome(connectome).summary()}")

    # The simulated circuit is built by the generator, which assigns the
    # arbitrary region labels from `brain.graph.regions` in the config.
    graph = synthetic_brain_graph(
        num_neurons=int(config.get("brain.graph.num_neurons", 64)),
        connectivity=float(config.get("brain.graph.connectivity", 0.15)),
        regions=config.get("brain.graph.regions", ("unassigned",)),
        seed=int(config.get("brain.graph.seed", 0)),
    )
    print(f"graph       : {graph.summary()}")

    # ------------------------------------------------------------ 3. simulation
    _step(3, "Neural simulation (simulated activity)")
    sim = NeuralSimulation(
        graph,
        SimulationConfig.from_mapping(config.get("brain.simulation")),
        NeuronParams.from_mapping(config.get("brain.neuron")),
    )
    recorder = ActivityRecorder(graph.node_ids, max_history=int(config.get("visualization.max_history", 500)))
    results = sim.run(n_steps=n_steps)
    recorder.record_all(results)

    activity_summary = summarize_activity(recorder)
    print(f"steps run   : {activity_summary['steps_recorded']}  (dt={sim.config.dt} s)")
    print(f"active neurons (mean / max): {activity_summary['mean_active_neurons']:.2f} / {activity_summary['max_active_neurons']}")
    print(f"mean firing rate  : {activity_summary['mean_firing_rate']:.4f}")
    print(f"neurons that fired: {activity_summary['neurons_that_fired']} / {graph.num_nodes}")
    print(f"busiest region    : {activity_summary['busiest_region']}")
    print(f"top neurons       : {recorder.most_active_neurons(3)}")
    detail = recorder.neuron_detail(recorder.most_active_neurons(1)[0][0], graph)
    print(f"selected neuron   : {json.dumps(detail)}")

    # ------------------------------------------------------------- 4. env roll
    _step(4, "Navigation environment (toy gridworld, no vision)")
    env = make_env(config.to_dict())
    print(f"observation dim: {env.observation_dim}   actions: {[a.name for a in Action]}")
    print(env.render())

    preprocessor = VisionPreprocessor.from_mapping(config.get("vision"))
    observation, _ = env.reset(seed=config.seed)
    frame = observation[: env.spec.size].reshape(env.spec.height, env.spec.width)
    processed = preprocessor.process(frame)
    print(f"\npreprocessing : {preprocessor.config()}")
    print(f"observation -> image {frame.shape} -> processed {processed.shape} "
          f"range [{float(processed.min()):.2f}, {float(processed.max()):.2f}]")

    net = TinyBaselineNet(
        BaselineConfig.from_mapping(config.get("model.baseline"), input_dim=env.observation_dim, num_actions=len(Action))
    )
    print(f"baseline net  : {count_parameters(net)} parameters on {net.device} (placeholder, not a fly circuit)")

    total_reward = 0.0
    for _ in range(20):
        action, _, _ = net.act(observation, greedy=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"greedy rollout: reward {total_reward:+.3f}, steps {env.steps}, final heading {env.heading}")
    print(env.render())

    # -------------------------------------------------------------- 5. training
    _step(5, "Training loop (REINFORCE on the placeholder policy)")
    trainer = Trainer(net, TrainerConfig.from_mapping(config.get("training")))
    report = trainer.train(env, num_episodes=n_episodes)
    print(f"\n{json.dumps(report.summary(), indent=2)}")

    # -------------------------------------------------------------- 6. figures
    _step(6, "Activity figures")
    figures_dir = config.ensure_dir("paths.figures")
    written = save_activity_dashboard(
        graph, recorder, figures_dir, prefix="smoke", window=int(config.get("visualization.timeline_window", 200))
    )
    for name, path in written.items():
        print(f"{name:9s}: {path}")

    checkpoint = trainer.save_checkpoint(config.ensure_dir("paths.checkpoints") / "smoke.pt")
    print(f"checkpoint: {checkpoint}")

    # ------------------------------------------------------------------ done
    _step(7, "Result")
    print("All scaffold stages completed.")
    print("Reminder: no real fruit-fly connectome is used, and no neural activity")
    print("in this project is a recording from a real fly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
