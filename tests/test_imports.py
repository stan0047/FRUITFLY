"""Every public module must import cleanly.

These tests are the cheapest possible guard against a broken scaffold: they catch
circular imports, typos in ``__init__`` re-exports, and missing optional imports
before anything else runs.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "flybrain",
    "flybrain.config",
    "flybrain.data",
    "flybrain.data.connectome",
    "flybrain.brain",
    "flybrain.brain.graph",
    "flybrain.brain.neuron",
    "flybrain.brain.simulation",
    "flybrain.vision",
    "flybrain.vision.preprocessing",
    "flybrain.navigation",
    "flybrain.navigation.environment",
    "flybrain.models",
    "flybrain.models.baseline",
    "flybrain.training",
    "flybrain.training.trainer",
    "flybrain.visualization",
    "flybrain.visualization.activity",
]

TOP_LEVEL_SYMBOLS = [
    "Action",
    "ActivityRecorder",
    "BrainGraph",
    "Config",
    "FlyBrainNavEnv",
    "NeuralSimulation",
    "STATUS",
    "TinyBaselineNet",
    "Trainer",
    "__version__",
    "make_env",
    "synthetic_brain_graph",
    "synthetic_connectome",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


@pytest.mark.parametrize("symbol", TOP_LEVEL_SYMBOLS)
def test_top_level_exports(symbol: str) -> None:
    import flybrain

    assert hasattr(flybrain, symbol), f"flybrain.{symbol} is missing"
    assert symbol in flybrain.__all__


@pytest.mark.parametrize("module_name", MODULES)
def test_declared_exports_exist(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for symbol in getattr(module, "__all__", []):
        assert hasattr(module, symbol), f"{module_name}.__all__ lists missing symbol {symbol!r}"


def test_separation_of_concerns() -> None:
    """The layer boundaries must hold, not just the imports succeed."""
    import flybrain.brain.simulation as simulation
    import flybrain.data.connectome as connectome
    import flybrain.navigation.environment as environment
    import flybrain.visualization.activity as activity

    # Data handling must not reach into the simulation layer.
    assert "flybrain.brain" not in vars(connectome)

    # The simulation engine must know nothing about RL, actions or gymnasium.
    forbidden_for_simulation = {"gymnasium", "spaces", "flybrain.training", "flybrain.navigation", "Action"}
    assert not forbidden_for_simulation & set(vars(simulation))

    # The environment must not import the brain or the policy.
    assert not {"flybrain.brain", "flybrain.models", "torch"} & set(vars(environment))

    # Visualisation may read recorded arrays but must not drive the simulation.
    assert "NeuralSimulation" not in vars(activity)
    assert not {"flybrain.training", "flybrain.navigation"} & set(vars(activity))


def test_config_loads_and_resolves_paths() -> None:
    import flybrain

    config = flybrain.load_config()
    assert config.get("project.name") == "flybrain"
    assert config.get("connectome.source") == "synthetic"
    assert config.path("paths.outputs").is_absolute()
    assert config.seed == 0
