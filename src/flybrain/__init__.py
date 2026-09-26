"""FlyBrain — bio-inspired vision and navigation research scaffold.

Package layout follows four separation rules:

1. :mod:`flybrain.data` holds biological data handling and knows nothing about
   simulation.
2. :mod:`flybrain.brain` simulates neuron activity and knows nothing about
   rewards, actions or gymnasium.
3. :mod:`flybrain.navigation` and :mod:`flybrain.training` consume simulation
   output; they never reach back into it.
4. :mod:`flybrain.visualization` only reads recorded arrays.

.. warning::
   This project does **not** currently contain a real fruit-fly connectome, and
   nothing in it is a biological model. The only graph available is a synthetic
   random graph used for development, and all activity is numerically simulated.
"""

from flybrain.config import Config, ConfigError, load_config
from flybrain.data.connectome import Connectome, DataSource, load_flywire_connectome, synthetic_connectome
from flybrain.brain.graph import BrainGraph, synthetic_brain_graph
from flybrain.brain.neuron import LIFPopulation, NeuronParams
from flybrain.brain.simulation import NeuralSimulation, SimulationConfig, StepResult
from flybrain.models.baseline import BaselineConfig, TinyBaselineNet
from flybrain.navigation.environment import Action, FlyBrainNavEnv, GridSpec, make_env
from flybrain.training.trainer import Trainer, TrainerConfig
from flybrain.vision.preprocessing import VisionPreprocessor
from flybrain.visualization.activity import ActivityRecorder, summarize_activity

__version__ = "0.1.0"

#: One-line statement of what this package does and does not contain.
STATUS = (
    "Scaffold only: synthetic graph + LIF simulation + toy gridworld + placeholder "
    "policy. No real fruit-fly connectome is integrated; all activity is simulated."
)

__all__ = [
    "Action",
    "ActivityRecorder",
    "BaselineConfig",
    "BrainGraph",
    "Config",
    "ConfigError",
    "Connectome",
    "DataSource",
    "FlyBrainNavEnv",
    "GridSpec",
    "LIFPopulation",
    "NeuralSimulation",
    "NeuronParams",
    "STATUS",
    "SimulationConfig",
    "StepResult",
    "TinyBaselineNet",
    "Trainer",
    "TrainerConfig",
    "VisionPreprocessor",
    "__version__",
    "load_config",
    "load_flywire_connectome",
    "make_env",
    "summarize_activity",
    "synthetic_brain_graph",
    "synthetic_connectome",
]
