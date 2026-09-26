"""Simulated neurons.

The models here are **simulations, not biology**. A leaky integrate-and-fire
(LIF) unit is a standard engineering approximation chosen because it is cheap,
deterministic and easy to vectorise. It reproduces none of the ionic
mechanisms, morphology, or spike-shape dynamics of a real *Drosophila* neuron,
and results obtained with it are computational simulations only.
"""

from flybrain.brain.neuron import (
    LIFPopulation,
    NeuronParams,
    NeuronState,
    membrane_update,
    spike_mask,
)

__all__ = [
    "LIFPopulation",
    "NeuronParams",
    "NeuronState",
    "membrane_update",
    "spike_mask",
]
