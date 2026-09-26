"""Biological data access layer.

This package is the *only* place where third-party data enters FlyBrain, and it
is deliberately ignorant of neural simulation: it can import neither
``flybrain.brain`` nor ``flybrain.training``.

Provenance is a first-class field. Anything produced by
:func:`synthetic_connectome` is flagged ``is_biological=False`` with
``source="synthetic"`` so it can never be mistaken for published data.
"""

from flybrain.data.connectome import (
    Connectome,
    ConnectomeEdge,
    ConnectomeError,
    DataSource,
    available_sources,
    describe_schema,
    is_source_available,
    load_connectome,
    load_flywire_connectome,
    save_connectome,
    synthetic_connectome,
)

__all__ = [
    "Connectome",
    "ConnectomeEdge",
    "ConnectomeError",
    "DataSource",
    "available_sources",
    "describe_schema",
    "is_source_available",
    "load_connectome",
    "load_flywire_connectome",
    "save_connectome",
    "synthetic_connectome",
]
