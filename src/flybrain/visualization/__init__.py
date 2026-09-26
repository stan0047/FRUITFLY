"""Simulation-driven visualisation.

Reads recorded activity and renders static figures. It never steps the
simulation, never mutates it, and never imports the training or navigation
layers. The live dashboard is a future milestone; this module is its data source.
"""

from flybrain.visualization.activity import (
    ActivityRecorder,
    region_activity_matrix,
    save_activity_dashboard,
    save_activity_timeline,
    save_brain_graph,
    summarize_activity,
)

__all__ = [
    "ActivityRecorder",
    "region_activity_matrix",
    "save_activity_dashboard",
    "save_activity_timeline",
    "save_brain_graph",
    "summarize_activity",
]
