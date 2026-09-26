"""Vision preprocessing: frames in, model-ready arrays out.

No model, no environment, no plotting. See
:mod:`flybrain.vision.preprocessing` for the implementation.
"""

from flybrain.vision.preprocessing import (
    VisionPreprocessor,
    apply_preprocessing,
    center_crop,
    frame_difference,
    resize_nearest,
    stack_frames,
    to_grayscale,
    unit_range,
)

__all__ = [
    "VisionPreprocessor",
    "apply_preprocessing",
    "center_crop",
    "frame_difference",
    "resize_nearest",
    "stack_frames",
    "to_grayscale",
    "unit_range",
]
