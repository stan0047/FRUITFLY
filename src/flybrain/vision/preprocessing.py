"""Visual preprocessing.

Turns raw frames (whatever produces them) into small, normalised arrays suitable
for a model. Deliberately dependency-light: no OpenCV, no GPU requirement.

Nothing here models the fly's optic flow. The motion channel is a plain frame
difference, provided as a hook for the future optic-flow work; it is not a
biological model of any direction-selective neuron.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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


def _as_float_image(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    return array.astype(np.float32)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Collapse ``(H, W, C)`` to ``(H, W)`` with luminance weights.

    Averages channels when there are more than three, since an arbitrary channel
    count has no known colour basis.
    """
    array = _as_float_image(frame)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D frame, got shape {array.shape}")
    if array.shape[2] == 1:
        return array[..., 0]
    if array.shape[2] < 3:
        return array.mean(axis=2)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return array[..., :3] @ weights


def center_crop(frame: np.ndarray, size: int | tuple[int, int] | None) -> np.ndarray:
    """Crop a centred window, padding with zeros if the frame is smaller."""
    if size is None:
        return frame
    if isinstance(size, int):
        size = (size, size)
    crop_h, crop_w = int(size[0]), int(size[1])
    height, width = frame.shape[0], frame.shape[1]
    padded = np.zeros((max(height, crop_h), max(width, crop_w), *frame.shape[2:]), dtype=frame.dtype)
    padded[:height, :width] = frame
    top = (padded.shape[0] - crop_h) // 2
    left = (padded.shape[1] - crop_w) // 2
    return padded[top : top + crop_h, left : left + crop_w]


def resize_nearest(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``(height, width)``.

    Nearest neighbour keeps this module dependency-free and exactly reproducible.
    A smoother resampler is a drop-in replacement if quality ever matters.
    """
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("target size must be positive")
    array = np.asarray(frame)
    if array.ndim == 2:
        rows = (np.arange(height) * array.shape[0] / height).astype(np.int64)
        cols = (np.arange(width) * array.shape[1] / width).astype(np.int64)
        return array[np.ix_(np.clip(rows, 0, array.shape[0] - 1), np.clip(cols, 0, array.shape[1] - 1))]
    return resize_nearest(array, (height, width)) if False else _resize_channels(array, (height, width))


def _resize_channels(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    channel_first = array.ndim == 3 and array.shape[0] not in (height, width) and array.shape[2] in (1, 3, 4)
    data = np.moveaxis(array, 0, -1) if channel_first else array
    squeezed = data
    if squeezed.ndim == 3 and squeezed.shape[2] == 1:
        squeezed = squeezed[..., 0]
    resized = resize_nearest(squeezed, (height, width))
    return resized[..., None] if array.ndim == 3 and resized.ndim == 2 else resized


def frame_difference(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Absolute frame difference, a simple motion/magnitude channel."""
    return np.abs(_as_float_image(current) - _as_float_image(previous)).astype(np.float32)


def unit_range(frame: np.ndarray) -> np.ndarray:
    """Min-max normalise to ``[0, 1]``; a constant frame maps to zeros."""
    array = _as_float_image(frame)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return np.zeros_like(array)
    return ((array - low) / (high - low)).astype(np.float32)


def stack_frames(frames: Sequence[np.ndarray], count: int | None = None) -> np.ndarray:
    """Stack a frame history along the channel axis, oldest first.

    Pads at the front by repeating the oldest frame when the history is short,
    so the output shape is always ``(count * channels, H, W)``.
    """
    if not frames:
        raise ValueError("need at least one frame to stack")
    if count is None:
        count = len(frames)
    if count <= 0:
        raise ValueError("count must be positive")
    history = list(frames)[-count:]
    while len(history) < count:
        history.insert(0, history[0])
    return np.concatenate([np.atleast_3d(frame) for frame in history], axis=-1).transpose(2, 0, 1).astype(np.float32)


@dataclass
class VisionPreprocessor:
    """Config-driven frame preprocessing.

    Fields mirror ``configs/default.yaml`` under ``vision:`` and are built via
    :meth:`from_mapping`, so behaviour is never hardcoded at the call site.
    """

    width: int = 32
    height: int = 32
    grayscale: bool = True
    normalize: bool = True
    crop: tuple[int, int] | None = None
    frame_stack: int = 1
    use_motion_channel: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "VisionPreprocessor":
        if not mapping:
            return cls()
        crop = mapping.get("crop")
        known = {"width", "height", "grayscale", "normalize", "crop", "frame_stack", "use_motion_channel"}
        kwargs = {str(k): v for k, v in mapping.items() if str(k) in known}
        if crop is not None:
            kwargs["crop"] = (int(crop[0]), int(crop[1]))
        return cls(**kwargs)

    def process(self, frame: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
        """Run the full pipeline on one frame.

        Returns ``(height, width)`` for a single channel and
        ``(height, width, channels)`` when a motion channel is appended.
        """
        array = _as_float_image(frame)
        if self.grayscale:
            array = to_grayscale(array)
        if self.crop is not None:
            array = center_crop(array, self.crop)
        array = resize_nearest(array, (self.height, self.width))
        if self.normalize:
            array = unit_range(array)

        channels = [np.asarray(array, dtype=np.float32)]
        if self.use_motion_channel and previous is not None:
            motion = unit_range(frame_difference(previous, frame))
            channels.append(resize_nearest(motion, (self.height, self.width)).astype(np.float32))

        if len(channels) == 1:
            return channels[0]
        return np.stack(channels, axis=-1)

    def observation_dim(self) -> int:
        """Length of the flattened observation produced by :meth:`process`."""
        channels = (1 if self.grayscale else 3) + (1 if self.use_motion_channel else 0)
        return channels * self.height * self.width

    def config(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "grayscale": self.grayscale,
            "normalize": self.normalize,
            "crop": list(self.crop) if self.crop else None,
            "frame_stack": self.frame_stack,
            "use_motion_channel": self.use_motion_channel,
        }


def apply_preprocessing(frame: np.ndarray, config: Mapping[str, Any] | VisionPreprocessor | None = None) -> np.ndarray:
    """Convenience entry point used by tests and the smoke test."""
    preprocessor = config if isinstance(config, VisionPreprocessor) else VisionPreprocessor.from_mapping(config)
    return preprocessor.process(frame)
