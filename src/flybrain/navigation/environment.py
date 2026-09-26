"""A tiny 2D gridworld with ``LEFT`` / ``FORWARD`` / ``RIGHT``.

This is a **development scaffold**, not a fly simulator and not a model of
navigation. Its only jobs are to give the project a standard
:mod:`gymnasium` interface, a real reward signal, and a headless-friendly
``ansi`` renderer so the training loop can be exercised end to end before any
biologically motivated environment exists.

Design notes
------------
* The agent carries a heading and can only rotate in place. That makes
  ``LEFT`` / ``FORWARD`` / ``RIGHT`` meaningful and keeps the action set at three
  discrete values, matching the eventual motor output.
* There is no vision. The observation is a flattened one-hot encoding of the
  grid. Wiring the visual circuit in is a roadmap item, not a scaffold feature.
* Observations and rewards are plain arrays and floats, so nothing here depends
  on the simulation, the policy, or the visualisation layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np

try:  # gymnasium is a hard dependency; keep the module importable without it
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - only in a broken environment
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

__all__ = ["Action", "FlyBrainNavEnv", "GridSpec", "Heading", "make_env"]

Coordinate = tuple[int, int]

#: Cardinal headings, in clockwise order. ``FORWARD`` moves along the heading.
_HEADING_ORDER: tuple[str, ...] = ("north", "east", "south", "west")
_HEADING_VECTORS: dict[str, Coordinate] = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
_HEADING_GLYPHS: dict[str, str] = {"north": "^", "east": ">", "south": "v", "west": "<"}
#: Heading change in quarter turns for each action.
_TURN_FROM_ACTION: dict[int, int] = {0: -1, 1: 0, 2: +1}

#: Maps the YAML reward keys onto :class:`GridSpec` field names.
_REWARD_FIELDS: dict[str, str] = {"goal": "goal_reward", "step": "step_penalty", "collision": "collision_penalty"}


class Action(IntEnum):
    """The three discrete motor actions."""

    LEFT = 0
    FORWARD = 1
    RIGHT = 2

    @property
    def delta_heading(self) -> int:
        """Heading change in quarter turns: -1, 0 or +1."""
        return _TURN_FROM_ACTION[int(self)]


class Heading(str, Enum):
    """Cardinal heading names used by :class:`GridSpec`."""

    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    @property
    def vector(self) -> Coordinate:
        return _HEADING_VECTORS[self.value]

    @property
    def next_name(self) -> str:
        """Name of the heading one clockwise quarter turn away."""
        return _HEADING_ORDER[(_HEADING_ORDER.index(self.value) + 1) % len(_HEADING_ORDER)]


@dataclass
class GridSpec:
    """Static description of the gridworld layout."""

    width: int = 7
    height: int = 7
    num_obstacles: int = 4
    max_steps: int = 100
    randomize_layout: bool = True
    initial_heading: str = Heading.NORTH.value
    goal_reward: float = 1.0
    step_penalty: float = -0.01
    collision_penalty: float = -0.1
    obstacles: list[Coordinate] = field(default_factory=list)
    agent_start: Coordinate | None = None
    target: Coordinate | None = None

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("grid must be at least 3x3")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.num_obstacles < 0:
            raise ValueError("num_obstacles must be non-negative")
        if self.initial_heading not in _HEADING_VECTORS:
            raise ValueError(f"initial_heading must be one of {sorted(_HEADING_VECTORS)}")
        self.obstacles = [self.clamp(cell) for cell in self.obstacles]

    @property
    def size(self) -> int:
        return self.width * self.height

    def clamp(self, cell: Sequence[int]) -> Coordinate:
        """Clamp a coordinate into the grid."""
        x = int(np.clip(int(cell[0]), 0, self.width - 1))
        y = int(np.clip(int(cell[1]), 0, self.height - 1))
        return (x, y)

    def in_bounds(self, cell: Coordinate) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height


class FlyBrainNavEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """A minimal 2D navigation task.

    Observation
        ``float32`` vector of length ``3 * width * height + 6``: three flattened
        planes (agent, target, obstacle) followed by the agent's normalised
        position, the target's normalised position, and the heading as
        ``(sin, cos)``. Bounds are ``[-1, 1]``.

    Action
        :class:`Action` — ``LEFT``, ``FORWARD``, ``RIGHT``.

    Reward
        ``goal_reward`` on reaching the target, ``collision_penalty`` when a move
        is blocked by a wall or an obstacle, ``step_penalty`` on every step.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, spec: GridSpec | None = None) -> None:
        if spaces is None:  # pragma: no cover
            raise ImportError("gymnasium is required; install it with `pip install gymnasium`")

        self.spec = spec or GridSpec()
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)
        self.render_mode: str | None = "ansi"

        self.agent_pos: Coordinate = (0, 0)
        self.target_pos: Coordinate = (self.spec.width - 1, self.spec.height - 1)
        self.heading: str = self.spec.initial_heading
        self.obstacles: set[Coordinate] = set()
        self.steps = 0

    # ------------------------------------------------------------ properties

    @property
    def observation_dim(self) -> int:
        """Length of the flattened observation vector."""
        return 3 * self.spec.size + 6

    @property
    def heading_vector(self) -> Coordinate:
        return _HEADING_VECTORS[self.heading]

    @property
    def num_obstacles(self) -> int:
        return len(self.obstacles)

    def free_cells(self) -> list[Coordinate]:
        """Cells that are neither obstacle, agent nor target."""
        blocked = self.obstacles | {self.agent_pos, self.target_pos}
        return [
            (x, y)
            for x in range(self.spec.width)
            for y in range(self.spec.height)
            if (x, y) not in blocked
        ]

    # ------------------------------------------------------------- lifecycle

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode and return ``(observation, info)``."""
        super().reset(seed=seed)
        self._build_layout(options or {})
        self.heading = self.spec.initial_heading
        self.steps = 0
        return self.observe(), self.info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the gymnasium 5-tuple."""
        try:
            motor = Action(int(action))
        except ValueError as exc:
            raise ValueError(f"invalid action {action!r}; expected one of {[int(a) for a in Action]}") from exc

        collided = False
        if motor is not Action.FORWARD:
            turn = motor.delta_heading
            self.heading = _HEADING_ORDER[(_HEADING_ORDER.index(self.heading) + turn) % len(_HEADING_ORDER)]
        else:
            dx, dy = self.heading_vector
            candidate = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if not self.spec.in_bounds(candidate) or candidate in self.obstacles:
                collided = True
            else:
                self.agent_pos = candidate

        self.steps += 1
        reached_goal = self.agent_pos == self.target_pos
        reward = float(self.spec.step_penalty)
        if collided:
            reward += float(self.spec.collision_penalty)
        if reached_goal:
            reward += float(self.spec.goal_reward)

        terminated = reached_goal
        truncated = (not terminated) and self.steps >= self.spec.max_steps
        return self.observe(), reward, terminated, truncated, self.info(success=reached_goal)

    # ----------------------------------------------------------- observation

    def observe(self) -> np.ndarray:
        """Flattened one-hot grid encoding plus position and heading."""
        planes = self.spec.size
        observation = np.zeros(self.observation_dim, dtype=np.float32)

        observation[self._flat(self.agent_pos, 0)] = 1.0
        observation[self._flat(self.target_pos, planes)] = 1.0
        for obstacle in self.obstacles:
            observation[self._flat(obstacle, 2 * planes)] = 1.0

        tail = 3 * planes
        scale = np.array([self.spec.width - 1, self.spec.height - 1], dtype=np.float32)
        observation[tail : tail + 2] = np.asarray(self.agent_pos, dtype=np.float32) / scale
        observation[tail + 2 : tail + 4] = np.asarray(self.target_pos, dtype=np.float32) / scale
        index = _HEADING_ORDER.index(self.heading)
        angle = 2.0 * np.pi * index / len(_HEADING_ORDER)
        observation[tail + 4] = np.sin(angle)
        observation[tail + 5] = np.cos(angle)
        return observation

    def grid(self) -> np.ndarray:
        """``(height, width)`` int array of cell codes, for rendering and tests.

        ``0`` free, ``1`` obstacle, ``2`` target, ``3`` agent.
        """
        cells = np.zeros((self.spec.height, self.spec.width), dtype=np.int8)
        for obstacle in self.obstacles:
            cells[obstacle[1], obstacle[0]] = 1
        cells[self.target_pos[1], self.target_pos[0]] = 2
        cells[self.agent_pos[1], self.agent_pos[0]] = 3
        return cells

    def info(self, success: bool = False) -> dict[str, Any]:
        """Diagnostics for logging, plotting and tests."""
        return {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "heading": self.heading,
            "heading_vector": self.heading_vector,
            "steps": self.steps,
            "num_obstacles": self.num_obstacles,
            "success": bool(success),
        }

    # ------------------------------------------------------------- internals

    def _flat(self, cell: Coordinate, offset: int) -> int:
        return offset + int(cell[1]) * self.spec.width + int(cell[0])

    def _build_layout(self, options: Mapping[str, Any]) -> None:
        """Place obstacles, agent and target for a new episode."""
        spec = self.spec
        self.agent_pos = spec.clamp(options.get("agent_start") or spec.agent_start or (0, 0))
        self.target_pos = spec.clamp(options.get("target") or spec.target or (spec.width - 1, spec.height - 1))

        explicit = "obstacles" in options
        if explicit or not spec.randomize_layout:
            self.obstacles = {spec.clamp(cell) for cell in (options.get("obstacles") or spec.obstacles)}
        else:
            self.obstacles = self._sample_obstacles()

        self.obstacles.discard(self.agent_pos)
        self.obstacles.discard(self.target_pos)

    def _sample_obstacles(self) -> set[Coordinate]:
        spec = self.spec
        reserved = {self.agent_pos, self.target_pos}
        candidates = [
            (x, y) for x in range(spec.width) for y in range(spec.height) if (x, y) not in reserved
        ]
        wanted = min(int(spec.num_obstacles), len(candidates))
        if wanted <= 0:
            return set()
        picks = self.np_random.choice(len(candidates), size=wanted, replace=False)
        return {candidates[int(pick)] for pick in np.atleast_1d(picks)}

    # ---------------------------------------------------------------- render

    def render(self) -> str:
        """Return an ASCII view of the grid (headless friendly)."""
        return self.render_ascii()

    def render_ascii(self) -> str:
        glyphs = {0: ".", 1: "#", 2: "T", 3: "A"}
        cells = self.grid()
        border = "+" + "-" * self.spec.width + "+"
        rows = [border]
        for y in range(self.spec.height):
            rows.append("|" + "".join(glyphs[int(cells[y, x])] for x in range(self.spec.width)) + "|")
        rows.append(border)
        arrow = _HEADING_GLYPHS[self.heading]
        rows.append(
            f"agent={self.agent_pos} target={self.target_pos} "
            f"heading={self.heading}{arrow} step={self.steps}"
        )
        return "\n".join(rows)

    def close(self) -> None:  # pragma: no cover - gymnasium API completeness
        return None


def make_env(
    config: Mapping[str, Any] | GridSpec | None = None,
    seed: int | None = None,
) -> FlyBrainNavEnv:
    """Build the environment from a config mapping or a :class:`GridSpec`.

    Accepts either a whole config object (the ``navigation`` section is read) or
    the navigation section on its own.
    """
    if isinstance(config, GridSpec):
        env = FlyBrainNavEnv(config)
        env.reset(seed=seed)
        return env

    mapping: Mapping[str, Any] = config or {}
    section: Mapping[str, Any] = mapping.get("navigation", mapping) if isinstance(mapping, Mapping) else {}

    known = {
        "width",
        "height",
        "num_obstacles",
        "max_steps",
        "randomize_layout",
        "initial_heading",
        "obstacles",
        "agent_start",
        "target",
    }
    kwargs: dict[str, Any] = {str(key): value for key, value in section.items() if str(key) in known}
    for key, value in dict(section.get("reward") or {}).items():
        if key in _REWARD_FIELDS:
            kwargs[_REWARD_FIELDS[key]] = value

    if seed is None and isinstance(mapping, Mapping):
        seed = mapping.get("project", {}).get("seed")

    env = FlyBrainNavEnv(GridSpec(**kwargs))
    env.reset(seed=int(seed) if seed is not None else None)
    return env
