"""Configuration loading and path resolution.

Every module takes its parameters from a dataclass that can be built from this
config object, so there are no hardcoded paths or magic numbers in the source.

Paths in the YAML file are relative to the repository root (the directory
containing ``pyproject.toml``) and are resolved lazily by :meth:`Config.path`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "CONFIG_FILENAME",
    "Config",
    "ConfigError",
    "find_project_root",
    "load_config",
    "load_yaml",
]

CONFIG_FILENAME = "default.yaml"
_ROOT_MARKERS = ("pyproject.toml", ".git")


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or malformed."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards looking for a repository marker file.

    Falls back to the current working directory so the package still works when
    installed outside a repository (for example inside Colab).
    """
    candidates = [Path(start)] if start is not None else [Path(__file__).resolve().parent, Path.cwd()]
    for candidate in candidates:
        resolved = candidate.resolve()
        for directory in (resolved, *resolved.parents):
            if any((directory / marker).exists() for marker in _ROOT_MARKERS):
                return directory
    return Path.cwd()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level of {file_path}, got {type(data).__name__}")
    return data


def _split_key(key: str) -> list[str]:
    return [part for part in key.split(".") if part]


@dataclass
class Config:
    """A read-mostly view over the parsed configuration.

    Supports dotted access (``config.get("brain.neuron.tau_membrane")``) and
    lazy, root-relative path resolution (``config.path("paths.outputs")``).
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=Path.cwd)
    source: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str | Path | None = None, root: Path | None = None) -> "Config":
        """Load a config file, defaulting to ``configs/default.yaml``."""
        project_root = Path(root).resolve() if root is not None else find_project_root()
        if path is None:
            config_path = project_root / "configs" / CONFIG_FILENAME
            if not config_path.is_file():
                raise ConfigError(
                    f"no config found at {config_path}; pass an explicit path or run from the repository"
                )
        else:
            config_path = Path(path)
            if not config_path.is_absolute():
                config_path = project_root / config_path
        return cls(data=load_yaml(config_path), root=project_root, source=config_path)

    # ------------------------------------------------------------------ read

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at a dotted ``key``, or ``default`` if absent."""
        node: Any = self.data
        for part in _split_key(key):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def require(self, key: str) -> Any:
        """Return the value at ``key`` or raise :class:`ConfigError`."""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required config key: {key!r}")
        return value

    def section(self, key: str) -> dict[str, Any]:
        """Return a mapping at ``key``, or an empty dict if absent."""
        value = self.get(key, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration data."""
        return copy.deepcopy(dict(self.data))

    # ----------------------------------------------------------------- paths

    def path(self, key: str) -> Path:
        """Resolve a configured, relative path against the project root."""
        value = self.get(key)
        if value is None:
            raise ConfigError(f"path key {key!r} is not set (or is null) in the config")
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (self.root / candidate)

    def ensure_path(self, key: str) -> Path:
        """Resolve a path and create the parent directory if needed."""
        target = self.path(key)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def ensure_dir(self, key: str) -> Path:
        """Resolve a path, treat it as a directory, and create it."""
        target = self.path(key)
        target.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------- utilities

    @property
    def seed(self) -> int:
        """Project-wide default seed."""
        return int(self.get("project.seed", 0))

    def child(self, key: str, root: Path | None = None) -> "Config":
        """Return a config rooted at ``root`` with the same data."""
        return Config(data=self.to_dict(), root=Path(root) if root is not None else self.root, source=self.source)

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "Config":
        """Return a copy with dotted-key overrides applied."""
        merged = self.to_dict()
        for key, value in (overrides or {}).items():
            _set_dotted(merged, key, value)
        return Config(data=merged, root=self.root, source=self.source)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key, _MISSING) is not _MISSING


_MISSING = object()


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    parts = _split_key(key)
    if not parts:
        raise ConfigError("empty config key")
    node: dict[str, Any] = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    """Convenience wrapper around :meth:`Config.load`."""
    return Config.load(path=path, root=root)
